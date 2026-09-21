"""計測値と根拠表から、生成AIに「改善点」を書かせる（根拠の tier を持ち込む）。

## なぜ判定ルール（`Domain.judge`）と別に置くか

`judge` は閾値を持つ決定的なコードで、閾値の正しさは出典で縛る（domains/README）。
一方、利用者が欲しいのは「この数字だから、こう直す」という**文章**で、
指標どうしの関係や順番の説明は決定的コードで書ききれない。そこを生成AIに任せる。

ただし生成AIは根拠を捏造する。だからここでは:

- 渡す材料を **計測値** と **根拠表**（`domains/<domain>_evidence.py`）に限る
- 「改善点」と書いてよいのは、根拠表の **Tier A / B** の項目に結びついたものだけ
- Tier C の項目は「観察」として数値を述べるだけ。欠点と呼ばない
- 出力は JSON にし、`validate()` が根拠 id と tier を機械的に照合する。
  照合に通らない項目は捨てる（文章の良し悪しに関係なく）
- 「プロと比べて」は言わない。比較対象が無いことを本文に明記させる

生成AIの出力は `web/data/<clip>/coach.json` に保存し、`clip.html` が表示する。
判定コード（`judge`）の finding とは別物として扱い、混ぜない。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date

from domains import get as get_domain
from domains.base import TIER_A, TIER_B

DEFAULT_MODEL = "claude-opus-5"

SYSTEM = """あなたはスポーツバイオメカニクスの助言者です。日本語で書きます。
与えられた「計測値」と「根拠表」だけを材料にします。根拠表に無い数値・文献・
プロの基準値を持ち出してはいけません。「プロと比べて」という言い方は禁止です。
比較対象が無いことを前提に、身体の使い方の効率と傷害リスクの観点で述べます。

出力は次の JSON だけ（前後に文章を付けない）:
{
  "improvements": [   // 根拠表の tier が A か B の項目に結びつくものだけ
    {"title": "短い見出し", "metric": "指標キー", "observed": "観測値と局面",
     "evidence_id": "根拠表の id", "why": "その数字がなぜ非効率/リスクか（根拠表の範囲で）",
     "advice": "何をどう変えるか。動作の言葉で", "confidence": "high|medium|low"}
  ],
  "observations": [   // tier C の項目、または根拠が無い数値。欠点と呼ばない
    {"metric": "指標キー", "observed": "観測値", "note": "何が言えて何が言えないか"}
  ],
  "not_measured": ["この計測では取れていないが、判断に要るもの"],
  "caveats": ["復元手法や撮影条件による限界"]
}
evidence_id は根拠表にある id をそのまま使うこと。1つの改善点に1つの id。
根拠表の verified が "未確認" の項目は improvements に使わず observations に回すこと。
"""


@dataclass(frozen=True)
class CoachInput:
    clip_name: str
    domain: str
    metrics: dict
    evidence: list[dict]
    processing_note: str = ""


def _metric_lines(domain: str, metrics: dict) -> list[str]:
    d = get_domain(domain)
    labels = getattr(d, "metric_labels", {})
    lines = []
    for key, value in metrics.items():
        if key in ("domain", "phases", "kinetic_chain", "phases_note"):
            continue
        if isinstance(value, bool) or value is None:
            lines.append(f"- {key}: {value}")
        elif isinstance(value, (int, float)):
            lab = labels.get(key)
            shown = "測定できません" if value != value else f"{value:.2f}"
            lines.append(f"- {key}: {shown}" + (f"  （{lab[0]}, {lab[1]}, {lab[3]}）" if lab else ""))
    ph = metrics.get("phases") or {}
    fps = metrics.get("fps") or 30.0
    if ph:
        pl = getattr(d, "phase_labels", {})
        lines.append("- phases: " + ", ".join(f"{pl.get(k, k)}=f{v}({v / fps:.2f}s)" for k, v in ph.items()))
    kc = metrics.get("kinetic_chain")
    if kc:
        lines.append(f"- kinetic_chain: {json.dumps(kc, ensure_ascii=False)}")
    if metrics.get("phases_note"):
        lines.append(f"- 注意: {metrics['phases_note']}")
    return lines


def _evidence_lines(evidence: list[dict]) -> list[str]:
    out = []
    for e in evidence:
        out.append(
            f"- id={e['id']} tier={e['tier']} verified={e['verified']}\n"
            f"  主張: {e['claim']}\n"
            f"  関係する指標: {', '.join(e['metrics']) or '（直接の指標なし）'}\n"
            f"  出典: {e['source']}" + (f"（{e['location']}）" if e.get("location") else "") +
            (f"\n  注意: {e['note']}" if e.get("note") else ""))
    return out


def build_prompt(inp: CoachInput) -> str:
    return "\n".join([
        f"## 対象: {inp.clip_name}（ドメイン {inp.domain}）",
        "", "## 計測値（この映像から測ったもの。単位は指標名のとおり）",
        *_metric_lines(inp.domain, inp.metrics),
        "", "## 根拠表",
        *_evidence_lines(inp.evidence),
        "", "## 撮影・復元の条件", inp.processing_note or "（記載なし）",
        "", "上の材料だけで JSON を出力してください。",
    ])


def validate(raw: dict, evidence: list[dict]) -> dict:
    """根拠 id と tier を機械的に照合する。通らない改善点は観察に落とす。"""
    by_id = {e["id"]: e for e in evidence}
    kept, demoted = [], []
    for item in raw.get("improvements", []) or []:
        e = by_id.get(item.get("evidence_id"))
        ok = (e is not None and e["tier"] in (TIER_A, TIER_B) and e["verified"] != "未確認"
              and item.get("metric") in e["metrics"])
        (kept if ok else demoted).append(item)
    observations = list(raw.get("observations", []) or [])
    for item in demoted:
        observations.append({"metric": item.get("metric"), "observed": item.get("observed"),
                             "note": "（根拠の照合に通らなかったため観察に降格）" + (item.get("why") or "")})
    return {"improvements": kept, "observations": observations,
            "not_measured": list(raw.get("not_measured", []) or []),
            "caveats": list(raw.get("caveats", []) or []),
            "demoted": len(demoted)}


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("生成AIの出力に JSON がありません")
    return json.loads(m.group(0))


def run(inp: CoachInput, client=None, model: str = DEFAULT_MODEL) -> dict:
    """client は `anthropic.Anthropic()` 互換（messages.create）。テストでは偽物を渡す。"""
    if client is None:
        import anthropic  # 任意依存。pyproject の [coach]
        client = anthropic.Anthropic()
    prompt = build_prompt(inp)
    msg = client.messages.create(model=model, max_tokens=2000, system=SYSTEM,
                                 messages=[{"role": "user", "content": prompt}])
    text = "".join(getattr(b, "text", "") for b in msg.content)
    out = validate(parse_json(text), inp.evidence)
    out.update({"generated_by": model, "generated_at": date.today().isoformat(),
                "clip": inp.clip_name, "domain": inp.domain,
                "evidence_ids": [e["id"] for e in inp.evidence]})
    return out
