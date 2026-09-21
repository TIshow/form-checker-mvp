#!/usr/bin/env python3
"""クリップの計測値から、根拠付きの改善点を生成AIに書かせて保存する。

    python tools/coach.py --clip golf_gvhmr            # → web/data/golf_gvhmr/coach.json
    python tools/coach.py --clip golf_gvhmr --dry-run  # プロンプトを表示するだけ

ANTHROPIC_API_KEY が要る（--dry-run は不要）。根拠表は domains/<domain>_evidence.py。
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import coach  # noqa: E402

WEB = Path(__file__).resolve().parents[1] / "web" / "data"


def load_evidence(domain: str) -> list[dict]:
    mod = importlib.import_module(f"domains.{domain}_evidence")
    return list(mod.EVIDENCE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, help="web/data/<clip>/clip.json")
    ap.add_argument("--model", default=coach.DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    clip = json.loads((WEB / args.clip / "clip.json").read_text())
    domain = clip["domain"]
    note = (f"復元: {clip.get('source', '?')}, {clip['metrics'].get('n_frames')}フレーム @{clip['fps']}fps, "
            f"座標 {clip.get('coords', '?')}. " + (clip.get("note") or ""))
    metrics, extra = dict(clip["metrics"]), {}
    if clip.get("audio"):
        # 音声の要約も計測値として渡す（キーは audio_ を前置）
        for h in clip["audio"]["summary"]:
            metrics[f"audio_{h['key']}"] = h["value"]
            extra[f"audio_{h['key']}"] = (h["label"], h["unit"], h["digits"], h["note"])
        note += f" 音声: {clip['audio']['note']}"
    inp = coach.CoachInput(args.clip, domain, metrics, load_evidence(domain), note, extra)
    if args.dry_run:
        print(coach.SYSTEM); print("-" * 60); print(coach.build_prompt(inp)); return
    out = coach.run(inp, model=args.model)
    dest = WEB / args.clip / "coach.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"✅ {dest}  改善点 {len(out['improvements'])} / 観察 {len(out['observations'])} / 降格 {out['demoted']}")


if __name__ == "__main__":
    main()
