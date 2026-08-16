#!/usr/bin/env python3
"""1回の練習（多数のサーブ）をまとめて集計する（issue #10）。

    python tools/make_session.py \
        --recon ~/.../recon --labels ~/.../clips/labels.csv \
        --fps 60 --out web/data/session.json

## なぜ必要か

これまでの数字はすべて**1本のサーブ**から出していた。「沈み込み膝角96°」と
言っても、それが典型なのかその1本だけなのかを判断する材料が無かった。

多数本あると、1本では原理的に分からない3つが見える。

1. **ばらつき** — 中央値と範囲。あなたの典型値と振れ幅
2. **測定の誤差** — 同じ人が同じ動作をしてどれだけ振れるか。指標ごとの安定度。
   これまで**どの指標にも誤差の幅が無かった**ので、初めて誤差棒がつく
3. **結果との関係** — 入った/入らなかったでフォームに差があるか

## 3 について、言えることと言えないこと

同一人物・同一セッションなので、体格も技術レベルも道具も天候も揃っている。
交絡がほぼ無い状態での比較になる。それでも:

- 「あなたの場合、入ったサーブの方が膝が深い**傾向**」→ 言える
- 「膝を深くすれば入る**ようになる**」→ 言えない（介入実験が要る）

指標を10個近く並べて差を探すと、偶然どれかに差が出る。順位和検定の p 値は
出すが、**Holm 法で多重比較を補正**し、補正後も残ったものだけを「差がある」
と扱う。それでも探索的な結果であって、確認された関係ではない。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 同じ tools/ の中を素の名前で import している。スクリプトとして
# 走らせるときは通るが、tools.make_session として import されると
# 通らない（テストがこれで落ちた）。明示的に足しておく。
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis  # noqa: E402
from analysis.serve import (  # noqa: E402
    FOOT_IDS, L_HAND, R_HAND, ServeKinematics, detect_up_axis,
)
from make_compare import PREFIXES, sanitize  # noqa: E402

# 集計する指標。(表示名, 取り出し方, 単位, 小数点)
METRICS = [
    ("沈み込み膝角", lambda m, x: m["min_knee_deg_overall"], "°", 0),
    ("打点の肘角", lambda m, x: m["elbow_at_contact_deg"], "°", 0),
    ("打点の体幹", lambda m, x: m["trunk_lean_at_contact_deg"], "°", 0),
    ("ラケットドロップ", lambda m, x: x["racket_drop_deg"], "°", 0),
    ("打点の高さ", lambda m, x: m["contact_height_m"], "m", 2),
    ("重心の伸び上がり", lambda m, x: m["com_rise_m"] * 100, "cm", 1),
    ("跳躍（足の浮き）", lambda m, x: x["foot_clearance_cm"], "cm", 1),
    ("駆動時間", lambda m, x: x["drive_s"], "秒", 2),
    ("打点タイミング", lambda m, x: m["contact_vs_compeak_s"], "秒", 2),
]


def extra(J: np.ndarray, fps: float) -> dict:
    """analysis に入っていない指標をここで足す。

    ラケットドロップは手法比較で目視評価と順位一致した唯一の指標だった
    （GVHMR 116° > TRAM 72° ≒ GEM-X 72°）。サーブの核心なので必ず載せる。
    """
    k = ServeKinematics(J, fps)
    ph = k.detect_phases()
    lo, ct = ph["loading"], ph["contact"]
    ax, sg = detect_up_axis(J)
    up = np.zeros(3)
    up[ax] = sg

    wr = k.idx("wrist")
    hd = R_HAND if k.racket_side == "R" else L_HAND
    v = J[:, hd] - J[:, wr]
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(v @ up, -1, 1)))

    feet = (J[..., ax] * sg)[:, FOOT_IDS].min(axis=1)
    ground = float(np.median(feet))
    return {
        "racket_drop_deg": round(float(ang[lo:ct + 1].max()), 1),
        "foot_clearance_cm": round(float(feet[lo:].max() - ground) * 100, 1),
        "drive_s": round((ct - lo) / fps, 3),
        "racket_side": k.racket_side,
    }


def mann_whitney_p(a: np.ndarray, b: np.ndarray) -> float:
    """順位和検定の両側 p（正規近似・同順位補正あり）。

    scipy を入れていないので自前。n が各群10以上あれば正規近似で足りる。
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return float("nan")
    both = np.concatenate([a, b])
    order = both.argsort()
    ranks = np.empty(len(both), float)
    ranks[order] = np.arange(1, len(both) + 1)
    # 同順位は平均順位に均す
    _, inv, cnt = np.unique(both, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    u_a = ranks[:na].sum() - na * (na + 1) / 2
    u = min(u_a, na * nb - u_a)
    mu = na * nb / 2
    ties = sum(c ** 3 - c for c in cnt)
    n = na + nb
    var = na * nb / 12 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return float("nan")
    # 連続性補正の 0.5 は、差が小さいと |u-mu| を下回って z が負になる。
    # そのまま両側 p を出すと 1 を超える（実測 1.03）。0 で止める。
    z = max(0.0, abs(u - mu) - 0.5) / math.sqrt(var)
    return float(2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))))


def holm(pvals: list[float]) -> list[float]:
    """Holm 法で多重比較を補正する。

    指標を10個近く並べて差を探せば、偶然どれかに差が出る。補正しない p を
    そのまま「差がある」と読むと、無い関係を報告することになる。
    """
    idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    order = sorted(idx, key=lambda i: pvals[i])
    out = [float("nan")] * len(pvals)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (len(order) - rank) * pvals[i])
        prev = max(prev, adj)          # 単調性を保つ
        out[i] = round(prev, 4)
    return out


def describe(v: np.ndarray) -> dict:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return {}
    q1, q3 = np.percentile(v, [25, 75])
    return {
        "n": int(len(v)),
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "q1": float(q1), "q3": float(q3),
        "min": float(v.min()), "max": float(v.max()),
        # 変動係数。単位の違う指標どうしで「安定度」を比べるため
        "cv": float(v.std(ddof=1) / abs(v.mean())) if len(v) > 1 and v.mean() else None,
        "values": [round(float(x), 4) for x in v],
    }


def load(recon_dir: Path, labels_csv: Path, fps: float) -> dict:
    with open(labels_csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(l for l in f if not l.startswith("#"))]

    serves, missing = [], []
    for r in rows:
        n = int(r["serve"])
        result = (r.get("result") or "").strip().lower()
        note = (r.get("note") or "").strip()
        if not result or "not serve" in note.lower():
            continue
        d = recon_dir / f"serve_{n:02d}"
        f = next((d / f"{p}joints.npy" for p in PREFIXES
                  if (d / f"{p}joints.npy").exists()), None)
        if f is None:
            missing.append(n)
            continue
        J = np.load(f)
        res = analysis.analyze_json(J, fps)
        x = extra(J, fps)
        serves.append({
            "serve": n, "result": result, "video": r.get("video", ""),
            "note": note, "n_frames": int(len(J)),
            "phases": res["metrics"]["phases"],
            "metrics": res["metrics"], "extra": x,
            "values": {name: get(res["metrics"], x)
                       for name, get, _, _ in METRICS},
            # 関節列は重い（42本ぶんで40MB超）。個別ファイルに分け、
            # ビューアが選んだときだけ読む。
            "_joints": res["joints"], "_up_axis": res["up_axis"],
        })

    if missing:
        print(f"⚠️ 復元が見つからない: {missing}")
    return {"serves": serves, "missing": missing, "fps": fps}


def summarize(data: dict) -> dict:
    serves = data["serves"]
    ins = [s for s in serves if s["result"] == "in"]
    others = [s for s in serves if s["result"] != "in"]

    rows, praw = [], []
    for name, _, unit, dp in METRICS:
        all_v = np.array([s["values"][name] for s in serves], float)
        a = np.array([s["values"][name] for s in ins], float)
        b = np.array([s["values"][name] for s in others], float)
        p = mann_whitney_p(a, b) if len(a) >= 3 and len(b) >= 3 else float("nan")
        praw.append(p)
        rows.append({
            "name": name, "unit": unit, "dp": dp,
            "all": describe(all_v), "in": describe(a), "other": describe(b),
            "diff": (float(np.median(a) - np.median(b))
                     if len(a) and len(b) else None),
            "p_raw": None if math.isnan(p) else round(p, 4),
        })
    for row, adj in zip(rows, holm(praw)):
        row["p_holm"] = None if math.isnan(adj) else adj

    return {
        "n_total": len(serves),
        "n_in": len(ins), "n_other": len(others),
        "result_labels": sorted({s["result"] for s in serves}),
        "metrics": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="1セッションのサーブをまとめて集計")
    ap.add_argument("--recon", required=True, help="serve_NN/ が並ぶディレクトリ")
    ap.add_argument("--labels", required=True, help="labels.csv")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--out", default="web/data/session.json")
    ap.add_argument("--label", default="", help="セッション名（表示用）")
    args = ap.parse_args()

    data = load(Path(os.path.expanduser(args.recon)),
                Path(os.path.expanduser(args.labels)), args.fps)
    if not data["serves"]:
        raise SystemExit("集計できるサーブがありません")

    out = {
        "label": args.label or Path(args.recon).parent.name,
        "fps": args.fps,
        "summary": summarize(data),
        "serves": data["serves"],
        "missing": data["missing"],
    }
    # 骨格は1本ずつ別ファイルへ。まとめると40MB超になり、集計を見るだけでも
    # 全部読み込むことになる。
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    sub = p.parent / "serves"
    sub.mkdir(exist_ok=True)
    for sv in out["serves"]:
        one = {"serve": sv["serve"], "result": sv["result"],
               "label": f"#{sv['serve']} ({sv['result']})",
               "fps": args.fps, "up_axis": sv.pop("_up_axis"),
               "joints": sv.pop("_joints"),
               "metrics": sv["metrics"]}
        c, _ = sanitize(one)
        (sub / f"serve_{sv['serve']:02d}.json").write_text(
            json.dumps(c, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        sv.pop("metrics", None)        # 集計側では使わない。JSONを小さく保つ
    print(f"✅ {sub}/serve_NN.json  ({len(out['serves'])}本)")

    clean, nans = sanitize(out)
    p.write_text(json.dumps(clean, ensure_ascii=False, allow_nan=False),
                 encoding="utf-8")

    s = out["summary"]
    print(f"✅ {p}  ({p.stat().st_size/1e6:.1f} MB)")
    print(f"   {s['n_total']}本  in={s['n_in']}  その他={s['n_other']}\n")
    w = 18
    print(f"{'指標':<{w}} {'中央値':>9} {'範囲':>15} {'CV':>7} "
          f"{'in−他':>8} {'p(補正後)':>9}")
    print("─" * (w + 52))
    for m in s["metrics"]:
        a = m["all"]
        if not a:
            continue
        d = m["dp"]
        rng = f"{a['min']:.{d}f}〜{a['max']:.{d}f}"
        cv = f"{a['cv']*100:.0f}%" if a.get("cv") else "—"
        diff = f"{m['diff']:+.{d}f}" if m["diff"] is not None else "—"
        ph = m.get("p_holm")
        mark = " *" if ph is not None and ph < 0.05 else ""
        print(f"{m['name']:<{w}} {a['median']:9.{d}f} {rng:>15} {cv:>7} "
              f"{diff:>8} {('—' if ph is None else f'{ph:.3f}'):>9}{mark}")
    if nans:
        print(f"\n   ℹ️ null にした項目: {', '.join(nans[:5])}")
    print("\n* は Holm 補正後も p<0.05。ただし探索的な結果であって、"
          "確認された関係ではない。")


if __name__ == "__main__":
    main()
