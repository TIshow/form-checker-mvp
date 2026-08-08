#!/usr/bin/env python3
"""復元手法の比較ページ用データを作る（issue #9）。

    python tools/make_models.py \
        --clip GVHMR=output \
        --clip GEM-X=output_gemx_mine \
        --clip TRAM=output_tram_mine \
        --fps 60 --out web/data/models.json

`make_compare.py` が「2本の動画を比べる」道具なのに対し、こちらは
**同じ動画を別々の手法で復元した結果**を並べる。用途が違うので分けてある:

- 同じクリップなのでフレーム番号が共通。スライダーは1本でよい
- 体格差の話は出てこない（同じ人）。差は全部、手法の差
- 判定の決め手になった**ラケットドロップ**を必ず載せる

## ラケットドロップをここで計算している理由

手首→手のベクトルが鉛直からどれだけ倒れるかで測る（手の向きがラケットの
シャフト方向の近似になる）。0°=真上、180°=真下。サーブでは打点前に背中側へ
落ち込み、そこから振り抜く。

実測では、この値の順位が目視評価の順位と一致した:

    GVHMR 116°（最も再現度が高い） > TRAM 72° = GEM-X 72°（劣る）

一方、腕の高さ・上腕の傾き・肘角はどれも逆の順位を示し、判断を誤らせた。
まだ analysis 側の正式な指標にはしていないので、ここで計算している。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis  # noqa: E402
from analysis.serve import (  # noqa: E402
    FOOT_IDS, L_HAND, R_HAND, ServeKinematics, detect_up_axis,
)
from make_compare import PREFIXES, sanitize  # noqa: E402


def racket_drop(J: np.ndarray, fps: float) -> dict:
    """ラケットヘッドの落ち込み。手首→手の向きが鉛直から倒れる角度[deg]。"""
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
    seg = ang[lo:ct + 1]
    return {
        "series": [round(float(a), 1) for a in ang],
        "max_deg": round(float(seg.max()), 1),
        "max_frame": int(seg.argmax()) + lo,
        "at_contact_deg": round(float(ang[ct]), 1),
    }


def foot_clearance(J: np.ndarray, fps: float) -> dict:
    """跳躍。床（最も低い足の中央値）から、足がどれだけ浮くか[cm]。"""
    k = ServeKinematics(J, fps)
    lo = k.detect_phases()["loading"]
    ax, sg = detect_up_axis(J)
    feet = (J[..., ax] * sg)[:, FOOT_IDS].min(axis=1)
    ground = float(np.median(feet))
    return {"max_cm": round(float(feet[lo:].max() - ground) * 100, 1)}


def _pick(d: Path) -> Path:
    for p in PREFIXES:
        f = d / f"{p}joints.npy"
        if f.exists():
            return f
    raise FileNotFoundError(f"{d} に *_joints.npy がありません")


def bundle(label: str, joints_dir: str, fps: float) -> dict:
    J = np.load(_pick(Path(joints_dir)))
    res = analysis.analyze_json(J, fps)
    res["label"] = label
    res["fps"] = fps
    res["source"] = joints_dir
    res["racket_drop"] = racket_drop(J, fps)
    res["foot_clearance"] = foot_clearance(J, fps)
    res.pop("feedback", None)      # 手法比較には要らない。JSONを小さくする
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="復元手法の比較データを作る")
    ap.add_argument("--clip", action="append", required=True,
                    metavar="ラベル=ディレクトリ",
                    help="例: GVHMR=output （複数回指定する）")
    ap.add_argument("--fps", type=float, required=True,
                    help="全手法に共通の撮影fps（同じ動画なので1つ）")
    ap.add_argument("--out", default="web/data/models.json")
    args = ap.parse_args()

    bundles = []
    for spec in args.clip:
        if "=" not in spec:
            raise SystemExit(f"--clip は ラベル=ディレクトリ の形で: {spec}")
        label, _, d = spec.partition("=")
        bundles.append(bundle(label, d, args.fps))

    n = {b["metrics"]["n_frames"] for b in bundles}
    if len(n) > 1:
        print(f"⚠️ フレーム数が揃っていません {n}。同じ動画の結果か確認してください")

    clean, nans = sanitize(bundles)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(clean, ensure_ascii=False, allow_nan=False),
                   encoding="utf-8")
    print(f"✅ {out}  ({out.stat().st_size / 1e6:.1f} MB)  {len(bundles)}手法")
    for b in bundles:
        ph = b["metrics"]["phases"]
        print(f"   {b['label']:8s} {b['metrics']['n_frames']}フレーム  "
              f"沈み込み{ph['loading']} → 打点{ph['contact']}  "
              f"ドロップ{b['racket_drop']['max_deg']:5.1f}°  "
              f"跳躍{b['foot_clearance']['max_cm']:+5.1f}cm")
    if nans:
        print(f"   ℹ️ null にした項目: {', '.join(nans[:6])}")


if __name__ == "__main__":
    main()
