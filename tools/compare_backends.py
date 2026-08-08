#!/usr/bin/env python3
"""復元手法（GVHMR と GEM-X）を同じ物差しで比べる（issue #9）。

    python tools/compare_backends.py \
        --a output/gv_joints.npy       --a-label GVHMR \
        --b output_gemx_mine/gx_joints.npy --b-label GEM-X \
        --fps 60

## 何を見るか

lm 論文の指標（WA-MPJPE 等）は正解データが要るので手元では測れない。
代わりに、**この用途で困るかどうか**が分かる3つを見る。

1. **跳躍が出るか。** GVHMR は接地の事前分布が強く、跳んでいるのに足が
   地面から離れない（自分のクリップで 8.0cm、Zverev で 3.6cm しか浮かない）。
   サーブの解析では致命的。issue #8 の核心。

2. **足が滑らないか。** GENMO 論文では GVHMR が Foot-Sliding で2.5倍勝つ。
   跳躍とトレードオフの関係にあり、滑ると接地と体重移動の判定が濁る。
   ここでは**接地しているはずの局面**で足がどれだけ水平移動するかを見る。

3. **関節角が妥当か。** 膝・肘・体幹。ここが違えばフィードバックが変わる。

どちらが「正しい」かを機械的には決められない。**動画で見た事実**
（跳んでいるか、足が滑っていないか）と突き合わせる材料を出すのが目的。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.serve import (  # noqa: E402
    L_ANKLE, L_FOOT, R_ANKLE, R_FOOT, ServeKinematics, detect_up_axis,
)

FOOT_IDS = [L_ANKLE, R_ANKLE, L_FOOT, R_FOOT]


def measure(path: str, fps: float) -> dict:
    J = np.load(path)
    k = ServeKinematics(J, fps)
    ph = k.detect_phases()
    lo, ct = ph["loading"], ph["contact"]

    ax, sg = detect_up_axis(J)
    hz = [a for a in (0, 1, 2) if a != ax]
    up = J[..., ax] * sg

    # --- 1. 跳躍 ---
    feet = up[:, FOOT_IDS].min(axis=1)
    ground = float(np.median(feet))
    clearance = float(feet[lo:].max() - ground)   # 駆動以降の最大の浮き

    # --- 2. 足の滑り ---
    # 立位とみなせるフレーム（足が床±2cm）で、接地側の足がどれだけ水平に動くか。
    # 本当に接地しているなら、ほぼ動かないはず。
    planted = np.abs(feet - ground) < 0.02
    low_id = np.array(FOOT_IDS)[np.argmin(up[:, FOOT_IDS], axis=1)]
    pos = J[np.arange(len(J)), low_id][:, hz]
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    both = planted[:-1] & planted[1:]
    slide = float(step[both].sum() * 100) if both.any() else float("nan")
    slide_per_s = slide / (both.sum() / fps) if both.any() else float("nan")

    kn = k.knee_angles()
    el = k.elbow_angle()
    tr = k.trunk_lean()
    return {
        "frames": len(J), "fps": fps, "racket": k.racket_side,
        "loading": lo, "contact": ct, "drive_s": (ct - lo) / fps,
        "ground_m": ground,
        "jump_cm": clearance * 100,
        "slide_cm_per_s": slide_per_s,
        "planted_frames": int(both.sum()),
        "knee_min_deg": float(kn.min()),
        "knee_at_loading_deg": float(kn[lo]),
        "elbow_at_contact_deg": float(el[ct]),
        "trunk_at_contact_deg": float(tr[ct]),
        "com_rise_cm": float(k.com_height[lo:].max() - k.com_height[lo]) * 100,
    }


ROWS = [
    ("フレーム数", "frames", "{:.0f}"),
    ("利き手", "racket", "{}"),
    ("沈み込み → 打点", "loading", "{:.0f}"),
    ("　　（打点フレーム）", "contact", "{:.0f}"),
    ("駆動時間", "drive_s", "{:.2f} 秒"),
    ("① 跳躍（足の浮き）", "jump_cm", "{:+.1f} cm"),
    ("② 足の滑り", "slide_cm_per_s", "{:.1f} cm/秒"),
    ("　　（接地とみなしたフレーム）", "planted_frames", "{:.0f}"),
    ("③ 膝の最小角", "knee_min_deg", "{:.0f}°"),
    ("　　沈み込み時の膝角", "knee_at_loading_deg", "{:.0f}°"),
    ("　　打点時の肘角", "elbow_at_contact_deg", "{:.0f}°"),
    ("　　打点時の体幹", "trunk_at_contact_deg", "{:.0f}°"),
    ("重心の伸び上がり", "com_rise_cm", "{:.1f} cm"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="復元手法を同じ物差しで比べる")
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--a-fps", type=float)
    ap.add_argument("--b-fps", type=float)
    args = ap.parse_args()

    a = measure(args.a, args.a_fps or args.fps)
    b = measure(args.b, args.b_fps or args.fps)

    w = 30
    print(f"\n{'':<{w}} {args.a_label:>14} {args.b_label:>14}")
    print("─" * (w + 30))
    for label, key, fmt in ROWS:
        print(f"{label:<{w}} {fmt.format(a[key]):>14} {fmt.format(b[key]):>14}")
    print()
    print("① 跳躍: 動画で跳んでいるなら、値が大きい方が実態に近い")
    print("② 滑り: 接地しているはずの局面での移動量。小さいほど良い")
    print("③ 角度: どちらが正しいかは値だけでは決まらない。差が大きい項目は")
    print("   レンダ動画で目視確認すること")


if __name__ == "__main__":
    main()
