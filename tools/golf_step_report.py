#!/usr/bin/env python3
"""ゴルフの復元結果を、ステップ間で同じ物差しで並べる（issue 018）。

    python tools/golf_step_report.py output_golf_gemx/gx_joints.npy output_golf_gemx_fit/gx_joints.npy

引数は SMPL24 順の関節列 (F,24,3)。何本でも。GVHMR の参照（非商用・確認用）を
並べたければ `output_golf/gv_joints.npy` も渡す。

指標は「復元が壊れていないか」を見るためのもの。再投影誤差はここには無い
（それは合わせ込みの目的関数なので、下がって当然）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis  # noqa: E402
from core.anchor import anchor_feet  # noqa: E402
from core.geometry import joint_angle  # noqa: E402
from core.kinematics import detect_up_axis  # noqa: E402
from core.skeleton import (  # noqa: E402
    FOOT_IDS, L_ANKLE, L_HIP, L_KNEE, L_WRIST, PELVIS, R_ELBOW, R_SHOULDER, R_WRIST,
)

COLUMNS = [
    ("phases", "局面 tk/top/imp/fin"),
    ("grip_sd", "両手首SD[cm]"),
    ("grip_top", "両手首 トップ"),
    ("grip_impact", "両手首 インパクト"),
    ("grip_finish", "両手首 フィニッシュ"),
    ("knee", "リード膝 imp[°]"),
    ("feet_dh", "足の高さ差 最大[cm]"),
    ("slide", "足の滑り[m]"),
    ("hand_peak", "手の最高速[m/s]@f"),
    ("elbow_jitter", "右肘ジッタ[°/f²]"),
    ("head_move", "頭の上下動[cm]"),
]


def stats(J: np.ndarray, fps: float) -> dict:
    up, sg = detect_up_axis(J)
    m = analysis.analyze_json(J, fps, "golf_swing", anchor=False)["metrics"]
    ph = m["phases"]
    wr = np.linalg.norm(J[:, L_WRIST] - J[:, R_WRIST], axis=-1) * 100
    knee = joint_angle(J[:, L_HIP], J[:, L_KNEE], J[:, L_ANKLE])
    elb = joint_angle(J[:, R_SHOULDER], J[:, R_ELBOW], J[:, R_WRIST])
    feet_h = J[:, FOOT_IDS, up] * sg
    dh = np.abs(np.minimum(feet_h[:, 0], feet_h[:, 2]) - np.minimum(feet_h[:, 1], feet_h[:, 3])) * 100
    hands = (J[:, L_WRIST] + J[:, R_WRIST]) / 2 - J[:, PELVIS]
    sp = np.linalg.norm(np.diff(hands, axis=0), axis=-1) * fps
    _, info = anchor_feet(J, fps)
    fin = slice(ph["finish"] + 5, min(ph["finish"] + 18, len(J)))
    return {
        "phases": f"{ph['takeaway']}/{ph['top']}/{ph['impact']}/{ph['finish']}",
        "grip_sd": f"{m['grip_spread_sd_cm']:.1f}",
        "grip_top": f"{wr[ph['top']]:.0f}",
        "grip_impact": f"{wr[ph['impact']]:.0f}",
        "grip_finish": f"{wr[fin].min():.0f}-{wr[fin].max():.0f}",
        "knee": f"{knee[ph['impact']]:.0f}",
        "feet_dh": f"{dh.max():.0f}",
        "slide": f"{info['slide_before_m']:.2f}",
        "hand_peak": f"{sp.max():.1f}@{int(sp.argmax()) + 1}",
        "elbow_jitter": f"{np.abs(np.diff(elb, n=2)).mean():.2f}",
        "head_move": f"{m['head_move_cm']:.1f}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("joints", nargs="+", help="SMPL24 順の関節列 .npy（複数可）")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    rows = {Path(p).parent.name or p: stats(np.load(p), args.fps) for p in args.joints}
    width = max(len(n) for n in rows) + 2
    print(" " * width + "".join(f"{label:>20s}" for _, label in COLUMNS))
    for name, r in rows.items():
        print(f"{name:{width}s}" + "".join(f"{r[key]:>20s}" for key, _ in COLUMNS))


if __name__ == "__main__":
    main()
