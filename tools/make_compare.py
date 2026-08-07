#!/usr/bin/env python3
"""比較ビューア用のデータを作る（issue #7）。

    python tools/make_compare.py \
        --mine output --mine-fps 60 \
        --ref  output_zverev --ref-fps 30 \
        --out  web/data

各クリップについて、指標・フェーズ・関節列・回転を1つのJSONにまとめる。
関節「位置」ではなく「回転」も出すのが要点で、これを同じアバターに流し込めば
体格差が消え、残る差は技術の差だけになる。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis  # noqa: E402
from analysis.serve import (  # noqa: E402
    L_ANKLE, L_FOOT, R_ANKLE, R_FOOT, detect_up_axis,
)


def quality(joints: np.ndarray, phases: dict) -> dict:
    """この復元が比較に使えるかの目安。UI で警告を出すために持たせる。

    復元が壊れていても数値は出てしまうので、素材の側の問題を
    「解析の結果」と取り違えないよう、判断材料を一緒に運ぶ。
    """
    ax, sg = detect_up_axis(joints)
    feet = joints[:, [L_ANKLE, R_ANKLE, L_FOOT, R_FOOT]][..., ax] * sg
    standing = float(np.median(feet.min(axis=1)))
    span = int(phases["contact"] - phases["loading"])

    issues = []
    # GVHMR は地面を y≈0 に置く。立位の足がそこから離れていれば追跡が怪しい。
    if standing > 0.15:
        issues.append(f"復元が接地していない（立位の足が {standing*100:.0f}cm 浮いている）")
    if span < 3:
        issues.append(f"沈み込みと打点がほぼ同じフレーム（{span}フレーム差）"
                      "＝サーブ動作が検出できていない")
    return {"standing_foot_m": round(standing, 3), "phase_span": span,
            "issues": issues}


def bundle(joints_dir: str, fps: float, label: str,
           video: str | None = None,
           trim: tuple[float | None, float | None] = (None, None)) -> dict:
    d = Path(joints_dir)
    joints = np.load(d / "gv_joints.npy")
    res = analysis.analyze_json(joints, fps)
    res["label"] = label
    res["fps"] = fps
    res["quality"] = quality(joints, res["metrics"]["phases"])

    # 元動画があるなら、カメラが動いていないかも見る（issue #8）。
    # 動いていると世界座標が汚染され、跳躍や伸び上がりの数値が意味を失う。
    # 関節列だけからは判別できない（接地足の滑りはサーブ本来の動きと区別がつかない）。
    if video:
        try:
            from camera_motion import analyze as cam_analyze, summarize
            cam = cam_analyze(video, *trim)
            res["quality"]["camera"] = {
                "motion_per_sec": round(cam["motion_per_sec"], 5),
                "is_static": cam["is_static"],
            }
            res["quality"]["issues"] += summarize(cam)
        except SystemExit as e:      # imageio 未導入など。解析自体は続ける
            print(f"⚠️ カメラ運動を調べられませんでした: {e}")

    pose_path = d / "gv_pose.npz"
    if pose_path.exists():
        p = np.load(pose_path)
        res["pose"] = {k: p[k].round(5).tolist() for k in p.files}
    else:
        print(f"⚠️ {pose_path} が無いのでアバター表示はできません")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="比較ビューア用データの生成")
    ap.add_argument("--mine", required=True, help="自分の結果ディレクトリ")
    ap.add_argument("--mine-fps", type=float, required=True)
    ap.add_argument("--mine-label", default="あなた")
    ap.add_argument("--mine-video", help="元動画（カメラ運動の点検に使う）")
    ap.add_argument("--ref", required=True, help="お手本の結果ディレクトリ")
    ap.add_argument("--ref-fps", type=float, required=True)
    ap.add_argument("--ref-label", default="お手本")
    ap.add_argument("--ref-video", help="元動画（カメラ運動の点検に使う）")
    ap.add_argument("--ref-trim", nargs=2, type=float, metavar=("開始", "終了"),
                    help="お手本を復元したときのトリム区間（秒）")
    ap.add_argument("--out", default="web/data")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref_trim = tuple(args.ref_trim) if args.ref_trim else (None, None)
    for key, src, fps, label, video, trim in [
        ("mine", args.mine, args.mine_fps, args.mine_label,
         args.mine_video, (None, None)),
        ("reference", args.ref, args.ref_fps, args.ref_label,
         args.ref_video, ref_trim),
    ]:
        b = bundle(src, fps, label, video, trim)
        path = out / f"{key}.json"
        path.write_text(json.dumps(b, ensure_ascii=False), encoding="utf-8")
        ph = b["metrics"]["phases"]
        size = path.stat().st_size / 1e6
        print(f"✅ {path}  ({size:.1f} MB)  {label}: "
              f"{b['metrics']['n_frames']}フレーム @{fps:.0f}fps  "
              f"沈み込み{ph['loading']} → 打点{ph['contact']}")
        for msg in b["quality"]["issues"]:
            print(f"   ⚠️ {msg.replace('**', '')}")   # 強調記法は UI 側で解釈する


if __name__ == "__main__":
    main()
