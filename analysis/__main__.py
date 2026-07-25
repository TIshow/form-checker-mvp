"""コマンドラインから解析する。

    # 標準出力にレポート
    python -m analysis --joints gv_joints.npy --com gv_com.npy \
                       --upaxis gv_upaxis.npy --fps 120

    # さらに output/ にレポートとグラフを保存
    python -m analysis --joints gv_joints.npy --com gv_com.npy \
                       --upaxis gv_upaxis.npy --fps 30 --save output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import analyze, format_report


def main() -> None:
    p = argparse.ArgumentParser(description="サーブの3D解析とフィードバック生成")
    p.add_argument("--joints", default="gv_joints.npy")
    p.add_argument("--com", default="gv_com.npy")
    p.add_argument("--upaxis", default="gv_upaxis.npy")
    p.add_argument("--fps", type=float, default=30.0,
                   help="撮影フレームレート。連鎖の順序判定には60以上が必要")
    p.add_argument("--top", type=int, default=2, help="提示する指摘の件数")
    p.add_argument("--save", metavar="DIR",
                   help="レポート(report.txt)とグラフ(com_height.png)を保存する先")
    args = p.parse_args()

    joints = np.load(args.joints)
    com = np.load(args.com)
    up_ax, up_sign = np.load(args.upaxis)
    metrics, feedback = analyze(joints, com, int(up_ax), float(up_sign), args.fps)

    report = format_report(metrics, feedback, top_n=args.top)
    print(report)

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.txt").write_text(report, encoding="utf-8")

        from .plot import save_com_height_graph

        save_com_height_graph(metrics, com, int(up_ax), float(up_sign),
                              str(out / "com_height.png"))
        print(f"\n✅ 保存: {out}/report.txt, {out}/com_height.png")


if __name__ == "__main__":
    main()
