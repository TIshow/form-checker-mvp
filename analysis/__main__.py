"""コマンドラインから解析する。

    python -m analysis --joints gv_joints.npy --com gv_com.npy \
                       --upaxis gv_upaxis.npy --fps 120
"""

from __future__ import annotations

import argparse

from . import analyze_from_files, format_report


def main() -> None:
    p = argparse.ArgumentParser(description="サーブの3D解析とフィードバック生成")
    p.add_argument("--joints", default="gv_joints.npy")
    p.add_argument("--com", default="gv_com.npy")
    p.add_argument("--upaxis", default="gv_upaxis.npy")
    p.add_argument("--fps", type=float, default=30.0,
                   help="撮影フレームレート。連鎖の順序判定には60以上が必要")
    p.add_argument("--top", type=int, default=2, help="提示する指摘の件数")
    args = p.parse_args()

    metrics, feedback = analyze_from_files(
        args.joints, args.com, args.upaxis, args.fps)
    print(format_report(metrics, feedback, top_n=args.top))


if __name__ == "__main__":
    main()
