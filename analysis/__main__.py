"""コマンドラインから解析する。

    python -m analysis --joints gv_joints.npy --fps 120
    python -m analysis --joints out/joints.npy --fps 60 --domain golf_swing
    python -m analysis --list

重心・上軸・利き側は関節から導出するため、入力は関節 .npy 1つでよい。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import analysis
import domains


def main() -> None:
    p = argparse.ArgumentParser(description="3D関節データの解析とフィードバック生成")
    p.add_argument("--joints", default="gv_joints.npy")
    p.add_argument("--fps", type=float, default=30.0,
                   help="撮影フレームレート。連鎖の順序判定には60以上が必要")
    p.add_argument("--domain", default=domains.DEFAULT,
                   help=f"競技。{' / '.join(domains.names())}")
    p.add_argument("--smooth-to-fps", type=float, default=None,
                   help="関節列をこの fps 相当まで平滑化してから計測（単一画像モデルのスロー映像向け）")
    p.add_argument("--anchor", action="store_true",
                   help="接地足を固定して並進のゆらぎを止める（カメラ空間の復元向け）")
    p.add_argument("--top", type=int, default=2, help="提示する指摘の件数")
    p.add_argument("--save", metavar="DIR", help="レポートとグラフの保存先")
    p.add_argument("--list", action="store_true", help="使えるドメインを表示して終了")
    args = p.parse_args()

    if args.list:
        for n in domains.names():
            print(f"  {n:<16} {domains.get(n).label}")
        return

    d, kin = analysis.kinematics_for(np.load(args.joints), args.fps, args.domain,
                                     args.smooth_to_fps, anchor=args.anchor)
    phases = d.detect_phases(kin)
    metrics = d.measure(kin, phases)
    feedback = d.judge(metrics)

    report = d.report(metrics, feedback, args.top)
    print(report)

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.txt").write_text(report, encoding="utf-8")

        # グラフに描く局面はドメインが決める（辞書の並び順に頼らない）。
        # タイトルも ASCII の name を使う。label は日本語で、matplotlib の
        # 既定フォントにグリフが無く豆腐になる。
        pair = getattr(d, "plot_phases", ())
        if len(pair) == 2 and all(k in phases for k in pair):
            from core.plot import save_com_height_graph
            save_com_height_graph(metrics, kin.com, kin.up_ax, kin.up_sign,
                                  str(out / "com_height.png"),
                                  phases=tuple(pair), title=d.name)
            print(f"\n✅ 保存: {out}/report.txt, {out}/com_height.png")
        else:
            print(f"\n✅ 保存: {out}/report.txt"
                  "（このドメインは重心グラフを描きません）")


if __name__ == "__main__":
    main()
