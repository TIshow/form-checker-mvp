"""サーブの3D解析とフィードバック生成。

3層に分かれている:
  serve.py     計測（幾何・運動学・指標）    変更頻度: 低
  feedback.py  判定ルールと閾値              変更頻度: 高
  report.py    表示                          将来UIに置換

使い方:
    import analysis
    metrics, feedback = analysis.analyze_from_files()
    print(analysis.format_report(metrics, feedback))
"""

from __future__ import annotations

import numpy as np

from .feedback import generate_feedback
from .report import format_report
from .serve import ServeKinematics, compute_metrics

__all__ = [
    "analyze",
    "analyze_from_files",
    "compute_metrics",
    "format_report",
    "generate_feedback",
    "ServeKinematics",
]


def analyze(joints: np.ndarray, com: np.ndarray, up_ax: int, up_sign: float,
            fps: float = 30.0) -> tuple[dict, list[dict]]:
    """関節データから指標とフィードバックを求める。

    joints  (F, 24, 3) SMPL 24関節の world座標 [m]
    com     (F, 3)     全身重心の world座標 [m]
    up_ax   上方向の軸index, up_sign その符号
    """
    kin = ServeKinematics(joints, com, up_ax, up_sign, fps)
    metrics = compute_metrics(kin)
    return metrics, generate_feedback(metrics)


def analyze_from_files(joints_path: str = "gv_joints.npy",
                       com_path: str = "gv_com.npy",
                       upaxis_path: str = "gv_upaxis.npy",
                       fps: float = 30.0) -> tuple[dict, list[dict]]:
    """GVHMRパイプラインが出力した .npy から解析する。"""
    up_ax, up_sign = np.load(upaxis_path)
    return analyze(np.load(joints_path), np.load(com_path),
                   int(up_ax), float(up_sign), fps)
