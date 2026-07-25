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


def analyze(joints: np.ndarray, fps: float = 30.0) -> tuple[dict, list[dict]]:
    """関節データから指標とフィードバックを求める。

    joints  (F, 24, 3) SMPL 24関節の world座標 [m]
    重心・上軸は関節から導出する（`serve.compute_com` / `serve.detect_up_axis`）。
    """
    kin = ServeKinematics(joints, fps)
    metrics = compute_metrics(kin)
    return metrics, generate_feedback(metrics)


def analyze_from_files(joints_path: str = "gv_joints.npy",
                       fps: float = 30.0) -> tuple[dict, list[dict]]:
    """GVHMRパイプラインが出力した関節 .npy から解析する。"""
    return analyze(np.load(joints_path), fps)
