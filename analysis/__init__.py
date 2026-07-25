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
    "analyze_json",
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


def analyze_json(joints: np.ndarray, fps: float = 30.0) -> dict:
    """Web が返す JSON 化可能な結果。指標・フィードバック・3Dビューア用の関節列。

    3D復元の外（サーバーのCPUやブラウザ）へ渡す境界。numpy を残さず、
    そのまま json.dumps できる形にする。
    """
    kin = ServeKinematics(joints, fps)
    metrics = compute_metrics(kin)
    feedback = generate_feedback(metrics)
    return {
        "metrics": metrics,
        "feedback": feedback,
        "up_axis": [kin.up_ax, kin.up_sign],
        "joints": np.asarray(joints).round(4).tolist(),  # (F,24,3) 3Dビューア用
    }
