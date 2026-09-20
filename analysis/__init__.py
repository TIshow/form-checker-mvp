"""アプリ層。関節データ → 指標・フィードバック・JSON。

3層に分かれている:

  core/      計測（幾何・運動学）        競技非依存  変更頻度: 低
  domains/   局面・指標・判定・表示      競技ごと    変更頻度: 高
  analysis/  ここ。両者をつないで返す形を決める

使い方:

    import analysis
    metrics, feedback = analysis.analyze(joints, fps=60)             # テニス
    metrics, feedback = analysis.analyze(joints, 60, "golf_swing")   # ゴルフ
    print(analysis.format_report(metrics, feedback))

ドメインを省略するとテニスのサーブになる（`domains.DEFAULT`）。
"""

from __future__ import annotations

import numpy as np

import domains
from core import Kinematics
from core.filter import level_from_upright, temporal_smooth, window_for

__all__ = [
    "analyze", "analyze_json", "analyze_from_files",
    "format_report", "kinematics_for", "domains",
]


def kinematics_for(joints: np.ndarray, fps: float, domain: str | None = None,
                   smooth_to_fps: float | None = None,
                   level_window: tuple[float, float] | None = None):
    """ドメインに利き側を決めさせて `Kinematics` を組み立てる。

    利き側の根拠は競技ごとに違う（サーブ=手首が高く上がる腕、
    ゴルフ=トップで伸びているリード腕、投球=速く動く腕）ため、
    ここでは決めずにドメインへ委ねる。

    smooth_to_fps  単一画像モデルのジッタ対策。関節列を「この fps 相当」まで
                   時間方向に平滑化してから計測する（`core/filter.py`）。
                   通常速度の映像では窓が1になり何も変わらない。
    level_window   (開始秒, 終了秒)。この区間で直立している前提で、頭−足首を
                   鉛直として関節列を回す（カメラの傾きの較正）。
    """
    d = domains.get(domain)
    joints = np.asarray(joints)
    if level_window is not None:
        joints, _ = level_from_upright(joints, level_window[0] * fps, level_window[1] * fps)
    joints = temporal_smooth(joints, window_for(fps, smooth_to_fps))
    return d, Kinematics(joints, fps, side=d.side(joints))


def analyze(joints: np.ndarray, fps: float = 30.0,
            domain: str | None = None,
            smooth_to_fps: float | None = None,
            level_window: tuple[float, float] | None = None) -> tuple[dict, list[dict]]:
    """関節データから指標とフィードバックを求める。

    joints  (F, 24, 3) SMPL 24関節の world座標 [m]
    他の骨格は `core.convert` で並べ替えてから渡す。
    """
    d, kin = kinematics_for(joints, fps, domain, smooth_to_fps, level_window)
    metrics = d.measure(kin, d.detect_phases(kin))
    return metrics, d.judge(metrics)


def analyze_json(joints: np.ndarray, fps: float = 30.0,
                 domain: str | None = None,
                 smooth_to_fps: float | None = None,
                 level_window: tuple[float, float] | None = None) -> dict:
    """Web が返す JSON 化可能な結果。指標・フィードバック・ビューア用の関節列。

    3D復元の外（サーバーのCPUやブラウザ）へ渡す境界。numpy を残さず、
    そのまま json.dumps できる形にする。
    """
    d, kin = kinematics_for(joints, fps, domain, smooth_to_fps, level_window)
    metrics = d.measure(kin, d.detect_phases(kin))
    return {
        "domain": d.name,
        "metrics": metrics,
        "feedback": d.judge(metrics),
        "up_axis": [kin.up_ax, kin.up_sign],
        "smoothed_window": window_for(fps, smooth_to_fps),
        "joints": kin.J.round(4).tolist(),  # (F,24,3) 3Dビューア用（平滑化後）
    }


def analyze_from_files(joints_path: str = "gv_joints.npy", fps: float = 30.0,
                       domain: str | None = None) -> tuple[dict, list[dict]]:
    """復元パイプラインが出力した関節 .npy から解析する。"""
    return analyze(np.load(joints_path), fps, domain)


def format_report(metrics: dict, feedback: list[dict], top_n: int = 2) -> str:
    """指標の `domain` を見て、そのドメインの表示にまわす。"""
    return domains.get(metrics.get("domain")).report(metrics, feedback, top_n)
