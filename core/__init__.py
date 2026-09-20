"""競技に依存しない計測の層。

  skeleton.py    SMPL 24関節の定義・体節質量比
  geometry.py    幾何ユーティリティ（純関数）
  kinematics.py  重心・床・関節角・捻転・連鎖
  convert.py     他の骨格 → SMPL24 の並べ替え
  plot.py        重心の高さのグラフ（任意依存 matplotlib）

競技ごとの局面・指標・判定は domains/ にある。この層を触るのは
「測り方そのもの」が変わるときだけで、頻度は低い。
"""

from __future__ import annotations

from .geometry import (
    angle_from_axis, angular_speed, horizontal, horizontal_angle,
    joint_angle, smooth, unit,
)
from .kinematics import (
    Kinematics, compute_com, detect_up_axis, dominant_side_by_peak_height,
)
from .skeleton import FOOT_IDS, N_JOINTS, SEGMENTS, SIDED

__all__ = [
    "Kinematics", "compute_com", "detect_up_axis",
    "dominant_side_by_peak_height",
    "angle_from_axis", "angular_speed", "horizontal", "horizontal_angle",
    "joint_angle", "smooth", "unit",
    "FOOT_IDS", "N_JOINTS", "SEGMENTS", "SIDED",
]
