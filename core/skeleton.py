"""SMPL 24関節の骨格定義。競技に依存しない。

ここは**骨格の事実**だけを置く。どの関節がどこか、体節の質量がいくらか。
競技ごとの解釈（どちらが利き手か、どこが打点か）は domains/ の担当。

復元手法ごとに骨格が違うため、内部表現は SMPL 24関節に統一し、
他の骨格は `core.convert` で並べ替えてから入れる。
"""

from __future__ import annotations

PELVIS, L_HIP, R_HIP, SPINE1 = 0, 1, 2, 3
L_KNEE, R_KNEE, SPINE2 = 4, 5, 6
L_ANKLE, R_ANKLE, SPINE3 = 7, 8, 9
L_FOOT, R_FOOT, NECK = 10, 11, 12
L_COLLAR, R_COLLAR, HEAD = 13, 14, 15
L_SHOULDER, R_SHOULDER = 16, 17
L_ELBOW, R_ELBOW = 18, 19
L_WRIST, R_WRIST = 20, 21
L_HAND, R_HAND = 22, 23

N_JOINTS = 24

#: 接地の判定に使う「足まわり」。足首とつま先の4点。
FOOT_IDS = [L_ANKLE, R_ANKLE, L_FOOT, R_FOOT]

#: 左右のある関節。`Kinematics.idx("elbow")` で利き側を引くための表。
SIDED: dict[str, tuple[int, int]] = {
    "shoulder": (R_SHOULDER, L_SHOULDER),
    "elbow": (R_ELBOW, L_ELBOW),
    "wrist": (R_WRIST, L_WRIST),
    "hand": (R_HAND, L_HAND),
    "hip": (R_HIP, L_HIP),
    "knee": (R_KNEE, L_KNEE),
    "ankle": (R_ANKLE, L_ANKLE),
    "foot": (R_FOOT, L_FOOT),
    "collar": (R_COLLAR, L_COLLAR),
}

# De Leva の体節質量比 (親関節, 子関節, 質量比, 近位からのCOM比)。合計 ≒ 1.0。
# 体幹49.7% / 頭8.1% / 大腿 各10% / 下腿 各4.65% / 上腕 各2.8% ほか。
SEGMENTS = [
    (PELVIS, NECK, 0.497, 0.50), (NECK, HEAD, 0.081, 0.50),
    (L_SHOULDER, L_ELBOW, 0.028, 0.436), (R_SHOULDER, R_ELBOW, 0.028, 0.436),
    (L_ELBOW, L_WRIST, 0.016, 0.430), (R_ELBOW, R_WRIST, 0.016, 0.430),
    (L_WRIST, L_HAND, 0.006, 0.50), (R_WRIST, R_HAND, 0.006, 0.50),
    (L_HIP, L_KNEE, 0.100, 0.433), (R_HIP, R_KNEE, 0.100, 0.433),
    (L_KNEE, L_ANKLE, 0.0465, 0.433), (R_KNEE, R_ANKLE, 0.0465, 0.433),
    (L_ANKLE, L_FOOT, 0.0145, 0.50), (R_ANKLE, R_FOOT, 0.0145, 0.50),
]
