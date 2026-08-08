"""SOMA の骨格を SMPL の24関節配置に並べ替える（issue #9）。

GEM-X は SOMA（NVIDIA の人体モデル）で77関節を返す。一方 `analysis/serve.py`
も `web/avatar.js` も SMPL の24関節前提で書かれている。

ここで**並べ替えるだけ**にしておけば、解析層もビューアもアバターも無改造で
GEM-X の出力を扱える。GVHMR 経路には一切触れない。

    from analysis.soma import to_smpl24
    joints24 = to_smpl24(joints_soma)      # (F,77or78,3) -> (F,24,3)

## 対応の根拠

SOMA は Mixamo 式の**ボーン名**を使う。「関節 LeftShin」は脛の付け根、つまり
**膝**を指す。階層（SOMA_neutral.npz の joint_parent_ids）で確認した:

    Hips → LeftLeg → LeftShin → LeftFoot → LeftToeBase
            (股関節)   (膝)      (足首)     (つま先)
    Chest → LeftShoulder → LeftArm → LeftForeArm → LeftHand
             (鎖骨)        (肩)      (肘)          (手首)

## Root の扱い

SOMA_neutral.npz の joint_names は **78個**（先頭に `Root`）だが、モデルカードは
**77関節**と書いている。GEM-X の実出力がどちらかで添字が1ずれるため、
入力の関節数を見て自動で吸収する。取り違えると「膝の角度」が別の関節の角度に
なり、しかも**それらしい数字が出てしまう**ので、ここは自動判定にしている。
"""

from __future__ import annotations

import numpy as np

# SOMA_neutral.npz の joint_names（Root を含む78個の並び）での添字。
# 名前も残すのは、将来 SOMA が並びを変えたときに気付けるようにするため。
SMPL24_FROM_SOMA78: list[tuple[str, int]] = [
    ("Hips", 1),               # 0  骨盤
    ("LeftLeg", 68),           # 1  左股関節
    ("RightLeg", 73),          # 2  右股関節
    ("Spine1", 2),             # 3  脊椎1
    ("LeftShin", 69),          # 4  左膝
    ("RightShin", 74),         # 5  右膝
    ("Spine2", 3),             # 6  脊椎2
    ("LeftFoot", 70),          # 7  左足首
    ("RightFoot", 75),         # 8  右足首
    ("Chest", 4),              # 9  脊椎3
    ("LeftToeBase", 71),       # 10 左つま先
    ("RightToeBase", 76),      # 11 右つま先
    ("Neck1", 5),              # 12 首
    ("LeftShoulder", 12),      # 13 左鎖骨
    ("RightShoulder", 40),     # 14 右鎖骨
    ("Head", 7),               # 15 頭
    ("LeftArm", 13),           # 16 左肩
    ("RightArm", 41),          # 17 右肩
    ("LeftForeArm", 14),       # 18 左肘
    ("RightForeArm", 42),      # 19 右肘
    ("LeftHand", 15),          # 20 左手首
    ("RightHand", 43),         # 21 右手首
    ("LeftHandMiddle1", 25),   # 22 左手
    ("RightHandMiddle1", 53),  # 23 右手
]

SOMA78_JOINTS = 78
SOMA77_JOINTS = 77   # Root を落とした並び


def _index_map(n_joints: int) -> np.ndarray:
    """入力の関節数に合わせた添字列を返す。"""
    idx = np.array([i for _, i in SMPL24_FROM_SOMA78], dtype=int)
    if n_joints == SOMA78_JOINTS:
        return idx
    if n_joints == SOMA77_JOINTS:
        # Root(0) が無い並び。Root より後ろの関節はすべて1つ前へ寄る。
        return idx - 1
    raise ValueError(
        f"SOMA の関節数は {SOMA77_JOINTS} か {SOMA78_JOINTS} のはずですが "
        f"{n_joints} でした。SOMA の骨格定義が変わった可能性があります"
    )


def to_smpl24(joints: np.ndarray) -> np.ndarray:
    """SOMA の関節列を SMPL の24関節順に並べ替える。

    joints: (F, 77 or 78, 3)
    戻り値: (F, 24, 3)
    """
    joints = np.asarray(joints)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"(F, J, 3) を期待しましたが {joints.shape} でした")
    return joints[:, _index_map(joints.shape[1]), :]


def verify_against_asset(npz_path: str) -> list[str]:
    """SOMA_neutral.npz の joint_names と照合し、食い違いを返す。

    添字を手で書いている以上、モデル側が並びを変えたら黙って壊れる。
    GEM-X を更新したときはこれを通す。問題なければ空リスト。
    """
    d = np.load(npz_path, allow_pickle=True)
    names = [str(x) for x in d["joint_names"]]
    bad = []
    if len(names) != SOMA78_JOINTS:
        bad.append(f"joint_names が {len(names)} 個（{SOMA78_JOINTS} を期待）")
        return bad
    for want, i in SMPL24_FROM_SOMA78:
        if names[i] != want:
            bad.append(f"添字 {i} は {want} のはずが {names[i]} でした")
    return bad
