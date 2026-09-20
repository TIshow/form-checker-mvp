"""SOMA の骨格を SMPL の24関節配置に並べ替える（issue #9）。

GEM-X は SOMA（NVIDIA の人体モデル）で77関節を返す。一方 `core/kinematics.py`
も `web/avatar.js` も SMPL の24関節前提で書かれている。

ここで**並べ替えるだけ**にしておけば、解析層もビューアもアバターも無改造で
GEM-X の出力を扱える。GVHMR 経路には一切触れない。

    from core.convert import to_smpl24
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


# --------------------------------------------------------------------------
# MHR（Meta Momentum Human Rig）→ SMPL 24
#
# SAM 3D Body が返すキーポイント。308点のうち**先頭70点**が体で、残りは顔
# （`sam_3d_body/metadata/mhr70.py`）。名前は COCO 式なので、SOMA のときとは
# 事情が違う:
#
#   SOMA は SMPL と同じ階層を持っていたので**並べ替えるだけ**で済んだ。
#   MHR-70 には **骨盤も脊椎も無い**。導出が要る。
#
# 導出したものは「ほぼ正しい位置」であって、SMPL の回帰器が出す関節とは
# 一致しない。**手法間で絶対値を比べるときはこの差を見込むこと。**
# --------------------------------------------------------------------------

MHR70_JOINTS = 70

#: SMPL24 の添字 -> MHR-70 の添字。None は下の `_derive` で作る。
SMPL24_FROM_MHR70: list[tuple[str, int | None]] = [
    ("pelvis（左右股関節の中点）", None),   # 0
    ("left-hip", 9),                        # 1
    ("right-hip", 10),                      # 2
    ("spine1（骨盤→首の 1/4）", None),      # 3
    ("left-knee", 11),                      # 4
    ("right-knee", 12),                     # 5
    ("spine2（骨盤→首の 1/2）", None),      # 6
    ("left-ankle", 13),                     # 7
    ("right-ankle", 14),                    # 8
    ("spine3（骨盤→首の 3/4）", None),      # 9
    ("left-big-toe-tip", 15),               # 10
    ("right-big-toe-tip", 18),              # 11
    ("neck", 69),                           # 12
    ("left-acromion", 67),                  # 13
    ("right-acromion", 68),                 # 14
    ("head（左右耳の中点）", None),          # 15
    ("left-shoulder", 5),                   # 16
    ("right-shoulder", 6),                  # 17
    ("left-elbow", 7),                      # 18
    ("right-elbow", 8),                     # 19
    ("left-wrist", 62),                     # 20
    ("right-wrist", 41),                    # 21
    ("left-middle-first-joint", 51),        # 22  手＝中指の付け根
    ("right-middle-first-joint", 30),       # 23
]

_L_HIP70, _R_HIP70, _NECK70 = 9, 10, 69
_L_EAR70, _R_EAR70 = 3, 4


def mhr70_to_smpl24(joints: np.ndarray) -> np.ndarray:
    """SAM 3D Body の MHR キーポイントを SMPL 24関節の並びにする。

    joints: (F, 70以上, 3) — 308点でも先頭70点だけ使う
    戻り値: (F, 24, 3)

    ## 導出しているもの

    骨盤 = 左右股関節の中点。脊椎3点 = 骨盤→首を4等分。
    頭 = 左右耳の中点（鼻ではない。SMPL の head は頭蓋の中心寄りで、
    鼻を使うと身長が 10cm ほど低く出て、体格で正規化した指標がずれる）。

    脊椎と鎖骨は `core/kinematics.py` の計測では**使っていない**
    （体節質量比は骨盤→首を1本の体幹として扱う）。ビューアとアバターの
    見た目のためだけに埋めている。
    """
    joints = np.asarray(joints, dtype=float)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"(F, J, 3) を期待しましたが {joints.shape} でした")
    if joints.shape[1] < MHR70_JOINTS:
        raise ValueError(
            f"MHR のキーポイントは {MHR70_JOINTS} 点以上のはずですが "
            f"{joints.shape[1]} でした。SAM 3D Body の出力が変わった可能性があります"
        )

    F = joints.shape[0]
    out = np.zeros((F, 24, 3))
    for smpl_i, (_, mhr_i) in enumerate(SMPL24_FROM_MHR70):
        if mhr_i is not None:
            out[:, smpl_i] = joints[:, mhr_i]

    pelvis = (joints[:, _L_HIP70] + joints[:, _R_HIP70]) / 2
    neck = joints[:, _NECK70]
    out[:, 0] = pelvis
    for smpl_i, t in ((3, 0.25), (6, 0.50), (9, 0.75)):
        out[:, smpl_i] = pelvis + (neck - pelvis) * t
    out[:, 15] = (joints[:, _L_EAR70] + joints[:, _R_EAR70]) / 2
    return out


def verify_mhr_names(names: list[str]) -> list[str]:
    """SAM 3D Body の `mhr70.py` の並びと照合し、食い違いを返す。

        from sam_3d_body.metadata.mhr70 import mhr_names
        assert not verify_mhr_names(mhr_names)

    添字を手で書いている以上、上流が並びを変えたら黙って壊れる。
    取り違えても**それらしい数字が出てしまう**ので、必ず通すこと。
    """
    bad = []
    if len(names) < MHR70_JOINTS:
        return [f"mhr_names が {len(names)} 個（{MHR70_JOINTS} 以上を期待）"]
    for label, i in SMPL24_FROM_MHR70:
        if i is None:
            continue
        if names[i] != label:
            bad.append(f"添字 {i} は {label} のはずが {names[i]} でした")
    for i, want in ((_L_HIP70, "left-hip"), (_R_HIP70, "right-hip"),
                    (_NECK70, "neck"), (_L_EAR70, "left-ear"),
                    (_R_EAR70, "right-ear")):
        if names[i] != want:
            bad.append(f"導出に使う添字 {i} は {want} のはずが {names[i]} でした")
    return bad
