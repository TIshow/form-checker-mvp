"""野球の投球。**指標のみ。判定ルールはまだ無い。**

## 「跳ばないから楽」は投球には当てはまらない

投球はストライドで身長の8〜9割を前方へ移動する。跳躍と同じ
**グローバル並進**の問題で、名前が違うだけ。issue 008 の問題圏内にある。

## 撮影フレームレートが本質的な制約になる

投球の肩の内旋角速度は約7,000°/秒と言われ、人体の運動で最速
（サーブは2,500〜3,000°/秒）。体節間の時間差もそのぶん短い。

`chain_min_fps` を 240 にしてあるが、**これは検証済みの値ではない。**
テニスの60fps（体節間 20〜40ms に対し2サンプル以上）と同じ考え方を、
より短い時間差に当てはめた見積もり。実測で決め直すこと。

**60fps のスマホ映像では、投球のキネティックチェーンは判定できない。**
これは実装の問題ではなく撮影の問題で、ハイスピード撮影が要る。

## 位置だけでは測れないもの

最大外旋（MER）は上腕の**軸回転**で、関節位置にほとんど現れない。
投球障害の研究が中心に据える量がこれなので、位置ベースの解析だけでは
この競技の核心に届かない。マーカーレスで測るなら手の姿勢まで要る
（MHR の127関節が効く可能性がある。issue 011）。
"""

from __future__ import annotations

import numpy as np

from core import Kinematics, joint_angle, smooth
from core.skeleton import (
    FOOT_IDS, L_ANKLE, L_FOOT, L_HIP, L_KNEE, R_ANKLE, R_FOOT, R_HIP, R_KNEE,
)
from domains.base import NotImplementedDomain

#: 投球で連鎖の順序を論じるのに必要と見積もったフレームレート。**未検証**。
PITCH_CHAIN_MIN_FPS = 240.0


class BaseballPitch(NotImplementedDomain):
    name = "baseball_pitch"
    label = "野球 投球（指標のみ）"
    chain_min_fps = PITCH_CHAIN_MIN_FPS
    headline = ("stride_ratio", "hip_shoulder_separation_deg",
                "lead_knee_at_release_deg", "trunk_lean_at_release_deg")

    evidence_needed = (
        "最大外旋(MER)を測る手段。上腕の軸回転は関節位置に出ない",
        "ハイスピード撮影（240fps以上）での実測。60fpsでは連鎖を判定できない",
        "リリース検出をボール追跡で置き換える（今は手首の最大速度という代用）",
        "ストライドで前方へ大きく移動するため、世界座標の並進精度の検証",
        "各指標の基準レンジを、本文を開いて確認した出典で",
    )

    def side(self, joints: np.ndarray) -> str:
        """投げる腕。リリースに向けて手首が最も速く動く方。"""
        lw = smooth(np.linalg.norm(np.diff(joints[:, 20], axis=0), axis=-1))
        rw = smooth(np.linalg.norm(np.diff(joints[:, 21], axis=0), axis=-1))
        return "R" if rw.max() >= lw.max() else "L"

    def detect_phases(self, kin: Kinematics) -> dict[str, int]:
        """足上げ → 踏み出し足の接地 → リリース（代用）。

        **リリースはボールを見ずに決めている。** 投球腕の手首速度が最大に
        なるフレームを使う代用値。
        """
        # 踏み出し足 = 投球腕と反対側
        lead_foot = [L_ANKLE, L_FOOT] if kin.side == "R" else [R_ANKLE, R_FOOT]
        lf = kin.height(kin.J[:, lead_foot]).min(axis=1)
        lift = int(np.argmax(lf))                       # 足を最も上げたところ
        after = lf[lift:]
        # 接地 = 足上げ以降で床の高さに戻る最初のところ
        ground = kin.ground()
        touched = np.where(after <= ground + 0.03)[0]
        contact = lift + int(touched[0]) if len(touched) else lift

        wr = kin.idx("wrist")
        speed = smooth(np.linalg.norm(np.diff(kin.J[:, wr], axis=0), axis=-1))
        speed = np.concatenate([[0.0], speed])
        release = contact + int(np.argmax(speed[contact:])) if contact < kin.F - 1 else kin.F - 1
        return {"lift": lift, "foot_contact": contact, "release": release}

    def measure(self, kin: Kinematics, phases: dict[str, int]) -> dict:
        fc, rel = phases["foot_contact"], phases["release"]
        body_h = kin.body_height_proxy()

        stride = float(np.linalg.norm(kin.J[fc, L_ANKLE] - kin.J[fc, R_ANKLE]))

        knee_l = joint_angle(kin.J[:, L_HIP], kin.J[:, L_KNEE], kin.J[:, L_ANKLE])
        knee_r = joint_angle(kin.J[:, R_HIP], kin.J[:, R_KNEE], kin.J[:, R_ANKLE])
        lead = knee_l if kin.side == "R" else knee_r
        xf = kin.x_factor()
        feet = kin.height(kin.J[:, FOOT_IDS]).min(axis=1) - kin.ground()

        return {
            "domain": self.name,
            "throwing_side": kin.side,
            "fps": kin.fps,
            "n_frames": kin.F,
            "phases": phases,
            # ストライド。身長で正規化する
            "stride_m": stride,
            "stride_ratio": float(stride / body_h) if body_h > 0 else float("nan"),
            # 股関節-肩の分離（TIER C。位置ベースなので弱い）
            "hip_shoulder_separation_deg": float(xf[fc]),
            "hip_shoulder_separation_max_deg": float(xf[fc : rel + 1].max())
            if rel > fc else float(xf[fc]),
            # リード脚。伸展するほど骨盤の回転が止まり上体へ力が移る
            "lead_knee_at_contact_deg": float(lead[fc]),
            "lead_knee_at_release_deg": float(lead[rel]),
            "lead_knee_extension_deg": float(lead[rel] - lead[fc]),
            # 上体
            "trunk_lean_at_release_deg": float(kin.trunk_lean()[rel]),
            "elbow_at_release_deg": float(kin.elbow_angle()[rel]),
            # 並進の健全性チェック（投球は前方へ大きく動く）
            "com_travel_m": float(np.linalg.norm(kin.com[rel] - kin.com[0])),
            "foot_clearance_m": float(feet[: rel + 1].max()),
            "kinetic_chain": kin.kinetic_chain(fc, min(rel + 2, kin.F)),
        }
