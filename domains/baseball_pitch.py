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

#: 接地とみなす下降速度の、最大下降速度に対する比。
#:
#: **高さの閾値で接地を判定してはいけない。** マウンドは傾斜で、踏み出し足は
#: 軸足より**低い**ところに着く。実測（KBOの投手・24fps）では踏み出し足が
#: 推定した床より 6cm 下まで行き、「床+3cm」では接地が 0.3秒遅れて
#: **接地とリリースが同じフレームに潰れた**。高さではなく
#: 「速く下りてきた足が止まる」ところを見る。
FOOT_PLANT_SPEED_RATIO = 0.25


class BaseballPitch(NotImplementedDomain):
    name = "baseball_pitch"
    label = "野球 投球（指標のみ）"
    chain_min_fps = PITCH_CHAIN_MIN_FPS
    headline = ("stride_ratio", "hip_shoulder_separation_deg",
                "lead_knee_at_release_deg", "trunk_lean_at_release_deg",
                "elbow_at_release_deg")
    plot_phases = ("foot_contact", "release")
    metric_labels = {
        "stride_ratio": ("ストライド", "身長比", 2, "大きいほど広い"),
        "hip_shoulder_separation_deg": ("股関節と肩の分離", "°", 0, "接地時"),
        "lead_knee_at_release_deg": ("リリース時のリード膝", "°", 0, "大きいほど伸びている"),
        "trunk_lean_at_release_deg": ("リリース時の体幹", "°", 0, "鉛直からの傾き"),
        "elbow_at_release_deg": ("リリース時の肘角", "°", 0, "大きいほど伸びている"),
    }
    phase_labels = {"lift": "足上げ", "foot_contact": "踏み出し足の接地",
                    "release": "リリース"}

    evidence_needed = (
        "最大外旋(MER)を測る手段。上腕の軸回転は関節位置に出ない",
        "接地の判定（下降速度が最大の "
        f"{FOOT_PLANT_SPEED_RATIO:.0%} を下回った点。実測で高さの閾値より"
        "はるかにましだが、比率そのものの根拠はまだ無い）",
        "接地→リリースは実時間で120〜150ms。24fpsでは3〜4フレームしかなく"
        "分離が粗い。120fps以上で撮れば局面の精度が一段上がる",
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

        # 接地 = 「速く下りてきた足が止まる」ところ。
        # 高さの閾値は使わない（マウンドの傾斜で踏み出し足は軸足より低く着く）。
        # リリースを先に決める。≒ 手首が最も速いフレーム。**足上げ以降の全体**
        # から探すこと。接地以降に限ると、接地の推定が少し遅れただけで
        # ピークを跨ぎ、接地とリリースが同じフレームに潰れる（実測で起きた）。
        wr = kin.idx("wrist")
        speed = smooth(np.linalg.norm(np.diff(kin.J[:, wr], axis=0), axis=-1))
        speed = np.concatenate([[0.0], speed])
        release = lift + int(np.argmax(speed[lift:])) if lift < kin.F - 1 else kin.F - 1

        # 接地 = 「速く下りてきた足が止まる」ところ。足上げ〜リリースの間で探す。
        # 高さの閾値は使わない（マウンドの傾斜で踏み出し足は軸足より低く着く）。
        vel = np.diff(lf, prepend=lf[0])                # 負 = 下降
        contact = lift
        if release - lift >= 2:
            fast = lift + int(np.argmin(vel[lift:release + 1]))
            slowed = np.flatnonzero(
                vel[fast:release + 1] > FOOT_PLANT_SPEED_RATIO * vel[fast])
            contact = fast + int(slowed[0]) if len(slowed) else fast
        return {"lift": lift, "foot_contact": min(contact, release),
                "release": release}

    #: 接地からリリースまでの、力学的にあり得る範囲 [秒]。
    #:
    #: 投球では踏み出し足が着いてから 120〜150ms でリリースする。ここを
    #: 大きく外れた検出は、接地を取り違えている。**取り違えたまま数字を
    #: 出さない**ための関門で、外れたら依存する指標を NaN にする。
    CONTACT_TO_RELEASE_S = (0.08, 0.30)

    def measure(self, kin: Kinematics, phases: dict[str, int]) -> dict:
        fc, rel = phases["foot_contact"], phases["release"]
        body_h = kin.body_height_proxy()
        lo_s, hi_s = self.CONTACT_TO_RELEASE_S
        gap_s = (rel - fc) / kin.fps
        separated = lo_s <= gap_s <= hi_s
        nan = float("nan")

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
            "stride_m": stride if separated else nan,
            "stride_ratio": (float(stride / body_h)
                             if separated and body_h > 0 else nan),
            # 股関節-肩の分離（TIER C。位置ベースなので弱い）
            "hip_shoulder_separation_deg": float(xf[fc]) if separated else nan,
            "hip_shoulder_separation_max_deg": (float(xf[fc : rel + 1].max())
                                                if separated else nan),
            # リード脚。伸展するほど骨盤の回転が止まり上体へ力が移る
            "lead_knee_at_contact_deg": float(lead[fc]) if separated else nan,
            "lead_knee_at_release_deg": float(lead[rel]),
            "lead_knee_extension_deg": (float(lead[rel] - lead[fc])
                                        if separated else nan),
            # 接地を信用してよいか。False の指標は NaN にしてある
            "phases_separated": bool(separated),
            "contact_to_release_s": float(gap_s),
            "phases_note": ("" if separated else
                            f"接地→リリースが {gap_s * 1000:.0f}ms。"
                            f"力学的な範囲（{lo_s * 1000:.0f}〜{hi_s * 1000:.0f}ms）"
                            "の外なので、接地の検出を信用していません。"
                            "マウンドの傾斜で踏み出し足が軸足より低く着くため、"
                            "床からの高さでは接地を取れません"),
            # 上体
            "trunk_lean_at_release_deg": float(kin.trunk_lean()[rel]),
            "elbow_at_release_deg": float(kin.elbow_angle()[rel]),
            # 並進の健全性チェック（投球は前方へ大きく動く）
            "com_travel_m": float(np.linalg.norm(kin.com[rel] - kin.com[0])),
            "foot_clearance_m": float(feet[: rel + 1].max()),
            "kinetic_chain": kin.kinetic_chain(fc, min(rel + 2, kin.F)),
        }
