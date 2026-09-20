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

#: 接地とみなす水平移動量の、最大移動量に対する比。
#:
#: **接地は「高さ」ではなく「前進が止まること」で決める。**
#:
#: 高さで判定しようとして2回失敗した記録:
#:
#:   1. 「床+3cm」— マウンドは傾斜で、踏み出し足は軸足より**低く**着く。
#:      実測で推定した床を突き抜けて 6〜16cm 下まで行き、接地が 0.3秒遅れて
#:      **接地とリリースが同じフレームに潰れた**
#:   2. 「下降が止まる点」— 足上げから降り始める最初の減速を拾ってしまい、
#:      接地が 1.4秒も早く出た
#:
#: 水平方向にはこの問題が無い。踏み出し足は 1フレーム 15〜18cm で前進し、
#: 着いた瞬間に 0 になる。同じクリップで GVHMR は f58、SAM 3D Body は f59 と、
#: **独立した2手法が1フレーム差で一致した**。どちらもリリース(f62)の
#: 125〜167ms 前で、投球の実測レンジ（120〜150ms）に収まる。
FOOT_PLANT_STEP_RATIO = 0.25


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
        f"接地の判定（前進量が最大の {FOOT_PLANT_STEP_RATIO:.0%} を下回った点。"
        "2手法が1フレーム差で一致し実測レンジにも入ったが、比率そのものの"
        "根拠はまだ無い。足の速度センサ等で裏を取りたい）",
        "接地→リリースは実時間で120〜150ms。24fpsでは3〜4フレームしかなく、"
        "±1フレームで±33%ずれる。120fps以上で撮れば局面の精度が一段上がる",
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

        # 接地 = 「前へ出ていた足が止まる」ところ。足上げ〜リリースの間で探す。
        # 高さは使わない（マウンドの傾斜で踏み出し足は軸足より低く着く）。
        # 水平面の移動量の**大きさ**ではなく、**踏み出す向きの成分**を見る。
        # カメラ空間で出す手法（SAM 3D Body）では水平2軸の片方が奥行きで、
        # 単一画像モデルはそこが最も苦手。大きさを取ると奥行きのジッタが
        # 混じり、止まったあとも動いているように見えた。
        # 踏み出しの向きは「足上げ→リリース」の変位で決める。
        hz = [a for a in (0, 1, 2) if a != kin.up_ax]
        foot_h = kin.J[:, lead_foot[0]][:, hz]
        travel = foot_h[release] - foot_h[lift]
        n = float(np.linalg.norm(travel))
        if n > 1e-6:
            along = foot_h @ (travel / n)
            step = np.diff(along, prepend=along[0])      # 前向きが正
        else:
            step = np.concatenate(
                [[0.0], np.linalg.norm(np.diff(foot_h, axis=0), axis=-1)])
        # 踏み出しの**最大前進のフレームから前向きに**探し、前進が最初に
        # 閾値を割った点を接地とする。
        #
        # 「リリースから遡って最後に動いていた点」も試したが、打撃で破綻した。
        # 接地後も踏み出し足は小さく動き続ける（かかとの着地、つま先の向き
        # 直し。単一画像モデルではそこにジッタも乗る）ので、遡ると接地が
        # インパクトまでずれ込む。最大前進から前へ辿れば、その後の小さな
        # 動きは閾値の下で無視される。投球の実測（GVHMR f58 / SAM 3D Body f59）
        # はこの方法でも変わらない。
        contact = lift
        if release - lift >= 2:
            seg = step[lift:release + 1]
            peak = int(np.argmax(seg))
            below = np.flatnonzero(seg[peak:] < FOOT_PLANT_STEP_RATIO * seg[peak])
            contact = min(lift + peak + int(below[0]), release) if len(below) else release
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
