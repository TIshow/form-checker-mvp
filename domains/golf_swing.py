"""ゴルフのスイング。**指標のみ。判定ルールはまだ無い。**

## なぜ判定が空なのか

閾値には出典が要る（`domains/base.py` の TIER A/B/C）。テニスでは、
検索結果から拾った数値を「文献値」と称して判定し、**論文に存在しない値で
プロの技術を欠点と呼んでいた**。同じ失敗を繰り返さないため、ゴルフは
**測るところまでで止めてある**。`evidence_needed` を埋めてから judge を書く。

## ゴルフがテニスより楽なところ / 難しいところ

楽: その場で振るのでグローバル並進がほぼ無い。跳躍が再現されない問題
    （issue 008）も、接地の事前分布が邪魔をする問題も起きない。
    単一画像モデル（SAM 3D Body）を使う場合、この差は特に大きい。

難: **自己遮蔽**（トップで腕とクラブが体を横切る）と**軸回転**。
    スイングは骨盤と胸郭の捻転そのものだが、体節が自分の軸まわりに回る
    成分は関節「位置」にほとんど現れない。`x_factor` は位置から出している
    ので、ゴルフでは特に弱い。

## クラブが無いことの影響

テニスは手の向きでラケットを代用できた（`hand_direction`）。ゴルフの
**シャフト面とフェース角は競技の中心**で、手の向きでは代用できない。
issue 006（道具の追跡）は、ゴルフでは回避不能な前提条件になる。
"""

from __future__ import annotations

import numpy as np

from core import Kinematics, joint_angle
from core.skeleton import (
    HEAD, L_ANKLE, L_ELBOW, L_FOOT, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    R_ANKLE, R_ELBOW, R_FOOT, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
)
from domains.base import DEFAULT_CHAIN_MIN_FPS, NotImplementedDomain


class GolfSwing(NotImplementedDomain):
    name = "golf_swing"
    label = "ゴルフ スイング（指標のみ）"
    chain_min_fps = DEFAULT_CHAIN_MIN_FPS
    headline = ("x_factor_at_top_deg", "head_move_cm", "spine_tilt_change_deg",
                "tempo_ratio", "lead_knee_at_impact_deg")

    evidence_needed = (
        "インパクトの検出をクラブ追跡で置き換える（今は手の高さの最低点という代用）",
        "捻転差を位置ベースでなく体節の軸回転として測る手段（issue 011 §「軸回転」）",
        "各指標のプロの実測レンジを、本文を開いて確認した出典で（セクション・図番号まで）",
        "自分の複数スイングでの測定ばらつき（CV）。テニスは 打点高1% / 足の浮き42% だった",
    )

    def side(self, joints: np.ndarray) -> str:
        """**リード側**（右打ちなら左）を返す。テニスの「ラケット側」とは意味が違う。

        根拠: トップでリード腕は伸び、トレール肘は曲がる。両手でクラブを
        握るため、テニスの「手首が高く上がる方」は使えない。
        これは近似であり、アドレスの向きから決める方が確実。
        """
        top = self._top_frame(joints)
        le = joint_angle(joints[:, L_SHOULDER], joints[:, L_ELBOW], joints[:, L_WRIST])
        re = joint_angle(joints[:, R_SHOULDER], joints[:, R_ELBOW], joints[:, R_WRIST])
        return "L" if le[top] >= re[top] else "R"

    @staticmethod
    def _top_frame(joints: np.ndarray) -> int:
        """トップ（手が最も高い）。上軸を知らない段階でも使えるよう素で計算する。"""
        from core import detect_up_axis
        ax, sg = detect_up_axis(joints)
        hands = ((joints[:, L_WRIST] + joints[:, R_WRIST]) / 2)[:, ax] * sg
        return int(np.argmax(hands))

    def detect_phases(self, kin: Kinematics) -> dict[str, int]:
        """アドレス → トップ → インパクト（代用） → フィニッシュ。

        **インパクトはクラブを見ずに決めている。** トップ以降で手が最も低く
        なるフレームを使う代用値で、真のインパクトではない。issue 006 が
        入るまでここは近似。
        """
        hands = kin.height((kin.J[:, L_WRIST] + kin.J[:, R_WRIST]) / 2)
        top = int(np.argmax(hands))
        after = hands[top:]
        impact = top + int(np.argmin(after)) if len(after) > 1 else top
        finish = int(np.argmax(hands[impact:])) + impact if impact < kin.F - 1 else kin.F - 1
        return {"address": 0, "top": top, "impact": impact, "finish": finish}

    def measure(self, kin: Kinematics, phases: dict[str, int]) -> dict:
        top, impact = phases["top"], phases["impact"]
        addr = phases["address"]
        fps = kin.fps

        head_h = kin.height(kin.J[:, HEAD])
        trunk = kin.trunk_lean()
        knee_l = joint_angle(kin.J[:, L_HIP], kin.J[:, L_KNEE], kin.J[:, L_ANKLE])
        knee_r = joint_angle(kin.J[:, R_HIP], kin.J[:, R_KNEE], kin.J[:, R_ANKLE])
        lead = knee_l if kin.side == "L" else knee_r
        xf = kin.x_factor()
        feet = kin.height(kin.J[:, [L_ANKLE, R_ANKLE, L_FOOT, R_FOOT]]).min(axis=1) - kin.ground()

        back_s = (top - addr) / fps
        down_s = max(impact - top, 1) / fps

        return {
            "domain": self.name,
            "lead_side": kin.side,
            "fps": fps,
            "n_frames": kin.F,
            "phases": phases,
            # 捻転（TIER C。位置ベースなので弱い）
            "x_factor_at_top_deg": float(xf[top]),
            "x_factor_max_deg": float(xf[: impact + 1].max()),
            # 頭の上下動。スイング中の軸のブレ
            "head_move_cm": float((head_h[: impact + 1].max()
                                   - head_h[: impact + 1].min()) * 100),
            # 前傾の維持。アドレスとインパクトの体幹角の差
            "spine_tilt_address_deg": float(trunk[addr]),
            "spine_tilt_impact_deg": float(trunk[impact]),
            "spine_tilt_change_deg": float(trunk[impact] - trunk[addr]),
            # リード膝
            "lead_knee_at_top_deg": float(lead[top]),
            "lead_knee_at_impact_deg": float(lead[impact]),
            # テンポ。バックスイング : ダウンスイング
            "backswing_s": float(back_s),
            "downswing_s": float(down_s),
            "tempo_ratio": float(back_s / down_s) if down_s > 0 else float("nan"),
            # 接地の健全性チェック。ゴルフは跳ばないので 0 付近のはず
            "foot_clearance_m": float(feet[: impact + 1].max()),
            "kinetic_chain": kin.kinetic_chain(top, min(impact + 2, kin.F)),
        }
