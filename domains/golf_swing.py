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
    #: 重心グラフはダウンスイング（トップ→インパクト）を見たい
    plot_phases = ("top", "impact")

    evidence_needed = (
        "インパクトの検出をクラブ追跡で置き換える（今は手の高さの最低点という代用）",
        "トップ・フィニッシュの検出を実際のスイング映像で検証する"
        "（手の高さだけで決めているので、手を上げない練習スイングでは崩れる）",
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
    def _hand_height(joints: np.ndarray) -> np.ndarray:
        """両手首の中点の高さ。上軸を知らない段階でも使えるよう素で計算する。"""
        from core import detect_up_axis
        ax, sg = detect_up_axis(joints)
        return ((joints[:, L_WRIST] + joints[:, R_WRIST]) / 2)[:, ax] * sg

    @staticmethod
    def _frames(hands: np.ndarray) -> tuple[int, int, int, int]:
        """(テイクバック, トップ, インパクト, フィニッシュ)。

        **順番が大事。先にインパクトを決める。**

        手の高さだけを見て「最も高い＝トップ」としてはいけない。ゴルフの
        フィニッシュは手がトップと同じか**それ以上に上がる**（ドライバーなら
        頭上に来る）ので、クリップ全体の argmax はフィニッシュを拾う。
        そうなるとトップ＝インパクト＝フィニッシュが同じフレームに潰れ、
        リード側の判定まで裏返る（トップでは伸びているのがリード腕だが、
        フィニッシュでは左右が入れ替わるため）。

        なので、
          インパクト   = 両端を除いた区間で手が最も低いフレーム
          トップ       = その**手前**で手が最も高いフレーム
          フィニッシュ = その**後**で手が最も高いフレーム
          テイクバック = トップの手前で手が**最後に**最低位置にいたフレーム
                         （前に立っているだけの映像がどれだけ付いていても、
                           バックスイングの長さが伸びないようにする。
                           最初の最低位置ではなく最後を採るのがポイント——
                           アドレスで構えている間は高さが変わらないので、
                           argmin だと必ず先頭が返ってしまう）
        """
        F = len(hands)
        if F < 4:
            return 0, 0, max(F - 1, 0), max(F - 1, 0)
        impact = 1 + int(np.argmin(hands[1:F - 1]))
        top = int(np.argmax(hands[:impact])) if impact > 0 else 0
        finish = impact + int(np.argmax(hands[impact:]))
        return GolfSwing._takeaway(hands, top), top, impact, finish

    @staticmethod
    def _takeaway(hands: np.ndarray, top: int, tol: float = 0.02) -> int:
        """手が上がり始めるフレーム。トップ手前で最低位置にいた**最後**の1枚。

        tol はバックスイングの振幅に対する比。アドレスの静止中は高さが
        ぴったり同じとは限らないので、厳密な最小値ではなく幅を持たせる。
        """
        if top <= 0:
            return 0
        pre = hands[: top + 1]
        low = float(pre.min())
        rise = float(hands[top]) - low
        if rise <= 0:
            return top
        at_low = np.flatnonzero(pre <= low + tol * rise)
        return int(at_low[-1]) if len(at_low) else 0

    @staticmethod
    def _top_frame(joints: np.ndarray) -> int:
        return GolfSwing._frames(GolfSwing._hand_height(joints))[1]

    def detect_phases(self, kin: Kinematics) -> dict[str, int]:
        """アドレス → テイクバック → トップ → インパクト（代用） → フィニッシュ。

        **インパクトはクラブを見ずに決めている。** 手が最も低くなるフレームを
        使う代用値で、真のインパクトではない。issue 006 が入るまでここは近似。
        """
        takeaway, top, impact, finish = self._frames(
            kin.height((kin.J[:, L_WRIST] + kin.J[:, R_WRIST]) / 2))
        return {"address": 0, "takeaway": takeaway, "top": top,
                "impact": impact, "finish": finish}

    def measure(self, kin: Kinematics, phases: dict[str, int]) -> dict:
        top, impact = phases["top"], phases["impact"]
        addr, takeaway = phases["address"], phases["takeaway"]
        fps = kin.fps

        head_h = kin.height(kin.J[:, HEAD])
        trunk = kin.trunk_lean()
        knee_l = joint_angle(kin.J[:, L_HIP], kin.J[:, L_KNEE], kin.J[:, L_ANKLE])
        knee_r = joint_angle(kin.J[:, R_HIP], kin.J[:, R_KNEE], kin.J[:, R_ANKLE])
        lead = knee_l if kin.side == "L" else knee_r
        xf = kin.x_factor()
        feet = kin.height(kin.J[:, [L_ANKLE, R_ANKLE, L_FOOT, R_FOOT]]).min(axis=1) - kin.ground()

        # バックスイングは**テイクバックから**測る。アドレスから測ると、
        # 前に立っているだけの映像の長さがそのまま tempo_ratio に乗る。
        back_s = (top - takeaway) / fps
        down_s = (impact - top) / fps
        # トップとインパクトが同じフレームに潰れたクリップ（スイングが
        # 入っていない／トップで切れている）では、テンポは測れない。
        # 1フレームに丸めて「2.1」のような妥当に見える数字を出さない。
        tempo = back_s / down_s if down_s > 0 and back_s > 0 else float("nan")

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
            "tempo_ratio": float(tempo),
            # 接地の健全性チェック。ゴルフは跳ばないので 0 付近のはず
            "foot_clearance_m": float(feet[: impact + 1].max()),
            "kinetic_chain": kin.kinetic_chain(top, min(impact + 2, kin.F)),
        }
