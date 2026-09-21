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

from core import Kinematics, joint_angle, smooth
from core.skeleton import (
    HEAD, PELVIS, L_ANKLE, L_ELBOW, L_FOOT, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    R_ANKLE, R_ELBOW, R_FOOT, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
)
from domains.base import DEFAULT_CHAIN_MIN_FPS, NotImplementedDomain


class GolfSwing(NotImplementedDomain):
    name = "golf_swing"
    label = "ゴルフ スイング（指標のみ）"
    chain_min_fps = DEFAULT_CHAIN_MIN_FPS
    headline = ("x_factor_at_top_deg", "head_move_cm", "spine_tilt_change_deg",
                "tempo_ratio", "lead_knee_at_impact_deg", "grip_spread_sd_cm")
    #: 重心グラフはダウンスイング（トップ→インパクト）を見たい
    plot_phases = ("top", "impact")
    metric_labels = {
        "x_factor_at_top_deg": ("トップの捻転差", "°", 0, "位置ベース・弱い指標"),
        "head_move_cm": ("頭の上下動", "cm", 1, "足元基準。小さいほど軸が安定"),
        "spine_tilt_change_deg": ("前傾の変化", "°", 0, "アドレス→インパクト"),
        "tempo_ratio": ("テンポ比", "", 2, "バックスイング:ダウンスイング"),
        "lead_knee_at_impact_deg": ("インパクトのリード膝", "°", 0, "大きいほど伸びている"),
        "grip_spread_sd_cm": ("両手首の距離のばらつき", "cm", 1, "復元の健全性。大きいと腕が崩れている"),
    }
    phase_labels = {"address": "アドレス", "takeaway": "テイクバック",
                    "top": "トップ", "impact": "インパクト（代用）",
                    "finish": "フィニッシュ"}

    evidence_needed = (
        "インパクトの検出をクラブ追跡で置き換える（今は手の高さの最低点という代用）",
        "トップ・フィニッシュの検出を実際のスイング映像で検証する"
        "（手の高さだけで決めているので、手を上げない練習スイングでは崩れる）",
        "捻転差を位置ベースでなく体節の軸回転として測る手段（issue 011 §「軸回転」）",
        "前傾角は正面撮りでは信用できない（GVHMR と 18.7° 差。単眼は奥行き方向の"
        "傾きを決められない）。前傾角が要るなら横（飛球線後方）から撮る。"
        "正面で撮るなら最初に直立する1秒を入れ `--level-from` で打ち消す（videos/README.md）",
        "各指標のプロの実測レンジを、本文を開いて確認した出典で（セクション・図番号まで）",
        "自分の複数スイングでの測定ばらつき（CV）。テニスは 打点高1% / 足の浮き42% だった",
    )

    def side(self, joints: np.ndarray) -> str:
        """**リード側**（右打ちなら左）を返す。テニスの「ラケット側」とは意味が違う。

        根拠: フォロースルーで手（とクラブ）はリード側の肩の上に巻き付く。
        フィニッシュのフレームで、手が骨盤から見て左右どちらの肩の側に
        あるかを見る。

        「トップで伸びている方の腕がリード」も試したが、実映像では両肘が
        152° と 157° のようにほぼ同じで、判定が不安定だった（右打ちの
        ゴルファーを右リードと誤判定）。フィニッシュの手の位置の方が
        左右差が大きく、動作の定義そのものに近い。
        """
        fps = 30.0
        _, _, _, finish = GolfSwing._frames(
            GolfSwing._hand_height(joints), GolfSwing._hand_speed(joints, fps), fps)
        hands = (joints[finish, L_WRIST] + joints[finish, R_WRIST]) / 2
        lr = joints[finish, R_SHOULDER] - joints[finish, L_SHOULDER]   # 左→右
        return "R" if float((hands - joints[finish, PELVIS]) @ lr) > 0 else "L"

    @staticmethod
    def _hand_height(joints: np.ndarray) -> np.ndarray:
        """両手首の中点の高さ（骨盤相対）。上軸を知らない段階でも使えるよう素で計算する。"""
        from core import detect_up_axis
        ax, sg = detect_up_axis(joints)
        return ((joints[:, L_WRIST] + joints[:, R_WRIST]) / 2 - joints[:, PELVIS])[:, ax] * sg

    @staticmethod
    def _frames(hands_h: np.ndarray, hand_speed: np.ndarray, fps: float
                ) -> tuple[int, int, int, int]:
        """(テイクバック, トップ, インパクト, フィニッシュ)。

        **順番が大事。先にインパクトを決める。**

        インパクト = **手が最も速いフレーム**（平滑化後）。
        「手が最も低いフレーム」で代用しようとして失敗した: 手はアドレスでも
        同じくらい低く、構えて待つ 2〜3秒の間の揺れが最低点になって、
        インパクトが構えの途中に飛んだ（実映像で f47。正解は f82）。
        ダウンスイングの手の速度は他のどの局面よりずっと速いので、
        速度なら構えの揺れに引きずられない。

        トップ = インパクトの直前 1.5秒で手が最も高いフレーム。
        クリップ全体の最高点だとフィニッシュを拾う（手はトップと同等以上に
        上がる）。インパクトの手前に窓を切れば、その中の最高点はトップしかない。

        フィニッシュ = インパクト以降で手が最も高いフレーム。
        テイクバック = トップ手前で手が最後に最低位置にいたフレーム。
        """
        F = len(hands_h)
        if F < 4:
            return 0, 0, max(F - 1, 0), max(F - 1, 0)
        impact = int(np.argmax(hand_speed))
        lo = max(0, impact - int(round(1.5 * fps)))
        top = lo + int(np.argmax(hands_h[lo:impact + 1])) if impact > lo else impact
        finish = impact + int(np.argmax(hands_h[impact:]))
        return GolfSwing._takeaway(hand_speed, top), top, impact, finish

    @staticmethod
    def _takeaway(hand_speed: np.ndarray, top: int, frac: float = 0.10) -> int:
        """手が動き始めるフレーム。バックスイングの速さの頂点の手前で、
        手の速さが最後に小さかった次。

        以前は手の高さの「最低位置にいた最後の1枚」で取っていたが、構えて待つ
        間の揺れ（数cm）に敏感で、基準（床／骨盤）を変えるだけで 82 → 47 と
        1秒以上ずれた。速さなら、構えの揺れ（ピークの数%）とバックスイング
        （20〜30%）の間に閾値を置ける。

        **トップから遡ってはいけない。** トップで手は一瞬止まる（切り返し）ので、
        「トップまでで最後に遅かった枚」がトップ自身になり、テイクバックが 0 に
        飛ぶ。復元が滑らかなほど起きる（GEM-X の DDIM 出力で実際に起きた。
        regression 出力は閾値を 0.05 m/s 上回っていただけで、たまたま通っていた）。
        遡る起点はバックスイング中の速さの頂点にする。そこより手前で遅かった
        最後の枚は、構えの終わりしかない。
        """
        if top <= 0:
            return 0
        thr = frac * float(hand_speed.max())
        peak = int(np.argmax(hand_speed[: top + 1]))
        slow = np.flatnonzero(hand_speed[:peak + 1] < thr)
        return int(slow[-1]) + 1 if len(slow) and slow[-1] < peak else 0

    @staticmethod
    def _hand_speed(joints: np.ndarray, fps: float) -> np.ndarray:
        # 骨盤相対（体全体の並進のゆらぎを含めない）
        hands = (joints[:, L_WRIST] + joints[:, R_WRIST]) / 2 - joints[:, PELVIS]
        v = np.concatenate([[0.0], np.linalg.norm(np.diff(hands, axis=0), axis=-1)]) * fps
        return smooth(v, 3)

    @staticmethod
    def _top_frame(joints: np.ndarray, fps: float = 30.0) -> int:
        hands_h = GolfSwing._hand_height(joints)
        return GolfSwing._frames(hands_h, GolfSwing._hand_speed(joints, fps), fps)[1]

    def detect_phases(self, kin: Kinematics) -> dict[str, int]:
        """アドレス → テイクバック → トップ → インパクト（代用） → フィニッシュ。

        **インパクトはクラブを見ずに決めている。** 手が最も速いフレームを
        使う代用値で、真のインパクトではない。issue 006 が入るまでここは近似。
        """
        # 手の高さは**骨盤相対**。並進のゆらぎ（と床の推定）に影響されない。
        # ゴルフでは骨盤の上下動が小さいので、トップ・フィニッシュの検出には
        # これで足りる。
        hands_h = kin.height((kin.J[:, L_WRIST] + kin.J[:, R_WRIST]) / 2 - kin.J[:, PELVIS])
        takeaway, top, impact, finish = self._frames(
            hands_h, self._hand_speed(kin.J, kin.fps), kin.fps)
        return {"address": 0, "takeaway": takeaway, "top": top,
                "impact": impact, "finish": finish}

    def measure(self, kin: Kinematics, phases: dict[str, int]) -> dict:
        top, impact = phases["top"], phases["impact"]
        addr, takeaway = phases["address"], phases["takeaway"]
        fps = kin.fps

        # 頭の高さは**足元（両足首の中点）基準**。カメラ空間で返す手法は
        # フレームごとに体全体が数cm並進して見えるので、絶対高さだと
        # それが「頭の上下動」に化ける（実測: GVHMR 1.2cm に対し 11cm）。
        base = (kin.J[:, L_ANKLE] + kin.J[:, R_ANKLE]) / 2
        head_h = kin.height(kin.J[:, HEAD] - base)
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

        # 両手首の距離。**グリップを握っている間は一定（10〜15cm）のはず。**
        # 復元が腕を崩したかを見る健全性の指標で、フォームの指標ではない。
        # 実測（同じスイング）: GVHMR 15±5cm、SAM 3D Body 13±7cm、GEM-X 25±21cm——
        # GEM-X は手が離れ、テニスでラケットドロップが消えたのと同じ種類の崩れ。
        grip = np.linalg.norm(kin.J[:, L_WRIST] - kin.J[:, R_WRIST], axis=-1)
        grip_seg = grip[addr : impact + 1]

        return {
            "domain": self.name,
            "lead_side": kin.side,
            "grip_spread_mean_cm": float(grip_seg.mean() * 100),
            "grip_spread_sd_cm": float(grip_seg.std() * 100),
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
