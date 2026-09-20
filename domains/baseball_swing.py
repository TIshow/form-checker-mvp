"""野球の打撃（バッティング）。**指標のみ。判定ルールはまだ無い。**

## 投球と何が違うか

同じ野球でも、投球（`baseball_pitch.py`）とは別のドメインとして持つ。

- **平らな地面で打つ。** マウンドの傾斜が無いので、投球で捨てた「足の高さ」も
  使える。ただし投球と同じく**前進が止まる点**で接地を取る方が素材を選ばない
- **両手でバットを握る。** ゴルフと同じで「手首が高く上がる腕」では利き側が
  決まらない。踏み出す方の足で**リード側**を決める
- **回転が主役。** 骨盤→胸郭→腕→バットの順で回る。`x_factor` は位置ベースで
  弱い指標なので、捻転の数字は参考に留める
- **バットを追っていない。** インパクトは「手が最も速いフレーム」で代用する。
  真のインパクトはバットの先端で起き、手の最速とは数フレームずれうる
  （issue 006）。ボールも見ていない

## 撮影について

投球（内旋 約7,000°/秒）ほどではないが、スイングも骨盤→肩→手の時間差は
数十 ms しかない。連鎖の順序を論じるなら **120fps 以上**（見積もり・未検証）。
24fps の放送映像では局面（接地・インパクト）までは取れるが、連鎖は判定しない。
"""

from __future__ import annotations

import numpy as np

from core import Kinematics, joint_angle, smooth
from core.skeleton import (
    FOOT_IDS, HEAD, L_ANKLE, L_HIP, L_KNEE, L_WRIST, PELVIS, R_ANKLE, R_HIP,
    R_KNEE, R_WRIST,
)
from domains.base import NotImplementedDomain

#: 連鎖の順序を論じるのに必要と見積もったフレームレート。**未検証**。
SWING_CHAIN_MIN_FPS = 120.0

#: 接地とみなす移動量の、最大移動量に対する比（投球と同じ考え方）。
FOOT_PLANT_STEP_RATIO = 0.25


class BaseballSwing(NotImplementedDomain):
    name = "baseball_swing"
    label = "野球 打撃（指標のみ）"
    chain_min_fps = SWING_CHAIN_MIN_FPS
    headline = ("stride_ratio", "hip_shoulder_separation_deg", "plant_to_contact_s",
                "lead_knee_at_contact_deg", "head_move_cm", "hand_speed_max_mps")
    plot_phases = ("foot_plant", "contact")
    metric_labels = {
        "stride_ratio": ("ストライド", "身長比", 2, "構え→接地の水平移動"),
        "hip_shoulder_separation_deg": ("股関節と肩の分離", "°", 0, "接地時・位置ベースで弱い指標"),
        "plant_to_contact_s": ("接地→インパクト", "秒", 2, "短いほど速い切り返し"),
        "lead_knee_at_contact_deg": ("インパクト時のリード膝", "°", 0, "大きいほど伸びている（壁）"),
        "head_move_cm": ("頭の移動", "cm", 1, "構え→インパクト。小さいほど軸が安定"),
        "hand_speed_max_mps": ("手の最高速度", "m/s", 1, "インパクトの代用に使った値"),
    }
    phase_labels = {"stance": "構え", "lift": "足上げ", "foot_plant": "踏み出し足の接地",
                    "contact": "インパクト（代用）", "finish": "フィニッシュ"}

    #: 接地→インパクトの、力学的にあり得る範囲 [秒]。**投球より幅を広くとる。**
    #: 打者は投球のタイミングに合わせて間を作るので、接地からインパクトまでの
    #: 時間は投球ほど一定ではない。0.05 未満なら潰れている、0.6 超なら
    #: 接地の取り違え、とみなす。根拠は無く、実映像で決め直すこと。
    PLANT_TO_CONTACT_S = (0.05, 0.60)

    evidence_needed = (
        "インパクトの検出をバット／ボール追跡で置き換える（今は手の最高速度という代用）",
        "接地→インパクトの妥当範囲（今は 50〜600ms と広く置いている。出典なし）",
        "捻転（股関節と肩の分離）を位置ベースでなく体節の軸回転として測る手段",
        "連鎖の判定に必要なフレームレートの実測（今は 120fps と見積もり）",
        "各指標の基準レンジを、本文を開いて確認した出典で",
    )

    # -- 利き側 ------------------------------------------------------------
    def side(self, joints: np.ndarray) -> str:
        """**リード側**（右打ちなら左）。踏み出しで大きく動く方の足。

        両手でバットを握るので「手首が高く上がる腕」では決まらない。
        構えから振り終わりまでで、足首の移動距離が長い方をリード足とする。
        """
        path = lambda j: float(np.linalg.norm(np.diff(joints[:, j], axis=0), axis=-1).sum())
        return "L" if path(L_ANKLE) >= path(R_ANKLE) else "R"

    # -- 局面 --------------------------------------------------------------
    def detect_phases(self, kin: Kinematics) -> dict[str, int]:
        """構え → 足上げ → 踏み出し足の接地 → インパクト（代用） → フィニッシュ。

        インパクトは**手（両手首の中点）が最も速いフレーム**で代用する。
        接地は投球と同じく「踏み出す向きの前進が最大になった後、最初に止まる点」。
        """
        F = kin.F
        lead = [L_ANKLE] if kin.side == "L" else [R_ANKLE]
        rear_id = R_ANKLE if kin.side == "L" else L_ANKLE
        # 踏み出し足の高さは**後ろ足に対して**（並進のゆらぎを含めない）
        lead_h = kin.height(kin.J[:, lead[0]] - kin.J[:, rear_id])

        # 手の速さは**骨盤相対**（体全体の並進のゆらぎを含めない）。
        # 平滑化してから最速を取る。単一画像モデルは1フレームだけ手が飛ぶことが
        # あり（実測: クリップ末尾のカメラの揺れで 5.8m/s の孤立ピーク）、
        # 生の argmax はそれを「インパクト」にしてしまった。
        hands = (kin.J[:, L_WRIST] + kin.J[:, R_WRIST]) / 2 - kin.J[:, PELVIS]
        speed = smooth(np.concatenate(
            [[0.0], np.linalg.norm(np.diff(hands, axis=0), axis=-1)]), 3)
        contact = int(np.argmax(speed))

        # 足上げ = インパクト前で踏み出し足が最も高いフレーム
        lift = int(np.argmax(lead_h[: max(contact, 1)]))

        # 接地 = 足上げ〜インパクトの間で、踏み出す向きの前進が最後に止まった点
        plant = lift
        if contact - lift >= 2:
            # 前進は**軸足（後ろ足）に対して**測る（並進のゆらぎを含めない）
            hz = [a for a in (0, 1, 2) if a != kin.up_ax]
            rear = R_ANKLE if kin.side == "L" else L_ANKLE
            foot_h = (kin.J[:, lead[0]] - kin.J[:, rear])[:, hz]
            travel = foot_h[contact] - foot_h[lift]
            n = float(np.linalg.norm(travel))
            if n > 1e-6:
                along = foot_h @ (travel / n)
                step = np.diff(along, prepend=along[0])
            else:
                step = np.concatenate(
                    [[0.0], np.linalg.norm(np.diff(foot_h, axis=0), axis=-1)])
            # 最大前進から前向きに探す（理由は baseball_pitch.py の同じ箇所）
            seg = step[lift:contact + 1]
            peak = int(np.argmax(seg))
            below = np.flatnonzero(seg[peak:] < FOOT_PLANT_STEP_RATIO * max(seg[peak], 1e-9))
            plant = min(lift + peak + int(below[0]), contact) if len(below) else contact

        hands_abs = (kin.J[:, L_WRIST] + kin.J[:, R_WRIST]) / 2
        finish = contact + int(np.argmax(kin.height(hands_abs)[contact:])) if contact < F - 1 else F - 1
        return {"stance": 0, "lift": lift, "foot_plant": plant,
                "contact": contact, "finish": finish}

    # -- 指標 --------------------------------------------------------------
    def measure(self, kin: Kinematics, phases: dict[str, int]) -> dict:
        lift, plant, contact = phases["lift"], phases["foot_plant"], phases["contact"]
        fps = kin.fps
        body_h = kin.body_height_proxy()
        lo_s, hi_s = self.PLANT_TO_CONTACT_S
        gap_s = (contact - plant) / fps
        separated = lo_s <= gap_s <= hi_s
        nan = float("nan")

        lead_ankle = L_ANKLE if kin.side == "L" else R_ANKLE
        knee_l = joint_angle(kin.J[:, L_HIP], kin.J[:, L_KNEE], kin.J[:, L_ANKLE])
        knee_r = joint_angle(kin.J[:, R_HIP], kin.J[:, R_KNEE], kin.J[:, R_ANKLE])
        lead_knee = knee_l if kin.side == "L" else knee_r
        xf = kin.x_factor()
        hands = (kin.J[:, L_WRIST] + kin.J[:, R_WRIST]) / 2
        speed = np.concatenate([[0.0], np.linalg.norm(np.diff(hands, axis=0), axis=-1)]) * fps
        head_move = float(np.linalg.norm(kin.J[contact, HEAD] - kin.J[0, HEAD]))
        # ストライド = **構え→接地**の水平移動。足上げ（高さの頂点）から測ると、
        # 頂点が踏み出しの途中に来るぶん短く出る（合成データで 0.30m が 0.21m）。
        hz = [a for a in (0, 1, 2) if a != kin.up_ax]
        stance = phases["stance"]
        stride = float(np.linalg.norm(
            kin.J[plant, lead_ankle][hz] - kin.J[stance, lead_ankle][hz]))
        feet = kin.height(kin.J[:, FOOT_IDS]).min(axis=1) - kin.ground()

        return {
            "domain": self.name,
            "lead_side": kin.side,
            "fps": fps,
            "n_frames": kin.F,
            "phases": phases,
            "stride_m": stride if separated else nan,
            "stride_ratio": (float(stride / body_h) if separated and body_h > 0 else nan),
            "hip_shoulder_separation_deg": float(xf[plant]) if separated else nan,
            "hip_shoulder_separation_max_deg": (float(xf[plant:contact + 1].max())
                                                if separated else nan),
            "plant_to_contact_s": float(gap_s) if separated else nan,
            "lead_knee_at_plant_deg": float(lead_knee[plant]) if separated else nan,
            "lead_knee_at_contact_deg": float(lead_knee[contact]),
            "lead_knee_extension_deg": (float(lead_knee[contact] - lead_knee[plant])
                                        if separated else nan),
            "trunk_lean_at_contact_deg": float(kin.trunk_lean()[contact]),
            "head_move_cm": head_move * 100,
            "hand_speed_max_mps": float(speed.max()),
            "foot_clearance_m": float(feet[: contact + 1].max()),
            "phases_separated": bool(separated),
            "phases_note": ("" if separated else
                            f"接地→インパクトが {gap_s * 1000:.0f}ms。"
                            f"想定範囲（{lo_s * 1000:.0f}〜{hi_s * 1000:.0f}ms）の外なので、"
                            "接地の検出を信用していません。"),
            "kinetic_chain": kin.kinetic_chain(plant, min(contact + 2, kin.F)),
        }
