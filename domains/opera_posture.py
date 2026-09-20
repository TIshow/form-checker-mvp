"""オペラ（発声時の姿勢）。**指標のみ。判定ルールはまだ無い。**

## 他の3競技と構造が違う

局面（打点・トップ・リリース）が無い。**持続的な状態**を測るため、
`detect_phases` は区間の端を返すだけで、指標は全部「平均とばらつき」になる。

これは有利に働く:

- 跳ばない・歩かない → グローバル並進の問題が消える
- 20〜40ms のイベントが無い → 時間分解能の要求が消える
- **単一画像モデル（SAM 3D Body）のジッタが、平均を取れば消える**
  他の3競技では角速度の微分でジッタが増幅されるが、ここでは起きない

## ただし、この競技には固有の壁がある

効くのは**頭頸部のアライメント・胸郭・顎・喉頭**だが、
**SMPL 24関節にはどれも無い。** 首は1関節、胸郭も顎も無い。
MHR（127関節）なら頭と顎に届く可能性があるが、喉頭はどの人体モデルにも無い。

ここで測っているのは「体幹と頭の大まかな配置」までで、
**歌唱の姿勢指導に足りるかは未検証**。設計より先に実機で確かめること。

## 結果（響き）は別系統

姿勢と響きの関係を言うには音声側の量が要る（SPL・スペクトル傾斜・
シンガーズフォルマント・ビブラート・F0の安定性）。**同じ動画の音声トラックに
入っている**ので追加機材は要らない。issue 010 がテニスで詰まっているのと違い、
ここは最初から結果が測れる。`core/audio.py` は未実装。

被験者内の反復測定（同じ人が同じフレーズを姿勢を変えて歌う）にできるため、
テニスで42本やっても差が出なかった被験者間のばらつきを回避できる。
"""

from __future__ import annotations

import numpy as np

from core import Kinematics
from core.skeleton import HEAD, L_SHOULDER, NECK, PELVIS, R_SHOULDER
from domains.base import NotImplementedDomain


class OperaPosture(NotImplementedDomain):
    name = "opera_posture"
    label = "オペラ 発声姿勢（指標のみ）"
    #: 局面が無いので連鎖の判定はしない
    chain_min_fps = float("inf")
    headline = ("head_forward_cm", "trunk_lean_mean_deg", "com_sway_cm",
                "shoulder_tilt_mean_deg", "trunk_length_change_cm")

    evidence_needed = (
        "SMPL 24関節で足りるかの実機検証。頸椎の詳細・胸郭・顎が無い",
        "MHR(127関節)にすると何が見えるようになるか（喉頭は依然として測れない）",
        "音声側の測定（core/audio.py 未実装）。SPL・スペクトル傾斜・"
        "シンガーズフォルマント・F0安定性",
        "同一歌手・同一フレーズで姿勢を変えた被験者内比較のプロトコル",
        "歌手本人の体感と、どの指標が対応するか",
    )

    def side(self, joints: np.ndarray) -> str:
        """左右の区別に意味が無いので既定を返す（左右差は別途測る）。"""
        return "R"

    def detect_phases(self, kin: Kinematics) -> dict[str, int]:
        """局面は無い。解析する区間の端だけを返す。"""
        return {"start": 0, "end": max(kin.F - 1, 0)}

    def measure(self, kin: Kinematics, phases: dict[str, int]) -> dict:
        lo, hi = phases["start"], phases["end"] + 1
        J = kin.J[lo:hi]
        up_ax = kin.up_ax

        # 頭部前方位: 頭が肩の中点より前後にどれだけ出ているか（水平距離）
        sh_mid = (J[:, L_SHOULDER] + J[:, R_SHOULDER]) / 2
        head_off = J[:, HEAD] - sh_mid
        head_off[:, up_ax] = 0.0
        head_fwd = np.linalg.norm(head_off, axis=-1)

        trunk = kin.trunk_lean()[lo:hi]

        # 体幹の長さ（骨盤→首）。胸郭の挙上の粗い代用にしかならない
        trunk_len = np.linalg.norm(J[:, NECK] - J[:, PELVIS], axis=-1)

        # 肩の左右の高さ差
        sh_tilt = np.degrees(np.arctan2(
            np.abs(kin.height(J[:, L_SHOULDER]) - kin.height(J[:, R_SHOULDER])),
            np.linalg.norm(J[:, L_SHOULDER] - J[:, R_SHOULDER], axis=-1) + 1e-9))

        # 重心の水平方向の揺れ
        com = kin.com[lo:hi].copy()
        com[:, up_ax] = 0.0
        sway = float(np.linalg.norm(com - com.mean(0), axis=-1).std())

        def stat(x, scale=1.0):
            return float(np.mean(x) * scale), float(np.std(x) * scale)

        hf_m, hf_s = stat(head_fwd, 100)
        tl_m, tl_s = stat(trunk)
        st_m, st_s = stat(sh_tilt)

        return {
            "domain": self.name,
            "fps": kin.fps,
            "n_frames": kin.F,
            "phases": phases,
            "head_forward_cm": hf_m,
            "head_forward_sd_cm": hf_s,
            "trunk_lean_mean_deg": tl_m,
            "trunk_lean_sd_deg": tl_s,
            "shoulder_tilt_mean_deg": st_m,
            "shoulder_tilt_sd_deg": st_s,
            "trunk_length_mean_cm": float(trunk_len.mean() * 100),
            "trunk_length_change_cm": float((trunk_len.max() - trunk_len.min()) * 100),
            "com_sway_cm": sway * 100,
            "com_height_sd_cm": float(kin.com_height[lo:hi].std() * 100),
            # 持続的な動作なので連鎖は測らない
            "kinetic_chain": [],
        }
