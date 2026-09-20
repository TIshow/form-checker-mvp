"""関節列から、競技に依存しない運動学的な量を取り出す。

ここに置いてよいのは「どの競技でも同じ意味を持つ量」だけ:
重心、床、関節角、体幹の傾き、捻転差、角速度、体格。

置いてはいけないのは「その競技だからそう測る」もの:
利き手の決め方、局面（打点・トップ・リリース）、指標の選択、閾値。
それらは domains/ にある。

入力は (F, 24, 3) の SMPL 24関節 world座標 [m]。他の骨格は
`core.convert` で並べ替えてから渡す。
"""

from __future__ import annotations

import numpy as np

from .geometry import (
    angle_from_axis, angular_speed, horizontal_angle, joint_angle, smooth,
)
from .skeleton import (
    FOOT_IDS, HEAD, L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, NECK, PELVIS,
    R_ANKLE, R_HIP, R_KNEE, R_SHOULDER, SEGMENTS, SIDED,
)

# --------------------------------------------------------------------------
# 計測上の定数
#
# 判定の閾値ではなく「その測定値が信用できるか」を決めるもの。
# ほとんど回転していないセグメントは角速度ピークの位置がノイズで決まるため、
# 順序を論じる材料にしてはいけない。
# --------------------------------------------------------------------------
CHAIN_MIN_SPEED_DEG_S = 30.0  # これ未満の角速度しか出ないセグメントは信用しない
CHAIN_REL_SPEED = 0.15        # 最大ピークに対する相対比


def compute_com(joints: np.ndarray) -> np.ndarray:
    """24関節から全身重心を計算する [m]。De Leva の体節質量比で加重平均。

    純関数。3D復元の外で導出できるため、ここが重心計算の唯一の定義。
    """
    com = np.zeros((joints.shape[0], 3))
    for a, b, mass, r in SEGMENTS:
        com += mass * (joints[:, a] * (1 - r) + joints[:, b] * r)
    return com / sum(s[2] for s in SEGMENTS)


def detect_up_axis(joints: np.ndarray) -> tuple[int, float]:
    """上方向の軸と符号を関節から推定する（頭 − 足首の平均ベクトル）。"""
    up_vec = (joints[:, HEAD] - (joints[:, L_ANKLE] + joints[:, R_ANKLE]) / 2).mean(0)
    up_ax = int(np.argmax(np.abs(up_vec)))
    return up_ax, float(np.sign(up_vec[up_ax]))


class Kinematics:
    """関節データから競技非依存の運動学量を取り出す。

    side  "R" / "L" … 利き側。**どちらかは競技が決める**（domains/ の責務）。
          サーブなら打点で高く上がる腕、投球なら投げる腕、ゴルフなら…と
          根拠が競技ごとに違うため、ここでは受け取るだけにしてある。
    """

    def __init__(self, joints: np.ndarray, fps: float = 30.0, side: str = "R"):
        self.J = np.asarray(joints)
        self.fps = float(fps)
        self.F = self.J.shape[0]
        self.side = side

        self.up_ax, self.up_sign = detect_up_axis(self.J)
        self.com = compute_com(self.J)
        self.com_height = self.height(self.com)

    # -- 基本 --------------------------------------------------------------
    def height(self, p: np.ndarray) -> np.ndarray:
        """world座標から「上方向の高さ」成分を取り出す(上向きが正)。"""
        return p[..., self.up_ax] * self.up_sign

    def up(self) -> np.ndarray:
        """上方向の単位ベクトル (3,)。"""
        v = np.zeros(3)
        v[self.up_ax] = self.up_sign
        return v

    def idx(self, name: str) -> int:
        """利き側の関節インデックスを返す。"""
        right, left = SIDED[name]
        return right if self.side == "R" else left

    def ground(self) -> float:
        """床の高さ。最も低い足の**中央値**。

        最低値だと一度きりの沈み込みが床になる。中央値なら跳躍や踏み込みに
        引っ張られない。世界座標の原点の置き方は復元手法ごとに違う
        （GVHMR は床を y≈0 に置くが TRAM は置かない）ため、高さは必ず
        原点ではなくここから測る。同じ動画で打点の高さが 2.00m と 0.48m に
        なった事故の再発防止。
        """
        return float(np.median(self.height(self.J[:, FOOT_IDS]).min(axis=1)))

    def height_above_ground(self, p: np.ndarray) -> np.ndarray:
        """床からの高さ [m]。"""
        return self.height(p) - self.ground()

    # -- 角度 --------------------------------------------------------------
    def knee_angles(self) -> np.ndarray:
        """左右の膝角度の小さい方 [deg]。180=伸展、**小さいほど深く曲げている**。"""
        left = joint_angle(self.J[:, L_HIP], self.J[:, L_KNEE], self.J[:, L_ANKLE])
        right = joint_angle(self.J[:, R_HIP], self.J[:, R_KNEE], self.J[:, R_ANKLE])
        return np.minimum(left, right)

    def elbow_angle(self) -> np.ndarray:
        """利き側の肘角度 [deg]。180=伸びきり、**大きいほど伸びている**。"""
        return joint_angle(self.J[:, self.idx("shoulder")],
                           self.J[:, self.idx("elbow")],
                           self.J[:, self.idx("wrist")])

    def trunk_lean(self) -> np.ndarray:
        """体幹(骨盤→首)が鉛直から傾いている角度 [deg]。0=直立。"""
        return angle_from_axis(self.J[:, NECK] - self.J[:, PELVIS], self.up())

    def x_factor(self) -> np.ndarray:
        """肩の軸と腰の軸の捻転差 [deg]（水平面上）。

        **弱い指標であることを承知で使うこと。** 体節が自分の軸まわりに回る
        成分は関節「位置」にほとんど現れない。回転が主役の競技
        （ゴルフ・野球）では、この値だけで捻転を語れない。
        """
        return horizontal_angle(
            self.J[:, R_SHOULDER] - self.J[:, L_SHOULDER],
            self.J[:, R_HIP] - self.J[:, L_HIP],
            self.up_ax,
        )

    def hand_direction(self) -> np.ndarray:
        """利き側の 手首→手 が鉛直から倒れる角度 [deg]。0=真上、180=真下。

        手の向きは、握っている道具のシャフト方向の近似になる。
        テニスでは**ラケットドロップ**（打点前に背中側へ落ちる動き）として、
        3手法の再現度の順位を目視評価どおりに並べた唯一の指標だった
        （GVHMR 116° > TRAM 72° = GEM-X 72°）。腕の高さ・上腕の傾き・肘角は
        いずれも逆の順位を示し、判断を誤らせた。

        ゴルフのフェース角やクラブ面はこれでは代用できない（issue 006）。
        """
        v = self.J[:, self.idx("hand")] - self.J[:, self.idx("wrist")]
        return angle_from_axis(v, self.up())

    def body_height_proxy(self) -> float:
        """立位の頭〜足の距離 [m]。高さを体格で正規化するために使う。

        頭の絶対高さではなく**足からの差**を採る。復元手法によって世界座標の
        原点の置き方が違うため（GVHMR は床を y≈0、TRAM は置かない）。
        絶対高さのままだと符号が負になり、正規化した指標が NaN になる。
        """
        n = max(1, self.F // 3)
        head_h = self.height(self.J[:, HEAD])[:n]
        foot_h = self.height(self.J[:, FOOT_IDS])[:n].min(axis=1)
        return float(np.median(head_h - foot_h))

    # -- キネティックチェーン ----------------------------------------------
    def chain_segments(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """既定の連鎖セグメント（腰→肩→上腕→前腕）。

        競技ごとに差し替えてよい。投球ならストライド脚、ゴルフなら
        クラブを足す、といった拡張はドメイン側で行う。
        """
        sh, el, wr = self.idx("shoulder"), self.idx("elbow"), self.idx("wrist")
        return [
            ("腰の回転", self.J[:, L_HIP], self.J[:, R_HIP]),
            ("肩の回転", self.J[:, L_SHOULDER], self.J[:, R_SHOULDER]),
            ("上腕", self.J[:, sh], self.J[:, el]),
            ("前腕", self.J[:, el], self.J[:, wr]),
        ]

    def kinetic_chain(self, lo: int, hi: int,
                      segments: list | None = None) -> list[dict]:
        """力の伝達順序を測る。近位から遠位へ順にピークが並ぶのが理想。

        lo, hi  加速区間のフレーム範囲。短すぎる場合はクリップ全体を使う。
        戻り値の `reliable` が False のものは回転が小さく、順序判定に使えない。
        """
        segs = segments if segments is not None else self.chain_segments()
        if hi - lo < 3:
            lo, hi = 0, self.F

        out = []
        for name, a, b in segs:
            speed = smooth(angular_speed(a, b, self.fps))[lo:hi]
            peak = int(np.argmax(speed))
            out.append({
                "segment": name,
                "peak_frame": lo + peak,
                "peak_speed": float(speed[peak]),
            })

        peak_max = max(c["peak_speed"] for c in out) if out else 0.0
        floor = max(CHAIN_MIN_SPEED_DEG_S, peak_max * CHAIN_REL_SPEED)
        for c in out:
            c["reliable"] = c["peak_speed"] >= floor
        return out


def dominant_side_by_peak_height(joints: np.ndarray) -> str:
    """手首が最も高く上がる方の腕を利き側とする。

    サーブ・スパイク・オーバーハンドスローのように**腕を頭上へ振る**
    動作でのみ妥当。ゴルフ（両手でクラブを握る）やオペラ（振らない）には
    使えないので、各ドメインが自分の根拠で決めること。
    """
    up_ax, up_sign = detect_up_axis(joints)
    h = joints[..., up_ax] * up_sign
    from .skeleton import L_WRIST, R_WRIST
    return "R" if h[:, R_WRIST].max() >= h[:, L_WRIST].max() else "L"
