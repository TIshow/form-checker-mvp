"""サーブの3D関節データから運動学的な指標を計測する。

ここには「測る」ことだけを置く。良し悪しの判定は feedback.py、
表示は report.py の担当。物理計算は滅多に変わらないが判定の閾値は
頻繁に変わるため、両者を分けている。

入力は GVHMR パイプラインが出力する numpy 配列のみ:
  gv_joints.npy  (F, 24, 3)  SMPL 24関節の world座標 [m]
  gv_com.npy     (F, 3)      全身重心の world座標 [m]
  gv_upaxis.npy  (2,)        [上方向の軸index, 符号]

純 numpy のため GVHMR 環境 (Python 3.10 venv) は不要。
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# SMPL 24関節のインデックス
# --------------------------------------------------------------------------
PELVIS, L_HIP, R_HIP, SPINE1 = 0, 1, 2, 3
L_KNEE, R_KNEE, SPINE2 = 4, 5, 6
L_ANKLE, R_ANKLE, SPINE3 = 7, 8, 9
L_FOOT, R_FOOT, NECK = 10, 11, 12
L_COLLAR, R_COLLAR, HEAD = 13, 14, 15
L_SHOULDER, R_SHOULDER = 16, 17
L_ELBOW, R_ELBOW = 18, 19
L_WRIST, R_WRIST = 20, 21
L_HAND, R_HAND = 22, 23

# --------------------------------------------------------------------------
# 計測上の定数
#
# 判定の閾値ではなく「その測定値が信用できるか」を決めるもの。
# ほとんど回転していないセグメントは角速度ピークの位置がノイズで決まるため、
# 順序を論じる材料にしてはいけない。
# --------------------------------------------------------------------------
CHAIN_MIN_SPEED_DEG_S = 30.0  # これ未満の角速度しか出ないセグメントは信用しない
CHAIN_REL_SPEED = 0.15        # 最大ピークに対する相対比


# --------------------------------------------------------------------------
# 幾何ユーティリティ
# --------------------------------------------------------------------------
def unit(v: np.ndarray) -> np.ndarray:
    """ゼロ除算を避けて正規化する。"""
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """b を頂点とする a-b-c の3D角度 [deg]。各引数は (F,3)。"""
    cos = np.sum(unit(a - b) * unit(c - b), axis=-1).clip(-1.0, 1.0)
    return np.degrees(np.arccos(cos))


def angular_speed(p_start: np.ndarray, p_end: np.ndarray, fps: float) -> np.ndarray:
    """線分 p_start->p_end の向きが変化する角速度 [deg/s]。(F,) を返す。"""
    d = unit(p_end - p_start)
    cos = np.sum(d[1:] * d[:-1], axis=-1).clip(-1.0, 1.0)
    speed = np.degrees(np.arccos(cos)) * fps
    return np.concatenate([[0.0], speed])


def smooth(x: np.ndarray, win: int = 5) -> np.ndarray:
    """移動平均。角速度はノイズが乗るため平滑化してからピークを取る。"""
    if win <= 1 or len(x) < win:
        return x
    return np.convolve(x, np.ones(win) / win, mode="same")


def horizontal(v: np.ndarray, up_ax: int) -> np.ndarray:
    """上方向成分を除去して水平面に射影する。"""
    out = v.copy()
    out[..., up_ax] = 0.0
    return out


def horizontal_angle(v1: np.ndarray, v2: np.ndarray, up_ax: int) -> np.ndarray:
    """水平面に射影した2ベクトルのなす角 [deg]。(F,) を返す。"""
    a, b = unit(horizontal(v1, up_ax)), unit(horizontal(v2, up_ax))
    cos = np.sum(a * b, axis=-1).clip(-1.0, 1.0)
    return np.degrees(np.arccos(cos))


# --------------------------------------------------------------------------
# 運動学
# --------------------------------------------------------------------------
class ServeKinematics:
    """関節データからサーブの運動学的な量を取り出す。"""

    #: 利き手側に読み替える関節名 -> (右, 左)
    _SIDED = {
        "shoulder": (R_SHOULDER, L_SHOULDER),
        "elbow": (R_ELBOW, L_ELBOW),
        "wrist": (R_WRIST, L_WRIST),
        "hand": (R_HAND, L_HAND),
        "hip": (R_HIP, L_HIP),
        "knee": (R_KNEE, L_KNEE),
        "ankle": (R_ANKLE, L_ANKLE),
    }

    def __init__(self, joints: np.ndarray, com: np.ndarray,
                 up_ax: int, up_sign: float, fps: float = 30.0):
        self.J = joints
        self.com = com
        self.up_ax = int(up_ax)
        self.up_sign = float(up_sign)
        self.fps = float(fps)
        self.F = joints.shape[0]

        self.com_height = self.height(com)
        self.racket_side = self._detect_racket_side()

    def height(self, p: np.ndarray) -> np.ndarray:
        """world座標から「地面からの高さ」成分を取り出す(上向きが正)。"""
        return p[..., self.up_ax] * self.up_sign

    def _detect_racket_side(self) -> str:
        """ラケットを持つ側を判定する。サーブでは打点で最も高く上がる腕。"""
        left = self.height(self.J[:, L_WRIST]).max()
        right = self.height(self.J[:, R_WRIST]).max()
        return "R" if right >= left else "L"

    def idx(self, name: str) -> int:
        """利き手側の関節インデックスを返す。"""
        right, left = self._SIDED[name]
        return right if self.racket_side == "R" else left

    # -- フェーズ検出 ------------------------------------------------------
    def detect_phases(self) -> dict:
        """打点・沈み込みのフレームを検出する。

        打点    = ラケット側手首が最も高くなるフレーム
        沈み込み = その手前で重心が最も低くなるフレーム
        """
        wrist_h = self.height(self.J[:, self.idx("wrist")])
        contact = int(np.argmax(wrist_h))
        loading = int(np.argmin(self.com_height[: max(contact, 1)]))
        com_peak = int(np.argmax(self.com_height))
        return {"loading": loading, "contact": contact, "com_peak": com_peak}

    # -- 角度 --------------------------------------------------------------
    def knee_angles(self) -> np.ndarray:
        """左右の膝角度の小さい方 [deg]。180=伸展、小さいほど深く曲げている。"""
        left = joint_angle(self.J[:, L_HIP], self.J[:, L_KNEE], self.J[:, L_ANKLE])
        right = joint_angle(self.J[:, R_HIP], self.J[:, R_KNEE], self.J[:, R_ANKLE])
        return np.minimum(left, right)

    def elbow_angle(self) -> np.ndarray:
        """ラケット側の肘角度 [deg]。打点では伸びきる(180に近い)のが理想。"""
        return joint_angle(self.J[:, self.idx("shoulder")],
                           self.J[:, self.idx("elbow")],
                           self.J[:, self.idx("wrist")])

    def trunk_lean(self) -> np.ndarray:
        """体幹(骨盤→首)が鉛直から傾いている角度 [deg]。"""
        up = np.zeros(3)
        up[self.up_ax] = self.up_sign
        trunk = unit(self.J[:, NECK] - self.J[:, PELVIS])
        cos = (trunk * up).sum(-1).clip(-1.0, 1.0)
        return np.degrees(np.arccos(cos))

    def x_factor(self) -> np.ndarray:
        """肩の軸と腰の軸の捻転差 [deg]（水平面上）。"""
        return horizontal_angle(
            self.J[:, R_SHOULDER] - self.J[:, L_SHOULDER],
            self.J[:, R_HIP] - self.J[:, L_HIP],
            self.up_ax,
        )

    def body_height_proxy(self) -> float:
        """立位の頭の高さ。打点の高さを体格で正規化するために使う。"""
        head_h = self.height(self.J[:, HEAD])
        return float(np.median(head_h[: max(1, self.F // 3)]))

    # -- キネティックチェーン ----------------------------------------------
    def kinetic_chain(self, loading: int, contact: int) -> list[dict]:
        """力の伝達順序を測る。

        腰→肩→上腕→前腕 の順に角速度のピークが並ぶのが理想。
        各セグメントの、加速区間内でのピーク時刻と速さを返す。
        `reliable` が False のものは回転が小さく、順序判定に使えない。
        """
        sh, el, wr = self.idx("shoulder"), self.idx("elbow"), self.idx("wrist")
        segments = [
            ("腰の回転", self.J[:, L_HIP], self.J[:, R_HIP]),
            ("肩の回転", self.J[:, L_SHOULDER], self.J[:, R_SHOULDER]),
            ("上腕", self.J[:, sh], self.J[:, el]),
            ("前腕", self.J[:, el], self.J[:, wr]),
        ]

        # 加速区間: 沈み込み〜打点直後。短すぎる場合はクリップ全体を使う。
        lo, hi = loading, min(contact + 2, self.F)
        if hi - lo < 3:
            lo, hi = 0, self.F

        out = []
        for name, a, b in segments:
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


# --------------------------------------------------------------------------
# 指標の集計
# --------------------------------------------------------------------------
def compute_metrics(kin: ServeKinematics) -> dict:
    """運動学から、判定に使う指標一式を辞書で返す。"""
    ph = kin.detect_phases()
    loading, contact, com_peak = ph["loading"], ph["contact"], ph["com_peak"]
    fps = kin.fps

    knee = kin.knee_angles()
    wrist_h = kin.height(kin.J[:, kin.idx("wrist")])
    body_h = kin.body_height_proxy()

    return {
        "racket_side": kin.racket_side,
        "fps": fps,
        "n_frames": kin.F,
        "phases": ph,
        # 脚のドライブ
        "com_low_m": float(kin.com_height[loading]),
        "com_peak_m": float(kin.com_height[com_peak]),
        "com_rise_m": float(kin.com_height[com_peak] - kin.com_height[loading]),
        "drive_time_s": float((com_peak - loading) / fps),
        # 沈み込み
        "min_knee_deg": float(knee[loading]),
        "min_knee_deg_overall": float(knee[: max(contact, 1)].min()),
        # 打点
        "contact_height_m": float(wrist_h[contact]),
        "contact_height_ratio": (float(wrist_h[contact] / body_h)
                                 if body_h > 0 else float("nan")),
        "contact_vs_compeak_s": float((contact - com_peak) / fps),
        "elbow_at_contact_deg": float(kin.elbow_angle()[contact]),
        "trunk_lean_at_contact_deg": float(kin.trunk_lean()[contact]),
        # 捻転
        "max_x_factor_deg": float(
            kin.x_factor()[loading : max(contact, loading + 1)].max()),
        # 連鎖
        "kinetic_chain": kin.kinetic_chain(loading, contact),
    }
