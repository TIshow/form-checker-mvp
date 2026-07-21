"""検証用の合成サーブデータを生成する。

タイミングは全て「秒」で定義する。フレーム番号で定義すると fps を変えたときに
動作の速さまで変わってしまい、「30fpsでは連鎖を判定しない」といった
フレームレート依存の挙動を正しく検証できないため。

回転はシグモイドで与える。角度をガウス関数にすると角速度（＝角度の変化率）の
ピークが中心から±σずれてしまい、「いつ最も速く回ったか」を意図通りに作れない。
"""

from __future__ import annotations

import numpy as np

from analysis import serve as S

UP_AXIS = 1       # y-up
UP_SIGN = 1.0

DUR = 6.9         # クリップ長 [s]
T_LOAD = 2.90     # 沈み込み [s]
T_CONTACT = 3.43  # 打点 [s]


def _gauss(t, center, sigma):
    return np.exp(-((t - center) ** 2) / (2 * sigma ** 2))


def _sigmoid(t, center, tau):
    return 1.0 / (1.0 + np.exp(-(t - center) / tau))


def _swing(t, t_peak, t_fall, amp, tau=0.05):
    """t_peak で加速し t_fall で戻る回転。角速度のピークが t_peak に来る。"""
    return amp * _sigmoid(t, t_peak, tau) * (1.0 - _sigmoid(t, t_fall, 0.12))


def synth_serve(fps: float = 30.0, good_chain: bool = True,
                contact_late: bool = False, deep_knee: bool = True):
    """右利きサーブを模した (joints, com) を返す。

    good_chain=False   肩が腰より先にピーク（上体が早く開く）
    contact_late=True  重心の頂点より遅れて打点が来る
    deep_knee=False    沈み込みが浅い
    """
    F = int(DUR * fps)
    t = np.arange(F) / fps
    J = np.zeros((F, 24, 3))

    pelvis_h = 1.0 - 0.05 * _gauss(t, T_LOAD, 0.40) + 0.12 * _gauss(t, T_CONTACT, 0.27)
    # 深い沈み込みで膝の関節角が文献のプロ相当(約115°)まで下がるようにする。
    # 浅い場合は 160°台にとどまる。
    knee_bend = 0.42 * _gauss(t, T_LOAD, 0.40) * (1.0 if deep_knee else 0.06)

    # 角速度がピークになる時刻 [s]。腰→肩→上腕→前腕 の順が理想。
    if good_chain:
        t_hip, t_sh = T_CONTACT - 0.30, T_CONTACT - 0.22
    else:
        t_hip, t_sh = T_CONTACT - 0.15, T_CONTACT - 0.34
    t_ua, t_fa = T_CONTACT - 0.14, T_CONTACT - 0.06
    t_fall = T_CONTACT + 0.40
    t_reach = T_CONTACT + (0.23 if contact_late else 0.0)

    hip_ang = _swing(t, t_hip, t_fall, 1.0)
    sh_ang = _swing(t, t_sh, t_fall, 1.4)
    ua_ang = _swing(t, t_ua, t_fall, np.pi / 2)   # 上腕: 水平→垂直
    fa_ang = _swing(t, t_fa, t_fall, np.pi / 2)   # 前腕: 上腕より少し遅れて

    for i in range(F):
        h = pelvis_h[i]
        J[i, S.PELVIS] = [0, h, 0]
        J[i, S.SPINE1] = [0, h + 0.10, 0]
        J[i, S.SPINE2] = [0, h + 0.20, 0]
        J[i, S.SPINE3] = [0, h + 0.30, 0]
        J[i, S.NECK] = [0, h + 0.45, 0]
        J[i, S.HEAD] = [0, h + 0.60, 0]

        # 脚: 膝を曲げると膝が前に出て、足首は接地したまま
        for sgn, knee, ankle, foot in [
            (-1, S.L_KNEE, S.L_ANKLE, S.L_FOOT),
            (+1, S.R_KNEE, S.R_ANKLE, S.R_FOOT),
        ]:
            J[i, knee] = [sgn * 0.10, h - 0.45 + knee_bend[i] * 0.5, knee_bend[i] * 0.9]
            J[i, ankle] = [sgn * 0.10, 0.08, 0]
            J[i, foot] = [sgn * 0.10, 0.02, 0.10]

        a, s = hip_ang[i], sh_ang[i]
        J[i, S.L_HIP] = [-0.10 * np.cos(a), h, -0.10 * np.sin(a)]
        J[i, S.R_HIP] = [0.10 * np.cos(a), h, 0.10 * np.sin(a)]
        J[i, S.L_SHOULDER] = [-0.18 * np.cos(s), h + 0.42, -0.18 * np.sin(s)]
        J[i, S.R_SHOULDER] = [0.18 * np.cos(s), h + 0.42, 0.18 * np.sin(s)]
        J[i, S.L_COLLAR] = J[i, S.L_SHOULDER] * 0.5
        J[i, S.R_COLLAR] = J[i, S.R_SHOULDER] * 0.5

        # 右腕: 上腕・前腕がそれぞれ水平→垂直に起き上がる（向きが大きく変わる）
        reach = 1.0 + 0.60 * _gauss(t[i], t_reach, 0.25)
        ua = np.array([np.cos(ua_ang[i]), np.sin(ua_ang[i]), 0.0])
        fa = np.array([np.cos(fa_ang[i]), np.sin(fa_ang[i]), 0.0])
        J[i, S.R_ELBOW] = np.array(J[i, S.R_SHOULDER]) + 0.28 * reach * ua
        J[i, S.R_WRIST] = J[i, S.R_ELBOW] + 0.26 * reach * fa
        J[i, S.R_HAND] = J[i, S.R_WRIST] + 0.09 * reach * fa

        # 左腕は下ろしたまま（利き手判定が右になるように）
        J[i, S.L_ELBOW] = [-0.30, h + 0.20, 0]
        J[i, S.L_WRIST] = [-0.32, h + 0.02, 0]
        J[i, S.L_HAND] = [-0.33, h - 0.05, 0]

    com = J[:, [S.PELVIS, S.NECK, S.L_HIP, S.R_HIP]].mean(axis=1)
    return J, com
