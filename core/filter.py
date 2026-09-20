"""関節列の時間方向の平滑化。

## 何のためにあるか

単一画像モデル（SAM 3D Body）はフレームごとに独立に推定するので、
フレーム間にジッタが乗る。通常の映像では「速い腕にだけ 1.9倍」程度で済んだが、
**スロー映像では致命的になる**。1フレームの実際の動きが数mmに落ちる一方、
ジッタは数cmのまま変わらないので、速度の微分がジッタだけになる。
実測（打撃・24fps再生の約10倍スロー）: 手の速度が 41 m/s の孤立ピークを出し、
接地の検出が潰れた。

時系列モデル（GVHMR）は内部で平滑化しているのでこの問題が無い。

## どれだけ平滑化するか

「実時間で何 fps 相当の解像度が要るか」で窓を決める。

    window = round(fps / target_fps)

サーブの連鎖判定に必要なのは 60fps 相当なので `target_fps=60`。
240fps 撮影なら 4フレーム窓（17ms）、24fps ならそのまま（窓1）。
**通常速度の映像では何も変わらない**ので、既存の結果は動かない。

平滑化は必ず**計測の前**に、関節座標に対して行う。角度や速度を出してから
平滑化すると、局面の時刻がずれる。
"""

from __future__ import annotations

import numpy as np


def window_for(fps: float, target_fps: float | None) -> int:
    """撮影fpsから平滑化の窓幅[フレーム]を決める。1 なら平滑化しない。"""
    if not target_fps or target_fps <= 0:
        return 1
    return max(1, int(round(fps / target_fps)))


def temporal_smooth(joints: np.ndarray, window: int) -> np.ndarray:
    """(F, J, 3) を時間方向に移動平均する。端は縮めた窓で処理し、長さは変えない。"""
    joints = np.asarray(joints, dtype=float)
    if window <= 1 or joints.shape[0] < 3:
        return joints
    F = joints.shape[0]
    half = window // 2
    out = np.empty_like(joints)
    csum = np.concatenate([np.zeros((1,) + joints.shape[1:]), np.cumsum(joints, axis=0)])
    for i in range(F):
        lo, hi = max(0, i - half), min(F, i + half + 1)
        out[i] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


# --------------------------------------------------------------------------
# 水平を取る（カメラの傾きの較正）
# --------------------------------------------------------------------------

def level_from_upright(joints: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, float]:
    """直立している区間 [lo, hi) の頭−足首の向きを鉛直として、関節列全体を回す。

    カメラ空間で返す手法（SAM 3D Body）の「上」はカメラの −Y で、カメラの
    傾きがそのまま乗る。さらに単眼の復元は**奥行き方向の体の傾き**を決められない
    （一枚の絵では「前に傾いた体」と「上を向いたカメラ」が同じに写る）。
    ゴルフの正面撮りで前傾角が GVHMR と 18.7° 食い違ったが、カメラの傾きと
    復元のバイアスの内訳は映像だけからは分からない。

    床の平面から法線を出す案は2度捨てた——SMPL24 の足の点は一直線に並び、
    MHR のかかと・つま先は奥行きの推定が不安定で、どちらも GVHMR 基準と
    一致しなかった。代わりに**撮影の最初に1秒、直立して腕を下ろす**という
    撮り方を決め、その区間の頭−足首を鉛直にする。カメラの傾きも、直立姿勢での
    復元のバイアスも、まとめて打ち消せる。人の直立は 1〜2° で再現できる。

    戻り値: (回転後の関節列, 直した角度[deg])。区間が無効なら回さない。
    """
    from .skeleton import HEAD, L_ANKLE, R_ANKLE
    J = np.asarray(joints, dtype=float)
    lo, hi = max(0, int(lo)), min(len(J), int(hi))
    if hi - lo < 1:
        return J, 0.0
    seg = J[lo:hi]
    up = (seg[:, HEAD] - (seg[:, L_ANKLE] + seg[:, R_ANKLE]) / 2).mean(0)
    n = np.linalg.norm(up)
    if n < 1e-6:
        return J, 0.0
    up /= n
    # 現在の上軸（成分が最大の軸）に up を重ねる回転
    ax = int(np.argmax(np.abs(up)))
    target = np.zeros(3); target[ax] = np.sign(up[ax])
    v = np.cross(up, target); s_ = np.linalg.norm(v); c = float(up @ target)
    if s_ < 1e-9:
        return J, 0.0
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / s_ ** 2)
    return J @ R.T, float(np.degrees(np.arctan2(s_, c)))
