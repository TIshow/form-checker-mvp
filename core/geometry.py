"""幾何ユーティリティ。純関数のみ、競技にも骨格にも依存しない。"""

from __future__ import annotations

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    """ゼロ除算を避けて正規化する。"""
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """b を頂点とする a-b-c の3D角度 [deg]。各引数は (F,3)。180=まっすぐ。"""
    cos = np.sum(unit(a - b) * unit(c - b), axis=-1).clip(-1.0, 1.0)
    return np.degrees(np.arccos(cos))


def angle_from_axis(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """ベクトル v が軸 axis から傾いている角度 [deg]。0=同じ向き、180=真逆。"""
    cos = (unit(v) * unit(axis)).sum(-1).clip(-1.0, 1.0)
    return np.degrees(np.arccos(cos))


def angular_speed(p_start: np.ndarray, p_end: np.ndarray, fps: float) -> np.ndarray:
    """線分 p_start->p_end の向きが変化する角速度 [deg/s]。(F,) を返す。"""
    d = unit(p_end - p_start)
    cos = np.sum(d[1:] * d[:-1], axis=-1).clip(-1.0, 1.0)
    speed = np.degrees(np.arccos(cos)) * fps
    return np.concatenate([[0.0], speed])


def smooth(x: np.ndarray, win: int = 5) -> np.ndarray:
    """移動平均。角速度はノイズが乗るため平滑化してからピークを取る。

    注意: 平滑化は時間解像度と引き換えになる。単一画像モデル（SAM 3D Body 等）の
    ジッタをこれで消すと、キネティックチェーンの順序判定に必要な 20〜40ms の
    分解能も一緒に失う。窓を広げる前に issue 011 を読むこと。
    """
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
