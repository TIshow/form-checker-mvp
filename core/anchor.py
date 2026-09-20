"""接地した足を動かさないよう、体の並進を足から組み立て直す（表示用）。

## 使いどころ

**表示にだけ使う。計測には使わない。** 単一画像モデル（SAM 3D Body）は体の
並進をフレームごとに独立に推定するので、立っているだけの人が床を滑って見える
（錨の足が 10秒で 2〜5m。GVHMR は 2〜24cm）。姿勢（関節角）は並進と独立なので、
滑りは姿勢の誤りではない。計測は生の関節列で行い、局面の検出は骨盤相対・
もう片方の足相対で取るので、この処理の有無で計測値は変わらない
（`analysis.analyze_json` が計測用と表示用を分けて返す）。

## 2段階

1. **並進**を接地足から組み立て直す（推定器の並進は捨てる）。全関節に同じ
   並進を足すだけなので、骨の長さも関節角も厳密に不変
2. `pin_feet=True` のとき、接地中の足を接地した位置に**留める**。両足が
   着いているときの足どうしの相対ジッタを消し、錨の足の滑りを 0 にする。
   **ただし脛の長さと膝角を変える**——ゴルフで最大 12.5°（平均 2°）。
   だから計測には決して使わず、表示にだけ使う

接地は「最も低い足」と「それに対して静止している足」で決める。高さだけで
決めると、低く滑る打撃の踏み出しまで留めてしまい踏み出しが消える。
跳躍は再現しない（両足が浮いている間は並進を保つ）。GVHMR と同じ限界。
"""

from __future__ import annotations

import numpy as np

from .kinematics import detect_up_axis
from .skeleton import FOOT_IDS, PELVIS

#: その瞬間の最も低い足からこの高さ以内にある足を「接地している」とみなす [m]。
#: マウンドでは踏み出し足が軸足より 16cm 低く着くので、そのとき軸足は
#: 接地から外れる（もう蹴り終えている）。
CONTACT_TOL_M = 0.05


def contact_mask(joints: np.ndarray, up_ax: int, up_sign: float,
                 tol: float = CONTACT_TOL_M) -> np.ndarray:
    """(F, 4) の bool。FOOT_IDS の順で、そのフレームで接地している足の点。"""
    h = joints[:, FOOT_IDS, up_ax] * up_sign               # (F,4)
    return h <= h.min(axis=1, keepdims=True) + tol


#: 足は (足首, つま先) の2点を1つの剛体として扱う。FOOT_IDS の並びに対応。
_FEET = ((FOOT_IDS[0], FOOT_IDS[2]), (FOOT_IDS[1], FOOT_IDS[3]))   # 左, 右

#: もう片方の足に対する速さ [m/s]。これより遅ければ留め始め（ENTER）、
#: 留めた後は RELEASE を超えるまで留め続ける（ヒステリシス）。
#: 打撃の踏み出しは低く滑るように出るので、高さだけで見ると「接地」になって
#: 留められ、踏み出しが消えた。ジッタ（1cm/フレーム × 24〜30fps ≒ 0.25 m/s）と
#: 踏み出し（数 m/s）の間に閾値を置く。ENTER と RELEASE を分けるのは、
#: ジッタが閾値の前後で揺れるたびに留め直して跳ぶのを防ぐため
#: （1つの閾値だとオペラで残りの滑りが 1.4m あった）。
RELATIVE_SPEED_ENTER = 0.5
RELATIVE_SPEED_RELEASE = 1.5


def anchor_feet(joints: np.ndarray, fps: float = 30.0,
                pin_feet: bool = False) -> tuple[np.ndarray, dict]:
    """接地足から並進を組み立て直す。`pin_feet` で足をその場に留める（表示用）。

    pin_feet=False: 全関節に同じ並進。関節角は不変。足の相対ジッタは残る
    pin_feet=True : 接地中の足をピン留め。滑りは 0 になるが膝角が変わる。表示専用
    """
    J = np.asarray(joints, dtype=float)
    F = J.shape[0]
    if F < 2:
        return J.copy(), {"slide_before_m": 0.0, "slide_after_m": 0.0}

    up_ax, up_sign = detect_up_axis(J)
    P = J - J[:, PELVIS:PELVIS + 1]                          # 骨盤相対の姿勢
    foot_h = np.stack([(J[:, list(ids), up_ax] * up_sign).min(1) for ids in _FEET], 1)  # (F,2)
    lowest = foot_h.argmin(1)                                # 各フレームの錨
    foot_c = np.stack([P[:, list(ids)].mean(1) for ids in _FEET], 1)   # (F,2,3) 骨盤相対の足の中心
    # 静止判定に使う相対位置は 3フレームで均す。ジッタは1フレームおきに向きを
    # 変えるので、生の差分だと静止している足でも ENTER を超えて留められない
    # （オペラで両足静止なのに滑りが 74cm 残った）。
    rel = foot_c[:, 1] - foot_c[:, 0]
    if F >= 3:
        rel = np.stack([rel[:1, :].mean(0)] + [rel[max(0, i - 1):i + 2].mean(0) for i in range(1, F - 1)] + [rel[-1:].mean(0)])

    def wanted(t: int, locked_now: dict) -> list[int]:
        a = int(lowest[t]); o = 1 - a
        if t == 0:
            near = foot_h[t, o] <= foot_h[t, a] + CONTACT_TOL_M
            return [a, o] if near else [a]
        speed = np.linalg.norm(rel[t] - rel[t - 1]) * fps
        if o in locked_now:      # 留め中: 速くなるか高く上がるまで留め続ける
            keep = speed < RELATIVE_SPEED_RELEASE and foot_h[t, o] <= foot_h[t, a] + 2 * CONTACT_TOL_M
        else:                    # 留め始め: 近くて静止
            keep = speed < RELATIVE_SPEED_ENTER and foot_h[t, o] <= foot_h[t, a] + CONTACT_TOL_M
        return [a, o] if keep else [a]

    out = np.empty_like(J)
    out[0] = J[0]
    locked: dict[int, np.ndarray] = {k: out[0, list(_FEET[k])].copy() for k in wanted(0, {})}

    for t in range(1, F):
        want = wanted(t, locked)
        common = [k for k in want if k in locked]
        if common:
            T = np.mean([locked[k].mean(0) - P[t, list(_FEET[k])].mean(0) for k in common], axis=0)
        else:
            T = out[t - 1, PELVIS].copy()
            prev_floor = (out[t - 1, FOOT_IDS, up_ax] * up_sign).min()
            now_floor = ((P[t, FOOT_IDS] + T)[:, up_ax] * up_sign).min()
            if now_floor < prev_floor:
                T[up_ax] += (prev_floor - now_floor) * up_sign
        A = P[t] + T
        if pin_feet:
            for k in common:
                ids = list(_FEET[k])
                A[ids] += locked[k].mean(0) - A[ids].mean(0)
        for k in want:
            if k not in locked:
                locked[k] = A[list(_FEET[k])].copy()
        for k in list(locked):
            if k not in want:
                del locked[k]
        out[t] = A

    def slide(X):
        """錨の足（各フレームで最も低い足）が、錨のまま動いた水平距離の合計。"""
        hz = [a for a in (0, 1, 2) if a != up_ax]
        total = 0.0
        for t in range(1, F):
            if lowest[t] == lowest[t - 1]:
                ids = list(_FEET[int(lowest[t])])
                total += float(np.linalg.norm((X[t, ids] - X[t - 1, ids]).mean(0)[hz]))
        return total

    return out, {"method": "pin_feet" if pin_feet else "translation_only",
                 "slide_before_m": slide(J), "slide_after_m": slide(out)}
