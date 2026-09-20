"""接地した足を動かさないよう、体の並進を足から組み立て直す（foot-skate 除去）。

## なぜ要るか

単一画像モデル（SAM 3D Body）は、**体がカメラからどこにあるか（並進）**を
フレームごとに独立に推定する。人物検出の箱が数ピクセル揺れれば並進も揺れ、
足を着けて立っているだけの人が床の上を滑って見える。実測:

    オペラ（直立して歌う）  重心の揺れ  GVHMR 0.8cm  /  SAM 3D Body 4.2cm
    ゴルフ（足は固定）      頭の上下動  GVHMR 1.2cm  /  SAM 3D Body 11cm

指標を足元基準にして逃げても、**表示の滑りは残る**。ゴルフでもオペラでも
野球でも「足がスライドして動いている」と見て分かった。

姿勢（関節角）は並進と独立に推定されているので、滑りそのものは姿勢の誤りでは
ない。ただし滑る体の上では姿勢のジッタも大きく見える。

GVHMR（時系列モデル）は内部に「接地した足は静止している」という拘束を持ち、
それが足を止めている。テニスの跳躍を潰した当のものでもある。ここでは同じ
拘束を**後処理**として掛ける。

## どう組み立てるか

各フレームで**接地している足**を決め（その瞬間の最も低い足と、それに近い高さの足）、
「接地している足の絶対位置は前のフレームから動かない」と置いて、
骨盤相対の姿勢に足す並進を前のフレームから積み上げる。

    絶対位置[t] = 骨盤相対[t] + 並進[t]
    並進[t]     = mean( 絶対位置[t−1, 接地足] − 骨盤相対[t, 接地足] )

推定器が出した並進は**捨てる**（それが揺れの元）。フレーム 0 だけ元の位置を使う。

両足とも浮いている区間（跳躍）は前の並進を保ち、足が床より下に潜らないよう
高さだけ持ち上げる。**跳躍そのものは再現しない**——GVHMR と同じ限界で、
テニス以外の3競技（投球・打撃・ゴルフ）とオペラには関係しない。

## 何が変わって何が変わらないか

- 変わらない: 関節角、体幹の傾き、捻転、足元基準の指標（骨盤相対の姿勢はそのまま）
- 変わる: 絶対位置、床の推定、表示。踏み出しの前進は「接地足に対する
  もう片方の足の相対移動」として残る（それが本来の踏み出し幅）
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


def anchor_feet(joints: np.ndarray, fps: float = 30.0) -> tuple[np.ndarray, dict]:
    """接地足が動かないよう並進を組み立て直し、接地中の足をその場に留める。

    各フレームで:

    - **最も低い足を必ず錨（anchor）にする。** 立っていれば誰かの足は床にある
    - **もう片方の足**は、高さが錨に近く（`CONTACT_TOL_M`）**かつ**錨に対して
      ほぼ静止しているときだけ留める。高さだけで決めると、低く滑る踏み出し
      （打撃）まで留めてしまい踏み出しが消える
    - 留めた足の位置から並進を決め、留めた足はその位置に**ピン留め**する
      （足首とつま先を同じ量だけ）。並進だけ直しても、両足が着いているときの
      足どうしの相対ジッタの半分ずつが両足に残って滑る（実測: オペラで 2.6m）

    錨が左右で入れ替わるとき（踏み出しの着地）は、それまでの並進で決まる
    位置でそのまま留めるので、絶対位置は連続する。

    ピン留めで脛の長さが 1cm 程度変わるが、膝角への影響は 1° 未満。
    戻り値の dict は診断用: 接地足の滑り量 [m]（直す前 / 後）。
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

    return out, {"slide_before_m": slide(J), "slide_after_m": slide(out)}
