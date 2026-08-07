#!/usr/bin/env python3
"""空中の重心の落ち方から、動画の実フレームレートを推定する。

    python tools/estimate_fps.py output/gv_joints.npy

## なぜ必要か

スローモーションとして引き伸ばして保存された動画は、コンテナ上の fps が
再生レートであって撮影レートではない。ファイルからは撮影レートを知れないが、
時間の指標（秒・角速度）はすべてそれに依存する。

## 原理

跳んでいる間、重心は自由落下する。重力加速度 9.81 m/s² は既知なので、
「1フレームあたり何 m/frame² 落ちたか」を測ればフレームと秒の換算比が出る。

    h(n) = h0 + v·n − ½·a·n²          （n はフレーム）
    a = g / fps²   ⇒   fps = sqrt(g / a)

「サーブはだいたい0.3秒」のような推測を使わずに済むのが利点。

## 窓の取り方が肝

自由落下しているのは**足が地面から離れている間だけ**。接地後のフレームを
含めると放物線が鈍り、fps を過大評価する。実データでの確認:

    重心ピーク±6 frames  → 残差 3.7mm、見かけの重力 10.49 m/s²（真値の1.07倍）
    重心ピーク±20 frames → 残差 22.7mm、見かけの重力 2.89 m/s²（真値の0.29倍）

そこで「当てはまりが十分よい範囲で最大の窓」を選ぶ。残差そのものが
滞空しているかどうかの判定材料になる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.serve import compute_com, detect_up_axis  # noqa: E402

G = 9.81            # m/s²
MAX_RMS_M = 0.005   # 放物線からのずれがこれ以下なら自由落下とみなす（5mm）
COMMON_FPS = [24, 25, 30, 50, 60, 120, 240]


def _fit(com_h: np.ndarray, lo: int, hi: int) -> tuple[float, float] | None:
    """[lo,hi) に放物線を当てはめ、(1フレームあたりの落下加速度, 残差RMS) を返す。"""
    n = np.arange(lo, hi)
    c2, c1, c0 = np.polyfit(n, com_h[lo:hi], 2)
    if c2 >= 0:                       # 上に凸でなければ落下していない
        return None
    resid = com_h[lo:hi] - np.polyval([c2, c1, c0], n)
    return -2.0 * c2, float(np.sqrt((resid ** 2).mean()))


def estimate(joints: np.ndarray, max_rms: float = MAX_RMS_M) -> dict:
    up_ax, up_sign = detect_up_axis(joints)
    com_h = compute_com(joints)[:, up_ax] * up_sign
    peak = int(np.argmax(com_h))

    scan, chosen = [], None
    for half in range(4, 31):
        lo, hi = peak - half, peak + half + 1
        if lo < 0 or hi > len(com_h):
            break
        r = _fit(com_h, lo, hi)
        if r is None:
            continue
        accel, rms = r
        row = {"half": half, "rms_m": rms, "fps": float(np.sqrt(G / accel))}
        scan.append(row)
        if rms <= max_rms:            # 当てはまる範囲で最大の窓を採る
            chosen = row

    if chosen is None:
        return {"error": "自由落下とみなせる区間が見つかりませんでした",
                "scan": scan, "peak": peak}

    near = min(COMMON_FPS, key=lambda c: abs(c - chosen["fps"]))
    return {
        "peak": peak,
        "half": chosen["half"],
        "airborne_frames": chosen["half"] * 2 + 1,
        "fps": chosen["fps"],
        "rms_m": chosen["rms_m"],
        "nearest_common": near,
        "scan": scan,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in sys.argv[1:]:
        joints = np.load(path)
        r = estimate(joints)
        print(f"\n{path}  ({joints.shape[0]} フレーム)")
        if "error" in r:
            print(f"  推定できず: {r['error']}")
            for s in r.get("scan", [])[:6]:
                print(f"    ±{s['half']:2d}: 残差{s['rms_m']*1000:5.1f}mm "
                      f"→ {s['fps']:.0f} fps")
            continue
        print(f"  重心ピーク : frame {r['peak']}")
        print(f"  滞空とみなした窓: ピーク±{r['half']} "
              f"({r['airborne_frames']} フレーム)")
        print(f"  放物線からのずれ: {r['rms_m']*1000:.1f} mm")
        print(f"  → 推定 fps: {r['fps']:.1f}   "
              f"（最も近い一般的な値: {r['nearest_common']} fps）")
        print("\n  窓を変えたときの感度（狭いほど自由落下に近い）:")
        for s in r["scan"]:
            if s["half"] % 2 == 0 and s["half"] <= 20:
                mark = " ←採用" if s["half"] == r["half"] else ""
                print(f"    ±{s['half']:2d}: 残差{s['rms_m']*1000:5.1f}mm "
                      f"→ {s['fps']:5.0f} fps{mark}")


if __name__ == "__main__":
    main()
