#!/usr/bin/env python3
"""動画のカメラが動いているかを調べる（issue #8）。

    python tools/camera_motion.py ~/Downloads/zverev_serve.mp4
    python tools/camera_motion.py video.mp4 --start 7.3 --end 11.0

## なぜ必要か

3D復元は GVHMR を `-s`（静止カメラ）で呼んでいる。これはカメラが動かない前提で、
崩れると**カメラの動きが人物の動きとして世界座標に足し込まれる**。

実例: Zverev のサーブ動画は実際にはジャンプしているのに、復元では足が
2.9cm しか浮かず、駆動中はむしろ下がった。背景を測るとカメラは継続的に
パンしていた。

壊れるのはグローバル軌跡（跳躍・水平移動・重心の絶対高さ）だけで、
関節角やキネティックチェーンの順序は `body_pose` 由来なので影響を受けにくい。
だから「解析できない」ではなく「**どの指標が使えないか**」を切り分けたい。

## 原理

連続するフレーム間の全体的なずれを位相相関で測る。カメラが静止していれば
背景が支配的なのでずれは 0 付近になる。人物が動いても、画面に占める割合が
小さいうちはピークは背景側に立つ。

必要なもの: `pip install -e '.[video]'`
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

# 「静止」とみなす1秒あたりのずれ（画面の短辺に対する割合）。
# 実測: 固定カメラの自撮り = 0.1%/s 未満、放送のパン = 2〜4%/s。
STATIC_PER_SEC = 0.005
SAMPLE_HZ = 4.0      # 1秒あたり何点測るか
DOWNSCALE = 4        # 位相相関にかける前の間引き（速度のため）


def _shift(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """a→b の全体的な平行移動を位相相関で求める [px]。"""
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-9
    c = np.fft.ifft2(r).real
    py, px = np.unravel_index(np.argmax(c), c.shape)
    dy = py - a.shape[0] if py > a.shape[0] // 2 else py
    dx = px - a.shape[1] if px > a.shape[1] // 2 else px
    return int(dx) * DOWNSCALE, int(dy) * DOWNSCALE


def analyze(path: str, start: float | None = None,
            end: float | None = None) -> dict:
    try:
        import imageio.v3 as iio
    except ImportError:
        raise SystemExit("imageio が要ります: pip install -e '.[video]'")

    meta = iio.immeta(path, plugin="FFMPEG")
    fps = float(meta.get("fps") or 30.0)
    lo = int((start or 0.0) * fps)
    hi = int(end * fps) if end is not None else None
    step = max(1, round(fps / SAMPLE_HZ))

    frames, times = [], []
    for i, fr in enumerate(iio.imiter(path, plugin="FFMPEG")):
        if hi is not None and i > hi:
            break
        if i < lo or (i - lo) % step:
            continue
        g = fr[..., :3].mean(axis=2)[::DOWNSCALE, ::DOWNSCALE]
        frames.append(g)
        times.append(i / fps)
    if len(frames) < 3:
        raise SystemExit("フレームが足りません（区間を広げてください）")

    short = min(frames[0].shape) * DOWNSCALE
    dt = step / fps
    steps = []
    for (t1, a), (t2, b) in zip(zip(times, frames), zip(times[1:], frames[1:])):
        dx, dy = _shift(a, b)
        steps.append({"t": t1, "dx": dx, "dy": dy,
                      "mag": float(np.hypot(dx, dy))})

    mags = np.array([s["mag"] for s in steps])
    # 外れ値（カット切り替わり）に引きずられないよう中央値で判定する
    per_sec = float(np.median(mags) / dt / short)
    return {
        "fps": fps, "short_side_px": short, "samples": len(frames),
        "steps": steps,
        "median_px_per_sec": float(np.median(mags) / dt),
        "motion_per_sec": per_sec,
        "is_static": per_sec < STATIC_PER_SEC,
        # 1点だけ突出していればカット切り替わりの可能性
        "cuts": [s["t"] for s in steps if s["mag"] > max(8 * np.median(mags), 0.02 * short)],
    }


def summarize(r: dict) -> list[str]:
    """UI に出す警告文。使えない指標まで書くのが要点。"""
    if r["is_static"]:
        return []
    return [
        f"カメラが動いています（背景が毎秒 {r['motion_per_sec']*100:.1f}% ずれている）。"
        "静止カメラ前提で復元しているため、**跳躍の高さ・水平移動・重心の絶対高さは"
        "信用できません**。関節角とキネティックチェーンの順序は影響を受けにくいので"
        "比較に使えます"
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="カメラが動いているかを調べる")
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    args = ap.parse_args()

    r = analyze(args.video, args.start, args.end)
    print(f"\n{args.video}")
    print(f"  再生fps  : {r['fps']:.2f}   短辺 {r['short_side_px']}px   "
          f"{r['samples']}点を測定")
    print("\n  区間ごとのずれ")
    for s in r["steps"]:
        mark = "  ← 動いている" if s["mag"] > 0.004 * r["short_side_px"] else ""
        print(f"    {s['t']:6.2f}s  横{s['dx']:+5d}px 縦{s['dy']:+5d}px{mark}")
    print(f"\n  中央値: 毎秒 {r['median_px_per_sec']:.1f}px "
          f"（短辺の {r['motion_per_sec']*100:.2f}%）")
    if r["cuts"]:
        print(f"  カット切り替わりらしき時刻: "
              f"{', '.join(f'{t:.2f}s' for t in r['cuts'])}")
    print()
    if r["is_static"]:
        print("  ✅ 静止カメラとみなせます。`-s` での復元は妥当です。")
    else:
        for m in summarize(r):
            print(f"  ⚠️ {m}")
        print("\n  → 対処は docs/issues/008-moving-camera-slam.md")


if __name__ == "__main__":
    sys.exit(main())
