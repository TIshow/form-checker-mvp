#!/usr/bin/env python3
"""1本の長い動画から、サーブの区間を自動で見つける（issue #10 の前段）。

    python tools/find_serves.py ~/Downloads/my_serve.MOV --expect 40
    python tools/find_serves.py X.MOV --sheet /tmp/sheet.png   # 目視確認用の一覧

## なぜ必要か

**長いクリップを丸ごと復元してはいけない。** 立っているだけの区間が続くと
そこで復元がドリフトし、サーブ部分の結果まで巻き添えで壊れる。実測（10.9秒の
うちサーブは末尾3秒）:

                        全長を投げた   サーブだけ切り出した
    利き手の判定          L（誤り）      R（正しい）
    沈み込み→打点         1 フレーム     51 フレーム
    立位の足の高さ        32.5cm 浮く    5.8cm

40本入った6分の動画なら影響はもっと大きい。1本ずつ切り出す必要があるが、
手作業は現実的でないので自動化する。

## やり方

フレーム間の差（動きの量）を見る。ただし**全画面の平均では埋もれる**。
実測で、6分の動画の全画面平均は 2.4〜3.5 でほぼ平坦、大きなスパイクは
手前を人が横切ったときだけだった。人物は画面に小さく写るので、平均すると
薄まってしまう。

そこで**動きが集中している領域を先に見つけ**、そこだけで測る。同じ動画で
中央値2.35・最大40.9まで分離できた。

サーブは「上に伸びる」動作なので、見つけた領域から**上方向に広く**取る。

## 出てくる区間について

動きの**ピーク**（打点付近）を1本につき1点見つけ、その前後を固定長で取る。
長さが揃うので、待機時間が混ざらない。切りすぎると「1フレーム目からすでに
トロフィー姿勢」になるので、前は 2.2秒 遡る（Zverev で踏んだ失敗）。

## 誤検出の癖（`--sheet` で必ず目視すること）

実測で40本中10本が誤検出だった。内訳がそのまま癖を示している:

    打ち終わって定位置に戻る動き   7本  ← 最も多い
    サーブしに位置へ歩いていく     2本
    ボールを拾いに行く             1本

**主因は「打った後に戻る動き」。** サーブの山から6秒以上あとに別の山ができるので、
最小間隔のフィルタを素通りする。間隔を広げても、本当のサーブ間隔（中央9.9秒）に
近すぎて分離できない。

動きの量だけを見ている以上、これは原理的に区別できない。サーブは腕が上へ伸びる
動作なので、領域の上部だけを見れば分けられる可能性はあるが未実装。

一覧画像を出して目で確かめ、要らない番号を `--drop` で外す前提で使う。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# 検出の既定値。実測で決めた値なので、素材が変わったら見直すこと。
DETECT_SIZE = (180, 320)     # 復号する解像度。これで足りる
MAP_STEP = 30                # 動きの地図を作るときの間引き（0.5秒ごと）
SCAN_STEP = 3                # 本スキャンの間引き（0.05秒ごと）
SMOOTH_S = 0.25              # 平滑化の窓
MIN_SEP_S = 6.0              # 山どうしの最小間隔。1本を二重に拾わないため
PAD_BEFORE_S = 2.2           # ピーク（打点付近）からトス・トロフィーまで遡る
PAD_AFTER_S = 1.3            # フォロースルーぶん


def _iter(path: str, step: int):
    import imageio.v3 as iio

    for i, fr in enumerate(iio.imiter(path, plugin="FFMPEG", size=DETECT_SIZE)):
        if i % step == 0:
            yield i, fr[..., :3].mean(axis=2)


def find_roi(path: str, margin_up: float = 1.6) -> tuple[int, int, int, int]:
    """動きが集中している領域を返す (r0, r1, c0, c1)。

    サーブは上へ伸びるので、見つけた重心から**上方向に広く**取る。
    """
    prev, acc = None, None
    for _, g in _iter(path, MAP_STEP):
        if prev is not None:
            d = np.abs(g - prev)
            acc = d if acc is None else acc + d
        prev = g
    if acc is None:
        raise SystemExit("フレームを読めませんでした")

    rows, cols = acc.sum(axis=1), acc.sum(axis=0)

    def span(v, frac=0.35):
        """ピークの frac 倍を超えている範囲を返す。

        「累積で総量の N% まで」という取り方だと、動きが薄く広がっている
        ぶんを全部拾ってしまい、実測でほぼ全画面（行0-241 列1-180）になった。
        背景のノイズはピークよりずっと低いので、ピーク基準の方が分離する。
        """
        base = np.median(v)
        hot = np.where(v - base > (v.max() - base) * frac)[0]
        return (int(hot.min()), int(hot.max())) if len(hot) else (0, len(v) - 1)

    r0, r1 = span(rows)
    c0, c1 = span(cols)
    h = r1 - r0
    r0 = max(0, int(r0 - h * margin_up))          # 腕とラケットのぶん上へ
    r1 = min(acc.shape[0], int(r1 + h * 0.2))
    w = c1 - c0
    c0 = max(0, int(c0 - w * 0.4))
    c1 = min(acc.shape[1], int(c1 + w * 0.4))
    return r0, r1, c0, c1


def motion(path: str, roi: tuple[int, int, int, int]) -> tuple[np.ndarray, float]:
    r0, r1, c0, c1 = roi
    prev, sig = None, []
    for _, g in _iter(path, SCAN_STEP):
        crop = g[r0:r1, c0:c1]
        if prev is not None:
            sig.append(float(np.abs(crop - prev).mean()))
        prev = crop
    return np.array(sig), SCAN_STEP


def segments(sig: np.ndarray, dt: float, pct: float,
             min_sep_s: float = MIN_SEP_S) -> list[tuple[float, float]]:
    """動きのピークを見つけ、その周りに**固定長の窓**を取る。

    最初は「しきい値を超えた区間を取り、前後に余白を足す」でやったが、区間の
    長さが 3.8〜11.4秒とばらつき、中央値も 5.6秒と長すぎた。人物の領域に
    絞ると待機中の身じろぎも閾値を超えてしまい、区間が伸びるため。

    サーブは「一番速く動く瞬間（打点付近）が1つある」動作なので、山の広がりを
    追うより**ピークを1点見つけて前後を固定で取る**方が素直で、長さも揃う。

    実測（6.3分に40本 ≒ 9.5秒間隔）では、最小間隔6秒・70%点でちょうど40本、
    間隔の中央値 9.1秒になった。
    """
    k = max(1, int(round(SMOOTH_S / dt)))
    sm = np.convolve(sig, np.ones(k) / k, mode="same")
    thr = np.percentile(sm, pct)
    sep = int(min_sep_s / dt)

    taken: list[int] = []
    for i in np.argsort(sm)[::-1]:          # 大きい山から順に
        if sm[i] < thr:
            break
        if all(abs(i - j) >= sep for j in taken):
            taken.append(int(i))

    n = len(sig)
    return [(max(0.0, (i - int(PAD_BEFORE_S / dt)) * dt),
             min(n * dt, (i + int(PAD_AFTER_S / dt)) * dt))
            for i in sorted(taken)]


def contact_sheet(path: str, segs, out_png: str, cols: int = 8) -> None:
    """各区間の中央のフレームを並べる。本当にサーブかを目で確かめる用。"""
    import imageio.v3 as iio
    from PIL import Image, ImageDraw

    import imageio.v3 as _  # noqa: F401
    meta = iio.immeta(path, plugin="FFMPEG")
    fps = meta["fps"]
    want = {int((s + e) / 2 * fps): i for i, (s, e) in enumerate(segs)}
    shots = {}
    for i, fr in enumerate(iio.imiter(path, plugin="FFMPEG", size=(160, 284))):
        if i in want:
            shots[want[i]] = fr
        if i > max(want):
            break
    if not shots:
        return
    w, h = 160, 284
    rows = (len(segs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (h + 16)), (17, 21, 28))
    d = ImageDraw.Draw(sheet)
    for i, (s, e) in enumerate(segs):
        if i not in shots:
            continue
        x, y = (i % cols) * w, (i // cols) * (h + 16)
        sheet.paste(Image.fromarray(shots[i]).resize((w, h)), (x, y + 16))
        d.text((x + 4, y + 2), f"{i+1:02d}  {int(s)//60}:{s%60:04.1f}", fill=(230, 235, 245))
    sheet.save(out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description="長い動画からサーブ区間を見つける")
    ap.add_argument("video")
    ap.add_argument("--expect", type=int,
                    help="想定本数。近づくようにしきい値を自動調整する")
    ap.add_argument("--pct", type=float, default=85.0,
                    help="しきい値の分位（--expect 指定時は無視）")
    ap.add_argument("--sheet", help="目視確認用の一覧画像の出力先")
    ap.add_argument("--json", help="区間を JSON で保存")
    ap.add_argument("--cache", help="動きの信号の保存先（既定: 動画と同じ場所）")
    ap.add_argument("--rescan", action="store_true",
                    help="キャッシュを無視して動画を読み直す")
    ap.add_argument("--drop", default="",
                    help="外す番号（1始まり、カンマ区切り）。"
                         "--sheet で目視して誤検出を落とす")
    args = ap.parse_args()

    path = os.path.expanduser(args.video)
    import imageio.v3 as iio
    fps = iio.immeta(path, plugin="FFMPEG")["fps"]

    # 動画の全復号に数分かかる。しきい値や余白を変えて試したいだけのときに
    # 毎回やり直すのは無駄なので、動きの信号を横に置いておく。
    cache = Path(args.cache) if args.cache else \
        Path(path).with_suffix(".motion.npz")
    if cache.exists() and not args.rescan:
        z = np.load(cache)
        sig, step, roi = z["sig"], int(z["step"]), tuple(z["roi"])
        print(f"{path}\n  信号をキャッシュから読みました（{cache}）")
        print(f"  人物の領域: 行{roi[0]}-{roi[1]} 列{roi[2]}-{roi[3]}")
    else:
        print(f"{path}\n動きの地図を作っています…")
        roi = find_roi(path)
        print(f"  人物の領域（{DETECT_SIZE[1]}x{DETECT_SIZE[0]}換算）: "
              f"行{roi[0]}-{roi[1]} 列{roi[2]}-{roi[3]}")
        print("動きを測っています…")
        sig, step = motion(path, roi)
        np.savez(cache, sig=sig, step=step, roi=np.array(roi))
        print(f"  信号を保存しました（{cache}）")
    dt = step / fps

    if args.expect:
        best, bestd = None, 1e9
        for pct in np.arange(70, 97, 0.5):
            segs = segments(sig, dt, pct)
            d = abs(len(segs) - args.expect)
            if d < bestd:
                best, bestd, bestpct = segs, d, pct
        segs, pct = best, bestpct
        print(f"  分位を自動選択: {pct:.1f}%点 → {len(segs)}本"
              f"（想定 {args.expect}本）")
    else:
        segs = segments(sig, dt, args.pct)
        pct = args.pct

    if args.drop:
        bad = {int(x) for x in args.drop.replace(" ", "").split(",") if x}
        kept = [g for i, g in enumerate(segs, 1) if i not in bad]
        print(f"  除外: {sorted(bad)} → {len(segs)}本 から {len(kept)}本")
        segs = kept

    durs = np.array([e - s for s, e in segs])
    print(f"\n{len(segs)}本  長さ 中央{np.median(durs):.1f}秒 "
          f"[{durs.min():.1f}〜{durs.max():.1f}]  合計{durs.sum()/60:.1f}分\n")
    for i, (s, e) in enumerate(segs):
        print(f"  {i+1:2d}  {int(s)//60}:{s%60:05.2f} 〜 {int(e)//60}:{e%60:05.2f}"
              f"  ({e-s:.1f}秒)")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"video": path, "fps": fps, "pct": pct,
             "segments": [{"start": round(s, 2), "end": round(e, 2)} for s, e in segs]},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n✅ {args.json}")

    if args.sheet:
        print("一覧画像を作っています…")
        contact_sheet(path, segs, args.sheet)
        print(f"✅ {args.sheet}  — 本当にサーブか目で確かめてください")


if __name__ == "__main__":
    sys.exit(main())
