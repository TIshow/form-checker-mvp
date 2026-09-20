#!/usr/bin/env python3
"""1本のクリップを web/clip.html で見られる形にする（競技を問わない）。

    python tools/make_clip.py \
        --joints output_pitch_s3/s3_joints.npy --fps 24 \
        --domain baseball_pitch \
        --video videos/baseball/baseball_pitcher_form.mp4 \
        --label "投球フォーム解析"

`make_compare.py`（2本を比べる）や `make_models.py`（1本を複数手法で）と違い、
**1本を1つの手法で見る**ための最小の道具。人に見せるのはたいていこれ。

## 元動画を並べて出す

3D骨格だけだと「本当にこの人の動きなのか」が伝わらない。元動画を横に置いて
フレームを同期させる。動画は `web/data/` へコピーする（`.gitignore` 済み）。

## 指標のラベルはドメインが持つ

キー名（`stride_ratio`）をそのまま見せても通じないので、各ドメインの
`metric_labels` / `phase_labels` を使う。**向きの説明も一緒に出す**
——「膝角は小さいほど深い」を書かずに数字だけ見せて、読み違えが起きた。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis  # noqa: E402
import domains  # noqa: E402

#: 局面の色。表示側の THREE と合わせる（0xRRGGBB）
PHASE_COLORS = [0xF59E0B, 0xEF4444, 0x22C55E, 0x3B82F6, 0xA855F7]


def sanitize(obj):
    """NaN / Inf を null にする。JSON.parse が落ちるのを防ぐ。

    指標が「測れなかった」ことを NaN で表すので、これは日常的に出る。
    表示側は null を「測定できません」と出す。
    """
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def copy_video(src: Path, dest: Path, max_width: int) -> None:
    """元動画を web/data/ へ。幅が大きすぎるときは縮める。

    4K のまま置くと、表示は 500px 足らずの箱なのにブラウザが 4K を
    デコードし続ける（実際に描画されず真っ黒になった）。送る用の
    ファイルも無駄に重くなる。
    """
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff = "ffmpeg"
    cmd = [ff, "-y", "-loglevel", "error", "-i", str(src),
           "-vf", f"scale='min({max_width},iw)':-2",
           "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dest)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception as e:
        print(f"⚠️ 縮小できなかったのでそのままコピーします: {e}")
        shutil.copy2(src, dest)


def build(joints_path: str, fps: float, domain: str, label: str,
          video: str | None, out: Path, max_width: int = 1280) -> dict:
    J = np.load(joints_path)
    d = domains.get(domain)
    res = analysis.analyze_json(J, fps, domain)
    m = res["metrics"]
    ph = m.get("phases", {})

    res["label"] = label
    res["fps"] = fps
    res["source"] = joints_path
    res["domain_label"] = d.label

    # 表示に必要な「言葉」をドメインから引いて同梱する。
    # 表示側にハードコードすると競技を足すたびに HTML を触ることになる。
    res["headline"] = [
        {"key": k,
         "label": d.metric_labels.get(k, (k, "", 2, ""))[0],
         "unit": d.metric_labels.get(k, (k, "", 2, ""))[1],
         "digits": d.metric_labels.get(k, (k, "", 2, ""))[2],
         "note": d.metric_labels.get(k, (k, "", 2, ""))[3],
         "value": m.get(k)}
        for k in d.headline
    ]
    res["phase_list"] = [
        {"key": k, "label": d.phase_labels.get(k, k), "frame": int(v),
         "t": round(int(v) / fps, 2),
         "color": PHASE_COLORS[i % len(PHASE_COLORS)]}
        for i, (k, v) in enumerate(ph.items()) if isinstance(v, int)
    ]
    # 骨格の色替えに使う（web/skeleton.js が clip.highlight を見る）
    res["highlight"] = [{"frame": p["frame"], "color": p["color"]}
                        for p in res["phase_list"]]
    res["note"] = m.get("phases_note", "")
    res["judged"] = bool(res.get("feedback"))

    if video:
        vp = Path(video)
        dest = out.parent / "clip_video.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        copy_video(vp, dest, max_width)
        res["video"] = dest.name
        print(f"✅ {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="1本のクリップを見る用のデータを作る")
    ap.add_argument("--joints", required=True, help="関節 .npy")
    ap.add_argument("--fps", type=float, required=True, help="撮影フレームレート")
    ap.add_argument("--domain", default=domains.DEFAULT,
                    help=f"競技。{' / '.join(domains.names())}")
    ap.add_argument("--video", help="並べて表示する元動画（任意）")
    ap.add_argument("--label", default="", help="画面に出す名前")
    ap.add_argument("--max-width", type=int, default=1280,
                    help="元動画をこの幅まで縮めて置く（既定1280）")
    ap.add_argument("--out", default="web/data/clip.json")
    args = ap.parse_args()

    out = Path(args.out)
    d = domains.get(args.domain)
    res = build(args.joints, args.fps, args.domain,
                args.label or d.label, args.video, out, args.max_width)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sanitize(res), ensure_ascii=False, allow_nan=False),
                   encoding="utf-8")
    print(f"✅ {out}  ({out.stat().st_size / 1e6:.1f} MB)  {d.label}")
    print(f"   {res['metrics']['n_frames']}フレーム @{args.fps:.0f}fps")
    for p in res["phase_list"]:
        print(f"   {p['label']:16s} frame {p['frame']:3d}  ({p['t']:.2f}s)")
    miss = [h["label"] for h in res["headline"] if h["value"] is None
            or (isinstance(h["value"], float) and not math.isfinite(h["value"]))]
    if miss:
        print(f"   ℹ️ 測定できなかった指標: {', '.join(miss)}")
    print("\n→ python web/devserver.py  →  http://127.0.0.1:8123/clip.html")


if __name__ == "__main__":
    main()
