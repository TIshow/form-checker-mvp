#!/usr/bin/env python3
"""動画の実フレームレートを調べる。ffprobe も追加ライブラリも要らない。

    python tools/videoinfo.py temp_my_serve.mp4

fps を取り違えると、時間に関わる指標（伸び上がり秒数・角速度・打点タイミング）が
すべて倍率ごとずれる。実際 60fps の動画を 30fps として解析し、2倍ずれていた。
解析前にここで確かめる。

注意: ここで出るのは**コンテナ上の再生フレームレート**。スローモーションとして
引き伸ばして保存された動画は、実際の撮影レートがこれより高い。その場合は
`--fps` に実撮影レートを手で指定すること（判別方法は下の「スローモーションの見分け方」）。
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

CONTAINERS = {"moov", "trak", "mdia", "minf", "stbl", "edts"}


def _boxes(data: bytes, start: int, end: int):
    """MP4/MOV の [size][type][payload] を順に返す。"""
    off = start
    while off + 8 <= end:
        size, typ = struct.unpack(">I4s", data[off:off + 8])
        head = 8
        if size == 1:                               # 64bit size
            size = struct.unpack(">Q", data[off + 8:off + 16])[0]
            head = 16
        elif size == 0:                             # 末尾まで
            size = end - off
        if size < head:
            return
        yield typ.decode("latin1"), off + head, off + size
        off += size


def _walk(data: bytes, start: int, end: int, out: list):
    for typ, s, e in _boxes(data, start, end):
        if typ in CONTAINERS:
            _walk(data, s, e, out)
        else:
            out.append((typ, s, e))


def video_info(path: str) -> dict | None:
    """映像トラックの fps・フレーム数・尺・解像度を返す。"""
    data = Path(path).read_bytes()
    leaves: list = []
    _walk(data, 0, len(data), leaves)

    track, best = {}, None
    for typ, s, e in leaves:
        if typ == "tkhd":
            w = struct.unpack(">I", data[e - 8:e - 4])[0] / 65536
            h = struct.unpack(">I", data[e - 4:e])[0] / 65536
            track = {"width": int(w), "height": int(h)}
        elif typ == "mdhd" and track:
            ver = data[s]
            if ver == 1:
                ts, dur = struct.unpack(">IQ", data[s + 20:s + 32])
            else:
                ts, dur = struct.unpack(">II", data[s + 12:s + 20])
            track["timescale"], track["duration"] = ts, dur
        elif typ == "stts" and track:
            n = struct.unpack(">I", data[s + 4:s + 8])[0]
            ent = [struct.unpack(">II", data[s + 8 + i * 8:s + 16 + i * 8])
                   for i in range(n)]
            track["frames"] = sum(c for c, _ in ent)
            track["ticks"] = sum(c * d for c, d in ent)
            track["vfr"] = n > 1
            # 解像度を持つ＝映像トラック。最もフレーム数が多いものを採用。
            if track.get("width") and (best is None
                                       or track["frames"] > best.get("frames", 0)):
                best = dict(track)
            track = {}
    if not best or not best.get("ticks"):
        return None
    best["seconds"] = best["ticks"] / best["timescale"]
    best["fps"] = best["frames"] / best["seconds"]
    return best


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in sys.argv[1:]:
        info = video_info(path)
        print(f"\n{path}")
        if not info:
            print("  映像トラックを読めませんでした")
            continue
        print(f"  解像度   : {info['width']}x{info['height']}")
        print(f"  フレーム数: {info['frames']}")
        print(f"  再生時間 : {info['seconds']:.2f} 秒")
        print(f"  fps      : {info['fps']:.2f}"
              + ("  ※可変フレームレート" if info["vfr"] else ""))
        print(f"  → 解析: python -m analysis --joints gv_joints.npy "
              f"--fps {info['fps']:.0f} --save output")


if __name__ == "__main__":
    main()
