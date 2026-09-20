#!/usr/bin/env python3
"""web/ を配信する開発用サーバ。キャッシュ無効 + Range 対応。

    python web/devserver.py        # → http://127.0.0.1:8123

## キャッシュを無効にする理由

標準の http.server はキャッシュ制御を返さないため、ブラウザが古い index.html や
avatar.js を掴んだままになり「直したのに変わらない」という誤解を生む。

## Range に対応する理由

標準の SimpleHTTPRequestHandler は **Range リクエストを実装していない**。
動画は全体を先頭から流すことしかできず、**シークが黙って無視される**。
`video.currentTime = 2.58` を書いても 0 のまま、エラーも `seeking` も出ない。

clip.html は「3D骨格のフレーム番号から動画の時刻を決める」作りなので、
これが無いと**動画だけが常に先頭のまま**になる。実際それで、リリースの
フレームを表示しているのに動画はセットポジションを映していた。
"""
import functools
import http.server
import os
import pathlib
import re

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class DevHandler(http.server.SimpleHTTPRequestHandler):
    """キャッシュを返さず、単一レンジの Range リクエストに応える。"""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def send_head(self):
        rng = self.headers.get("Range")
        if not rng:
            # Accept-Ranges は下の send_response で 200 のときに足している。
            # これが無いと、ブラウザによってはシークを試みることすらしない。
            return super().send_head()

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(f.fileno()).st_size
        m = RANGE_RE.fullmatch(rng.strip())
        if not m:
            f.close()
            self.send_error(400, "Bad Range")
            return None

        start_s, end_s = m.group(1), m.group(2)
        if start_s:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        else:
            # "bytes=-500" は末尾 500 バイト
            if not end_s:
                f.close()
                self.send_error(400, "Bad Range")
                return None
            start = max(0, size - int(end_s))
            end = size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            f.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        f.seek(start)
        return _Slice(f, end - start + 1)

    def send_response(self, code, message=None):
        super().send_response(code, message)
        if code == 200:
            self.send_header("Accept-Ranges", "bytes")


class _Slice:
    """copyfile に渡す、指定バイト数だけ読める読み出し口。"""

    def __init__(self, fp, remaining: int):
        self.fp, self.remaining = fp, remaining

    def read(self, n=-1):
        if self.remaining <= 0:
            return b""
        if n is None or n < 0:
            n = self.remaining
        data = self.fp.read(min(n, self.remaining))
        self.remaining -= len(data)
        return data

    def close(self):
        self.fp.close()


if __name__ == "__main__":
    handler = functools.partial(DevHandler, directory=str(pathlib.Path(__file__).parent))
    print("http://127.0.0.1:8123 （キャッシュ無効 / Range 対応）  Ctrl-C で停止")
    http.server.ThreadingHTTPServer(("127.0.0.1", 8123), handler).serve_forever()
