#!/usr/bin/env python3
"""web/ を配信する開発用サーバ。キャッシュを無効にする。

標準の http.server はキャッシュ制御を返さないため、ブラウザが古い index.html や
avatar.js を掴んだままになり「直したのに変わらない」という誤解を生む。

    python web/devserver.py        # → http://127.0.0.1:8123
"""
import functools
import http.server
import pathlib


class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


if __name__ == "__main__":
    handler = functools.partial(NoCache, directory=str(pathlib.Path(__file__).parent))
    print("http://127.0.0.1:8123 （キャッシュ無効）  Ctrl-C で停止")
    http.server.ThreadingHTTPServer(("127.0.0.1", 8123), handler).serve_forever()
