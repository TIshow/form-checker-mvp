"""結果を画像として保存する（人が視覚的に確認するため）。

matplotlib は任意依存（`.[plot]`）。使うときだけ遅延 import する。
グラフは「主役」ではなく、レポートの補助。重心の高さだけを1枚にする。
"""

from __future__ import annotations

import numpy as np


def save_com_height_graph(metrics: dict, com: np.ndarray,
                          up_ax: int, up_sign: float, path: str) -> None:
    """重心の高さ vs 時間。沈み込みと打点に注釈を付ける。

    以前「重心の折れ線を見ても意味が分からない」となった反省を踏まえ、
    生の曲線ではなく「どこが沈み込みでどこが打点か」を図中に明示する。
    """
    import matplotlib

    matplotlib.use("Agg")  # 画面のない環境でファイル出力するため
    import matplotlib.pyplot as plt

    fps = metrics["fps"]
    ph = metrics["phases"]
    tr, pk = ph["loading"], ph["contact"]
    h = com[:, up_ax] * up_sign
    t = np.arange(len(h)) / fps

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t, h, lw=2.5, color="#2563eb")
    ax.axvspan(tr / fps, pk / fps, color="orange", alpha=0.12)

    ax.scatter([tr / fps], [h[tr]], s=130, color="orange", zorder=5)
    ax.annotate(f"loading\nknees bent\n{h[tr]:.3f} m",
                xy=(tr / fps, h[tr]), xytext=(tr / fps - 1.4, h.min() + 0.005),
                arrowprops=dict(arrowstyle="->", lw=1.4), fontsize=10)

    ax.scatter([pk / fps], [h[pk]], s=130, color="red", zorder=5)
    # 打点はグラフ上端に近いので、注釈は右下に置きタイトルと重ならないようにする
    ax.annotate(f"contact\nfully extended\n{h[pk]:.3f} m",
                xy=(pk / fps, h[pk]), xytext=(pk / fps + 0.5, h[pk] - 0.045),
                arrowprops=dict(arrowstyle="->", lw=1.4), fontsize=10)

    rise = (h[pk] - h[tr]) * 100
    span = h.max() - h.min()
    ax.text((tr + pk) / 2 / fps, h.min() - 0.02 * span,
            f"leg drive  +{rise:.1f} cm in {(pk - tr) / fps:.2f}s",
            ha="center", fontsize=10, color="darkorange", weight="bold")

    # 上下に余白を足して注釈やタイトルの衝突を防ぐ
    ax.set_ylim(h.min() - 0.10 * span, h.max() + 0.12 * span)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("center of mass height (m)")
    ax.set_title("Serve — center of mass height")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
