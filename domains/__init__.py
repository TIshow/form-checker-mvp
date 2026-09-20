"""競技ドメインのレジストリ。

1つのリポジトリで複数の競技を扱う。共有するのは core/（計測）と
保存・比較・表示で、競技ごとに違うのは局面・指標・判定だけ。

    import domains
    d = domains.get("golf_swing")
    domains.names()        # -> ['tennis_serve', 'golf_swing', ...]

## 実装の状態

  tennis_serve     実装済み。TIER A 2件 / TIER B 2件
  golf_swing       指標のみ。判定は出典を確認してから
  baseball_pitch   指標のみ。240fps 以上の撮影が前提
  baseball_swing   指標のみ。打撃。接地とインパクト（手の最速で代用）
  opera_posture    指標のみ。音声側（core/audio.py）が未実装

新しい競技を足すときは `NotImplementedDomain` を継承し、
`measure` まで書いて `judge` は空のまま出すこと。根拠のない閾値を
置くより、判定しない方が常に良い（`domains/base.py` を読むこと）。
"""

from __future__ import annotations

from .baseball_pitch import BaseballPitch
from .baseball_swing import BaseballSwing
from .golf_swing import GolfSwing
from .opera_posture import OperaPosture
from .tennis_serve import TennisServe

#: 既定のドメイン。既存の呼び出しが引数なしで動くようにするため。
DEFAULT = "tennis_serve"

_REGISTRY = {d.name: d for d in (
    TennisServe(), GolfSwing(), BaseballPitch(), BaseballSwing(), OperaPosture(),
)}

__all__ = ["get", "names", "DEFAULT",
           "TennisServe", "GolfSwing", "BaseballPitch", "BaseballSwing", "OperaPosture"]


def get(name: str | None = None):
    """名前でドメインを引く。"""
    key = name or DEFAULT
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"未知のドメイン {key!r}。使えるのは: {', '.join(_REGISTRY)}"
        ) from None


def names() -> list[str]:
    return list(_REGISTRY)
