"""競技ドメインの共通の型と、競技をまたいで成り立つ判定。

## ドメインが実装する4つ

    side(joints)             利き側をどう決めるか
    detect_phases(kin)       名前つきフレーム（打点・トップ・リリース…）
    measure(kin, phases)     指標の辞書
    judge(metrics)           改善点のリスト

これだけが競技ごとに違う。計測の中身（重心・関節角・連鎖）は core/ にあり、
保存・比較・表示は全ドメインで共有する。

## 判定の信頼度を3段階に分ける（全ドメイン共通の規律）

閾値の正しさがそのままフィードバックの正しさになるため、
「何を根拠にその数字を決めたか」をコード上で区別する。

  TIER A 力学的原理     反例が考えにくい。断定してよい
  TIER B 文献参照       実測レンジがある。参考として提示する
  TIER C 根拠なし       判定しない。数値を表示するだけ

TIER C を「欠点」として指摘してはいけない。テニスでは初期実装が体幹の傾きを
35°超で「傾きすぎ」と警告していたが、文献ではプロの接球時の体幹は約42°傾いており、
**プロの技術を欠点と判定していた**。さらに、その根拠にした「水平から48°」という
数値は、本文を確認したところ**論文に存在しなかった**。

**新しい競技を足すときは、判定ゼロから始めること。** 指標を測るのは
力学的に定義できれば足りるが、閾値は出典を確認するまで置かない。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from core import Kinematics

TIER_A, TIER_B, TIER_C = "A", "B", "C"

# --------------------------------------------------------------------------
# 全ドメイン共通の TIER A ルール: キネティックチェーンの順序
#
# 「力は近位から遠位へ順に伝わる」は力学的原理であり、サーブでもスイングでも
# 投球でも同じ。閾値ではなく物理なので、ここに1つだけ書いて共有する。
# --------------------------------------------------------------------------

#: 隣接する体節のピーク時間差。これ未満を「逆転」と呼ぶのはノイズを読んでいるだけ。
TH_CHAIN_MIN_GAP_S = 0.05

#: 順序を論じるのに最低限必要な撮影フレームレート（既定）。
#: 30fps では 1フレーム=33ms あり、体節間の 20〜40ms を分解できない。
#: **競技ごとに上書きすること。** 投球の内旋は約7000°/秒で、60fps では足りない。
DEFAULT_CHAIN_MIN_FPS = 60.0

P_PRINCIPLE = 100   # 力学的原理の違反
P_TIMING = 90       # タイミング
P_REFERENCE = 50    # 文献レンジからの逸脱


def chain_order_finding(m: dict, min_fps: float = DEFAULT_CHAIN_MIN_FPS) -> dict | None:
    """[TIER A] 力の伝達順序の逆転を検出する。全ドメイン共通。

    撮影フレームレートが足りない場合は測定できないので判定しない。
    """
    fps = m["fps"]
    if fps < min_fps:
        return None

    chain = [c for c in m.get("kinetic_chain", []) if c.get("reliable", True)]
    min_gap = max(TH_CHAIN_MIN_GAP_S * fps, 2.0)

    for cur, nxt in zip(chain, chain[1:]):
        gap = cur["peak_frame"] - nxt["peak_frame"]
        if gap >= min_gap:
            return {
                "priority": P_PRINCIPLE,
                "tier": TIER_A,
                "id": "kinetic_chain_order",
                "title": "力の伝わる順序が逆転しています",
                "detail": (
                    f"{cur['segment']}(frame {cur['peak_frame']}) より "
                    f"{nxt['segment']}(frame {nxt['peak_frame']}) が "
                    f"{gap / fps * 1000:.0f}ms 先にピークに達しています。"
                ),
                "cue": (
                    f"{cur['segment']}から先に動かす意識を。"
                    "体の中心から順に加速すると、力が末端まで乗ります。"
                ),
            }
    return None


@runtime_checkable
class Domain(Protocol):
    """1つの競技動作を解析するための実装。"""

    #: レジストリ上の名前（`domains.get("tennis_serve")`）
    name: str
    #: 人が読むラベル
    label: str
    #: 連鎖の順序を判定できる最低フレームレート
    chain_min_fps: float
    #: measure() が返す主要指標のキー。比較画面がこの順で並べる。
    headline: tuple[str, ...]

    def side(self, joints: np.ndarray) -> str:
        """利き側 "R"/"L" を決める。根拠は競技ごとに違う。"""

    def detect_phases(self, kin: Kinematics) -> dict[str, int]:
        """名前つきフレームを返す。持続的な動作なら区間の端でよい。"""

    def measure(self, kin: Kinematics, phases: dict[str, int]) -> dict:
        """指標の辞書。JSON 化できる値のみ。"""

    def judge(self, metrics: dict) -> list[dict]:
        """改善点。根拠のない閾値は置かず、空リストを返してよい。"""

    def report(self, metrics: dict, findings: list[dict], top_n: int) -> str:
        """人が読むテキスト。"""


class NotImplementedDomain:
    """まだ判定を持たないドメインの土台。

    `measure` までは実装し、`judge` は空を返す——という状態を
    「未完成」ではなく**正しい途中状態**として扱えるようにする。
    根拠のない閾値を置くより、判定しない方が常に良い。
    """

    name = "unnamed"
    label = "未実装"
    chain_min_fps = DEFAULT_CHAIN_MIN_FPS
    headline: tuple[str, ...] = ()
    #: 判定を入れる前に何を確かめる必要があるか。report がそのまま表示する。
    evidence_needed: tuple[str, ...] = ()

    def judge(self, metrics: dict) -> list[dict]:
        found = []
        c = chain_order_finding(metrics, self.chain_min_fps)
        if c:
            found.append(c)
        return found

    def report(self, metrics: dict, findings: list[dict], top_n: int = 2) -> str:
        rule = "=" * 62
        lines = [rule, f"  {self.label}", rule,
                 f"  フレーム数: {metrics['n_frames']}  ({metrics['fps']:.0f}fps)",
                 ""]
        for k in self.headline or sorted(metrics):
            v = metrics.get(k)
            if isinstance(v, float):
                lines.append(f"  {k:<28} {v:8.2f}")
            elif isinstance(v, (int, str)):
                lines.append(f"  {k:<28} {v}")
        if self.evidence_needed:
            lines += ["", "── 判定を入れる前に確かめること ──"]
            lines += [f"  ・{e}" for e in self.evidence_needed]
        lines += ["", rule, "  改善ポイント", rule]
        if not findings:
            lines.append("  判定ルールがまだありません（指標の表示のみ）。")
        else:
            for i, f in enumerate(findings[:top_n], 1):
                lines += [f"  {i}. [{f['tier']}] {f['title']}",
                          f"     {f['detail']}", f"     → {f['cue']}", ""]
        lines.append(rule)
        return "\n".join(lines)
