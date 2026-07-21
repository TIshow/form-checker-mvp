"""計測値から改善点を判定する（方針の層）。

## 判定の信頼度を3段階に分ける

閾値の正しさがそのままフィードバックの正しさになるため、
「何を根拠にその数字を決めたか」をコード上で区別する。

  TIER A 力学的原理     反例が考えにくい。断定してよい
  TIER B 文献参照       プロの実測レンジがある。参考として提示する
  TIER C 根拠なし       判定しない。数値を report で見せるだけ

TIER C を「欠点」として指摘してはいけない。実際、初期実装では体幹の傾きを
35°超で「傾きすぎ」と警告していたが、文献ではプロの接球時の体幹は
鉛直から約42°傾いており、**プロの技術を欠点と判定していた**。

## プロとの比較は行わない

プロ同士でもフォームは大きく異なり、差は必ずしも欠点ではない。体格が違えば
同じ関節角度は再現できない。TIER B で文献レンジを「参考情報」として示すことはあっても、
「プロと違うから直せ」とは言わない。

## 出典

- Kinematics characteristics of key point of interest during tennis serve:
  a systematic review and meta-analysis (Front. Sports Act. Living, 2024)
- Biomechanics of the Tennis Serve: Implications for Strength Training
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# TIER A — 力学的原理。反例が考えにくいので断定してよい
# --------------------------------------------------------------------------

# 重心が落ち始めてから打つとエネルギーが逃げる、というのは力学的に正しい。
# ただし「何秒遅れたら問題か」の実測値は見つかっていないため、
# 明らかに遅れている場合のみ指摘する保守的な値にしてある。
TH_CONTACT_LATE_S = 0.10

# サーブの連鎖は全体で0.1秒ほど、隣接する体節の時間差は 20〜40ms しかない。
# 1フレーム差を「順序の逆転」と呼ぶのはノイズを読んでいるだけ。
TH_CHAIN_MIN_GAP_S = 0.05
# 順序を論じるのに最低限必要な撮影フレームレート。
# 30fps では 1フレーム=33ms あり、上記の時間差を分解できない。
CHAIN_MIN_FPS = 60.0

# --------------------------------------------------------------------------
# TIER B — 文献にプロの実測レンジがあるもの
#
# 注意: 文献は「屈曲角」(0°=まっすぐ)、本実装は「関節角」(180°=まっすぐ) で
# 表す。関節角 = 180 - 屈曲角。
# --------------------------------------------------------------------------

# トロフィー時の前脚: 屈曲 64.5 ± 9.7° → 関節角 約 115°
# ただし屈曲量と球速の相関は弱いという報告があり(7.6°と14.7°で球速がほぼ同じ)、
# 「浅い＝悪い」と断じられない。平均より 2SD 以上浅い場合のみ情報として出す。
REF_KNEE_JOINT_DEG = 115.0
TH_KNEE_SHALLOW_DEG = 145.0   # 屈曲 35° 相当。プロ平均より約 3SD 浅い

# 接球時の肘: 屈曲 30.1 ± 15.9° → 関節角 約 150°
# 腕が伸びきるほど打点が高くなり、てこも長くなる。
REF_ELBOW_JOINT_DEG = 150.0
TH_ELBOW_BENT_DEG = 134.0     # 屈曲 46° 相当。プロ平均より約 1SD 曲がっている

# 接球時の体幹: 水平から 48 ± 7° = 鉛直から約 42°
# 傾けること自体は正しい技術（側屈が大きいほど打点が高くなる）。
# したがって「傾きすぎ」の判定は行わない。参考値としてのみ保持する。
REF_TRUNK_LEAN_DEG = 42.0

# --------------------------------------------------------------------------
# TIER C — 根拠が無く、判定に使わないもの
#
# これらは report で数値として表示するだけに留める。閾値を設けて
# 「欠点」と呼ぶだけの裏付けが無い。
#
#   打点の高さ比 (手首/頭)  … 標準的な指標ですらなく、比較対象が無い
#   X-factor (肩-腰の捻転差) … 研究の大半はゴルフのもの。サーブでの
#                              適正レンジを確認できていない
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 優先度: 力学的原理の違反 > 文献レンジからの逸脱
# --------------------------------------------------------------------------
P_PRINCIPLE = 100
P_TIMING = 90
P_ELBOW = 60
P_KNEE = 50


def _chain_order(m: dict) -> dict | None:
    """[TIER A] キネティックチェーンの順序の逆転を検出する。

    撮影フレームレートが足りない場合は測定できないので判定しない。
    """
    fps = m["fps"]
    if fps < CHAIN_MIN_FPS:
        return None

    chain = [c for c in m["kinetic_chain"] if c.get("reliable", True)]
    min_gap = max(TH_CHAIN_MIN_GAP_S * fps, 2.0)

    for cur, nxt in zip(chain, chain[1:]):
        gap = cur["peak_frame"] - nxt["peak_frame"]
        if gap >= min_gap:
            return {
                "priority": P_PRINCIPLE,
                "tier": "A",
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


def generate_feedback(m: dict) -> list[dict]:
    """指標から改善点を抽出し、優先度の高い順に返す。"""
    found: list[dict] = []

    # --- [TIER A] 力の伝わる順序 ------------------------------------------
    chain_issue = _chain_order(m)
    if chain_issue:
        found.append(chain_issue)

    # --- [TIER A] 打点のタイミング ----------------------------------------
    late = m["contact_vs_compeak_s"]
    if late > TH_CONTACT_LATE_S:
        found.append({
            "priority": P_TIMING,
            "tier": "A",
            "id": "contact_late",
            "title": "重心が落ち始めてから打っています",
            "detail": f"打点が重心の最高点より {late:.2f} 秒遅れています。",
            "cue": "伸び上がりの頂点で捉える意識を。体が落ち始めると力が逃げます。",
        })

    # --- [TIER B] 接球時の肘の伸展 ----------------------------------------
    elbow = m["elbow_at_contact_deg"]
    if np.isfinite(elbow) and elbow < TH_ELBOW_BENT_DEG:
        found.append({
            "priority": P_ELBOW,
            "tier": "B",
            "id": "elbow_bent_at_contact",
            "title": "打点で肘が曲がっています",
            "detail": (
                f"接球時の肘の関節角が {elbow:.0f}°。"
                f"プロの平均は約 {REF_ELBOW_JOINT_DEG:.0f}°（ほぼ伸展）です。"
            ),
            "cue": "腕を伸ばしきって高い打点で捉えると、てこが長くなり打点も上がります。",
        })

    # --- [TIER B] 沈み込みの深さ ------------------------------------------
    knee = m["min_knee_deg_overall"]
    if knee > TH_KNEE_SHALLOW_DEG:
        found.append({
            "priority": P_KNEE,
            "tier": "B",
            "id": "knee_shallow",
            "title": "沈み込みが浅めです",
            "detail": (
                f"最も曲げた時の膝の関節角が {knee:.0f}°（180°=伸びきり）。"
                f"プロのトロフィー時は約 {REF_KNEE_JOINT_DEG:.0f}° です。"
            ),
            "cue": (
                "膝を深く曲げてタメを作ると脚の力を使えます。"
                "ただし屈曲量と球速の相関は強くないため、優先度は高くありません。"
            ),
        })

    found.sort(key=lambda f: -f["priority"])
    return found
