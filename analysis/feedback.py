"""計測値から改善点を判定する（方針の層）。

プロとの比較は行わない。プロ同士でもフォームは大きく異なり、差は必ずしも
欠点ではないし、体格が違えば真似できない。代わりに誰にでも当てはまる
力学的原理で判定する。参照動画も権利処理も不要で、かつ最も価値の高い指摘になる。

この層は閾値の調整で頻繁に変わる。物理計算 (serve.py) を巻き込まないよう分離している。
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# 判定の閾値
#
# 順序・タイミング系は理論から断定できるが、角度・量の系は暫定値。
# 「膝を何度まで曲げるべきか」は文献値かデータ蓄積で確定させる必要がある。
# --------------------------------------------------------------------------
TH_CONTACT_LATE_S = 0.10      # 重心ピークからこれ以上遅れたら「落ちながら打っている」
TH_KNEE_SHALLOW_DEG = 150.0   # これより曲がっていなければ沈み込みが浅い
TH_CONTACT_RATIO_LOW = 1.15   # 打点が頭の高さのこの倍未満なら伸びが不足
TH_TRUNK_LEAN_DEG = 35.0      # 体幹が傾きすぎ
TH_X_FACTOR_SMALL_DEG = 20.0  # 捻転差が小さい

# 実際のサーブの連鎖は全体で0.1秒ほど、隣接する体節の時間差は 20〜40ms しかない。
# 1フレーム差を「順序の逆転」と呼ぶのはノイズを読んでいるだけなので、
# これ以上はっきり逆転している場合しか指摘しない。
TH_CHAIN_MIN_GAP_S = 0.05
# 順序を論じるのに最低限必要な撮影フレームレート。
# 30fps では 1フレーム=33ms あり、連鎖の時間差を分解できない。
CHAIN_MIN_FPS = 60.0

# 優先度: 力学的原理の違反 > 大きな逸脱 > 小さな逸脱
P_PRINCIPLE = 100
P_TIMING = 90
P_CONTACT = 60
P_KNEE = 50
P_XFACTOR = 45
P_LEAN = 40


def _chain_order(m: dict) -> dict | None:
    """キネティックチェーンの順序の逆転を検出する。

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

    # --- 原理1: 力の伝わる順序 --------------------------------------------
    chain_issue = _chain_order(m)
    if chain_issue:
        found.append(chain_issue)

    # --- 原理2: 打点のタイミング ------------------------------------------
    late = m["contact_vs_compeak_s"]
    if late > TH_CONTACT_LATE_S:
        found.append({
            "priority": P_TIMING,
            "id": "contact_late",
            "title": "重心が落ち始めてから打っています",
            "detail": f"打点が重心の最高点より {late:.2f} 秒遅れています。",
            "cue": "伸び上がりの頂点で捉える意識を。体が落ち始めると力が逃げます。",
        })

    # --- 打点の高さ --------------------------------------------------------
    ratio = m["contact_height_ratio"]
    if np.isfinite(ratio) and ratio < TH_CONTACT_RATIO_LOW:
        found.append({
            "priority": P_CONTACT,
            "id": "contact_low",
            "title": "打点が低めです",
            "detail": f"打点の高さが頭の位置の {ratio:.2f} 倍です。",
            "cue": "ボールに向かって伸び上がり、腕を最大限に伸ばして捉えましょう。",
        })

    # --- 沈み込みの深さ ----------------------------------------------------
    knee = m["min_knee_deg_overall"]
    if knee > TH_KNEE_SHALLOW_DEG:
        found.append({
            "priority": P_KNEE,
            "id": "knee_shallow",
            "title": "沈み込みが浅いです",
            "detail": f"最も曲げた時の膝角度が {knee:.0f}°（180°=伸びきり）。",
            "cue": "膝をもう少し深く曲げてタメを作ると、脚の力を使えます。",
        })

    # --- 捻転差 ------------------------------------------------------------
    xf = m["max_x_factor_deg"]
    if xf < TH_X_FACTOR_SMALL_DEG:
        found.append({
            "priority": P_XFACTOR,
            "id": "x_factor_small",
            "title": "上半身と下半身の捻転差が小さいです",
            "detail": f"肩と腰の最大の捻転差が {xf:.0f}°。",
            "cue": "腰を先に回し、肩を残す意識を。捻れが大きいほど力が生まれます。",
        })

    # --- 体幹の傾き --------------------------------------------------------
    lean = m["trunk_lean_at_contact_deg"]
    if lean > TH_TRUNK_LEAN_DEG:
        found.append({
            "priority": P_LEAN,
            "id": "trunk_lean",
            "title": "打点で体幹が傾きすぎています",
            "detail": f"打点時の体幹の傾きが {lean:.0f}°。",
            "cue": "軸を保って伸び上がりましょう。傾きすぎると安定性が落ちます。",
        })

    found.sort(key=lambda f: -f["priority"])
    return found
