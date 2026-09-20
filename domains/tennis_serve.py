"""テニスのサーブ。局面・指標・判定・表示。

計測の中身は core/ にある。ここにあるのは「サーブだからそう測る」部分だけ:
打点をどう見つけるか、どの指標を選ぶか、どの閾値で欠点と呼ぶか。

## 判定の信頼度

`domains/base.py` の TIER A/B/C に従う。下の TIER C を欠点として
指摘してはいけない。

## プロとの比較は行わない

プロ同士でもフォームは大きく異なり、差は必ずしも欠点ではない。体格が違えば
同じ関節角度は再現できない。TIER B で文献レンジを「参考情報」として示すことは
あっても、「プロと違うから直せ」とは言わない。

## 出典

[1] Kinematics characteristics of key point of interest during tennis serve
    among tennis players: a systematic review and meta-analysis.
    Frontiers in Sports and Active Living (2024). 27件の研究のメタアナリシス。
    https://doi.org/10.3389/fspor.2024.1432030

Tier B の値は上記の本文を開き、セクション・図番号まで確認したものだけを使う。
検索結果の要約だけを根拠に「文献値」と称してはいけない。
"""

from __future__ import annotations

import numpy as np

from core import FOOT_IDS, Kinematics, dominant_side_by_peak_height
from domains.base import (
    DEFAULT_CHAIN_MIN_FPS, P_TIMING, TIER_A, TIER_B, chain_order_finding,
)

# --------------------------------------------------------------------------
# TIER A — 力学的原理
# --------------------------------------------------------------------------

# 重心が落ち始めてから打つとエネルギーが逃げる、というのは力学的に正しい。
# ただし「何秒遅れたら問題か」の実測値は見つかっていないため、
# 明らかに遅れている場合のみ指摘する保守的な値にしてある。
TH_CONTACT_LATE_S = 0.10

# --------------------------------------------------------------------------
# TIER B — 論文本文で該当箇所を確認した値のみ
#
# 注意1: 文献は「屈曲角」(0°=まっすぐ)、本実装は「関節角」(180°=まっすぐ)。
#        関節角 = 180 - 屈曲角。
# 注意2: **角度の定義が一致している保証はない。** 論文がどの基準面・
#        どの体節ベクトルで測ったかまでは確認できていない。定義が違えば
#        数値を直接比べる意味がないため、Tier B は「参考」に留める。
# --------------------------------------------------------------------------

# [1] Section 3.4, Figure 3: 前脚の膝屈曲(トロフィー時) 64.5 ± 9.7°
#     → 関節角 約 115°。個別研究のばらつきは 45.0°〜82.8° と非常に大きい。
REF_KNEE_JOINT_DEG = 115.0
TH_KNEE_SHALLOW_DEG = 145.0   # 屈曲 35° 相当。プロ平均より約 3SD 浅い

# [1] Section 3.6, Figure 7: 肘屈曲(接球時) 30.1 ± 15.9°
#     → 関節角 約 150°。論文は「研究間・研究内のばらつきが大きい」と注記。
REF_ELBOW_JOINT_DEG = 150.0
TH_ELBOW_BENT_DEG = 134.0     # 屈曲 46° 相当。プロ平均より約 1SD 曲がっている

# [1] Section 3.4, Figure 2: 体幹傾斜(トロフィー時) 25.0 ± 7.1°
#     ただしこれは**トロフィー時**であり、本実装が測る**接球時**ではない。
REF_TRUNK_LEAN_TROPHY_DEG = 25.0

# --------------------------------------------------------------------------
# TIER C — 判定に使わないもの（表示のみ）
#
#   接球時の体幹の傾き … [1] はトロフィー時しか報告しておらず基準値が無い
#   打点の高さ比       … 標準的な指標ではなく比較対象が無い
#   X-factor           … 研究の大半はゴルフ。サーブでの適正レンジ未確認
#   ラケットドロップ   … 手法比較では目視評価を再現したが、
#                        「何度あれば良いか」の基準値は持っていない
# --------------------------------------------------------------------------

P_ELBOW = 60
P_KNEE = 50

_RULE = "=" * 62


class TennisServe:
    name = "tennis_serve"
    label = "テニス サーブ"
    chain_min_fps = DEFAULT_CHAIN_MIN_FPS
    headline = ("contact_height_m", "com_rise_m", "min_knee_deg_overall",
                "elbow_at_contact_deg", "racket_drop_deg")
    plot_phases = ("loading", "contact")

    # -- 利き側 ------------------------------------------------------------
    def side(self, joints: np.ndarray) -> str:
        """打点で最も高く上がる腕をラケット側とする。"""
        return dominant_side_by_peak_height(joints)

    # -- 局面 --------------------------------------------------------------
    def detect_phases(self, kin: Kinematics) -> dict[str, int]:
        """打点・沈み込み・重心頂点。

        打点     = ラケット側手首が最も高くなるフレーム
        沈み込み = その手前で重心が最も低くなるフレーム
        """
        wrist_h = kin.height(kin.J[:, kin.idx("wrist")])
        contact = int(np.argmax(wrist_h))
        loading = int(np.argmin(kin.com_height[: max(contact, 1)]))
        com_peak = int(np.argmax(kin.com_height))
        return {"loading": loading, "contact": contact, "com_peak": com_peak}

    # -- 指標 --------------------------------------------------------------
    def measure(self, kin: Kinematics, phases: dict[str, int]) -> dict:
        loading, contact = phases["loading"], phases["contact"]
        com_peak = phases["com_peak"]
        fps = kin.fps

        knee = kin.knee_angles()
        body_h = kin.body_height_proxy()
        ground = kin.ground()
        wrist_h = kin.height(kin.J[:, kin.idx("wrist")]) - ground
        feet_h = kin.height(kin.J[:, FOOT_IDS]).min(axis=1) - ground
        drop = kin.hand_direction()
        seg = drop[loading : contact + 1]

        return {
            "domain": self.name,
            "racket_side": kin.side,
            "fps": fps,
            "n_frames": kin.F,
            "phases": phases,
            # 脚のドライブ
            "com_low_m": float(kin.com_height[loading]),
            "com_peak_m": float(kin.com_height[com_peak]),
            "com_rise_m": float(kin.com_height[com_peak] - kin.com_height[loading]),
            "drive_time_s": float((com_peak - loading) / fps),
            "foot_clearance_m": float(feet_h[loading:].max()),
            # 沈み込み
            "min_knee_deg": float(knee[loading]),
            "min_knee_deg_overall": float(knee[: max(contact, 1)].min()),
            # 打点
            "contact_height_m": float(wrist_h[contact]),
            "contact_height_ratio": (float(wrist_h[contact] / body_h)
                                     if body_h > 0 else float("nan")),
            "contact_vs_compeak_s": float((contact - com_peak) / fps),
            "elbow_at_contact_deg": float(kin.elbow_angle()[contact]),
            "trunk_lean_at_contact_deg": float(kin.trunk_lean()[contact]),
            # ラケットドロップ（TIER C。判定はしない）
            "racket_drop_deg": float(seg.max()) if len(seg) else float("nan"),
            "racket_drop_frame": (int(seg.argmax()) + loading
                                  if len(seg) else contact),
            "racket_drop_at_contact_deg": float(drop[contact]),
            # 捻転
            "max_x_factor_deg": float(
                kin.x_factor()[loading : max(contact, loading + 1)].max()),
            # 連鎖
            "kinetic_chain": kin.kinetic_chain(loading, min(contact + 2, kin.F)),
        }

    # -- 判定 --------------------------------------------------------------
    def judge(self, m: dict) -> list[dict]:
        found: list[dict] = []

        chain_issue = chain_order_finding(m, self.chain_min_fps)
        if chain_issue:
            found.append(chain_issue)

        # [TIER A] 打点のタイミング
        late = m["contact_vs_compeak_s"]
        if late > TH_CONTACT_LATE_S:
            found.append({
                "priority": P_TIMING,
                "tier": TIER_A,
                "id": "contact_late",
                "title": "重心が落ち始めてから打っています",
                "detail": f"打点が重心の最高点より {late:.2f} 秒遅れています。",
                "cue": "伸び上がりの頂点で捉える意識を。体が落ち始めると力が逃げます。",
            })

        # [TIER B] 接球時の肘の伸展
        elbow = m["elbow_at_contact_deg"]
        if np.isfinite(elbow) and elbow < TH_ELBOW_BENT_DEG:
            found.append({
                "priority": P_ELBOW,
                "tier": TIER_B,
                "id": "elbow_bent_at_contact",
                "title": "打点で肘が曲がっています",
                "detail": (
                    f"接球時の肘の関節角が {elbow:.0f}°（180=伸びきり）。"
                    f"プロの平均は約 {REF_ELBOW_JOINT_DEG:.0f}° です。"
                ),
                "cue": "腕を伸ばしきって高い打点で捉えると、てこが長くなり打点も上がります。",
            })

        # [TIER B] 沈み込みの深さ
        knee = m["min_knee_deg_overall"]
        if knee > TH_KNEE_SHALLOW_DEG:
            found.append({
                "priority": P_KNEE,
                "tier": TIER_B,
                "id": "knee_shallow",
                "title": "沈み込みが浅めです",
                "detail": (
                    f"最も曲げた時の膝の関節角が {knee:.0f}°"
                    "（180=伸びきり / 小さいほど深く曲げている）。"
                    f"プロのトロフィー時は約 {REF_KNEE_JOINT_DEG:.0f}° です。"
                ),
                "cue": (
                    "膝を深く曲げてタメを作ると脚の力を使えます。"
                    "ただしプロでも45°〜83°と幅が大きく、優先度は高くありません。"
                ),
            })

        found.sort(key=lambda f: -f["priority"])
        return found

    # -- 表示 --------------------------------------------------------------
    def report(self, m: dict, feedback: list[dict], top_n: int = 2) -> str:
        ph = m["phases"]
        fps = m["fps"]
        side = "右" if m["racket_side"] == "R" else "左"

        lines = [
            _RULE, "  サーブ解析レポート", _RULE,
            f"  利き手(推定): {side}利き   フレーム数: {m['n_frames']}  ({fps:.0f}fps)",
            "",
            "── 動作フェーズ ──",
            f"  沈み込み  : frame {ph['loading']:3d}  ({ph['loading'] / fps:.2f}s)",
            f"  重心の頂点: frame {ph['com_peak']:3d}  ({ph['com_peak'] / fps:.2f}s)",
            f"  打点      : frame {ph['contact']:3d}  ({ph['contact'] / fps:.2f}s)",
            "",
            "── 脚のドライブ ──",
            f"  重心の最低点 : {m['com_low_m']:.3f} m",
            f"  重心の最高点 : {m['com_peak_m']:.3f} m",
            f"  伸び上がり   : {m['com_rise_m'] * 100:+.1f} cm / {m['drive_time_s']:.2f} 秒",
            f"  沈み込み膝角 : {m['min_knee_deg_overall']:.0f}° "
            f"(180=伸びきり / **小さいほど深い** / プロ約{REF_KNEE_JOINT_DEG:.0f}°)",
            "",
            "── 打点 ──",
            f"  高さ          : {m['contact_height_m']:.3f} m (床から)",
            f"  重心頂点との差: {m['contact_vs_compeak_s']:+.2f} 秒 (0に近いほど良い)",
            f"  肘の角度      : {m['elbow_at_contact_deg']:.0f}° "
            f"(180=伸びきり / **大きいほど伸びている** / プロ約{REF_ELBOW_JOINT_DEG:.0f}°)",
            "",
            "── 参考値（判定に使っていない）──",
            f"  ラケットドロップ  : {m['racket_drop_deg']:.0f}° "
            "(0=真上 / 180=真下 / 大きいほど深く落ちている)  ※基準値なし",
            f"  跳躍(足の浮き)    : {m['foot_clearance_m'] * 100:+.1f} cm  "
            "※3手法とも過小評価する（issue 008）",
            f"  体幹の傾き(接球時): {m['trunk_lean_at_contact_deg']:.0f}°  "
            f"※文献はトロフィー時{REF_TRUNK_LEAN_TROPHY_DEG:.0f}°のみ報告",
            f"  打点/身長の比     : {m['contact_height_ratio']:.2f}  ※比較対象なし",
            f"  最大捻転差        : {m['max_x_factor_deg']:.0f}°  ※適正レンジ未確認",
            "",
            "── キネティックチェーン（理想は上から順にピーク）──",
        ]

        for c in m["kinetic_chain"]:
            note = "" if c.get("reliable", True) else "  ※回転が小さく判定対象外"
            lines.append(
                f"  {c['segment']:<8} peak frame {c['peak_frame']:3d}  "
                f"({c['peak_speed']:6.0f} deg/s){note}"
            )

        if fps < self.chain_min_fps:
            lines += [
                "",
                f"  ⚠️ {fps:.0f}fps では 1フレーム={1000 / fps:.0f}ms。",
                "     連鎖の時間差は 20〜40ms のため、この撮影では順序を判定できません。",
                f"     評価するには {self.chain_min_fps:.0f}fps 以上"
                "（できれば120/240fps）で撮影してください。",
                "     上の数値は参考値です。",
            ]

        lines += ["", _RULE, "  改善ポイント", _RULE]
        if not feedback:
            lines.append("  ルール上の問題は検出されませんでした。")
        else:
            for i, f in enumerate(feedback[:top_n], 1):
                lines += [f"  {i}. {f['title']}", f"     {f['detail']}",
                          f"     → {f['cue']}", ""]
            rest = len(feedback) - top_n
            if rest > 0:
                lines.append(f"  (他 {rest} 件は今回は省略。一度に直すのは1〜2点まで)")

        lines.append(_RULE)
        return "\n".join(lines)
