"""解析結果を人が読める形に整形する（表示の層）。

将来 Web UI に置き換わる想定なので、計測・判定からは独立させている。
"""

from __future__ import annotations

from .feedback import (
    CHAIN_MIN_FPS,
    REF_ELBOW_JOINT_DEG,
    REF_KNEE_JOINT_DEG,
    REF_TRUNK_LEAN_DEG,
)

#: 一度に提示する指摘の上限。人は複数の修正キューを同時に処理できず、
#: 15件の指摘は0件と同じになる。
DEFAULT_TOP_N = 2

_RULE = "=" * 62


def format_report(m: dict, feedback: list[dict], top_n: int = DEFAULT_TOP_N) -> str:
    ph = m["phases"]
    fps = m["fps"]
    side = "右" if m["racket_side"] == "R" else "左"

    lines = [
        _RULE,
        "  サーブ解析レポート",
        _RULE,
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
        f"(180=伸びきり / プロ約{REF_KNEE_JOINT_DEG:.0f}°)",
        "",
        "── 打点 ──",
        f"  高さ          : {m['contact_height_m']:.3f} m",
        f"  重心頂点との差: {m['contact_vs_compeak_s']:+.2f} 秒 (0に近いほど良い)",
        f"  肘の角度      : {m['elbow_at_contact_deg']:.0f}° "
        f"(プロ約{REF_ELBOW_JOINT_DEG:.0f}°)",
        f"  体幹の傾き    : {m['trunk_lean_at_contact_deg']:.0f}° "
        f"(プロ約{REF_TRUNK_LEAN_DEG:.0f}° / 傾けるのは正しい技術)",
        "",
        "── 参考値（判定には使っていない）──",
        f"  打点/頭の高さ比: {m['contact_height_ratio']:.2f}  ※標準指標が無く比較対象なし",
        f"  最大捻転差     : {m['max_x_factor_deg']:.0f}°  ※サーブでの適正レンジ未確認",
        "",
        "── キネティックチェーン（理想は上から順にピーク）──",
    ]

    for c in m["kinetic_chain"]:
        note = "" if c.get("reliable", True) else "  ※回転が小さく判定対象外"
        lines.append(
            f"  {c['segment']:<8} peak frame {c['peak_frame']:3d}  "
            f"({c['peak_speed']:6.0f} deg/s){note}"
        )

    if fps < CHAIN_MIN_FPS:
        lines += [
            "",
            f"  ⚠️ {fps:.0f}fps では 1フレーム={1000 / fps:.0f}ms。",
            "     連鎖の時間差は 20〜40ms のため、この撮影では順序を判定できません。",
            f"     評価するには {CHAIN_MIN_FPS:.0f}fps 以上"
            "（できれば120/240fps）で撮影してください。",
            "     上の数値は参考値です。",
        ]

    lines += ["", _RULE, "  改善ポイント", _RULE]
    if not feedback:
        lines.append("  ルール上の問題は検出されませんでした。")
    else:
        for i, f in enumerate(feedback[:top_n], 1):
            lines += [
                f"  {i}. {f['title']}",
                f"     {f['detail']}",
                f"     → {f['cue']}",
                "",
            ]
        rest = len(feedback) - top_n
        if rest > 0:
            lines.append(f"  (他 {rest} 件は今回は省略。一度に直すのは1〜2点まで)")

    lines.append(_RULE)
    return "\n".join(lines)
