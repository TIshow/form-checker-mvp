"""ゴルフスイングの根拠表。生成AI（analysis/coach.py）に渡す唯一の「文献」。

## 規律（domains/README と同じ）

- **tier A**: 力学的原理。断定してよい
- **tier B**: 本文を開いてセクション番号まで確認した文献の値・関係。参考として提示する
- **tier C**: 根拠が無い、または単一研究・バイアス大・通説のみ。**判定に使わない**。
  観察として数値を述べるだけ

`verified` は「本文を開いて確認した」か「未確認」か。未確認の項目は tier に関わらず
改善点の根拠にできない（`coach.validate` が落とす）。

「プロとの比較」は持ち込まない。ここにあるのは「身体の使い方として効率が悪い／
傷害と関係がある」と文献が言っている範囲と、**言っていない範囲**（負の根拠）。
負の根拠は、生成AIが通説で断定するのを止めるために要る。

## 一次資料

[B22] Bourgain M, Rouch P, Rouillon O, Thoreux P, Sauret C. Golf Swing Biomechanics:
      A Systematic Review and Methodological Recommendations for Kinematics.
      Sports 2022;10(6):91. https://doi.org/10.3390/sports10060091（オープンアクセス、本文確認 2026-09-21）
[W24] Watson M, Coughlan D, Clement ND, Murray IR, Murray AD, Miller SC. Biomechanical
      parameters of the golf swing associated with lower back pain: A systematic review.
      J Sports Sci 2023;41(24):2236-2250. https://doi.org/10.1080/02640414.2024.2319443
      （オープンアクセス、本文確認 2026-09-21）
[E20] Edwards N, Dickin C, Wang H. Low back pain and golf: A review of biomechanical
      risk factors. Sports Med Health Sci 2020. PMC9219256
      （本文を自動抽出で参照。セクション名は取得したが人手で通読していない → 未確認扱い）
"""

from __future__ import annotations

from domains.base import TIER_A, TIER_B, TIER_C

VERIFIED = "本文確認"
UNVERIFIED = "未確認"

EVIDENCE: list[dict] = [
    {
        "id": "G-A1", "tier": TIER_A, "verified": VERIFIED,
        "claim": "力は近位（骨盤）→胸郭→腕→クラブの順に伝えるのが、末端速度を最大にする（速度の時間的加算）。"
                 "順序が逆転すると末端が損をする。ただし体節間の時間差は 4〜56ms で、"
                 "30fps（33ms/フレーム）では順序を判定できない。",
        "metrics": ["kinetic_chain"],
        "source": "[B22] §3.7.1 Rationale, §3.7.2（timing differences 4–56 ms; downswing 0.3 s）、§3.7.3（手法依存で計測が難しい）",
        "note": "判定は fps が chain_min_fps 以上のときだけ。30fps の映像では『順序は測れない』と述べること。",
    },
    {
        "id": "G-B1", "tier": TIER_B, "verified": VERIFIED,
        "claim": "ダウンスイングの所要時間は技量によらず約 0.3 秒で、再現性が高い（男性 SD < 0.04 s）。",
        "metrics": ["downswing_s"],
        "source": "[B22] §3.2.1 Phases, Table 1; §3.3.6 Recommendations（'downswing ... lasts about 0.3 s'）",
        "note": "0.3 秒から外れていても欠点とは言えない（レンジの SD は小さいが、外れる理由は多い）。参考値として述べる。",
    },
    {
        "id": "G-B2", "tier": TIER_B, "verified": VERIFIED,
        "claim": "X-factor（骨盤と肩の捻転差）の典型値は、肩ライン対骨盤で約 60°、胸郭対骨盤で約 30°。"
                 "定義で値が倍違う。クラブヘッド速度との関係は『ある』とする研究と『ない』とする研究が併存し、"
                 "プロと一般の差は約 11%。閾値の合意は無い。",
        "metrics": ["x_factor_at_top_deg", "x_factor_max_deg"],
        "source": "[B22] §3.4.2 Commentary（~60° vs ~30°; link to clubhead speed [38,55,59,60] vs none [7,58]; ~11%）, §3.4.4 Typical Values, Table 3",
        "note": "本システムの X-factor は肩と股関節の位置ベース（肩ライン対骨盤に近い）。数値の大小で欠点と言わない。",
    },
    {
        "id": "G-B3", "tier": TIER_B, "verified": VERIFIED,
        "claim": "捻転差の最大値はトップではなく、ダウンスイング開始直後（トップから 1〜18% 後）に来る（X-factor stretch）。"
                 "骨盤が先に回り始め、胸郭がまだ戻っていない間に生じ、体幹筋の伸張短縮サイクルを使う。",
        "metrics": ["x_factor_at_top_deg", "x_factor_max_deg"],
        "source": "[B22] §3.4.1 Rationale（Cheetham et al. [55]）, §3.4.2（'approximately 1 to 18% after the conventional X-factor'）",
        "note": "x_factor_max − x_factor_at_top が 0 に近ければ『骨盤先行の伸張が見えていない』と言えるが、"
                "30fps・位置ベースでは数°の差は測れない。断定しない。",
    },
    {
        "id": "G-B4", "tier": TIER_B, "verified": VERIFIED,
        "claim": "エリートゴルファーでは、リード側への腰椎側屈のピークが大きい群に腰痛が多い"
                 "（Lindsay & Horton 2007; Quinn et al. 2022 で支持。一般ゴルファーでは差なし Cole & Grimshaw 2014）。",
        "metrics": [],
        "source": "[W24] §3.4.1 Kinematics（'Peak lead-side lumbar lateral flexion was greater in elite golfers with LBP'）",
        "note": "本システムは側屈を測っていない。『測れていない』として挙げる。",
    },
    {
        "id": "G-B5", "tier": TIER_B, "verified": VERIFIED,
        "claim": "腰痛と『悪いスイング技術』の関係を裏づける決定的な証拠は無い。関係を示した研究は"
                 "少数・バイアス大・結果が食い違う。",
        "metrics": [],
        "source": "[W24] Abstract（'There is no conclusive evidence to support the commonly held belief that LBP is associated with \"poor\" golf swing technique'）, §3.3 Risk of bias",
        "note": "傷害リスクを断定しないための負の根拠。改善点を『腰を守るため』と言い切らないこと。",
    },
    {
        "id": "G-B6", "tier": TIER_B, "verified": VERIFIED,
        "claim": "クランチファクター（体幹側屈×軸回転速度）と腰痛の関係は示されていない。"
                 "クラブヘッド速度とはわずかに負の相関。",
        "metrics": [],
        "source": "[B22] §3.5.2（'no study has demonstrated a link between crunch factors and low-back pain'）; [W24] §3.4.1（recreational: no difference）",
        "note": "負の根拠。",
    },
    {
        "id": "G-C1", "tier": TIER_C, "verified": VERIFIED,
        "claim": "腰痛を発症したエリートゴルファーは、発症前の時点でリード膝の屈曲とリード足首の背屈が小さかった（前向き研究 1 件、複数局面で再現）。",
        "metrics": ["lead_knee_at_top_deg", "lead_knee_at_impact_deg"],
        "source": "[W24] §3.4.1（Quinn, Olivier & McKinon 2022; 'reduced lead knee flexion and reduced lead ankle dorsiflexion, were replicated at more than one point of the swing'）",
        "note": "単一研究・バイアス『serious』評価。方向（伸びすぎが悪い？）は示唆に留まる。判定に使わない。",
    },
    {
        "id": "G-C2", "tier": TIER_C, "verified": VERIFIED,
        "claim": "頭の上下動が小さいほど良い、という通説を支持する文献値は今回の一次資料に無い。",
        "metrics": ["head_move_cm"],
        "source": "なし（[B22] [W24] に該当項目なし）",
        "note": "観察として数値を述べるだけ。『軸がぶれている』と言わない。",
    },
    {
        "id": "G-C3", "tier": TIER_C, "verified": VERIFIED,
        "claim": "バックスイング:ダウンスイング = 3:1 のテンポ比（Tour Tempo）を支持する査読文献は見つかっていない。",
        "metrics": ["tempo_ratio", "backswing_s"],
        "source": "なし（Novosel & Garrity『Tour Tempo』は書籍。未確認）",
        "note": "ダウンスイングの所要時間だけは G-B1 で参考値がある。比そのものは判定しない。",
    },
    {
        "id": "G-C4", "tier": TIER_C, "verified": UNVERIFIED,
        "claim": "アドレスの至適姿勢は体幹前屈 45°・中立脊柱、バックスイングで体重の約 40% を後ろ足へ（Hume, Keogh & Reid 2005 の引用として）。",
        "metrics": ["spine_tilt_address_deg", "spine_tilt_change_deg"],
        "source": "[E20]（自動抽出。'Lumbar hyperextension' 節。原典 Hume 2005 は未確認）",
        "note": "本システムの前傾角の定義（直立からの体幹の傾き）と一致する保証が無い。比較しない。"
                "『アーリーエクステンション』は [E20] [W24] [B22] のいずれにも項目が無い。",
    },
    {
        "id": "G-C5", "tier": TIER_C, "verified": VERIFIED,
        "claim": "両手首の距離のばらつきは復元の健全性の指標であり、身体の使い方の指標ではない。",
        "metrics": ["grip_spread_sd_cm", "grip_spread_mean_cm"],
        "source": "本システムの定義（domains/golf_swing.py）",
        "note": "大きいときは『腕の復元が崩れている』と述べ、フォームの話にしない。",
    },
]
