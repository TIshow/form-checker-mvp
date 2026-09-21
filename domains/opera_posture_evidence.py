"""オペラ（発声姿勢 × 響き）の根拠表。生成AI（analysis/coach.py）に渡す唯一の「文献」。

規律は golf_swing_evidence.py と同じ。**今のところ本文まで確認した文献は無い。**
姿勢と声の関係を扱う論文は Journal of Voice 等の有料誌に多く、ここにあるのは
抄録（Europe PMC 経由）で読めた範囲だけ。だから tier B は一つも無く、
「改善点」と言えるものは無い。観察と、測れていないものの列挙が出力になる。

## 一次資料（抄録のみ確認 2026-09-21）

[KA20] Knight EJ, Austin SF. The Effect of Head Flexion/Extension on Acoustic Measures of
       Singing Voice Quality. J Voice 2020;34(6):964.e11-e21. doi:10.1016/j.jvoice.2019.06.019
[LO20] Longo L, Di Stadio A, Ralli M, et al. Voice Parameter Changes in Professional
       Musician-Singers Singing with and without an Instrument: The Effect of Body Posture.
       Folia Phoniatr Logop 2020;72(4):309-315. doi:10.1159/000501202
[OM96] Omori K, et al. Singing power ratio: quantitative evaluation of singing voice quality.
       J Voice 1996（定義の原典。未確認）
[SU74] Sundberg J. Articulatory interpretation of the "singing formant". JASA 1974（未確認）
"""

from __future__ import annotations

from domains.base import TIER_A, TIER_C

VERIFIED = "本文確認"
UNVERIFIED = "未確認"

EVIDENCE: list[dict] = [
    {
        "id": "O-A1", "tier": TIER_A, "verified": VERIFIED,
        "claim": "音圧・SPR・帯域比の絶対値はマイク距離・会場・圧縮で変わる。意味があるのは"
                 "**同一録音・同一条件内の相対比較**だけ。別の日・別の場所の数値は比べられない。",
        "metrics": ["audio_spr_mean_db", "audio_formant_mean_db", "audio_tilt_mean_db", "audio_voiced_fraction"],
        "source": "計測の定義（core/audio.py §時間軸・§帯域）。反例が考えにくい",
        "note": "改善点の根拠にはならない（何かを直せとは言っていない）。数値を語るときの前提。",
    },
    {
        "id": "O-A2", "tier": TIER_A, "verified": VERIFIED,
        "claim": "伴奏が同じマイクに入っている録音では、響き・声量・F0 は歌手だけの値ではない。"
                 "F0 は声域に絞っても、SPR・帯域比は歌手＋伴奏の混合になる。",
        "metrics": ["audio_spr_mean_db", "audio_formant_mean_db", "audio_f0_jump_fraction"],
        "source": "計測の定義（core/audio.py）。この動画のスペクトログラムで伴奏の倍音線を確認（issue 013）",
        "note": "撮り直し（無伴奏・伴奏はイヤホン）の根拠。",
    },
    {
        "id": "O-C1", "tier": TIER_C, "verified": UNVERIFIED,
        "claim": "頭の屈曲（顎を引く方向）で SPR が上がり、伸展（顎を上げる方向）で F0 と低域の"
                 "成分が大きくなった。SPR の上昇は高域が増えたのではなく低域が減ったため。"
                 "母音・音高との交互作用なし。",
        "metrics": ["head_forward_cm", "audio_spr_mean_db"],
        "source": "[KA20] 抄録（Europe PMC）。本文・N・効果量は未確認",
        "note": "本システムの head_forward_cm は『肩の中点からの頭の水平距離』で、頭の屈曲・伸展角ではない。"
                "対応づけには首〜頭ベクトルの角度が要る（未実装）。判定に使わない。",
    },
    {
        "id": "O-C2", "tier": TIER_C, "verified": UNVERIFIED,
        "claim": "プロの歌手 17 名で、楽器を弾きながら歌うとシンガーズフォルマントが弱まり、"
                 "肩と背中の位置が声の指標に影響した。頭と首の位置は影響しなかった。",
        "metrics": ["shoulder_tilt_mean_deg", "trunk_lean_mean_deg", "trunk_length_change_cm"],
        "source": "[LO20] 抄録（Europe PMC）。本文・効果量は未確認",
        "note": "『肩・背中が効く』という方向の示唆。閾値は無い。判定に使わない。",
    },
    {
        "id": "O-C3", "tier": TIER_C, "verified": UNVERIFIED,
        "claim": "SPR（2〜4 kHz のピーク − 0〜2 kHz のピーク）は歌声の質の客観指標として使われ、"
                 "訓練年数と関係するとされる。",
        "metrics": ["audio_spr_mean_db"],
        "source": "[OM96] 定義の原典（未確認）。検索結果の要約のみ",
        "note": "『大きいほど良い』とは本文で確認していない。同一人物の相対比較にだけ使う。",
    },
    {
        "id": "O-C4", "tier": TIER_C, "verified": UNVERIFIED,
        "claim": "シンガーズフォルマントは 3 kHz 付近の共鳴の山（訓練された男声で約 2.8〜3.4 kHz）。"
                 "女声・高音域では帯域が異なりうる。",
        "metrics": ["audio_formant_mean_db"],
        "source": "[SU74]（未確認）、NCVS の解説ページ（Web）",
        "note": "本システムの帯域 2.8〜3.4 kHz はこの解説に合わせた仮置き。ソプラノでは合わない可能性がある。",
    },
    {
        "id": "O-C5", "tier": TIER_C, "verified": VERIFIED,
        "claim": "重心の揺れ・体幹長の変化・頭部の前後位置は、姿勢の粗い代用量であり、"
                 "響きとの関係を示した文献は根拠表に無い。",
        "metrics": ["com_sway_cm", "trunk_length_change_cm", "head_forward_cm", "trunk_lean_mean_deg", "shoulder_tilt_mean_deg"],
        "source": "本システムの定義（domains/opera_posture.py）",
        "note": "観察として数値を述べるだけ。",
    },
]
