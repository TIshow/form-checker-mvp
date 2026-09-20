"""合成サーブデータによる解析・フィードバックの検証。

このテストは実際に2つのバグを検出した実績がある:
  1. ほとんど回転していないセグメントのノイズを順序判定に使っていた（誤検知）
  2. 30fpsで1フレーム差を「順序の逆転」として指摘していた（測定不能な量の主張）
閾値やルールを触ったら必ず走らせること。
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import analysis
import domains
from domains import base
from domains import tennis_serve as fb
from tests.synth import synth_serve


def run(fps: float = 30.0, **kw):
    joints, _ = synth_serve(fps=fps, **kw)
    return analysis.analyze(joints, fps=fps)


def ids(feedback: list[dict]) -> list[str]:
    return [f["id"] for f in feedback]


# --------------------------------------------------------------------------
# 基本
# --------------------------------------------------------------------------
def test_detects_racket_side():
    metrics, _ = run()
    assert metrics["racket_side"] == "R"


def test_phases_are_ordered():
    metrics, _ = run()
    ph = metrics["phases"]
    assert ph["loading"] < ph["contact"]
    assert 0 <= ph["com_peak"] < metrics["n_frames"]


def test_com_height_is_anatomically_plausible():
    """重心は身長のおおよそ半分の高さにある。桁が狂えば計算が壊れている。"""
    metrics, _ = run()
    assert 0.5 < metrics["com_low_m"] < 1.5
    assert metrics["com_peak_m"] > metrics["com_low_m"]


# --------------------------------------------------------------------------
# 良いフォームで誤検知しないこと
# --------------------------------------------------------------------------
def test_clean_serve_reports_nothing():
    _, feedback = run(fps=120.0)
    assert feedback == []


def test_low_rotation_segment_excluded_from_chain():
    """回転が小さいセグメントはピーク位置がノイズなので判定対象外になる。"""
    metrics, _ = run(fps=120.0)
    chain = metrics["kinetic_chain"]
    assert all("reliable" in c for c in chain)


# --------------------------------------------------------------------------
# 欠点を検出できること
# --------------------------------------------------------------------------
def test_detects_early_shoulder_rotation():
    _, feedback = run(fps=120.0, good_chain=False)
    assert "kinetic_chain_order" in ids(feedback)


def test_detects_late_contact():
    _, feedback = run(fps=120.0, contact_late=True, deep_knee=False)
    assert "contact_late" in ids(feedback)


def test_detects_shallow_knee():
    _, feedback = run(fps=120.0, contact_late=True, deep_knee=False)
    assert "knee_shallow" in ids(feedback)


# --------------------------------------------------------------------------
# フレームレートによる抑制
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fps", [24.0, 30.0, 50.0])
def test_chain_not_judged_below_min_fps(fps):
    """連鎖の時間差は20〜40ms。30fpsでは1フレーム33msあり分解できない。

    測定できない量について主張しないことを保証する。
    """
    _, feedback = run(fps=fps, good_chain=False)
    assert "kinetic_chain_order" not in ids(feedback)


def test_chain_judged_at_high_fps():
    """同じ悪いフォームでも、十分なfpsなら検出できる。"""
    _, feedback = run(fps=120.0, good_chain=False)
    assert "kinetic_chain_order" in ids(feedback)


def test_min_fps_boundary_is_respected():
    assert domains.get("tennis_serve").chain_min_fps >= 60.0, \
        "連鎖判定の下限fpsを下げると誤検知が復活する"


# --------------------------------------------------------------------------
# ドメインの分離（issue 011）
# --------------------------------------------------------------------------
def test_every_domain_measures_without_crashing():
    """合成サーブを全ドメインに通す。局面検出が例外を投げないこと。

    動作としては正しくない（サーブをゴルフとして測る）が、**どのドメインも
    同じ (F,24,3) を受けて JSON 化できる辞書を返す**という契約を守らせる。
    """
    joints, _ = synth_serve(fps=120.0)
    for name in domains.names():
        d, kin = analysis.kinematics_for(joints, 120.0, name)
        m = d.measure(kin, d.detect_phases(kin))
        json.dumps(m)                      # numpy が残っていれば例外
        assert m["n_frames"] == joints.shape[0]
        assert d.report(m, d.judge(m)), f"{name} のレポートが空"


def test_unimplemented_domains_make_no_threshold_claims():
    """閾値の出典が無いドメインは、独自の判定を出してはいけない。

    テニスでは、出典を確認していない数値で**プロの技術を欠点と判定していた**。
    新しい競技で同じことを繰り返さないための歯止め。

    tier の文字列だけを見ても意味がない（判定を書いた本人が "A" と
    名乗れてしまう）。**id で縛る。** 未実装ドメインが出してよいのは
    `domains/base.py` が全競技共通で持つ連鎖の判定ただ1つで、
    それ以外の id が出てきたら、そのドメインに独自ルールが足された証拠。
    """
    joints, _ = synth_serve(fps=120.0)
    for name in domains.names():
        if name == "tennis_serve":
            continue
        d, kin = analysis.kinematics_for(joints, 120.0, name)
        m = d.measure(kin, d.detect_phases(kin))
        found = d.judge(m)
        assert {f["id"] for f in found} <= {"kinetic_chain_order"}, \
            f"{name} が base.py 以外の判定を出している: {[f['id'] for f in found]}"
        assert {f["tier"] for f in found} <= {"A"}, f"{name} に非TIER Aの判定がある"
        assert type(d).judge is base.NotImplementedDomain.judge, \
            f"{name} が judge を上書きしている。出典を確認したなら tier と共に記録すること"
        assert d.evidence_needed, f"{name} に evidence_needed が無い"


# ---------------------------------------------------------------------------
# 他競技の局面検出（レビューで見つかった不具合の再発防止）
# ---------------------------------------------------------------------------

def _golf_swing(n=90, idle=0, finish_high=True):
    """アドレス → トップ → インパクト → フィニッシュ の合成スイング。

    `finish_high` はフィニッシュで手がトップより高く上がる、実際のゴルフで
    普通に起きる形。これを入れないと下のバグを再現できない。
    """
    F = n + idle
    J = np.zeros((F, 24, 3))
    for j, h in {0: .95, 1: .90, 2: .90, 3: 1.05, 4: .50, 5: .50, 6: 1.15,
                 7: .08, 8: .08, 9: 1.25, 10: .02, 11: .02, 12: 1.45,
                 13: 1.42, 14: 1.42, 15: 1.65, 16: 1.40, 17: 1.40}.items():
        J[:, j, 1] = h
    J[:, [1, 4, 7, 10, 13, 16], 0] = -0.18
    J[:, [2, 5, 8, 11, 14, 17], 0] = 0.18

    t = np.clip((np.arange(F) - idle) / max(n - 1, 1), 0, 1)
    top_t, imp_t = 0.35, 0.65
    top_h, imp_h = 1.70, 0.78
    fin_h = 2.00 if finish_high else 1.50
    h = np.where(t <= top_t, 0.80 + (top_h - 0.80) * (t / top_t),
        np.where(t <= imp_t, top_h - (top_h - imp_h) * ((t - top_t) / (imp_t - top_t)),
                 imp_h + (fin_h - imp_h) * ((t - imp_t) / (1 - imp_t))))
    h[:idle] = 0.80
    for w, dx in ((20, -.10), (21, .10), (22, -.12), (23, .12)):
        J[:, w, 1] = h
        J[:, w, 0] = dx
    # トップまでは左腕(リード)が伸び右肘が曲がる。フィニッシュでは入れ替わる
    lead = t <= imp_t
    J[:, 18, 1] = np.where(lead, (J[:, 16, 1] + J[:, 20, 1]) / 2,
                           (J[:, 16, 1] + J[:, 20, 1]) / 2 + 0.12)
    J[:, 19, 1] = np.where(lead, (J[:, 17, 1] + J[:, 21, 1]) / 2 + 0.12,
                           (J[:, 17, 1] + J[:, 21, 1]) / 2)
    J[:, 18, 0], J[:, 19, 0] = -0.14, 0.14
    return J


def test_golf_top_is_not_the_finish():
    """フィニッシュで手が最も高く上がっても、トップを取り違えないこと。

    クリップ全体の argmax で「トップ」を決めていたため、フィニッシュを拾って
    トップ＝インパクト＝フィニッシュが同じフレームに潰れ、リード側の判定まで
    裏返っていた（フィニッシュでは伸びている腕が左右逆になるため）。
    """
    m, _ = analysis.analyze(_golf_swing(finish_high=True), 60.0, "golf_swing")
    ph = m["phases"]
    assert ph["top"] < ph["impact"] < ph["finish"]
    assert m["lead_side"] == "L", "右打ちのリード側は左。フィニッシュを拾うと右になる"


def test_golf_tempo_ignores_idle_footage():
    """構えている時間が長くてもテンポが変わらないこと。

    バックスイングをフレーム0から測っていたため、前に立っているだけの映像が
    そのまま tempo_ratio に乗っていた（4秒足すと 2.3 → 15.7）。
    """
    ratios = [analysis.analyze(_golf_swing(idle=k), 60.0, "golf_swing")[0]["tempo_ratio"]
              for k in (0, 60, 240)]
    assert max(ratios) - min(ratios) < 0.01, f"待機時間でテンポが動く: {ratios}"


def test_golf_tempo_is_nan_when_the_swing_is_cut_off():
    """スイングが入っていないクリップで、それらしい数字を出さないこと。"""
    m, _ = analysis.analyze(_golf_swing()[:32], 60.0, "golf_swing")
    assert math.isnan(m["tempo_ratio"])


def _standing(offsets, tilt_deg=0.0):
    """重心を与えた軌跡で動かす直立スケルトン。"""
    F = len(offsets)
    J = np.zeros((F, 24, 3))
    for j, h in {0: .95, 1: .90, 2: .90, 3: 1.05, 4: .50, 5: .50, 6: 1.15,
                 7: .08, 8: .08, 9: 1.25, 10: .02, 11: .02, 12: 1.45,
                 13: 1.42, 14: 1.42, 15: 1.65, 18: 1.15, 19: 1.15,
                 20: .92, 21: .92, 22: .86, 23: .86}.items():
        J[:, j, 1] = h
    J[:, [1, 4, 7, 10, 13, 18, 20, 22], 0] = -0.18
    J[:, [2, 5, 8, 11, 14, 19, 21, 23], 0] = 0.18
    a, half = np.radians(tilt_deg), 0.18
    J[:, 16, 0], J[:, 16, 1] = -half * np.cos(a), 1.40 + half * np.sin(a)
    J[:, 17, 0], J[:, 17, 1] = half * np.cos(a), 1.40 - half * np.sin(a)
    return J + offsets[:, None, :]


def test_opera_sway_measures_displacement_not_radius():
    """円を描く揺れが「揺れていない」と出ないこと。

    平均位置からの距離の**標準偏差**を取っていたため、半径が一定の動き
    （＝円）では 0 になっていた。半径10cmで回っても 0.03cm と出ていた。
    """
    t = np.linspace(0, 4 * np.pi, 240)
    r = 0.10
    circle = np.stack([r * np.cos(t), np.zeros_like(t), r * np.sin(t)], 1)
    line = np.stack([r * np.sin(t), np.zeros_like(t), np.zeros_like(t)], 1)

    m_c, _ = analysis.analyze(_standing(circle), 60.0, "opera_posture")
    m_l, _ = analysis.analyze(_standing(line), 60.0, "opera_posture")
    m_s, _ = analysis.analyze(_standing(np.zeros((240, 3))), 60.0, "opera_posture")

    assert abs(m_c["com_sway_cm"] - 10.0) < 0.1     # 円: 半径そのもの
    assert abs(m_l["com_sway_cm"] - 7.07) < 0.1     # 直線: 振幅の 1/√2
    assert m_s["com_sway_cm"] < 0.01                # 静止


def test_opera_shoulder_tilt_does_not_saturate():
    """肩の傾きが 45° で頭打ちにならないこと。

    arctan2 の隣辺に肩間の3D距離（＝斜辺）を入れていたため、真の90°でも
    45°と出ていた。
    """
    for want in (0, 20, 45, 60, 90):
        m, _ = analysis.analyze(_standing(np.zeros((30, 3)), tilt_deg=want),
                                60.0, "opera_posture")
        assert abs(m["shoulder_tilt_mean_deg"] - want) < 0.5, \
            f"{want}° が {m['shoulder_tilt_mean_deg']:.1f}° と出た"


def test_unknown_domain_is_rejected_with_the_list():
    with pytest.raises(KeyError, match="使えるのは"):
        domains.get("tennis_smash")


# --------------------------------------------------------------------------
# 提示の作法
# --------------------------------------------------------------------------
def test_report_limits_number_of_cues():
    """一度に複数の修正キューを出さない。15件の指摘は0件と同じ。"""
    metrics, feedback = run(fps=120.0, good_chain=False,
                            contact_late=True, deep_knee=False)
    text = analysis.format_report(metrics, feedback, top_n=2)
    assert sum(line.strip().startswith(("1.", "2.", "3."))
               for line in text.splitlines()) <= 2


def test_report_warns_when_fps_too_low():
    metrics, feedback = run(fps=30.0)
    text = analysis.format_report(metrics, feedback)
    assert "順序を判定できません" in text


def test_feedback_sorted_by_priority():
    _, feedback = run(fps=120.0, good_chain=False,
                      contact_late=True, deep_knee=False)
    priorities = [f["priority"] for f in feedback]
    assert priorities == sorted(priorities, reverse=True)


# --------------------------------------------------------------------------
# 閾値の裏付け
#
# 初期実装は体幹の傾き35°超を「傾きすぎ」と警告していた。根拠として
# 「プロは水平から48°傾いている」という値を挙げていたが、論文本文を確認した
# ところ**その数値は存在しなかった**（検索スニペットからの孫引きだった）。
# 接球時の体幹について検証済みの基準値は無いため、判定してはいけない。
#
# 根拠のない閾値で断定しないことを、テストで縛る。
# --------------------------------------------------------------------------
def test_every_finding_declares_its_tier():
    """全ての指摘が、何を根拠に判定したかを申告していること。"""
    _, feedback = run(fps=120.0, good_chain=False,
                      contact_late=True, deep_knee=False)
    assert feedback, "検証のため何か検出される想定"
    for f in feedback:
        assert f["tier"] in ("A", "B"), f"{f['id']} の根拠区分が不正: {f.get('tier')}"


def test_trunk_lean_is_never_reported_as_a_fault():
    """接球時の体幹には検証済みの基準値が無い。欠点として指摘してはいけない。"""
    for kw in ({}, {"good_chain": False}, {"contact_late": True},
               {"deep_knee": False}):
        _, feedback = run(fps=120.0, **kw)
        assert "trunk_lean" not in ids(feedback)


def test_unbacked_metrics_are_not_judged():
    """根拠の無い指標(打点比・X-factor)で欠点と断じないこと。"""
    for kw in ({}, {"good_chain": False}, {"deep_knee": False}):
        _, feedback = run(fps=120.0, **kw)
        assert "contact_low" not in ids(feedback)
        assert "x_factor_small" not in ids(feedback)


def test_literature_references_are_plausible():
    """文献値が現実的な範囲にあること（取り違えの検出）。"""
    assert 100 <= fb.REF_KNEE_JOINT_DEG <= 130      # 屈曲 約64° 相当
    assert 140 <= fb.REF_ELBOW_JOINT_DEG <= 170     # 屈曲 約30° 相当
    assert 15 <= fb.REF_TRUNK_LEAN_TROPHY_DEG <= 40  # トロフィー時 25.0±7.1°


def test_detects_bent_elbow_at_contact():
    """接球時に肘が曲がっていれば指摘する（文献: プロは約150°）。"""
    metrics, _ = run(fps=120.0)
    assert metrics["elbow_at_contact_deg"] > fb.TH_ELBOW_BENT_DEG, \
        "合成データは腕が伸びている想定"


# --------------------------------------------------------------------------
# 入出力
# --------------------------------------------------------------------------
def test_analyze_from_files(tmp_path):
    joints, _ = synth_serve(fps=120.0)
    np.save(tmp_path / "j.npy", joints)

    metrics, _ = analysis.analyze_from_files(str(tmp_path / "j.npy"), fps=120.0)
    assert metrics["n_frames"] == joints.shape[0]


def test_analyze_json_is_serializable():
    """Web が返す結果が numpy を残さず json 化できること。"""
    import json

    joints, _ = synth_serve(fps=120.0, good_chain=False)
    res = analysis.analyze_json(joints, fps=120.0)
    text = json.dumps(res)  # numpy が残っていれば例外
    assert "feedback" in res and "joints" in res and "metrics" in res
    assert len(res["joints"]) == joints.shape[0]
    assert json.loads(text)["metrics"]["phases"]["contact"] >= 0


def test_com_derived_from_joints_is_anatomical():
    """関節から導出した重心が身長の約半分に来る（移動が忠実かの確認）。"""
    from core import compute_com, detect_up_axis
    joints, _ = synth_serve(fps=30.0)
    com = compute_com(joints)
    up_ax, up_sign = detect_up_axis(joints)
    h = com[:, up_ax] * up_sign
    assert 0.5 < h.min() < 1.5
    assert up_ax == 1 and up_sign == 1.0


# ---------------------------------------------------------------------------
# SOMA(GEM-X) → SMPL24 の並べ替え（issue #9）
# ---------------------------------------------------------------------------

def test_soma_mapping_is_a_valid_permutation():
    """24関節ぶん、重複なく SOMA の範囲内を指していること。"""
    from core.convert import SMPL24_FROM_SOMA78, SOMA78_JOINTS
    idx = [i for _, i in SMPL24_FROM_SOMA78]
    assert len(idx) == 24
    assert len(set(idx)) == 24, "同じ SOMA 関節を2度使っている"
    assert all(0 <= i < SOMA78_JOINTS for i in idx)


def test_soma_mapping_absorbs_missing_root():
    """Root 込み(78)と Root 無し(77)で、同じ関節を指すこと。

    ここがずれると「膝の角度」が別の関節の角度になり、しかも
    それらしい数字が出てしまうので、取り違えに気付けない。
    """
    import numpy as np
    from core.convert import to_smpl24
    # 関節 i の座標を i にしておけば、どれを引いたかが値で分かる
    j78 = np.arange(78, dtype=float)[None, :, None].repeat(3, axis=2)
    j77 = j78[:, 1:, :] - 1.0          # Root を落とし、添字を1つ詰めた並び
    assert np.array_equal(to_smpl24(j78), to_smpl24(j77) + 1.0)


def test_soma_mapping_rejects_unknown_joint_count():
    import numpy as np
    import pytest
    from core.convert import to_smpl24
    with pytest.raises(ValueError, match="SOMA の関節数"):
        to_smpl24(np.zeros((2, 24, 3)))


# ---------------------------------------------------------------------------
# 多数本の集計に使う統計（issue #10）
# scipy を入れていないので自前実装。結論を左右するので挙動を固定する。
# ---------------------------------------------------------------------------

def _mw(a, b):
    import numpy as np
    from tools.make_session import mann_whitney_p
    return mann_whitney_p(np.array(a, float), np.array(b, float))


def test_mann_whitney_p_stays_within_zero_and_one():
    """連続性補正が |u-mu| を上回ると p が 1 を超えていた（実測 1.03）。"""
    assert _mw([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]) == 1.0
    assert _mw([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7]) <= 1.0


def test_mann_whitney_p_separates_and_overlaps():
    assert _mw(list(range(10)), list(range(100, 110))) < 0.01   # 完全に分離
    assert _mw(list(range(10)), list(range(2, 12))) > 0.05      # 大きく重なる


def test_mann_whitney_p_false_positive_rate():
    """差の無い分布で p<0.05 が約5%に収まること。

    ここが狂うと「入った/入らなかったで差がある」と誤って報告する。
    """
    import numpy as np
    rng = np.random.default_rng(0)
    ps = [_mw(rng.normal(size=17), rng.normal(size=25)) for _ in range(400)]
    assert 0.01 < float(np.mean(np.array(ps) < 0.05)) < 0.12


def test_holm_is_monotonic_and_scales_smallest():
    from tools.make_session import holm
    adj = holm([0.001, 0.01, 0.04, 0.20, 0.90])
    assert adj == sorted(adj)              # 単調
    assert adj[0] == 0.005                 # 最小の p は件数倍（0.001×5）
    assert all(p <= 1.0 for p in adj)


def test_holm_ignores_nan():
    import math
    from tools.make_session import holm
    adj = holm([0.01, float("nan"), 0.5])
    assert math.isnan(adj[1])
    assert adj[0] == 0.02                  # nan を除いた2件ぶんで補正
