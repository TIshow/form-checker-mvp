"""合成サーブデータによる解析・フィードバックの検証。

このテストは実際に2つのバグを検出した実績がある:
  1. ほとんど回転していないセグメントのノイズを順序判定に使っていた（誤検知）
  2. 30fpsで1フレーム差を「順序の逆転」として指摘していた（測定不能な量の主張）
閾値やルールを触ったら必ず走らせること。
"""

from __future__ import annotations

import numpy as np
import pytest

import analysis
from analysis import feedback as fb
from tests.synth import UP_AXIS, UP_SIGN, synth_serve


def run(fps: float = 30.0, **kw):
    joints, com = synth_serve(fps=fps, **kw)
    return analysis.analyze(joints, com, UP_AXIS, UP_SIGN, fps=fps)


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
    assert fb.CHAIN_MIN_FPS >= 60.0, "連鎖判定の下限fpsを下げると誤検知が復活する"


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
    joints, com = synth_serve(fps=120.0)
    np.save(tmp_path / "j.npy", joints)
    np.save(tmp_path / "c.npy", com)
    np.save(tmp_path / "u.npy", np.array([UP_AXIS, UP_SIGN]))

    metrics, _ = analysis.analyze_from_files(
        str(tmp_path / "j.npy"), str(tmp_path / "c.npy"),
        str(tmp_path / "u.npy"), fps=120.0)
    assert metrics["n_frames"] == joints.shape[0]
