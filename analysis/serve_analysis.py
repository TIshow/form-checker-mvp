"""サーブの3D関節データからバイオメカ指標を算出し、改善フィードバックを生成する。

入力は GVHMR パイプラインが出力する numpy 配列のみ:
  gv_joints.npy  (F, 24, 3)  SMPL 24関節の world座標 [m]
  gv_com.npy     (F, 3)      全身重心の world座標 [m]
  gv_upaxis.npy  (2,)        [上方向の軸index, 符号]

純 numpy のため GVHMR 環境 (Python 3.10 venv) は不要。通常の Python で動く。

設計方針:
  プロとの比較は行わない。誰にでも当てはまる力学的原理
  （キネティックチェーンの順序、打点タイミング）で判定する。
  そのため参照動画も権利処理も不要で、かつ最も価値の高い指摘が得られる。
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# SMPL 24関節のインデックス
# --------------------------------------------------------------------------
PELVIS, L_HIP, R_HIP, SPINE1 = 0, 1, 2, 3
L_KNEE, R_KNEE, SPINE2 = 4, 5, 6
L_ANKLE, R_ANKLE, SPINE3 = 7, 8, 9
L_FOOT, R_FOOT, NECK = 10, 11, 12
L_COLLAR, R_COLLAR, HEAD = 13, 14, 15
L_SHOULDER, R_SHOULDER = 16, 17
L_ELBOW, R_ELBOW = 18, 19
L_WRIST, R_WRIST = 20, 21
L_HAND, R_HAND = 22, 23


# --------------------------------------------------------------------------
# 幾何ユーティリティ
# --------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """b を頂点とする a-b-c の3D角度 [deg]。各引数は (F,3)。"""
    cos = np.sum(_unit(a - b) * _unit(c - b), axis=-1).clip(-1.0, 1.0)
    return np.degrees(np.arccos(cos))


def angular_speed(p_start: np.ndarray, p_end: np.ndarray, fps: float) -> np.ndarray:
    """線分 p_start->p_end の向きが変化する角速度 [deg/s]。(F,) を返す。"""
    d = _unit(p_end - p_start)
    cos = np.sum(d[1:] * d[:-1], axis=-1).clip(-1.0, 1.0)
    speed = np.degrees(np.arccos(cos)) * fps
    return np.concatenate([[0.0], speed])


def smooth(x: np.ndarray, win: int = 5) -> np.ndarray:
    """移動平均。角速度はノイズが乗るため平滑化してからピークを取る。"""
    if win <= 1 or len(x) < win:
        return x
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


def horizontal(v: np.ndarray, up_ax: int) -> np.ndarray:
    """上方向成分を除去して水平面に射影する。"""
    out = v.copy()
    out[..., up_ax] = 0.0
    return out


def signed_horizontal_angle(v1: np.ndarray, v2: np.ndarray, up_ax: int) -> np.ndarray:
    """水平面に射影した2ベクトルのなす角 [deg]。(F,) を返す。"""
    a, b = _unit(horizontal(v1, up_ax)), _unit(horizontal(v2, up_ax))
    cos = np.sum(a * b, axis=-1).clip(-1.0, 1.0)
    return np.degrees(np.arccos(cos))


# --------------------------------------------------------------------------
# 基本量
# --------------------------------------------------------------------------
class ServeKinematics:
    """関節データからサーブの運動学的な量を取り出す。"""

    def __init__(self, joints: np.ndarray, com: np.ndarray,
                 up_ax: int, up_sign: float, fps: float = 30.0):
        self.J = joints
        self.com = com
        self.up_ax = int(up_ax)
        self.up_sign = float(up_sign)
        self.fps = float(fps)
        self.F = joints.shape[0]

        self.com_height = self._height(com)
        self.racket_side = self._detect_racket_side()

    # -- 高さ --------------------------------------------------------------
    def _height(self, p: np.ndarray) -> np.ndarray:
        """world座標から「地面からの高さ」成分を取り出す(上向きが正)。"""
        return p[..., self.up_ax] * self.up_sign

    # -- 利き手 ------------------------------------------------------------
    def _detect_racket_side(self) -> str:
        """ラケットを持つ側を判定する。サーブでは打点で最も高く上がる腕。"""
        lh = self._height(self.J[:, L_WRIST]).max()
        rh = self._height(self.J[:, R_WRIST]).max()
        return "R" if rh >= lh else "L"

    def idx(self, name: str) -> int:
        """利き手側の関節インデックスを返す。"""
        table = {
            "shoulder": (R_SHOULDER, L_SHOULDER),
            "elbow": (R_ELBOW, L_ELBOW),
            "wrist": (R_WRIST, L_WRIST),
            "hand": (R_HAND, L_HAND),
            "hip": (R_HIP, L_HIP),
            "knee": (R_KNEE, L_KNEE),
            "ankle": (R_ANKLE, L_ANKLE),
        }
        r, l = table[name]
        return r if self.racket_side == "R" else l

    # -- フェーズ検出 ------------------------------------------------------
    def detect_phases(self) -> dict:
        """打点・沈み込みのフレームを検出する。

        打点  = ラケット側手首が最も高くなるフレーム
        沈み込み = その手前で重心が最も低くなるフレーム
        """
        wrist_h = self._height(self.J[:, self.idx("wrist")])
        contact = int(np.argmax(wrist_h))

        search_end = max(contact, 1)
        loading = int(np.argmin(self.com_height[:search_end]))

        com_peak = int(np.argmax(self.com_height))
        return {"loading": loading, "contact": contact, "com_peak": com_peak}

    # -- 角度 --------------------------------------------------------------
    def knee_angles(self) -> np.ndarray:
        """左右の膝角度の小さい方 [deg]。180=伸展、小さいほど深く曲げている。"""
        left = joint_angle(self.J[:, L_HIP], self.J[:, L_KNEE], self.J[:, L_ANKLE])
        right = joint_angle(self.J[:, R_HIP], self.J[:, R_KNEE], self.J[:, R_ANKLE])
        return np.minimum(left, right)

    def elbow_angle(self) -> np.ndarray:
        """ラケット側の肘角度 [deg]。"""
        return joint_angle(self.J[:, self.idx("shoulder")],
                           self.J[:, self.idx("elbow")],
                           self.J[:, self.idx("wrist")])

    def trunk_lean(self) -> np.ndarray:
        """体幹(骨盤→首)が鉛直から傾いている角度 [deg]。"""
        up = np.zeros(3)
        up[self.up_ax] = self.up_sign
        trunk = _unit(self.J[:, NECK] - self.J[:, PELVIS])
        cos = (trunk * up).sum(-1).clip(-1.0, 1.0)
        return np.degrees(np.arccos(cos))

    def x_factor(self) -> np.ndarray:
        """肩の軸と腰の軸の捻転差 [deg]（水平面上）。"""
        return signed_horizontal_angle(
            self.J[:, R_SHOULDER] - self.J[:, L_SHOULDER],
            self.J[:, R_HIP] - self.J[:, L_HIP],
            self.up_ax,
        )

    def body_height_proxy(self) -> float:
        """立位の頭の高さ。打点の高さを体格で正規化するために使う。"""
        head_h = self._height(self.J[:, HEAD])
        return float(np.median(head_h[: max(1, self.F // 3)]))

    # -- キネティックチェーン ----------------------------------------------
    def kinetic_chain(self, loading: int, contact: int) -> list[dict]:
        """力の伝達順序を測る。

        脚→腰→体幹→肩→肘 の順に角速度のピークが並ぶのが理想。
        各セグメントについて、加速区間内でのピーク時刻を返す。
        """
        sh, el, wr = self.idx("shoulder"), self.idx("elbow"), self.idx("wrist")
        segments = [
            ("腰の回転", self.J[:, L_HIP], self.J[:, R_HIP]),
            ("肩の回転", self.J[:, L_SHOULDER], self.J[:, R_SHOULDER]),
            ("上腕", self.J[:, sh], self.J[:, el]),
            ("前腕", self.J[:, el], self.J[:, wr]),
        ]

        # 加速区間: 沈み込み〜打点直後。短すぎる場合はクリップの全体を使う。
        lo, hi = loading, min(contact + 2, self.F)
        if hi - lo < 3:
            lo, hi = 0, self.F

        out = []
        for name, a, b in segments:
            speed = smooth(angular_speed(a, b, self.fps))
            window = speed[lo:hi]
            peak_local = int(np.argmax(window))
            out.append({
                "segment": name,
                "peak_frame": lo + peak_local,
                "peak_speed": float(window[peak_local]),
            })

        # ほとんど回転していないセグメントはピーク位置がノイズで決まってしまう。
        # 順序判定に使えるのは、十分な角速度が出ているものだけ。
        peak_max = max(c["peak_speed"] for c in out) if out else 0.0
        floor = max(CHAIN_MIN_SPEED_DEG_S, peak_max * CHAIN_REL_SPEED)
        for c in out:
            c["reliable"] = c["peak_speed"] >= floor
        return out


# --------------------------------------------------------------------------
# 指標の集計
# --------------------------------------------------------------------------
def compute_metrics(kin: ServeKinematics) -> dict:
    ph = kin.detect_phases()
    loading, contact, com_peak = ph["loading"], ph["contact"], ph["com_peak"]
    fps = kin.fps

    knee = kin.knee_angles()
    wrist_h = kin._height(kin.J[:, kin.idx("wrist")])
    body_h = kin.body_height_proxy()

    chain = kin.kinetic_chain(loading, contact)

    return {
        "racket_side": kin.racket_side,
        "fps": fps,
        "n_frames": kin.F,
        "phases": ph,
        # 脚のドライブ
        "com_low_m": float(kin.com_height[loading]),
        "com_peak_m": float(kin.com_height[com_peak]),
        "com_rise_m": float(kin.com_height[com_peak] - kin.com_height[loading]),
        "drive_time_s": float((com_peak - loading) / fps),
        # 沈み込み
        "min_knee_deg": float(knee[loading]),
        "min_knee_deg_overall": float(knee[: max(contact, 1)].min()),
        # 打点
        "contact_height_m": float(wrist_h[contact]),
        "contact_height_ratio": float(wrist_h[contact] / body_h) if body_h > 0 else float("nan"),
        "contact_vs_compeak_s": float((contact - com_peak) / fps),
        "elbow_at_contact_deg": float(kin.elbow_angle()[contact]),
        "trunk_lean_at_contact_deg": float(kin.trunk_lean()[contact]),
        # 捻転
        "max_x_factor_deg": float(kin.x_factor()[loading : max(contact, loading + 1)].max()),
        # 連鎖
        "kinetic_chain": chain,
    }


# --------------------------------------------------------------------------
# フィードバック生成（ルールベース）
# --------------------------------------------------------------------------
# しきい値は暫定。順序・タイミング系は理論から断定できるが、
# 角度・量の系はデータを蓄積してから調整する前提。
TH_CONTACT_LATE_S = 0.10      # 重心ピークからこれ以上遅れたら「落ちながら打っている」
TH_KNEE_SHALLOW_DEG = 150.0   # これより曲がっていなければ沈み込みが浅い
TH_CONTACT_RATIO_LOW = 1.15   # 打点が頭の高さのこの倍未満なら伸びが不足
TH_TRUNK_LEAN_DEG = 35.0      # 体幹が傾きすぎ
TH_X_FACTOR_SMALL_DEG = 20.0  # 捻転差が小さい

# キネティックチェーンの順序判定に使うセグメントの下限。
# これ未満しか回転していないものは、ピーク位置がノイズなので判定に使わない。
CHAIN_MIN_SPEED_DEG_S = 30.0
CHAIN_REL_SPEED = 0.15        # 最大ピークに対する相対比


def generate_feedback(m: dict) -> list[dict]:
    """指標から改善点を抽出し、優先度順に返す。

    優先度: 力学的原理の違反 > 大きな逸脱 > 小さな逸脱
    """
    found: list[dict] = []

    # --- 原理1: キネティックチェーンの順序 --------------------------------
    # 十分に回転しているセグメントだけを、体の中心から末端の順に並べて比較する。
    chain = [c for c in m["kinetic_chain"] if c.get("reliable", True)]
    for i in range(len(chain) - 1):
        cur, nxt = chain[i], chain[i + 1]
        if cur["peak_frame"] > nxt["peak_frame"]:
            found.append({
                "priority": 100,
                "id": "kinetic_chain_order",
                "title": "力の伝わる順序が逆転しています",
                "detail": (
                    f"{cur['segment']}(frame {cur['peak_frame']}) より "
                    f"{nxt['segment']}(frame {nxt['peak_frame']}) が先にピークに達しています。"
                ),
                "cue": (
                    f"{cur['segment']}から先に動かす意識を。"
                    "体の中心から順に加速すると、力が末端まで乗ります。"
                ),
            })
            break  # 最初の逆転だけ指摘する（一度に複数出さない）

    # --- 原理2: 打点のタイミング ------------------------------------------
    late = m["contact_vs_compeak_s"]
    if late > TH_CONTACT_LATE_S:
        found.append({
            "priority": 90,
            "id": "contact_late",
            "title": "重心が落ち始めてから打っています",
            "detail": f"打点が重心の最高点より {late:.2f} 秒遅れています。",
            "cue": "伸び上がりの頂点で捉える意識を。体が落ち始めると力が逃げます。",
        })

    # --- 打点の高さ --------------------------------------------------------
    ratio = m["contact_height_ratio"]
    if np.isfinite(ratio) and ratio < TH_CONTACT_RATIO_LOW:
        found.append({
            "priority": 60,
            "id": "contact_low",
            "title": "打点が低めです",
            "detail": f"打点の高さが頭の位置の {ratio:.2f} 倍です。",
            "cue": "ボールに向かって伸び上がり、腕を最大限に伸ばして捉えましょう。",
        })

    # --- 沈み込みの深さ ----------------------------------------------------
    knee = m["min_knee_deg_overall"]
    if knee > TH_KNEE_SHALLOW_DEG:
        found.append({
            "priority": 50,
            "id": "knee_shallow",
            "title": "沈み込みが浅いです",
            "detail": f"最も曲げた時の膝角度が {knee:.0f}°（180°=伸びきり）。",
            "cue": "膝をもう少し深く曲げてタメを作ると、脚の力を使えます。",
        })

    # --- 捻転差 ------------------------------------------------------------
    xf = m["max_x_factor_deg"]
    if xf < TH_X_FACTOR_SMALL_DEG:
        found.append({
            "priority": 45,
            "id": "x_factor_small",
            "title": "上半身と下半身の捻転差が小さいです",
            "detail": f"肩と腰の最大の捻転差が {xf:.0f}°。",
            "cue": "腰を先に回し、肩を残す意識を。捻れが大きいほど力が生まれます。",
        })

    # --- 体幹の傾き --------------------------------------------------------
    lean = m["trunk_lean_at_contact_deg"]
    if lean > TH_TRUNK_LEAN_DEG:
        found.append({
            "priority": 40,
            "id": "trunk_lean",
            "title": "打点で体幹が傾きすぎています",
            "detail": f"打点時の体幹の傾きが {lean:.0f}°。",
            "cue": "軸を保って伸び上がりましょう。傾きすぎると安定性が落ちます。",
        })

    found.sort(key=lambda f: -f["priority"])
    return found


# --------------------------------------------------------------------------
# レポート出力
# --------------------------------------------------------------------------
def format_report(m: dict, feedback: list[dict], top_n: int = 2) -> str:
    """人が読める形に整形する。指摘は上位 top_n 件のみ（一度に多く出さない）。"""
    ph = m["phases"]
    fps = m["fps"]
    side = "右" if m["racket_side"] == "R" else "左"

    lines = [
        "=" * 62,
        "  サーブ解析レポート",
        "=" * 62,
        f"  利き手(推定): {side}利き   フレーム数: {m['n_frames']}  ({fps:.0f}fps)",
        "",
        "── 動作フェーズ ──",
        f"  沈み込み : frame {ph['loading']:3d}  ({ph['loading']/fps:.2f}s)",
        f"  重心の頂点: frame {ph['com_peak']:3d}  ({ph['com_peak']/fps:.2f}s)",
        f"  打点     : frame {ph['contact']:3d}  ({ph['contact']/fps:.2f}s)",
        "",
        "── 脚のドライブ ──",
        f"  重心の最低点 : {m['com_low_m']:.3f} m",
        f"  重心の最高点 : {m['com_peak_m']:.3f} m",
        f"  伸び上がり   : {m['com_rise_m']*100:+.1f} cm / {m['drive_time_s']:.2f} 秒",
        f"  沈み込み膝角 : {m['min_knee_deg_overall']:.0f}° (180=伸びきり)",
        "",
        "── 打点 ──",
        f"  高さ         : {m['contact_height_m']:.3f} m (頭の {m['contact_height_ratio']:.2f} 倍)",
        f"  重心頂点との差: {m['contact_vs_compeak_s']:+.2f} 秒 (0に近いほど良い)",
        f"  肘の角度     : {m['elbow_at_contact_deg']:.0f}°",
        f"  体幹の傾き   : {m['trunk_lean_at_contact_deg']:.0f}°",
        f"  最大捻転差   : {m['max_x_factor_deg']:.0f}°",
        "",
        "── キネティックチェーン（理想は上から順にピーク）──",
    ]
    for c in m["kinetic_chain"]:
        note = "" if c.get("reliable", True) else "  ※回転が小さく判定対象外"
        lines.append(
            f"  {c['segment']:<8} peak frame {c['peak_frame']:3d}  "
            f"({c['peak_speed']:6.0f} deg/s){note}"
        )

    lines += ["", "=" * 62, "  改善ポイント", "=" * 62]
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
    lines.append("=" * 62)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# エントリポイント
# --------------------------------------------------------------------------
def analyze(joints: np.ndarray, com: np.ndarray, up_ax: int, up_sign: float,
            fps: float = 30.0) -> tuple[dict, list[dict]]:
    kin = ServeKinematics(joints, com, up_ax, up_sign, fps)
    metrics = compute_metrics(kin)
    return metrics, generate_feedback(metrics)


def analyze_from_files(joints_path: str = "/content/gv_joints.npy",
                       com_path: str = "/content/gv_com.npy",
                       upaxis_path: str = "/content/gv_upaxis.npy",
                       fps: float = 30.0) -> tuple[dict, list[dict]]:
    joints = np.load(joints_path)
    com = np.load(com_path)
    up_ax, up_sign = np.load(upaxis_path)
    return analyze(joints, com, int(up_ax), float(up_sign), fps)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="サーブの3D解析とフィードバック生成")
    p.add_argument("--joints", default="/content/gv_joints.npy")
    p.add_argument("--com", default="/content/gv_com.npy")
    p.add_argument("--upaxis", default="/content/gv_upaxis.npy")
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args()

    metrics, feedback = analyze_from_files(args.joints, args.com, args.upaxis, args.fps)
    print(format_report(metrics, feedback))
