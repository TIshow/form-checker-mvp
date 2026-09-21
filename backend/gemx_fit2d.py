"""Step 2: GEM-X の出力を、GEM-X 自身の 2D 検出に合わせ直す（issue 018）。

    .venv/bin/modal run backend/gemx_fit2d.py --src output_golf_gemx --out output_golf_gemx_fit --fps 30

## 何をするか

Step 1 の出力（`gx_pose.npz` = SOMA の姿勢パラメータ、`gx_kp2d.npy` = ViTPose の
77 点）を受け取り、**SOMA の関節回転**を変数にして、関節の投影が 2D 検出に
近づくよう最適化する。順運動学は上流の `SomaLayer` そのもの。上流のコードには
手を入れず、出力パラメータを書き換えて、上流の描画関数でメッシュを描く。

## 015 との違い（不採用になった理由を潰す）

015 は関節の**位置**を動かした。骨長を固定しても奥行きへ逃げ放題で、膝が
149→112° に折れた。ここでは:

- 変数は回転（axis-angle の補正量 delta）。骨長・体型（identity/scale）は GEM-X の
  値のまま。順運動学を通るので、関節位置の自由度は最初から無い
- root の奥行きはほぼ固定（σ 5mm）。画像面の並進だけ σ 5cm で許す
- 2D は GEM-X 自身の閾値（スコア 0.5）以上だけ。ロバスト損失（soft-L1）で
  外れ値を切る
- 事前分布は「初期値からどれだけ回したか」（σ 0.15 rad）と「補正量の加速度」
  （σ 0.02 rad）。生の速度は均さないので、インパクトの速度ピークは残る
- 5 フレームに 1 枚は合わせに使わず、評価にだけ使う（過学習の検出）

## 何が直って何が直らないか（事前の見立て）

直る候補: 2D が見えている区間の画像面の誤り（トップで両手首が離れる、など）。
直らない: 2D が無い区間（フィニッシュの腕）、奥行きの誤り（後ろ足の +40cm）。
K は上流の既定値（fx = fy = max(W,H)）で未較正。
"""

from __future__ import annotations

import glob
import io
import json
import subprocess
from pathlib import Path

import modal

from backend.reconstruct_gemx import ASSETS, GEMX, GEMX_COMMIT, _link_assets, image, vol

image = image.add_local_python_source("backend")
app = modal.App("gemx-fit2d")

#: 損失の尺度。単位はそれぞれ px / rad / rad / m。
SIGMA = {"px": 8.0, "rot": 0.15, "temporal": 0.02, "root_xy": 0.05, "root_z": 0.005}
#: 2D をこのスコア未満なら「見えていない」扱い。GEM-X 本体と同じ閾値。
CONF_MIN = 0.5


@app.function(image=image, gpu="L4", volumes={ASSETS: vol}, timeout=1800)
def fit(pose_npz: bytes, kp2d_npy: bytes, video_bytes: bytes, fps: float,
        holdout_every: int = 5, iters: int = 400, lr: float = 0.005) -> dict:
    import hashlib
    import importlib.util
    import os
    import sys
    import time
    import types

    import numpy as np
    import torch

    os.chdir(GEMX)
    sys.path.insert(0, GEMX)
    _link_assets()
    spec = importlib.util.spec_from_file_location("demo_soma", f"{GEMX}/scripts/demo/demo_soma.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    from gem.utils.cam_utils import estimate_K
    from gem.utils.soma_utils.soma_layer import SomaLayer
    from gem.utils.video_io_utils import get_video_lwh

    z = np.load(io.BytesIO(pose_npz))
    kp = np.asarray(np.load(io.BytesIO(kp2d_npy)), dtype=np.float32)          # (F,77,3)
    F = len(kp)
    stem = f"fit_{hashlib.sha256(video_bytes).hexdigest()[:12]}"
    Path("inputs").mkdir(exist_ok=True)
    src = f"inputs/{stem}.mp4"
    Path(src).write_bytes(video_bytes)
    L, W, H = get_video_lwh(src)
    if L != F:
        raise ValueError(f"動画 {L} フレーム、2D {F} フレーム。Step 1 と同じ動画を渡すこと")
    K = estimate_K(W, H).float().cuda()

    dev = "cuda"
    soma = SomaLayer(data_root="inputs/soma_assets", low_lod=True, device=dev,
                     identity_model_type="mhr", mode="warp")
    init = {k: torch.as_tensor(z[f"incam_{k}"]).to(dev) for k in
            ("body_pose", "global_orient", "transl", "identity_coeffs", "scale_params")}
    obs = torch.as_tensor(kp).to(dev)
    score = obs[..., 2].clamp(0, 1)
    vis = score >= CONF_MIN
    fit_frames = torch.ones(F, dtype=torch.bool, device=dev)
    if holdout_every > 0:
        fit_frames[::holdout_every] = False
    w = torch.sqrt(score) * vis * fit_frames[:, None]

    delta = torch.zeros(F, 76, 3, device=dev, requires_grad=True)
    dt = torch.zeros(F, 3, device=dev, requires_grad=True)

    def joints(delta, dt):
        bp = (init["body_pose"].view(F, 76, 3) + delta).reshape(F, 228)
        return soma(global_orient=init["global_orient"], body_pose=bp,
                    identity_coeffs=init["identity_coeffs"], scale_params=init["scale_params"],
                    transl=init["transl"] + dt)["joints"]                       # (F,77,3) カメラ座標

    def project(J):
        p = J @ K.T
        return p[..., :2] / p[..., 2:3].clamp(min=1e-6)

    with torch.no_grad():
        J0 = joints(delta, dt)
        e0 = torch.linalg.norm(project(J0) - obs[..., :2], dim=-1)                # (F,77) px

    def errors(J):
        e = torch.linalg.norm(project(J) - obs[..., :2], dim=-1)
        m = lambda mask: float(e[mask].median()) if mask.any() else None
        return {"fit_px": m(vis & fit_frames[:, None]), "heldout_px": m(vis & ~fit_frames[:, None])}

    def loss_terms(delta, dt):
        J = joints(delta, dt)
        r = (project(J) - obs[..., :2]) / SIGMA["px"]
        rep = ((torch.sqrt(1 + r.pow(2).sum(-1)) - 1) * w).sum() / w.sum()
        prior = (delta / SIGMA["rot"]).pow(2).mean()
        acc = (delta[2:] - 2 * delta[1:-1] + delta[:-2]) * (fps / 30.0) ** 2
        temporal = (acc / SIGMA["temporal"]).pow(2).mean()
        root = (dt[:, :2] / SIGMA["root_xy"]).pow(2).mean() + (dt[:, 2] / SIGMA["root_z"]).pow(2).mean()
        return rep, prior, temporal, root

    if not joints(delta, dt).requires_grad:
        raise RuntimeError("SomaLayer の出力に勾配が流れない")
    opt = torch.optim.Adam([delta, dt], lr=lr)
    t0 = time.time()
    history = []
    for it in range(iters):
        opt.zero_grad()
        rep, prior, temporal, root = loss_terms(delta, dt)
        total = rep + prior + temporal + root
        total.backward()
        opt.step()
        if it % 50 == 0 or it == iters - 1:
            history.append({"iter": it, "total": float(total), "reproj": float(rep), "prior": float(prior),
                            "temporal": float(temporal), "root": float(root)})
            print(f"[fit] it {it:4d} total {float(total):.4f} reproj {float(rep):.4f} prior {float(prior):.4f} "
                  f"temporal {float(temporal):.4f} root {float(root):.4f}")
    t_fit = time.time() - t0

    with torch.no_grad():
        J1 = joints(delta, dt)
        before, after = errors(J0), errors(J1)
        disp = torch.linalg.norm(J1 - J0, dim=-1)                                # (F,77) m
        bp_fit = (init["body_pose"].view(F, 76, 3) + delta).reshape(F, 228)
        incam = {"body_pose": bp_fit, "global_orient": init["global_orient"], "transl": init["transl"] + dt,
                 "identity_coeffs": init["identity_coeffs"], "scale_params": init["scale_params"]}
        # 世界座標: 局所回転は同じ。root の向きと並進は Step 1 の世界座標のものをそのまま使う
        glob = {"body_pose": bp_fit,
                "global_orient": torch.as_tensor(z["global_global_orient"]).to(dev),
                "transl": torch.as_tensor(z["global_transl"]).to(dev),
                "identity_coeffs": init["identity_coeffs"], "scale_params": init["scale_params"]}
        Jg = soma(**glob)["joints"]
        rot_deg = (delta.norm(dim=-1) * 180 / np.pi)                             # (F,76)

    report = {
        "frames": F, "fit_frames": int(fit_frames.sum()), "heldout_frames": int((~fit_frames).sum()),
        "visible_fraction": float(vis.float().mean()), "conf_min": CONF_MIN, "sigma": SIGMA,
        "iters": iters, "lr": lr, "fit_s": round(t_fit, 1),
        "reproj_median_px": {"before": before, "after": after},
        "per_joint_median_px_before": [round(float(v), 1) for v in
                                       torch.where(vis, e0, torch.nan).nanmedian(dim=0).values],
        "max_joint_displacement_cm": round(float(disp.max()) * 100, 1),
        "mean_joint_displacement_cm": round(float(disp.mean()) * 100, 2),
        "rotation_correction_deg": {"mean": round(float(rot_deg.mean()), 2), "max": round(float(rot_deg.max()), 1)},
        "root_shift_cm": {"xy_max": round(float(dt[:, :2].norm(dim=-1).max()) * 100, 2),
                          "z_max": round(float(dt[:, 2].abs().max()) * 100, 2)},
        "history": history,
        "meaning": "再投影の一致。3D の正しさそのものではない（採用判断は 3D の指標と目視で）",
    }
    print("[fit] 再投影 中央値 px:", report["reproj_median_px"])

    # 上流の描画関数でメッシュを描く（描画コードも上流のまま）
    args = types.SimpleNamespace(video=src, output_root="outputs/demo_soma_fit", static_cam=True, verbose=False,
                                 render_mhr=False, sam3d_ckpt_path=None, sam3d_mhr_path=None,
                                 ckpt=None, exp="gem_soma_regression", retarget=False)
    cfg = demo._build_cfg(args)
    cpu = lambda d: {k: v.detach().cpu() for k, v in d.items()}
    pred = {"body_params_incam": cpu(incam), "body_params_global": cpu(glob),
            "K_fullimg": K.cpu()[None].repeat(F, 1, 1)}
    torch.save(pred, cfg.paths.hpe_results)
    demo.render_incam(cfg, fps=int(round(fps)))
    render = Path(cfg.paths.incam_video).read_bytes()

    return {"joints_incam_77": J1.cpu().numpy(), "joints_global_77": Jg.cpu().numpy(),
            "poses": {"incam": {k: v.cpu().numpy() for k, v in incam.items()},
                      "global": {k: v.cpu().numpy() for k, v in glob.items()}},
            "render_incam": render, "report": report,
            "provenance": {"step": 2, "gemx_commit": GEMX_COMMIT, "upstream_modified": False,
                           "video_sha256": hashlib.sha256(video_bytes).hexdigest(),
                           "input_pose_sha256": hashlib.sha256(pose_npz).hexdigest(),
                           "method": "SOMA joint-rotation refinement to ViTPose 77 via SomaLayer"}}


@app.local_entrypoint()
def main(src: str = "output_golf_gemx", out: str = "output_golf_gemx_fit", fps: float = 30.0,
         holdout_every: int = 5, iters: int = 400, lr: float = 0.005):
    import sys

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.convert import to_smpl24

    d = Path(out)
    if d.exists() and any(d.iterdir()):
        raise SystemExit(f"{d} にすでに結果がある。別の --out を指定すること")
    s = Path(src)
    videos = sorted(glob.glob(str(s / "*_baked.mp4")))
    if len(videos) != 1:
        raise SystemExit(f"{s} に焼き込み済み動画が {len(videos)} 本。Step 1 の出力を指定すること")
    print(f"送信: {s}/gx_pose.npz, gx_kp2d.npy, {Path(videos[0]).name} → Modal GPU で合わせ込み中…")
    r = fit.remote((s / "gx_pose.npz").read_bytes(), (s / "gx_kp2d.npy").read_bytes(),
                   Path(videos[0]).read_bytes(), fps, holdout_every, iters, lr)

    d.mkdir(parents=True, exist_ok=True)
    g = np.asarray(r["joints_global_77"]); c = np.asarray(r["joints_incam_77"])
    np.save(d / "gx_joints.npy", to_smpl24(g))
    np.save(d / "gx_joints_soma.npy", g)
    np.save(d / "gx_joints_incam.npy", to_smpl24(c))
    np.savez(d / "gx_pose.npz", **{f"{space}_{k}": v for space, params in r["poses"].items()
                                   for k, v in params.items()})
    (d / "incam.mp4").write_bytes(r["render_incam"])
    (d / "report.json").write_text(json.dumps(r["report"], indent=2, ensure_ascii=False) + "\n")
    (d / "provenance.json").write_text(json.dumps(r["provenance"], indent=2) + "\n")
    rep = r["report"]
    print(f"✅ {d}  再投影 中央値: {rep['reproj_median_px']}  最大移動 {rep['max_joint_displacement_cm']}cm  "
          f"回転補正 平均 {rep['rotation_correction_deg']['mean']}° 最大 {rep['rotation_correction_deg']['max']}°")
    print(f"次: python tools/golf_step_report.py {src}/gx_joints.npy {out}/gx_joints.npy")
