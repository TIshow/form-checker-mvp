"""GEM-X の推論方法（regression / DDIM）と既存の接地後処理を、同じ入力で比べる（issue 017）。

    .venv/bin/modal run backend/gemx_experiments.py --video output_golf_gemx/temp_golf.mp4 --out output_golf_gemx_exp

## なぜこれが要るか

上流 DEMO.md は `--ddim`（50 steps）を「遅いが高品質」と書くが、固定版 3299255 では
そのフラグは `model.pipeline.regression_only = False` を書くだけで、実際の分岐
`GEMDiffusion.forward_test` が読むのは `model.pipeline.denoiser3d.regression_only`。
Pipeline はその属性を読まないので、**上流の --ddim では DDIM は一度も走らない**
（Codex の静的分析、issue 017。ここで実行して裏を取る）。

もう一つ。公開チェックポイントの学習設定 `configs/pipeline/regression_only.yaml` は
`train_modes: ["regression"]`——最終タイムステップに零入力で 1 回だけ通す学習。
途中のタイムステップで denoiser を評価する DDIM は、学習で一度も見ていない入力を
与えることになる。だから「DDIM が効くか」は理屈では決まらず、走らせて見るしかない。

## 何を固定し、何を変えるか

固定: 動画（同じバイト列）、ViTPose の色順修正（RGB）、bbox、SAM 3D Body の特徴、
K、静止カメラ、重み、SOMA のデコード（`core/gemx.py`）。
変える: 推論の分岐（regression か DDIM か）、GEM-X 自身の接地後処理
（`refine_translation_with_contacts` + `refine_pose_with_contact_ik`）、乱数 seed。

各条件で記録する: denoiser の実際の評価回数（CFG サンプラは 1 step で 2 回呼ぶので
「ちょうど 50」で判定しない）、DDIM の有効ステップ数、後処理が本当に適用されたか
（`static_conf_logits` が出ていて postproc=True のときだけ走る）、所要時間。

アプリ側の足固定や、不採用の再投影補正は一切かけない。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

from backend.reconstruct_gemx import (
    ASSETS, GEMX, GEMX_COMMIT, _link_assets, _patch_vitpose_color, image, vol,
)

image = image.add_local_python_source("backend")
app = modal.App("gemx-experiments")

#: 実験条件。名前は issue 017 の表に対応（B/D は seed 違いを添える）。
VARIANTS = [
    {"name": "A_reg_nopp",     "regression_only": True,  "postproc": False, "seed": 0},
    {"name": "B_ddim_nopp_s0", "regression_only": False, "postproc": False, "seed": 0},
    {"name": "B_ddim_nopp_s1", "regression_only": False, "postproc": False, "seed": 1},
    {"name": "C_reg_pp",       "regression_only": True,  "postproc": True,  "seed": 0},
    {"name": "D_ddim_pp_s0",   "regression_only": False, "postproc": True,  "seed": 0},
]
#: メッシュを描画する条件（描画は 1 本 1 分ほどかかるので絞る）
RENDER = {"A_reg_nopp", "B_ddim_nopp_s0", "D_ddim_pp_s0"}


@app.function(image=image, gpu="L4", volumes={ASSETS: vol}, timeout=2400)
def run_variants(video_bytes: bytes, variants: list[dict], render: list[str]) -> dict:
    import copy
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
    _patch_vitpose_color(True)

    # 上流デモの関数をそのまま使う（前処理・データ辞書・描画）。
    spec = importlib.util.spec_from_file_location("demo_soma", f"{GEMX}/scripts/demo/demo_soma.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    import hydra
    from gem.utils.net_utils import detach_to_cpu
    from core.gemx import decode_prediction

    source_hash = hashlib.sha256(video_bytes).hexdigest()
    Path("inputs").mkdir(exist_ok=True)
    stem = f"exp_{source_hash[:12]}"
    raw = f"inputs/{stem}_raw.mp4"
    Path(raw).write_bytes(video_bytes)
    src = f"inputs/{stem}.mp4"
    # reconstruct_gemx.py と同じ焼き込み（回転フラグを画素に落とす）
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-an", src], check=True)

    args = types.SimpleNamespace(
        video=src, output_root="outputs/demo_soma_exp", static_cam=True, verbose=False,
        render_mhr=False, sam3d_ckpt_path=None, sam3d_mhr_path=None,
        ckpt=f"{GEMX}/inputs/pretrained/gem_soma.ckpt", exp="gem_soma_regression",
        retarget=False)
    cfg = demo._build_cfg(args)
    demo._copy_video_if_needed(cfg)
    t0 = time.time()
    demo.run_preprocess(cfg)
    t_pre = time.time() - t0
    data = demo.load_data_dict(cfg)
    kp2d = np.asarray(data["kp2d"])

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(demo.resolve_ckpt_path(cfg))
    model = model.eval().cuda()
    den3d = model.pipeline.denoiser3d
    calls = []
    den3d.denoiser.register_forward_hook(lambda m, i, o: calls.append(1))
    steps = int(len(den3d.test_gen_only_diffusion.timestep_map))
    infer_version = int(model.pipeline.args.get("infer_version", 2))
    print(f"[setup] DDIM の有効ステップ数 {steps}, infer_version {infer_version}, "
          f"学習時 regression_only={den3d.regression_only}, "
          f"CFG sampler={model.pipeline.args.get('use_cfg_sampler_for_gen', False)}")

    results = {}
    for v in variants:
        name = v["name"]
        den3d.regression_only = bool(v["regression_only"])
        torch.manual_seed(int(v["seed"]))
        torch.cuda.manual_seed_all(int(v["seed"]))
        calls.clear()
        t0 = time.time()
        with torch.no_grad():
            pred = model.predict(copy.deepcopy(data), static_cam=True, postproc=bool(v["postproc"]))
        t_inf = time.time() - t0
        n_calls = len(calls)
        pred = detach_to_cpu(pred)
        net = pred.get("net_outputs", {})
        has_logits = "static_conf_logits" in net
        applied_pp = bool(v["postproc"]) and has_logits and infer_version != 3
        branch = "regression" if n_calls == 1 else ("ddim" if n_calls >= steps else f"unknown({n_calls})")
        joints, _ = decode_prediction(pred)
        static_conf = (torch.as_tensor(net["static_conf_logits"]).float().sigmoid()[0].numpy()
                       if has_logits else None)
        report = {**v, "branch_taken": branch, "denoiser_calls": n_calls,
                  "ddim_steps_configured": steps, "postproc_applied": applied_pp,
                  "has_static_conf_logits": has_logits, "inference_s": round(t_inf, 2)}
        print(f"[{name}] {report}")
        entry = {"joints_global_77": joints["global"], "joints_incam_77": joints["incam"],
                 "static_conf": static_conf, "report": report}
        if name in render:
            t0 = time.time()
            cfg.paths.hpe_results = f"{cfg.output_dir}/hpe_{name}.pt"
            cfg.paths.incam_video = f"{cfg.output_dir}/{name}_incam.mp4"
            torch.save(pred, cfg.paths.hpe_results)
            demo.render_incam(cfg, fps=30)
            entry["render_incam"] = Path(cfg.paths.incam_video).read_bytes()
            report["render_s"] = round(time.time() - t0, 2)
        results[name] = entry

    return {"variants": results, "kp2d": kp2d,
            "provenance": {"video_sha256": source_hash, "gemx_commit": GEMX_COMMIT,
                           "vitpose_input": "rgb", "static_camera": True,
                           "decoder": "official_soma_adapter", "units": "m",
                           "preprocess_s": round(t_pre, 2), "ddim_steps": steps,
                           "infer_version": infer_version}}


@app.local_entrypoint()
def main(video: str, out: str = "output_golf_gemx_exp"):
    import sys

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.convert import to_smpl24

    d = Path(out)
    if d.exists() and any(d.iterdir()):
        raise SystemExit(f"{d} にすでに結果がある。別の --out を指定すること")
    data = Path(video).read_bytes()
    print(f"送信: {video} ({len(data) / 1e6:.1f} MB) → {len(VARIANTS)} 条件を Modal で実行中…")
    r = run_variants.remote(data, VARIANTS, sorted(RENDER))

    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "gx_kp2d.npy", np.asarray(r["kp2d"]))
    reports = {}
    for name, e in r["variants"].items():
        vd = d / name
        vd.mkdir(exist_ok=True)
        g = np.asarray(e["joints_global_77"]); c = np.asarray(e["joints_incam_77"])
        np.save(vd / "gx_joints.npy", to_smpl24(g))
        np.save(vd / "gx_joints_soma.npy", g)
        np.save(vd / "gx_joints_incam.npy", to_smpl24(c))
        if e["static_conf"] is not None:
            np.save(vd / "static_conf.npy", np.asarray(e["static_conf"]))
        if "render_incam" in e:
            (vd / "incam.mp4").write_bytes(e["render_incam"])
        reports[name] = e["report"]
        print(f"✅ {vd}  {e['report']}")
    (d / "report.json").write_text(json.dumps(
        {"provenance": r["provenance"], "variants": reports}, indent=2, ensure_ascii=False) + "\n")
    print(f"provenance: {r['provenance']}")
