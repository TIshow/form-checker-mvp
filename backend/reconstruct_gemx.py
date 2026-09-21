"""GEM-X (NVIDIA) を Modal のサーバーレスGPUで動かす（issue #9）。

GVHMR 版 `backend/reconstruct.py` と**並行して**置く。評価が済むまで既存経路は
一切変えない。両方が同時に動くよう、Modal の app 名も Volume も分けてある。

    modal run backend/reconstruct_gemx.py::fetch_checkpoints   # 初回のみ
    modal run backend/reconstruct_gemx.py --video X.mp4 --out output_gemx
    modal run backend/reconstruct_gemx.py --video X.mp4 --out output_gemx --start 4.8 --end 10.3

## なぜ GEM-X を評価するのか

GVHMR も SMPL も**非商用ライセンス**で、部活動向けの製品化ができない。
GEM-X はコードが Apache-2.0、重みが NVIDIA Open Model License（"Models are
commercially usable"）、人体モデルは NVIDIA 独自の SOMA で、SMPL を必要としない。
検出も YOLOX(Apache-2.0) + ByteTrack(MIT) なので YOLOv8 の AGPL も外れる。
経緯と判断材料は docs/issues/009-licensing-for-productization.md。

## GVHMR との違いで、下流に効くところ

- **関節が SOMA の77個**（GVHMR は SMPL の24個）。`core/convert.py` で
  並べ替えれば既存の解析層・ビューア・アバターが無改造で動く。
- **カメラ空間の姿勢も返る**（`body_params_incam`）。GVHMR は世界座標しか
  返さなかった。跳躍がどの段で失われるかを直接比べられる（issue #8 の案D）。
- 手と顔も入っている。ラケット周りに使えるかは別途評価（issue #6）。

## 手順の出どころ

GEM-X の Dockerfile をそのまま移した（INSTALL.md は Python 3.12 と書いているが、
**Dockerfile は 3.10**。動く方に合わせる）。描画は EGL のヘッドレス。
"""

from __future__ import annotations

import glob
import subprocess
from pathlib import Path

import modal

GEMX = "/root/GEM-X"
ASSETS = "/assets"

# 再現性のため固定する。更新したら core/convert.py の verify_against_asset() を
# 通すこと（SOMA が関節の並びを変えると、手書きの対応表が黙って壊れる）。
GEMX_COMMIT = "3299255"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "git-lfs", "wget", "curl", "ffmpeg", "build-essential",
        # open3d / OpenCV のヘッドレス描画に要る（Dockerfile 準拠）
        "libegl1-mesa-dev", "libgl1-mesa-glx", "libglib2.0-0", "xvfb",
    )
    .run_commands(
        "git lfs install",
        # --recursive にしない。soma-retargeter submodule だけ SSH URL
        # (git@github.com:...) で、鍵の無いビルド環境では必ず失敗する。
        # あれは Unitree G1 へのリターゲット用で、ここでは使わない。
        f"git clone https://github.com/NVlabs/GEM-X {GEMX}",
        f"cd {GEMX} && git checkout {GEMX_COMMIT} && "
        f"git submodule update --init third_party/soma third_party/sam-3d-body",
        "pip install -U pip setuptools wheel uv",
        # torch はバージョン無指定（Dockerfile と同じ）。cu126 の index から取る
        "pip install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu126",
        f"cd {GEMX} && pip install -e third_party/soma",
        # SOMA の重み(827MB)は LFS。ポインタのままだと実行時に落ちる
        f"cd {GEMX}/third_party/soma && git lfs pull",
        # scripts/install_env.sh の中身を展開している。理由は detectron2 を
        # 入れないため。あれは third_party/sam-3d-body 単体デモ用の検出器で、
        # 遅延 import されるだけ。GEM-X のデモは YOLOX + ByteTrack を使う
        # （macOS 経路は detectron2 を丸ごとスキップしていて、検出は動く）。
        #
        # CC/CXX を渡すのは、Modal の Python が clang でビルドされていて
        # sysconfig が clang++ を要求するため。PyTorch は linux では g++ で
        # ビルドされており、拡張を clang++ で作ると ABI が合わない
        # （PyTorch 自身がそう警告する）。
        f"cd {GEMX} && CC=gcc CXX=g++ pip install -e .",
        "CC=gcc CXX=g++ pip install cloudpickle fvcore iopath pycocotools "
        "braceexpand roma 'setuptools<75'",
        # requirements.txt にも install_env.sh の Linux 分岐にも入っていないが、
        # demo_soma.py は両方要る。上流の抜け。
        #   onnxruntime … 人物検出は全プラットフォームで ONNX の YOLOX
        #                 (gem/utils/yolox_detector.py)。install_env.sh は
        #                 macOS のときしか入れない
        #   open3d     … レンダ動画の描画。関数内 import なので推論だけなら
        #                 無くても通るが、目視での確認に使うので入れる
        # onnxruntime は CPU 版。GPU 版は cuDNN を要求し、この CUDA イメージには
        # 入っていないので CUDA EP の読み込みに失敗して結局 CPU に落ちる。
        # 検出は数百フレームで十数秒なので CPU で十分。
        "pip install onnxruntime open3d",
        # デモは inputs/soma_assets を見る（setup スクリプトが張る symlink）
        f"cd {GEMX} && mkdir -p inputs && "
        f"ln -sfn {GEMX}/third_party/soma/assets inputs/soma_assets",
    )
    .env({"PYOPENGL_PLATFORM": "egl", "EGL_PLATFORM": "surfaceless"})
    .add_local_python_source("analysis", "core", "domains")
)

app = modal.App("gemx-reconstruct")
vol = modal.Volume.from_name("gemx-assets", create_if_missing=True)

# 重みは全部 nvidia/GEM-X にある。SAM-3D-Body も NVIDIA が再配布しているので、
# 配布元が同じでも SAM/MHR 等の第三者条項はそれぞれ確認する。
HF_REPO = "nvidia/GEM-X"
HF_FILES = [
    ("gem_soma.ckpt", "pretrained"),
    ("vitpose.pth", "checkpoints/vitpose"),
    ("sam3d_body.ckpt", "checkpoints/sam-3d-body-dinov3"),
    ("model_config.yaml", "checkpoints/sam-3d-body-dinov3"),
    ("mhr_model.pt", "mhr_data"),
    ("scale_mean.pth", "soma_data"),
    ("scale_comps.pth", "soma_data"),
]


@app.function(image=image, volumes={ASSETS: vol}, timeout=3600)
def fetch_checkpoints():
    """nvidia/GEM-X から重みを Volume に取得（初回のみ）。"""
    import os

    from huggingface_hub import hf_hub_download

    for fname, sub in HF_FILES:
        dest = f"{ASSETS}/{sub}"
        os.makedirs(dest, exist_ok=True)
        p = hf_hub_download(repo_id=HF_REPO, filename=fname, local_dir=dest)
        print(f"✅ {p}  ({os.path.getsize(p) / 1e6:.0f} MB)")
    vol.commit()


def _probe_fps(path: str) -> float | None:
    """コンテナ上の再生フレームレートを ffprobe で読む。

    スローモーションとして引き伸ばして保存された動画では、実際の撮影レートは
    これより高い。その場合は解析時に手で指定する（tools/camera_motion.py と
    docs/issues/008 を参照）。
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", path],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        num, _, den = out.partition("/")
        return float(num) / float(den or 1)
    except Exception as e:
        print(f"[fps] 検出できませんでした: {e}")
        return None


def _link_assets() -> None:
    """Volume に置いた重みを、GEM-X が探す位置へ繋ぐ。"""
    import os

    for fname, sub in HF_FILES:
        src = f"{ASSETS}/{sub}/{fname}"
        dst = Path(GEMX) / "inputs" / sub / fname
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(src, dst)


@app.function(image=image, gpu="L4", volumes={ASSETS: vol}, timeout=3600)
def reconstruct(video_bytes: bytes, name: str,
                start: float | None = None, end: float | None = None,
                static_cam: bool = True) -> dict:
    """動画1本を GEM-X で復元し、関節・姿勢・レンダ動画を返す。

    static_cam: True で `-s`（静止カメラ前提、VO を切る）。GVHMR と条件を
    揃えて比べたいときは True。カメラが動く素材では False も試す価値がある。

    上流のコードには一切手を入れない。入力の動画を用意し、公式デモをそのまま
    走らせ、公式の SOMA アダプタで関節に落とすだけ（issue 018）。
    """
    import numpy as np
    import torch

    os_chdir = __import__("os").chdir
    os_chdir(GEMX)
    _link_assets()

    Path("inputs").mkdir(exist_ok=True)
    import hashlib
    # Distinguish video contents and selected interval, not just a basename.
    source_hash = hashlib.sha256(video_bytes).hexdigest()
    cache_key = hashlib.sha256(f"{source_hash}:{start}:{end}:{GEMX_COMMIT}:v2".encode()).hexdigest()[:16]
    stem = f"{Path(name).stem}_{cache_key}"
    src = f"inputs/{stem}.mp4"
    Path(src).write_bytes(video_bytes)

    baked = f"inputs/{stem}_baked.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-an", baked], check=True)
    src = baked

    if start is not None or end is not None:
        trimmed = f"inputs/{stem}_trim.mp4"
        cmd = ["ffmpeg", "-y", "-i", src]
        if start is not None:
            cmd += ["-ss", str(start)]
        if end is not None:
            cmd += ["-to", str(end)]
        # 再エンコードする（コピーだとキーフレーム境界までしか切れない）
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-an", trimmed]
        subprocess.run(cmd, check=True, capture_output=True)
        src = trimmed
        print(f"[trim] {start}〜{end} 秒を切り出しました")

    # デモは入力ファイル名で出力先を決める（GVHMR と同じ罠）
    stem = Path(src).stem
    video_fps = _probe_fps(src)
    print(f"[fps] 動画から検出: {video_fps}")

    out_root = "outputs/demo_soma"
    cmd = ["python", "scripts/demo/demo_soma.py",
           f"--video={src}", f"--output_root={out_root}",
           # 明示しないと毎回 HuggingFace 取得の経路に入る（実体があれば
           # スキップされるが、Volume に置いたものを使う意図を明確にする）
           f"--ckpt={GEMX}/inputs/pretrained/gem_soma.ckpt"]
    if static_cam:
        cmd.append("-s")
    subprocess.run(cmd, check=True)

    # configs/demo_soma.yaml は `${output_dir}/hpe_results.pt`。
    # DEMO.md の表は preprocess/ 配下と書いているが、そちらが誤り。
    pred_path = f"{out_root}/{stem}/hpe_results.pt"
    pred = torch.load(pred_path, map_location="cpu")
    print(f"[hpe_results] 最上位のキー: {list(pred)}")

    # 推論結果を Volume に残す。ここから先（姿勢パラメータ → 関節座標）で
    # つまずくたびに3分のフルパイプラインを回し直すのは高くつくので、
    # 中間結果を取っておいて joints_from_pred() だけ試せるようにする。
    import shutil

    keep = Path(ASSETS) / "debug"
    keep.mkdir(parents=True, exist_ok=True)
    shutil.copy(pred_path, keep / f"{stem}_hpe_results.pt")
    vol.commit()
    print(f"[debug] {keep}/{stem}_hpe_results.pt に保存しました")

    # Use the same adapter as the pinned official demo: meters, explicit scale
    # split, and repose_to_bind_pose=False. Success of a guessed API shape was
    # not evidence that the anatomical rest pose was correct.
    from core.gemx import decode_prediction
    joints, poses = decode_prediction(pred)

    renders = {}
    for mp4 in sorted(glob.glob(f"{out_root}/{stem}/*.mp4")):
        renders[Path(mp4).name] = Path(mp4).read_bytes()

    # GEM-X 本体に入った 2D 関節（ViTPose, SOMA 77点, x,y,score）も返す。
    # 3D が崩れたとき、入力の 2D がすでに崩れていたかをローカルで確かめられる。
    kp2d = None
    for pt in glob.glob(f"{out_root}/{stem}/**/*vitpose*.pt", recursive=True):
        v = torch.load(pt, map_location="cpu")
        v = v[0] if isinstance(v, tuple) else v
        kp2d = np.asarray(v)
        print(f"[kp2d] {pt} {kp2d.shape}")
        break

    return {"joints": joints, "poses": poses, "renders": renders, "kp2d": kp2d,
            "video_fps": video_fps,
            "provenance": {"video_sha256": source_hash, "gemx_commit": GEMX_COMMIT,
                           "decoder": "official_soma_adapter", "units": "m",
                           "static_camera": static_cam, "start": start, "end": end,
                           "upstream_modified": False}}


@app.local_entrypoint()
def main(video: str, out: str = "output_gemx",
         start: float | None = None, end: float | None = None,
         moving_cam: bool = False):
    """ローカルの動画を Modal で復元し、結果をローカルへ保存する。

    保存されるもの:
      gx_joints.npy       SMPL24順の世界座標（既存の `python -m analysis` がそのまま食える）
      gx_joints_soma.npy  SOMA 77関節の世界座標（手・顔を使いたくなったとき用）
      gx_joints_incam.npy SMPL24順のカメラ空間（issue #8 の切り分け用）
      gx_pose.npz         SOMA の姿勢パラメータ（アバターへのリターゲット用）
      gx_kp2d.npy         GEM-X に入った ViTPose の 2D 関節 (F,77,3) x,y,score
      provenance.json     動画のハッシュ・固定コミット・上流に手を入れていないこと
    """
    import sys

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.convert import to_smpl24

    video_path = Path(video)
    data = video_path.read_bytes()
    print(f"送信: {video_path} ({len(data) / 1e6:.1f} MB) → Modal GPU で復元中…")

    r = reconstruct.remote(data, video_path.name, start, end,
                           not moving_cam)

    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    saved = []

    soma_g = np.asarray(r["joints"]["global"])
    soma_c = np.asarray(r["joints"]["incam"])
    np.save(d / "gx_joints.npy", to_smpl24(soma_g))
    np.save(d / "gx_joints_soma.npy", soma_g)
    np.save(d / "gx_joints_incam.npy", to_smpl24(soma_c))
    saved += ["gx_joints.npy", "gx_joints_soma.npy", "gx_joints_incam.npy"]

    np.savez(d / "gx_pose.npz", **{f"{space}_{k}": v
                                   for space, params in r["poses"].items()
                                   for k, v in params.items()})
    import json
    (d / "provenance.json").write_text(json.dumps(r["provenance"], indent=2) + "\n")
    saved += ["gx_pose.npz", "provenance.json"]
    if r.get("kp2d") is not None:
        np.save(d / "gx_kp2d.npy", np.asarray(r["kp2d"]))
        saved.append("gx_kp2d.npy")

    for fname, blob in r["renders"].items():
        (d / fname).write_bytes(blob)
        saved.append(fname)

    for s in saved:
        p = d / s
        print(f"✅ {p} ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"\n検出fps: {r['video_fps']}")
    print(f"次: python -m analysis --joints {d}/gx_joints.npy "
          f"--fps <実撮影レート> --save {d}")
