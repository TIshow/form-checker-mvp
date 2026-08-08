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

- **関節が SOMA の77個**（GVHMR は SMPL の24個）。`analysis/soma.py` で
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

# 再現性のため固定する。更新したら analysis/soma.py の verify_against_asset() を
# 通すこと（SOMA が関節の並びを変えると、手書きの対応表が黙って壊れる）。
GEMX_COMMIT = "3299255"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "git-lfs", "wget", "curl", "ffmpeg",
        # open3d / OpenCV のヘッドレス描画に要る（Dockerfile 準拠）
        "libegl1-mesa-dev", "libgl1-mesa-glx", "libglib2.0-0", "xvfb",
    )
    .run_commands(
        "git lfs install",
        f"git clone --recursive https://github.com/NVlabs/GEM-X {GEMX}",
        f"cd {GEMX} && git checkout {GEMX_COMMIT} && "
        f"git submodule update --init --recursive",
        "pip install -U pip setuptools wheel uv",
        # torch はバージョン無指定（Dockerfile と同じ）。cu126 の index から取る
        "pip install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu126",
        f"cd {GEMX} && pip install -e third_party/soma",
        # SOMA の重み(827MB)は LFS。ポインタのままだと実行時に落ちる
        f"cd {GEMX}/third_party/soma && git lfs pull",
        # install_env.sh は uv 前提。venv を作らないので system python を対象にする
        f"cd {GEMX} && UV_SYSTEM_PYTHON=1 bash scripts/install_env.sh",
        # デモは inputs/soma_assets を見る（setup スクリプトが張る symlink）
        f"cd {GEMX} && mkdir -p inputs && "
        f"ln -sfn {GEMX}/third_party/soma/assets inputs/soma_assets",
    )
    .env({"PYOPENGL_PLATFORM": "egl", "EGL_PLATFORM": "surfaceless"})
    .add_local_python_source("analysis")
)

app = modal.App("gemx-reconstruct")
vol = modal.Volume.from_name("gemx-assets", create_if_missing=True)

# 重みは全部 nvidia/GEM-X にある。SAM-3D-Body も NVIDIA が再配布しているので、
# Meta の申請制リポジトリを通す必要はない（＝全部 NVIDIA Open Model License）。
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
    """
    import numpy as np
    import torch

    os_chdir = __import__("os").chdir
    os_chdir(GEMX)
    _link_assets()

    Path("inputs").mkdir(exist_ok=True)
    stem = Path(name).stem
    src = f"inputs/{stem}.mp4"
    Path(src).write_bytes(video_bytes)

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
           f"--video={src}", f"--output_root={out_root}"]
    if static_cam:
        cmd.append("-s")
    subprocess.run(cmd, check=True)

    pred = torch.load(f"{out_root}/{stem}/preprocess/hpe_results.pt",
                      map_location="cpu")

    # SOMA の姿勢パラメータ → 関節座標。世界座標とカメラ空間の両方を出す。
    # デモの描画側は見栄えのために y の最小値を引いて接地させているが、
    # ここでは**生のまま**返す。床合わせは解析側の責務（中央値を使う）で、
    # 最小値で合わせるとクリップ中の一度の沈み込みが床になってしまう。
    from soma import SomaLayer  # noqa: E402  イメージ内にのみ存在

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    soma = SomaLayer(data_root="inputs/soma_assets", low_lod=True,
                     device=dev, identity_model_type="mhr", mode="warp")

    joints = {}
    poses = {}
    for space in ("global", "incam"):
        params = pred[f"body_params_{space}"]
        with torch.no_grad():
            out = soma(**{k: v.to(dev) for k, v in params.items()})
        joints[space] = out["joints"].cpu().numpy()
        poses[space] = {k: v.detach().cpu().numpy() for k, v in params.items()}
        print(f"[{space}] joints {joints[space].shape}")

    renders = {}
    for mp4 in sorted(glob.glob(f"{out_root}/{stem}/*.mp4")):
        renders[Path(mp4).name] = Path(mp4).read_bytes()

    return {"joints": joints, "poses": poses, "renders": renders,
            "video_fps": video_fps}


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
    """
    import sys

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.soma import to_smpl24

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

    np.savez(d / "gx_pose.npz", **{f"global_{k}": v
                                   for k, v in r["poses"]["global"].items()})
    saved.append("gx_pose.npz")

    for fname, blob in r["renders"].items():
        (d / fname).write_bytes(blob)
        saved.append(fname)

    for s in saved:
        p = d / s
        print(f"✅ {p} ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"\n検出fps: {r['video_fps']}")
    print(f"次: python -m analysis --joints {d}/gx_joints.npy "
          f"--fps <実撮影レート> --save {d}")
