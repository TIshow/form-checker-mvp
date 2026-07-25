"""GVHMR を Modal のサーバーレスGPUで動かす。

issue #2 のフェーズA/B。Colab で確定した環境レシピをそのまま Modal Image に焼き、
毎回の環境再構築を無くす。GPUを使うのは3D復元のみ。出力(.npy)を解析層に渡す。

## 準備（初回のみ）

  # 1. チェックポイントを Volume に取得（HuggingFace から、Modal上で実行）
  modal run backend/reconstruct.py::fetch_checkpoints

  # 2. body models を Volume にアップロード（ライセンス制のためローカルから）
  modal volume put gvhmr-assets \
      ~/Desktop/gvhmr_body_models/SMPL_NEUTRAL.pkl  /checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
  modal volume put gvhmr-assets \
      ~/Desktop/gvhmr_body_models/SMPLX_NEUTRAL.npz /checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz

## 実行

  modal run backend/reconstruct.py --video temp_my_serve.mp4
  # → gv_joints.npy / gv_com.npy / gv_upaxis.npy がカレントに返る
  # → python -m analysis で解析
"""

from pathlib import Path

import modal

GVHMR = "/root/GVHMR"
ASSETS = "/assets"  # Volume のマウント先。中に checkpoints/ を置く

# --------------------------------------------------------------------------
# Image — P0 で確定したレシピ（memory: redesign-direction の確定レシピ）
#   Python 3.10 / torch 2.3.0+cu121 / chumpy は --no-build-isolation /
#   GVHMR の `from turtle import` バグを除去 / requirements.txt は無改変
# --------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "build-essential", "wget")
    .run_commands(
        f"git clone https://github.com/zju3dv/GVHMR {GVHMR}",
        # GVHMR のバグ: body_model.py の不要な `from turtle import forward`
        f"cd {GVHMR} && find . -name '*.py' -exec sed -i '/from turtle import/d' {{}} +",
        # torch を先に固定（GVHMR 想定の版）
        "pip install torch==2.3.0 torchvision==0.18.0 "
        "--index-url https://download.pytorch.org/whl/cu121",
        # chumpy は setup.py が pip を要求するため、ビルド分離を切って個別に
        "pip install -U pip setuptools wheel",
        f"cd {GVHMR} && pip install --no-build-isolation chumpy==0.70",
        # 残りの依存（chumpy は解決済みなのでスキップされる）と GVHMR 本体
        f"cd {GVHMR} && pip install -r requirements.txt",
        f"cd {GVHMR} && pip install -e .",
        "pip install huggingface_hub",
    )
)

app = modal.App("gvhmr-reconstruct")
vol = modal.Volume.from_name("gvhmr-assets", create_if_missing=True)

# HuggingFace ミラー（Colab で実績。Google Drive はクォータ制限で落ちる）
HF_REPO = "ryanrudes/gvhmr"
HF_FILES = [
    "gvhmr/gvhmr_siga24_release.ckpt",
    "hmr2/epoch=10-step=25000.ckpt",
    "vitpose/vitpose-h-multi-coco.pth",
    "yolo/yolov8x.pt",
]

# De Leva 体節質量比 (親関節, 子関節, 質量比, 近位からのCOM比)。analysis と同一。
_SEG = [
    (0, 12, 0.497, 0.50), (12, 15, 0.081, 0.50),
    (16, 18, 0.028, 0.436), (17, 19, 0.028, 0.436),
    (18, 20, 0.016, 0.430), (19, 21, 0.016, 0.430),
    (20, 22, 0.006, 0.50), (21, 23, 0.006, 0.50),
    (1, 4, 0.100, 0.433), (2, 5, 0.100, 0.433),
    (4, 7, 0.0465, 0.433), (5, 8, 0.0465, 0.433),
    (7, 10, 0.0145, 0.50), (8, 11, 0.0145, 0.50),
]


@app.function(image=image, volumes={ASSETS: vol}, timeout=1800)
def fetch_checkpoints():
    """HuggingFace から4つのチェックポイントを Volume に取得（初回のみ）。"""
    import os
    from huggingface_hub import hf_hub_download

    dest = f"{ASSETS}/checkpoints"
    os.makedirs(dest, exist_ok=True)
    for rel in HF_FILES:
        p = hf_hub_download(repo_id=HF_REPO, filename=rel, local_dir=dest)
        print("✅", p)
    vol.commit()
    print("=== Volume の中身 ===")
    for root, _, files in os.walk(dest):
        for f in files:
            print(os.path.join(root, f))


@app.function(image=image, gpu="T4", volumes={ASSETS: vol}, timeout=900)
def reconstruct(video_bytes: bytes, name: str) -> dict:
    """1本の動画を GVHMR で3D復元し、joints/com/upaxis を numpy で返す。

    静止カメラ前提で `-s`（SLAM回避）。GPUを使うのはこの関数だけ。
    """
    import os
    import subprocess
    import time

    import numpy as np
    import torch

    t_start = time.time()
    os.chdir(GVHMR)

    # Volume のチェックポイントを GVHMR の想定パスへ結びつける
    os.makedirs("inputs", exist_ok=True)
    link = "inputs/checkpoints"
    if not os.path.islink(link):
        if os.path.exists(link):
            subprocess.run(["rm", "-rf", link], check=True)
        os.symlink(f"{ASSETS}/checkpoints", link)

    # body models が Volume にあるか確認（無ければ明確に失敗させる）
    for req in ["body_models/smpl/SMPL_NEUTRAL.pkl",
                "body_models/smplx/SMPLX_NEUTRAL.npz"]:
        if not os.path.exists(f"{ASSETS}/checkpoints/{req}"):
            raise FileNotFoundError(
                f"{req} が Volume にありません。README の準備手順（modal volume put）を実行してください。")

    # 動画を書き出して demo 実行
    stem = Path(name).stem
    src = f"inputs/{stem}.mp4"
    with open(src, "wb") as f:
        f.write(video_bytes)

    subprocess.run(
        ["python", "tools/demo/demo.py", f"--video={src}", "-s"],
        check=True,
    )

    # 出力 .pt を読み、GVHMR本体と同じ経路で 24関節を復元
    res = torch.load(f"outputs/demo/{stem}/hmr4d_results.pt", map_location="cpu")
    g = res["smpl_params_global"]  # 名前に反して SMPL-X パラメータ

    from hmr4d.utils.smplx_utils import make_smplx

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_smplx("supermotion").to(dev)
    with torch.no_grad():
        out = model(**{k: v.to(dev) for k, v in g.items()})
    s2s = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").to(dev)
    verts = torch.stack([torch.matmul(s2s, v) for v in out.vertices])
    jreg = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").to(dev)
    joints = torch.einsum("jv,fvc->fjc", jreg, verts).cpu().numpy()  # (F,24,3) m

    # 上方向の軸（頭 - 足首）
    up_vec = (joints[:, 15] - (joints[:, 7] + joints[:, 8]) / 2).mean(0)
    up_ax = int(np.argmax(np.abs(up_vec)))
    up_sign = float(np.sign(up_vec[up_ax]))

    # De Leva 体節質量比で全身重心
    com = np.zeros((joints.shape[0], 3))
    for a, b, m, r in _SEG:
        com += m * (joints[:, a] * (1 - r) + joints[:, b] * r)
    com /= sum(s[2] for s in _SEG)

    def to_bytes(arr):
        import io

        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()

    out = {
        "gv_joints.npy": to_bytes(joints),
        "gv_com.npy": to_bytes(com),
        "gv_upaxis.npy": to_bytes(np.array([up_ax, up_sign])),
    }

    # GVHMR がレンダリングした3D動画も返す（入力動画のコピーは除く）。
    #   *_incam*  : 元動画に3Dメッシュを重ねたもの（復元の当てはまりを確認できる）
    #   *_global* : 世界座標での3D視点（重力基準の"3D動画"）
    import glob

    for mp4 in sorted(glob.glob(f"outputs/demo/{stem}/*.mp4")):
        base = os.path.basename(mp4)
        if "input" in base:
            continue
        with open(mp4, "rb") as f:
            out[f"render_{base}"] = f.read()
        print(f"[render] {base} ({os.path.getsize(mp4) / 1e6:.1f} MB)")

    elapsed = time.time() - t_start
    # T4 は $0.000164/秒（modal.com/pricing）。コンテナ起動分は別途上乗せ。
    print(f"joints {joints.shape} up_axis {up_ax} sign {up_sign:+.0f} "
          f"COM mean {com.mean(0).round(3)}")
    print(f"[TIMING] GPU関数の実働 {elapsed:.1f}s  "
          f"≒ ${elapsed * 0.000164:.4f} (T4, 起動分は別)")
    return out


@app.local_entrypoint()
def main(video: str, out: str = "."):
    """ローカルの動画を Modal で復元し、結果 .npy をローカルへ保存する。"""
    video_path = Path(video)
    data = video_path.read_bytes()
    print(f"送信: {video_path} ({len(data) / 1e6:.1f} MB) → Modal GPU で復元中…")

    results = reconstruct.remote(data, video_path.name)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, content in results.items():
        (out_dir / fname).write_bytes(content)
        print(f"✅ {out_dir / fname} ({len(content) / 1e6:.1f} MB)")
    print("→ 次: python -m analysis "
          f"--joints {out_dir}/gv_joints.npy --com {out_dir}/gv_com.npy "
          f"--upaxis {out_dir}/gv_upaxis.npy --fps <実fps> --save {out_dir}")
