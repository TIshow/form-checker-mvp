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

from __future__ import annotations

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
    # analysis（純numpy）をイメージに含める。run_job がサーバー側で指標を導出するため。
    .add_local_python_source("analysis")
)

# Web エンドポイント用の軽量イメージ（GPU不要。fastapi だけ）
web_image = modal.Image.debian_slim().pip_install("fastapi[standard]")

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


def _gvhmr_joints(video_bytes: bytes, name: str):
    """GVHMR を実行し、24関節(F,24,3 world座標[m])とレンダ動画を返す。

    コンテナ内で動く純ヘルパー。CLI用 reconstruct と Web用 run_job が共有する。
    静止カメラ前提で `-s`（SLAM回避）。
    """
    import glob
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

    # GVHMR がレンダリングした3D動画（入力動画のコピーは除く）
    renders = {}
    for mp4 in sorted(glob.glob(f"outputs/demo/{stem}/*.mp4")):
        base = os.path.basename(mp4)
        if "input" in base:
            continue
        with open(mp4, "rb") as f:
            renders[f"render_{base}"] = f.read()

    elapsed = time.time() - t_start
    # T4 は $0.000164/秒（modal.com/pricing）。コンテナ起動分は別途上乗せ。
    print(f"joints {joints.shape}  [TIMING] {elapsed:.1f}s "
          f"≒ ${elapsed * 0.000164:.4f} (T4, 起動分は別)")
    return joints, renders


@app.function(image=image, gpu="T4", volumes={ASSETS: vol}, timeout=900)
def reconstruct(video_bytes: bytes, name: str) -> dict:
    """CLI用。関節を .npy にし、レンダ動画と共に返す（`modal run` → ローカル保存）。"""
    import io

    import numpy as np

    joints, renders = _gvhmr_joints(video_bytes, name)
    buf = io.BytesIO()
    np.save(buf, joints)
    return {"gv_joints.npy": buf.getvalue(), **renders}


@app.function(image=image, gpu="T4", volumes={ASSETS: vol}, timeout=900)
def run_job(video_bytes: bytes, name: str, fps: float) -> dict:
    """Web用。復元 → analysis で指標・フィードバックを導出し JSON で返す。

    「現実+3Dメッシュの重ね合わせ」動画(incam)も base64 で同梱する
    （フォームの当てはまりを目で確認できるため）。
    """
    import base64

    import analysis

    joints, renders = _gvhmr_joints(video_bytes, name)
    res = analysis.analyze_json(joints, fps)

    # 横並び(horiz) = 左:元動画+3Dメッシュ重ね / 右:別角度の3D。両方見える版を優先。
    def pick(rs):
        for key in ("horiz",):
            v = next((v for k, v in rs.items() if key in k), None)
            if v:
                return v
        v = next((v for k, v in rs.items() if "incam" in k), None)
        return v or (next(iter(rs.values()), None) if rs else None)

    overlay = pick(renders)
    if overlay is not None:
        res["overlay_video_b64"] = base64.b64encode(overlay).decode()
    return res


# --------------------------------------------------------------------------
# Web API（非同期）
#
# 復元は数分かかるため、投入と取得を分ける:
#   POST /submit  … 動画(base64)を受けてジョブを投入し job_id を返す（即時）
#   GET  /result  … job_id の状態を返す（pending / done+結果 / error）
#
# ブラウザから叩くため CORS を許可する。単一の ASGI アプリにまとめると
# CORS ミドルウェアを付けられ、URL も1つで済む。
# --------------------------------------------------------------------------
@app.function(image=web_image)
@modal.asgi_app(label="serve-api")
def web():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI(title="Tennis serve analysis")
    api.add_middleware(
        CORSMiddleware, allow_origins=["*"],
        allow_methods=["*"], allow_headers=["*"],
    )

    @api.post("/submit")
    def submit(item: dict):
        """item = {video_b64: str, name?: str, fps?: float}"""
        import base64

        data = base64.b64decode(item["video_b64"])
        call = run_job.spawn(
            data, item.get("name", "serve.mp4"), float(item.get("fps", 30.0)))
        return {"job_id": call.object_id}

    @api.get("/result")
    def result(job_id: str):
        fc = modal.FunctionCall.from_id(job_id)
        try:
            return {"status": "done", "result": fc.get(timeout=0)}
        except TimeoutError:
            return {"status": "pending"}
        except Exception as e:  # ジョブ内で失敗した場合
            return {"status": "error", "message": str(e)}

    return api


@app.local_entrypoint()
def main(video: str, out: str = "."):
    """ローカルの動画を Modal で復元し、関節 .npy とレンダ動画をローカルへ保存する。"""
    video_path = Path(video)
    data = video_path.read_bytes()
    print(f"送信: {video_path} ({len(data) / 1e6:.1f} MB) → Modal GPU で復元中…")

    results = reconstruct.remote(data, video_path.name)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, content in results.items():
        (out_dir / fname).write_bytes(content)
        print(f"✅ {out_dir / fname} ({len(content) / 1e6:.1f} MB)")
    print(f"→ 次: python -m analysis --joints {out_dir}/gv_joints.npy "
          f"--fps <実fps> --save {out_dir}")
