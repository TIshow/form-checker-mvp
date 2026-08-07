"""GVHMR を Modal のサーバーレスGPUで動かす（CLI と Web の両方）。

Colab で確定した環境レシピをそのまま Modal Image に焼き、毎回の環境再構築を無くす。
GPUを使うのは3D復元のみで、24関節を返す。重心・角度・フィードバックの導出は
analysis 側（純numpy）の責務。issue #2（Modal化）と issue #3 フェーズA（HTTP化）。

## 準備（初回のみ）

  # 1. チェックポイントを Volume に取得（HuggingFace から、Modal上で実行）
  modal run backend/reconstruct.py::fetch_checkpoints

  # 2. body models を Volume にアップロード（ライセンス制のためローカルから）
  modal volume put gvhmr-assets \
      ~/Desktop/gvhmr_body_models/SMPL_NEUTRAL.pkl  /checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
  modal volume put gvhmr-assets \
      ~/Desktop/gvhmr_body_models/SMPLX_NEUTRAL.npz /checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz

## CLI（ローカルで解析）

  modal run backend/reconstruct.py --video temp_my_serve.mp4
  # → gv_joints.npy / gv_pose.npz / レンダ動画 render_*.mp4 がカレントに返る
  #   gv_pose.npz = 関節の回転（アバターへのリターゲット用。issue #5）
  # → python -m analysis --joints gv_joints.npy --fps 60 --save output

## Web（ブラウザから。issue #3）

  modal deploy backend/reconstruct.py
  # POST <workspace>--serve-api.modal.run/submit  {video_b64, fps} → {job_id}
  # GET  ...                        /result?job_id=...  → {status, result}
  #   result = 指標・フィードバック・関節列(3Dビューア用)・重ね合わせ動画(base64)
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


def _probe_fps(path: str) -> float | None:
    """動画の実フレームレートを ffprobe で読む。

    fps を人が申告する設計だと、取り違えても誰も気づけない。実際 60fps の動画を
    30fps として解析し、時間の指標がすべて2倍ずれていた。動画から取れるものは
    動画から取る。

    注意: これはコンテナ上の再生フレームレート。スローモーションとして
    「引き伸ばして保存された」動画では、実際の撮影レートはこれより高い。
    その場合は呼び出し側で上書きする必要がある。
    """
    import json
    import subprocess

    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate,r_frame_rate",
             "-of", "json", path],
            capture_output=True, text=True, timeout=60, check=True).stdout
        st = json.loads(out)["streams"][0]
        for key in ("avg_frame_rate", "r_frame_rate"):
            num, _, den = st.get(key, "").partition("/")
            if num and den and float(den) != 0:
                fps = float(num) / float(den)
                if fps > 0:
                    return fps
    except Exception as e:
        print(f"[fps] 検出できず: {e}")
    return None


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

    video_fps = _probe_fps(src)
    print(f"[fps] 動画から検出: {video_fps}")

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

    # 関節の「回転」も返す（アバターへのリターゲット用。issue #5）。
    # 位置は体格差で破綻するが、回転なら体型の違うキャラにも移せる。
    #   global_orient (F,3) 体全体の向き / body_pose (F,63)=21関節×3 / transl (F,3)
    pose = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
            for k, v in g.items()}

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
    print(f"joints {joints.shape}  pose {{{', '.join(f'{k}:{tuple(v.shape)}' for k, v in pose.items())}}}")
    print(f"[TIMING] {elapsed:.1f}s ≒ ${elapsed * 0.000164:.4f} (T4, 起動分は別)")
    return joints, pose, renders, video_fps


@app.function(image=image, gpu="T4", volumes={ASSETS: vol}, timeout=900)
def reconstruct(video_bytes: bytes, name: str) -> dict:
    """CLI用。関節・回転・レンダ動画を返す（`modal run` → ローカル保存）。"""
    import io

    import numpy as np

    joints, pose, renders, video_fps = _gvhmr_joints(video_bytes, name)

    jbuf = io.BytesIO()
    np.save(jbuf, joints)
    pbuf = io.BytesIO()
    np.savez(pbuf, **pose)  # アバター用の回転（issue #5）
    return {"gv_joints.npy": jbuf.getvalue(),
            "gv_pose.npz": pbuf.getvalue(),
            "_fps": str(video_fps or ""), **renders}


@app.function(image=image, gpu="T4", volumes={ASSETS: vol}, timeout=900)
def run_job(video_bytes: bytes, name: str, fps: float) -> dict:
    """Web用。復元 → analysis で指標・フィードバックを導出し JSON で返す。

    「現実+3Dメッシュの重ね合わせ」動画(incam)も base64 で同梱する
    （フォームの当てはまりを目で確認できるため）。
    アバター用の回転(pose)も載せる（issue #5）。指標の導出には使わない。
    """
    import base64

    import analysis

    joints, pose, renders, video_fps = _gvhmr_joints(video_bytes, name)
    # 申告 fps より、動画から読めた値を優先する（取り違えを防ぐ）。
    # ただしスローモーション動画では実撮影レートの方が高いため、
    # 申告値が検出値より大きい場合は申告を尊重する。
    used = float(fps) if (video_fps is None or float(fps) > video_fps + 1) else video_fps
    res = analysis.analyze_json(joints, used)
    res["video_fps"] = video_fps
    res["fps_used"] = used

    # 回転はアバター表示専用。analysis（指標）は関節位置のみで完結させる。
    res["pose"] = {k: v.round(5).tolist() for k, v in pose.items()}

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
    detected = results.pop("_fps", "")
    for fname, content in results.items():
        (out_dir / fname).write_bytes(content)
        print(f"✅ {out_dir / fname} ({len(content) / 1e6:.1f} MB)")

    fps = f"{float(detected):.0f}" if detected else "<実fps>"
    if detected:
        print(f"\n[fps] 動画から検出: {float(detected):.2f}")
        print("      スローモーションとして保存された動画なら、実際の撮影レートは"
              "これより高い。その場合は手で指定すること。")
    print(f"→ 次: python -m analysis --joints {out_dir}/gv_joints.npy "
          f"--fps {fps} --save {out_dir}")
