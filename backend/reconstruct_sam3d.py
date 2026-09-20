"""SAM 3D Body + MHR を Modal のサーバーレスGPUで動かす（issue #11）。

    modal run backend/reconstruct_sam3d.py::explore                    # 初回。APIを調べる
    modal run backend/reconstruct_sam3d.py --video videos/baseball/x.mp4 --out output_s3

GVHMR 版・GEM-X 版と**並行して**置く。既存経路は一切変えない。

## なぜ評価するのか

GVHMR も SMPL も非商用で、収益化の出口が無い（[009]）。SAM 3D Body は

  コード・重み   SAM License（royalty-free で商用可。MAU の閾値なし）
  人体モデル MHR  **Apache-2.0**（SMPL を必要としない）

で、ライセンスだけ見れば今まで調べた中で最も条件が良い。

## GEM-X のイメージをそのまま使う

**ビルド定義を GEM-X 版と1文字も変えないこと。** Modal は定義が同じなら
ビルド済みレイヤを再利用するので、ここは数秒で立ち上がる。GEM-X は
`third_party/sam-3d-body` を submodule で持っており、重みも
`nvidia/GEM-X` から `sam3d_body.ckpt` / `model_config.yaml` / `mhr_model.pt`
を落としてあるので、**既に全部揃っている**。

重みを NVIDIA 経由で取ることには意味がある。Meta の申請制リポジトリを
通さずに済むうえ、NVIDIA が商用ライセンスの製品に同梱して再配布している
という事実が残る。ただし**これは学習データの素性を保証しない。**
金を受け取る前に Meta に確認すること（[011] §4 の第3層）。

## GVHMR との決定的な違い

**単一画像モデルで、時系列を持たない。** フレームごとに独立に推定するので、

- 世界座標が無い。**カメラ空間の3D関節だけ**が出る
  （三脚前提なら固定の回転1回で世界座標になる。[011] §2）
- フレーム間のジッタがある。角速度の微分で増幅されるので、
  キネティックチェーンの順序判定に効く。**これが唯一の構造的弱点**

## 関節

MHR の 308点のうち先頭70点が体（COCO式の名前）。`core/convert.py` の
`mhr70_to_smpl24()` で SMPL 24関節に落とす。**骨盤と脊椎は MHR に無いので
導出している**——SOMA のときのような単なる並べ替えではない。

[009]: docs/issues/009-licensing-for-productization.md
[011]: docs/issues/011-commercial-architecture.md
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

GEMX = "/root/GEM-X"
SAM3D = f"{GEMX}/third_party/sam-3d-body"
ASSETS = "/assets"

GEMX_COMMIT = "3299255"

# ---------------------------------------------------------------------------
# ★ ここから下の image 定義は backend/reconstruct_gemx.py と同一にしておく。
#    1文字でも違うと Modal が別イメージとして丸ごとビルドし直す（20分以上）。
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "git-lfs", "wget", "curl", "ffmpeg", "build-essential",
        "libegl1-mesa-dev", "libgl1-mesa-glx", "libglib2.0-0", "xvfb",
    )
    .run_commands(
        "git lfs install",
        f"git clone https://github.com/NVlabs/GEM-X {GEMX}",
        f"cd {GEMX} && git checkout {GEMX_COMMIT} && "
        f"git submodule update --init third_party/soma third_party/sam-3d-body",
        "pip install -U pip setuptools wheel uv",
        "pip install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu126",
        f"cd {GEMX} && pip install -e third_party/soma",
        f"cd {GEMX}/third_party/soma && git lfs pull",
        f"cd {GEMX} && CC=gcc CXX=g++ pip install -e .",
        "CC=gcc CXX=g++ pip install cloudpickle fvcore iopath pycocotools "
        "braceexpand roma 'setuptools<75'",
        "pip install onnxruntime open3d",
        f"cd {GEMX} && mkdir -p inputs && "
        f"ln -sfn {GEMX}/third_party/soma/assets inputs/soma_assets",
    )
    .env({"PYOPENGL_PLATFORM": "egl", "EGL_PLATFORM": "surfaceless"})
    .add_local_python_source("analysis", "core", "domains")
)

app = modal.App("sam3d-reconstruct")
vol = modal.Volume.from_name("gemx-assets", create_if_missing=True)

CKPT = f"{ASSETS}/checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt"
MHR = f"{ASSETS}/mhr_data/mhr_model.pt"


# ---------------------------------------------------------------------------
# 調査用。上流の API は変わるので、推論を書く前にこれで確かめる
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={ASSETS: vol}, timeout=900)
def explore() -> str:
    """sam-3d-body の API と、Volume の重みの在処を報告する（CPU）。"""
    import os
    import sys

    lines = []

    def say(s):
        print(s)
        lines.append(str(s))

    say("=== Volume の中身 ===")
    for root, _, files in os.walk(ASSETS):
        for f in files:
            p = os.path.join(root, f)
            say(f"  {p}  ({os.path.getsize(p) / 1e6:.0f} MB)")

    say(f"\n=== {SAM3D} ===")
    say(f"  存在: {os.path.isdir(SAM3D)}")
    sys.path.insert(0, SAM3D)

    say("\n=== mhr70 の並びが core/convert.py と一致するか ===")
    try:
        from sam_3d_body.metadata.mhr70 import mhr_names

        from core.convert import verify_mhr_names
        bad = verify_mhr_names(list(mhr_names))
        say(f"  {'✅ 一致' if not bad else bad}")
    except Exception as e:
        say(f"  ✗ {type(e).__name__}: {e}")

    say("\n=== import できるか ===")
    for mod in ("sam_3d_body", "tools.build_detector", "tools.build_sam",
                "tools.build_fov_estimator"):
        try:
            __import__(mod)
            say(f"  ✅ {mod}")
        except Exception as e:
            say(f"  ✗ {mod}: {type(e).__name__}: {e}")

    say("\n=== GEM-X 側の人物検出（detectron2 を避けるため流用したい）===")
    for mod in ("gem.utils.yolox_detector",):
        try:
            m = __import__(mod, fromlist=["*"])
            say(f"  ✅ {mod}: {[n for n in dir(m) if not n.startswith('_')][:12]}")
        except Exception as e:
            say(f"  ✗ {mod}: {type(e).__name__}: {e}")

    return "\n".join(lines)


def _probe_fps(path: str) -> float | None:
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


@app.function(image=image, gpu="L4", volumes={ASSETS: vol}, timeout=3600)
def reconstruct(video_bytes: bytes, name: str,
                start: float | None = None, end: float | None = None) -> dict:
    """動画を1フレームずつ SAM 3D Body に通し、SMPL24 の関節列を返す。

    出力は**カメラ空間**。世界座標は持たない（単一画像モデルなので当然）。
    三脚で撮った素材なら、床を1回指定すれば世界座標になる。
    """
    import io
    import os
    import sys

    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, SAM3D)
    os.chdir(SAM3D)

    stem = Path(name).stem
    src = f"/tmp/{stem}.mp4"
    Path(src).write_bytes(video_bytes)

    if start is not None or end is not None:
        trimmed = f"/tmp/{stem}_trim.mp4"
        cmd = ["ffmpeg", "-y", "-i", src]
        if start is not None:
            cmd += ["-ss", str(start)]
        if end is not None:
            cmd += ["-to", str(end)]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-an", trimmed]
        subprocess.run(cmd, check=True, capture_output=True)
        src = trimmed
        print(f"[trim] {start}〜{end} 秒を切り出しました")

    video_fps = _probe_fps(src)

    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from sam_3d_body.metadata.mhr70 import mhr_names

    from core.convert import mhr70_to_smpl24, verify_mhr_names

    bad = verify_mhr_names(list(mhr_names))
    if bad:
        raise RuntimeError(
            "MHR のキーポイントの並びが core/convert.py の表と食い違っています。"
            f"添字を直すまで進めてはいけません（それらしい数字が出てしまう）: {bad}")

    device = torch.device("cuda")
    model, model_cfg = load_sam_3d_body(CKPT, device=device, mhr_path=MHR)
    # 検出器は付けない。detectron2(vitdet) はイメージに入れていない。
    # 代わりに、下で「前フレームの人物を追う」自前の箱を渡す。
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model, model_cfg=model_cfg,
        human_detector=None, human_segmentor=None, fov_estimator=None,
    )

    cap = cv2.VideoCapture(src)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"フレームを読めませんでした: {src}")
    H, W = frames[0].shape[:2]
    print(f"[read] {len(frames)} フレーム  {W}x{H}  fps={video_fps}")

    # --- 被写体の箱をフレームごとに取る -------------------------------
    # SAM 3D Body 同梱の検出器は vitdet（detectron2）だが、イメージに
    # 入れていない。代わりに GEM-X 側の YOLOX(ONNX, Apache-2.0) を使う。
    # ライセンスも軽さも都合が良く、同じイメージに既に入っている。
    #
    # 「動いた画素の外接矩形」で代用しようとして失敗した記録を残す:
    # カメラがわずかでもパンすると**全画素が動く**ので、箱が毎回全画面に
    # なる。静止カメラでしか使えない手で、素材を選ばない方法ではなかった。
    from gem.utils.yolox_detector import YOLOXDetector, detect_and_track

    rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
    detector = YOLOXDetector(device="cuda")
    boxes, track_ids = detect_and_track(rgb, detector)
    lost = int((np.asarray(track_ids) < 0).sum())
    print(f"[box] 検出 {len(boxes)} フレーム  追跡できず {lost}")

    # 追跡が切れたフレームは直前の箱で埋める（欠けたまま渡すと全画面になる）
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    last = None
    for i in range(len(boxes)):
        if track_ids[i] < 0 or not np.isfinite(boxes[i]).all():
            if last is not None:
                boxes[i] = last
        else:
            last = boxes[i]
    print(f"[box] 先頭 {boxes[0].round().tolist()}  末尾 {boxes[-1].round().tolist()}")

    # **RGB で渡すこと。** numpy を渡すと process_one_image は中身を RGB と
    # みなす（パスを渡したときだけ BGR として読む）。cv2 は BGR なので、
    # そのまま渡すと赤と青が入れ替わった画像を推定させることになる。
    # 落ちも警告もせず、黙って精度だけ下がる。
    kps, missing = [], []
    for i in range(len(rgb)):
        out = estimator.process_one_image(
            rgb[i], bboxes=boxes[i].reshape(1, 4), bbox_thr=0.0, use_mask=False)
        if not out:
            missing.append(i)
            kps.append(kps[-1] if kps else np.zeros((len(mhr_names), 3)))
            continue
        d = out[0]
        k = np.asarray(d["pred_keypoints_3d"], dtype=float)
        # カメラ空間へ。pred_keypoints_3d は根基準なので平行移動を足す
        k = k + np.asarray(d["pred_cam_t"], dtype=float).reshape(1, 3)
        kps.append(k)
        if i % 20 == 0:
            print(f"  {i}/{len(rgb)}")

    kp = np.stack(kps)
    print(f"[out] keypoints {kp.shape}  推定できなかったフレーム {len(missing)}")
    joints = mhr70_to_smpl24(kp)

    jbuf = io.BytesIO()
    np.save(jbuf, joints)
    kbuf = io.BytesIO()
    np.save(kbuf, kp)          # 手と顔を含む生のキーポイント（issue 006 用）
    bbuf = io.BytesIO()
    # 画像上の箱 (F,4) xyxy。生座標の x と画像の x の向きが同じかを検算できる
    # （骨盤の x の増減と箱の中心の増減が同符号なら OpenCV 慣習で合っている）。
    np.save(bbuf, boxes.astype(np.float32))
    return {
        "s3_joints.npy": jbuf.getvalue(),
        "s3_mhr_keypoints.npy": kbuf.getvalue(),
        "s3_boxes.npy": bbuf.getvalue(),
        "_fps": str(video_fps or ""),
        "_missing": str(len(missing)),
        "_lost": str(lost),
    }


@app.local_entrypoint()
def main(video: str, out: str = "output_s3",
         start: float | None = None, end: float | None = None):
    video_path = Path(video)
    data = video_path.read_bytes()
    print(f"送信: {video_path} ({len(data) / 1e6:.1f} MB) → Modal GPU で復元中…")

    results = reconstruct.remote(data, video_path.name, start, end)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    detected = results.pop("_fps", "")
    missing = results.pop("_missing", "0")
    lost = results.pop("_lost", "0")
    for fname, content in results.items():
        (out_dir / fname).write_bytes(content)
        print(f"✅ {out_dir / fname} ({len(content) / 1e6:.1f} MB)")

    if int(lost or 0):
        print(f"\n⚠️ {lost} フレームで人物を追跡できず、直前の箱で代用しました")
    if int(missing or 0):
        print(f"\n⚠️ {missing} フレームで人物を推定できず、直前の値で埋めました")
    fps = f"{float(detected):.0f}" if detected else "<実fps>"
    print(f"\n→ 次: python -m analysis --joints {out_dir}/s3_joints.npy "
          f"--fps {fps} --domain baseball_pitch")
    print("   注意: これは**カメラ空間**の座標です。世界座標ではありません")
