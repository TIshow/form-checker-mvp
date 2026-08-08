"""TRAM (MIT) を Modal のサーバーレスGPUで動かす（issue #9）。

GVHMR 版・GEM-X 版と**並行して**置く。app 名も Volume も分けてあるので3つ同時に動く。

    modal run backend/reconstruct_tram.py::fetch_checkpoints   # 初回のみ
    modal run backend/reconstruct_tram.py --video X.mp4 --out output_tram
    modal run backend/reconstruct_tram.py --video X.mp4 --out output_tram --start 4.8 --end 10.3

## なぜ TRAM を評価するのか

GVHMR は品質は良いが**非商用**。GEM-X は商用可だが、実測で**サーブの動作が壊れる**
（ラケットドロップが消える）。TRAM は **MIT** で、しかも GENMO 論文 Table 1 の
EMDB 世界座標評価では GVHMR を上回る:

    WA-MPJPE₁₀₀   TRAM 76.4  <  GVHMR 111.0  <  WHAM 135.6
    W-MPJPE₁₀₀    TRAM 222.4 <  GVHMR 276.5
    RTE %         TRAM 1.4   <  GVHMR 2.0

残る非商用の部品は SMPL だけで、これは Meshcapade から**買える**。
「大学研究室の裁量」だった GVHMR と違い、可否ではなく金額の問題になる。
経緯は docs/issues/009-licensing-for-productization.md。

## 下流にとって嬉しいこと

**TRAM は SMPL の24関節をそのまま返す。** GEM-X のような並べ替え
（analysis/soma.py）が要らず、解析層・ビューア・アバターがそのまま動く。

## 3段構成

TRAM は処理を3つのスクリプトに分けている。ここでは前半2つだけ動かす。

    1. estimate_camera.py … DROID-SLAM でカメラ運動 + 人物の検出・追跡
    2. estimate_humans.py … VIMO で体の動き（カメラ空間）
    3. visualize_tram.py  … 合成して描画 ← **使わない**

3段目は描画のためだけに pytorch3d を要求する（CUDA ビルドで20〜40分）。
関節が欲しいだけなので、合成の計算（カメラ姿勢で回して足す）だけ自前でやり、
pytorch3d を丸ごと外す。lib/pipeline/__init__.py がその import を持っているので
イメージ構築時に落としている。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

TRAM = "/root/tram"
ASSETS = "/assets"
GVHMR_ASSETS = "/gvhmr"

# 再現性のため固定
TRAM_COMMIT = "4861c11"
DETECTRON2_COMMIT = "a59f05630a8f205756064244bf5beb8661f96180"

# ビルド機に GPU は無いので、対象アーキテクチャを明示しないと
# CUDA 拡張が「自分の GPU 向け」を決められない。T4/A100/A10G/L4 を含む。
ARCH = "7.5;8.0;8.6;8.9"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "wget", "curl", "ffmpeg", "build-essential", "unzip",
        "libsuitesparse-dev",              # install.sh の conda suitesparse に相当
        "libgl1-mesa-glx", "libglib2.0-0",  # OpenCV
    )
    .env({"TORCH_CUDA_ARCH_LIST": ARCH, "FORCE_CUDA": "1"})
    .run_commands(
        # submodule は lietorch / eigen / DEVA。いずれも HTTPS なので --recursive で通る
        f"git clone --recursive https://github.com/yufu-wang/tram {TRAM}",
        f"cd {TRAM} && git checkout {TRAM_COMMIT} && "
        f"git submodule update --init --recursive",
        "pip install -U pip setuptools wheel",
        # install.sh と同じ組み合わせ（CUDA 11.8 / torch 2.4.0）
        "pip install torch==2.4.0 torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu118",
        # numpy を先に固定する。後から入ると detectron2 等が別版に対して
        # ビルドされ、実行時に ABI で落ちる
        "pip install numpy==1.23.5",
        f"pip install 'git+https://github.com/facebookresearch/detectron2.git@{DETECTRON2_COMMIT}'",
        "pip install torch-scatter "
        "-f https://data.pyg.org/whl/torch-2.4.0+cu118.html",
        "pip install pytorch-lightning pulp supervision opencv-python loguru "
        "einops plyfile segment_anything scikit-image smplx timm==0.6.7 evo "
        "pytorch-minimize 'imageio[ffmpeg]' gdown openpyxl yacs matplotlib "
        "scipy tqdm pycocotools",
        # chumpy は setup.py が pip を要求するのでビルド分離を切る（GVHMR と同じ）
        "pip install --no-build-isolation git+https://github.com/mattloper/chumpy",
        # --- pytorch3d を外すための最小の改変 ---
        # 描画に使う lib/vis/renderer.py だけが pytorch3d を要求する。
        # lib/pipeline/__init__ が visualization を読み、visualization が
        # renderer を読むので、import の連鎖を1本切れば依存ごと消える。
        f"cd {TRAM} && sed -i '/from .visualization import visualize_tram/d' "
        f"lib/pipeline/__init__.py",
        f"cd {TRAM} && sed -i 's/, visualize_tram//' scripts/estimate_camera.py",
        f"cd {TRAM} && sed -i '/from lib.pipeline import visualize_tram/d' "
        f"scripts/estimate_humans.py",
        # DROID-SLAM の CUDA 拡張（TRAM 版はマスク対応の改造が入っている）
        f"cd {TRAM}/thirdparty/DROID-SLAM && python setup.py install",
    )
    .add_local_python_source("analysis")
)

app = modal.App("tram-reconstruct")
vol = modal.Volume.from_name("tram-assets", create_if_missing=True)
# SMPL は GVHMR 用に取得済みのものを使い回す。ライセンス上、再配布はできないので
# 同じ Volume を読むのが正しい（新たにダウンロードもしない）。
gvhmr_vol = modal.Volume.from_name("gvhmr-assets", create_if_missing=True)

# Google Drive の3つは scripts/download_models.sh と同じもの。
# GVHMR のときクォータ制限で落ちた経路なので、失敗したら手元で落として
# `modal volume put tram-assets <file> /pretrain/<name>` で入れる。
GDRIVE = [
    ("droid.pth", "1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh"),
    ("camcalib_sa_biased_l2.ckpt", "1t4tO0OM5s8XDvAzPW-5HaOkQuV3dHBdO"),
    ("vimo_checkpoint.pth.tar", "1fdeUxn_hK4ERGFwuksFpV_-_PHZJuoiW"),
]
DIRECT = [
    ("DEVA-propagation.pth",
     "https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/"
     "download/v1.0/DEVA-propagation.pth"),
    ("sam_vit_h_4b8939.pth",
     "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"),
]


@app.function(image=image, volumes={ASSETS: vol}, timeout=3600)
def fetch_checkpoints():
    """重みを Volume に取得（初回のみ）。SMPL は gvhmr-assets を使うので触らない。"""
    import os

    dest = f"{ASSETS}/pretrain"
    os.makedirs(dest, exist_ok=True)

    for name, url in DIRECT:
        p = f"{dest}/{name}"
        if os.path.exists(p):
            print(f"既にあります: {name}")
            continue
        subprocess.run(["wget", "-q", "-O", p, url], check=True)
        print(f"✅ {name}  ({os.path.getsize(p) / 1e6:.0f} MB)")

    failed = []
    for name, fid in GDRIVE:
        p = f"{dest}/{name}"
        if os.path.exists(p):
            print(f"既にあります: {name}")
            continue
        r = subprocess.run(["gdown", "--fuzzy", "-O", p,
                            f"https://drive.google.com/file/d/{fid}/view"],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(p):
            failed.append(name)
            print(f"❌ {name}: Google Drive から取得できませんでした\n{r.stderr[-400:]}")
        else:
            print(f"✅ {name}  ({os.path.getsize(p) / 1e6:.0f} MB)")

    vol.commit()
    if failed:
        print("\n手元で落として Volume に入れてください:")
        for name in failed:
            fid = dict(GDRIVE)[name]
            print(f"  https://drive.google.com/file/d/{fid}/view")
            print(f"  modal volume put tram-assets {name} /pretrain/{name}")
        raise SystemExit(f"{len(failed)} 件が未取得です")


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


def _link_assets() -> None:
    """Volume の重みと SMPL を、TRAM が探す位置へ繋ぐ。"""
    import os

    d = Path(TRAM) / "data"
    (d / "pretrain").mkdir(parents=True, exist_ok=True)
    (d / "smpl").mkdir(parents=True, exist_ok=True)

    for name, _ in DIRECT + [(n, None) for n, _ in GDRIVE]:
        src = f"{ASSETS}/pretrain/{name}"
        dst = d / "pretrain" / name
        if os.path.exists(src) and not dst.exists():
            os.symlink(src, dst)

    smpl = f"{GVHMR_ASSETS}/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"
    dst = d / "smpl" / "SMPL_NEUTRAL.pkl"
    if os.path.exists(smpl) and not dst.exists():
        os.symlink(smpl, dst)
    print(f"[assets] SMPL: {'あり' if dst.exists() else '見つかりません'}")


@app.function(image=image, gpu="L4",
              volumes={ASSETS: vol, GVHMR_ASSETS: gvhmr_vol}, timeout=3600)
def reconstruct(video_bytes: bytes, name: str,
                start: float | None = None, end: float | None = None,
                static_cam: bool = True) -> dict:
    """動画1本を TRAM で復元し、世界座標とカメラ空間の24関節を返す。

    static_cam: True で `--static_camera`。GVHMR/GEM-X と条件を揃えるとき用。
    False にすると DROID-SLAM がカメラ運動を推定する（TRAM の本来の売り）。
    """
    import os

    import numpy as np
    import torch

    os.chdir(TRAM)
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
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-an", trimmed]
        subprocess.run(cmd, check=True, capture_output=True)
        src = trimmed
        print(f"[trim] {start}〜{end} 秒を切り出しました")

    stem = Path(src).stem     # 出力先は入力ファイル名で決まる
    video_fps = _probe_fps(src)
    print(f"[fps] 動画から検出: {video_fps}")

    cam_cmd = ["python", "scripts/estimate_camera.py", f"--video={src}"]
    if static_cam:
        cam_cmd.append("--static_camera")
    subprocess.run(cam_cmd, check=True)
    subprocess.run(["python", "scripts/estimate_humans.py", f"--video={src}"],
                   check=True)

    # --- カメラと体の動きを合成して世界座標へ ---
    # scripts/visualize_tram.py がやっていることのうち、描画を除いた部分。
    seq = f"results/{stem}"
    cam = np.load(f"{seq}/camera.npy", allow_pickle=True).item()
    tracks = sorted(Path(f"{seq}/hps").glob("hps_track_*.npy"))
    if not tracks:
        raise RuntimeError("人物が検出されませんでした")
    # estimate_humans.py は追跡の長い順に番号を振る。0 が主対象。
    print(f"[tracks] {len(tracks)} 人ぶん。track 0 を使います")

    from lib.models.smpl import SMPL
    from lib.vis.traj import traj_filter

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    smpl = SMPL().to(dev)

    hps = np.load(tracks[0], allow_pickle=True).item()
    rotmat = hps["pred_rotmat"].to(dev)
    shape = hps["pred_shape"].to(dev)
    trans = hps["pred_trans"].to(dev)
    frame = hps["frame"]
    # 体型はフレームごとに揺れるので平均で固定する（TRAM 本体と同じ扱い）
    shape = shape.mean(dim=0, keepdim=True).repeat(len(shape), 1)

    with torch.no_grad():
        pred = smpl(body_pose=rotmat[:, 1:], global_orient=rotmat[:, [0]],
                    betas=shape, transl=trans.squeeze(),
                    pose2rot=False, default_smpl=True)
    j3d_cam = pred.joints[:, :24]          # SMPL の24関節。並べ替え不要

    R = torch.tensor(cam["world_cam_R"]).to(dev)[frame]
    T = torch.tensor(cam["world_cam_T"]).to(dev)[frame]
    j3d_w = torch.einsum("bij,bnj->bni", R, j3d_cam) + T[:, None]
    v_w = torch.einsum("bij,bnj->bni", R, pred.vertices) + T[:, None]
    _, j3d_w = traj_filter(v_w.cpu(), j3d_w.cpu())

    print(f"[joints] 世界座標 {tuple(j3d_w.shape)} / カメラ空間 {tuple(j3d_cam.shape)}")
    return {
        "joints_world": j3d_w.cpu().numpy(),
        "joints_incam": j3d_cam.cpu().numpy(),
        "frame": np.asarray(frame),
        "pose": {"pred_rotmat": rotmat.cpu().numpy(),
                 "pred_shape": shape.cpu().numpy(),
                 "pred_trans": trans.cpu().numpy()},
        "video_fps": video_fps,
        "n_tracks": len(tracks),
    }


@app.local_entrypoint()
def main(video: str, out: str = "output_tram",
         start: float | None = None, end: float | None = None,
         moving_cam: bool = False):
    """ローカルの動画を Modal で復元し、結果をローカルへ保存する。

      tr_joints.npy       世界座標の24関節（`python -m analysis` がそのまま食える）
      tr_joints_incam.npy カメラ空間の24関節（issue #8 の切り分け用）
      tr_pose.npz         SMPL の回転（アバターへのリターゲット用）
    """
    import numpy as np

    video_path = Path(video)
    data = video_path.read_bytes()
    print(f"送信: {video_path} ({len(data) / 1e6:.1f} MB) → Modal GPU で復元中…")

    r = reconstruct.remote(data, video_path.name, start, end, not moving_cam)

    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "tr_joints.npy", r["joints_world"])
    np.save(d / "tr_joints_incam.npy", r["joints_incam"])
    np.savez(d / "tr_pose.npz", frame=r["frame"], **r["pose"])

    for f in ("tr_joints.npy", "tr_joints_incam.npy", "tr_pose.npz"):
        p = d / f
        print(f"✅ {p} ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"\n検出fps: {r['video_fps']} / 検出人数: {r['n_tracks']}")
    print(f"次: python -m analysis --joints {d}/tr_joints.npy "
          f"--fps <実撮影レート> --save {d}")
