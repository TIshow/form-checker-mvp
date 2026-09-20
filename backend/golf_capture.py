"""Export cached GEM-X through its official SOMA layer and recover 2D evidence.

No GVHMR data is used. Run on the exact decoded video used for the cached job:
  .venv/bin/modal run backend/golf_capture.py --video output_golf_gemx/temp_golf.mp4
"""
from pathlib import Path
import modal
from backend.reconstruct_gemx import ASSETS, GEMX, GEMX_COMMIT, image, vol

image = image.add_local_python_source("backend")

app = modal.App("golf-accuracy")


@app.function(image=image, gpu="L4", volumes={ASSETS: vol}, timeout=1200)
def capture(video_bytes: bytes, prediction: str) -> dict:
    import hashlib
    import io
    import os
    import sys
    import numpy as np
    import torch

    os.chdir(GEMX)
    sys.path.insert(0, GEMX)
    from gem.utils.soma_utils.soma_layer import SomaLayer
    from gem.utils.vitpose_extractor import VitPoseExtractor
    from gem.utils.yolox_detector import YOLOXDetector, detect_and_track
    from gem.utils.kp2d_utils import smooth_bbx_xyxy
    from gem.utils.geo_transform import get_bbx_xys_from_xyxy
    from gem.utils.cam_utils import estimate_K
    from gem.utils.video_io_utils import read_video_np
    from core.convert import to_smpl24

    if Path(prediction).name != prediction:
        raise ValueError("prediction must be a filename in the existing debug directory")
    pred = torch.load(f"{ASSETS}/debug/{prediction}", map_location="cpu", weights_only=False)
    soma = SomaLayer(data_root="inputs/soma_assets", low_lod=True, device="cuda",
                     identity_model_type="mhr", mode="warp")
    data = {}
    for space in ("global", "incam"):
        params = pred[f"body_params_{space}"]
        with torch.no_grad():
            out = soma(**{k: v.cuda() if torch.is_tensor(v) else v for k,v in params.items()})
        native = out["joints"].cpu().numpy()
        data[f"joints_{space}_77"] = native
        data[f"joints_{space}"] = to_smpl24(native)
        print(space, native.shape)
    parents = soma.parents
    data["parents_77"] = parents.cpu().numpy() if torch.is_tensor(parents) else np.asarray(parents)
    Path("/tmp/golf_evidence.mp4").write_bytes(video_bytes)
    frames = read_video_np("/tmp/golf_evidence.mp4")
    if len(frames) != len(data["joints_global"]):
        raise ValueError("Decoded video and cached prediction frame counts differ")
    boxes, track_ids = detect_and_track(frames, YOLOXDetector(device="cuda"))
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    boxes = smooth_bbx_xyxy(boxes, window=5)
    H, W = frames.shape[1:3]
    boxes[:,[0,2]] = boxes[:,[0,2]].clamp(0,W-1)
    boxes[:,[1,3]] = boxes[:,[1,3]].clamp(0,H-1)
    bbx_xys = get_bbx_xys_from_xyxy(boxes, base_enlarge=1.2).float()
    kp = VitPoseExtractor(device="cuda:0", pose_type="soma").extract(frames, bbx_xys)
    if isinstance(kp, tuple):
        kp = kp[0]
    data["keypoints_2d_77"] = kp.cpu().numpy()
    data["keypoints_2d"] = to_smpl24(data["keypoints_2d_77"])
    data["boxes_xyxy"] = boxes.numpy()
    data["track_ids"] = np.asarray(track_ids)
    data["K"] = estimate_K(W,H).cpu().numpy()
    data["image_size_wh"] = np.array([W,H])
    data["frame_ids"] = np.arange(len(frames))
    data["gemx_commit"] = np.array(GEMX_COMMIT)
    data["video_sha256"] = np.array(hashlib.sha256(video_bytes).hexdigest())
    data["camera_intrinsics_source"] = np.array("GEM-X default estimate_K; uncalibrated")
    buf = io.BytesIO()
    np.savez_compressed(buf, **data)
    return {"evidence.npz": buf.getvalue()}


@app.local_entrypoint()
def main(video: str, prediction: str = "temp_golf_hpe_results.pt",
         out: str = "output_golf_accuracy"):
    results = capture.remote(Path(video).read_bytes(), prediction)
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    for name, blob in results.items():
        (dest/name).write_bytes(blob)
        print(dest/name)
