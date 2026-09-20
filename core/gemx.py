"""Decode GEM-X with the exact SOMA adapter used by its official renderer.

Heavy dependencies are imported only in the GPU environment. No guesses about
scale layout, bind pose, or output units are made here.
"""
from __future__ import annotations

import numpy as np


def decode_prediction(pred: dict, device: str = "cuda") -> tuple[dict, dict]:
    import torch
    from gem.utils.soma_utils.soma_layer import SomaLayer

    soma = SomaLayer(data_root="inputs/soma_assets", low_lod=True, device=device,
                     identity_model_type="mhr", mode="warp")
    joints, poses = {}, {}
    for space in ("global", "incam"):
        params = pred[f"body_params_{space}"]
        with torch.no_grad():
            result = soma(**{k: v.to(device) if torch.is_tensor(v) else v
                             for k, v in params.items()})
        J = result["joints"].detach().cpu().numpy()
        if J.ndim != 3 or J.shape[1:] != (77, 3) or not np.isfinite(J).all():
            raise ValueError(f"Invalid official SOMA output: {space}, {J.shape}")
        joints[space] = J
        poses[space] = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                        for k, v in params.items()}
    if joints["global"].shape != joints["incam"].shape:
        raise ValueError("GEM-X camera and world frames differ")
    return joints, poses
