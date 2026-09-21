"""Observation-assisted refinement with fixed limb lengths.

This reduces reprojection error; it cannot establish monocular depth accuracy.
Uses no reference 3D model or commercial-incompatible teacher. Optional SciPy is
loaded only on invocation. Inputs/outputs are in camera coordinates, meters.

This experiment was REJECTED on the golf clip after visual review. Preserving
bone lengths and improving 2D agreement did not preserve plausible 3D pose.
The returned `accepted` flag concerns the optimizer objective only, never
product adoption or validated pose accuracy. See docs/issues/016.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

PARENTS = np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21])
MOVABLE = np.array([4,5,7,8,16,17,18,19,20,21])
# Spine interpolation and hand/toe definitions are not reliable 2D observations.
OBSERVED = np.array([1,2,4,5,7,8,16,17,18,19,20,21])


@dataclass(frozen=True)
class RefinementConfig:
    pixel_sigma: float = 8.0
    position_sigma_m: float = 0.08
    root_sigma_m: float = 0.04
    temporal_sigma_m: float = 0.012
    confidence_min: float = 0.3
    max_nfev: int = 60


def project(joints: np.ndarray, K: np.ndarray) -> np.ndarray:
    J = np.asarray(joints, dtype=float)
    matrix = np.asarray(K, dtype=float)
    p = np.einsum("ij,...j->...i", matrix, J)
    return p[..., :2] / np.maximum(p[..., 2:3], 1e-6)


def fit_pose(joints: np.ndarray, keypoints: np.ndarray, K: np.ndarray,
             fps: float, *, config: RefinementConfig | None = None,
             fit_frames: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Fit root translation and limb directions; held-out frames get no 2D term.

    Confidence is the detector's score, not a calibrated probability. Missing
    observations remain missing. Only limb directions and a common translation
    change; lengths are held at their input sequence medians by construction.
    A prior on the correction, rather than raw motion, preserves fast events.
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    cfg = config or RefinementConfig()
    J = np.asarray(joints, dtype=float)
    kp = np.asarray(keypoints, dtype=float)
    K = np.asarray(K, dtype=float)
    if J.ndim != 3 or J.shape[1:] != (24,3) or len(J) < 3:
        raise ValueError("Need at least three frames of (F,24,3) joints")
    if kp.shape != J.shape or K.shape != (3,3):
        raise ValueError("Keypoints must be (F,24,3) xy-score, K must be (3,3)")
    if not np.isfinite(J).all() or not np.isfinite(K).all() or (J[...,2] <= 0).any():
        raise ValueError("Need finite joints in front of a positive-Z camera")
    if not np.isfinite(fps) or fps <= 0 or K[0,0] <= 0 or K[1,1] <= 0:
        raise ValueError("Invalid fps or intrinsics")
    if any(not np.isfinite(x) or x <= 0 for x in
           [cfg.pixel_sigma,cfg.position_sigma_m,cfg.root_sigma_m,cfg.temporal_sigma_m]):
        raise ValueError("Residual scales must be positive and finite")
    F = len(J); N = 3 + len(MOVABLE)*3
    fit = np.ones(F,dtype=bool) if fit_frames is None else np.asarray(fit_frames,dtype=bool)
    if fit.shape != (F,):
        raise ValueError("fit_frames must have one flag per frame")
    obs = kp[:,OBSERVED]
    valid = np.isfinite(obs).all(-1) & (obs[...,2] >= cfg.confidence_min)
    score = np.clip(np.nan_to_num(obs[...,2],nan=0,posinf=0,neginf=0),0,1)
    weight = np.sqrt(score)*valid*fit[:,None]
    targets = np.nan_to_num(obs[...,:2],nan=0,posinf=0,neginf=0)
    offsets = J.copy()
    offsets[:,1:] -= J[:,PARENTS[1:]]
    lengths = np.median(np.linalg.norm(offsets[:,MOVABLE],axis=-1),axis=0)
    if (lengths < 1e-4).any():
        raise ValueError("Zero-length limb")
    dirs = offsets[:,MOVABLE]/np.maximum(np.linalg.norm(offsets[:,MOVABLE],axis=-1,keepdims=True),1e-9)
    x0 = np.concatenate([J[:,0],dirs.reshape(F,-1)],axis=1)
    slot = {int(j):i for i,j in enumerate(MOVABLE)}

    def decode(x):
        X=x.reshape(F,N); out=np.empty_like(J); out[:,0]=X[:,:3]
        D=X[:,3:].reshape(F,len(MOVABLE),3)
        D=D/np.maximum(np.linalg.norm(D,axis=-1,keepdims=True),1e-9)
        for j in range(1,24):
            delta=D[:,slot[j]]*lengths[slot[j]] if j in slot else offsets[:,j]
            out[:,j]=out[:,PARENTS[j]]+delta
        return out

    def residual(x):
        X=x.reshape(F,N); out=decode(x); correction=out-J
        terms=[((project(out[:,OBSERVED],K)-targets)*weight[...,None]/cfg.pixel_sigma).ravel(),
               (correction[:,MOVABLE]/cfg.position_sigma_m).ravel(),
               ((X[:,:3]-J[:,0])/cfg.root_sigma_m).ravel(),
               ((np.linalg.norm(X[:,3:].reshape(F,-1,3),axis=-1)-1)*0.2).ravel(),
               (np.maximum(0.05-out[:,OBSERVED,2],0)/0.005).ravel()]
        # Normalize acceleration of the correction to a 30fps sampling interval.
        terms.append((np.diff(correction[:,MOVABLE],n=2,axis=0)*(fps/30)**2/cfg.temporal_sigma_m).ravel())
        return np.concatenate(terms)

    # All same-frame residuals depend on that frame's small parameter block;
    # the final temporal block spans exactly three frames. This also handles
    # downstream wrists/hands in forward kinematics without a dense Jacobian.
    counts=[len(OBSERVED)*2,len(MOVABLE)*3,3,len(MOVABLE),len(OBSERVED)]
    rows=F*sum(counts)+(F-2)*len(MOVABLE)*3
    sparsity=lil_matrix((rows,F*N),dtype=int);r=0
    for count in counts:
        for f in range(F):
            sparsity[r:r+count,f*N:(f+1)*N]=1;r+=count
    for f in range(F-2):
        count=len(MOVABLE)*3
        sparsity[r:r+count,f*N:(f+3)*N]=1;r+=count
    if not weight.any():
        return J.copy(), {"status":"no_observations", "accepted":False, "config":asdict(cfg)}
    result=least_squares(residual,x0.ravel(),jac_sparsity=sparsity.tocsr(),
                         loss="soft_l1",f_scale=1,max_nfev=cfg.max_nfev,
                         ftol=1e-5,xtol=1e-5,gtol=1e-5)
    output=decode(result.x)
    initial_cost=float(np.sum(2*(np.sqrt(1+residual(x0.ravel())**2)-1))/2)
    accepted=bool(np.isfinite(output).all() and (output[...,2]>0).all()
                  and result.cost < initial_cost)
    if not accepted:
        output=J.copy()
    def error(A,mask):
        e=np.linalg.norm(project(A[:,OBSERVED],K)-targets,axis=-1)
        selection=valid & mask[:,None]
        return float(np.median(e[selection])) if selection.any() else None
    return output, {
        "status":str(result.message),"accepted":accepted,"converged":bool(result.success),
        "nfev":int(result.nfev),"config":asdict(cfg),
        "initial_cost":initial_cost,"final_cost":float(result.cost),
        "fit_median_px_before":error(J,fit),"fit_median_px_after":error(output,fit),
        "heldout_median_px_before":error(J,~fit),"heldout_median_px_after":error(output,~fit),
        "max_displacement_m":float(np.linalg.norm(output-J,axis=-1).max()),
        "meaning":"Reprojection agreement; not ground-truth 3D accuracy",
    }


def camera_to_world(camera: np.ndarray, world: np.ndarray, new_camera: np.ndarray) -> np.ndarray:
    """Recover the per-frame rigid coordinate transform, not a pose correction.

    The paired estimates must describe the exact same pose (e.g. GEM-X camera
    and global outputs). A residual check rejects fitting unrelated models.
    """
    C,W,N=map(lambda x:np.asarray(x,dtype=float),(camera,world,new_camera))
    if C.shape != W.shape or C.shape != N.shape or C.ndim != 3 or C.shape[-1] != 3:
        raise ValueError("Coordinate transform inputs must have equal (F,J,3) shapes")
    cc=C.mean(1,keepdims=True);wc=W.mean(1,keepdims=True)
    u,_,vt=np.linalg.svd(np.einsum('fji,fjk->fik',C-cc,W-wc))
    d=np.broadcast_to(np.eye(3),(len(C),3,3)).copy();d[:,-1,-1]=np.linalg.det(u@vt)
    R=u@d@vt
    check=np.einsum('fji,fik->fjk',C-cc,R)+wc
    if not np.isfinite(check).all() or np.max(np.linalg.norm(check-W,axis=-1))>1e-4:
        raise ValueError("Camera/world inputs are not the same rigid pose")
    return np.einsum('fji,fik->fjk',N-cc,R)+wc
