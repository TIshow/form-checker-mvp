#!/usr/bin/env python3
"""Reproduce the rejected golf reprojection experiment; not a production pipeline."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import analysis
from core.refinement import fit_pose, camera_to_world, project, RefinementConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experimental', action='store_true',
                   help='Explicitly reproduce the rejected experiment for diagnosis only')
    p.add_argument('--evidence',required=True)
    p.add_argument('--out',default='output_golf_refined')
    p.add_argument('--fps',type=float,required=True)
    p.add_argument('--holdout-every',type=int,default=5,
                   help='Every Nth frame supplies no 2D fitting term; 0 fits all frames')
    p.add_argument('--max-nfev',type=int,default=60)
    args=p.parse_args()
    if not args.experimental:
        p.error('This experiment was rejected after visual review. Use --experimental only for diagnosis.')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):
        p.error(f'{out} already contains data; choose a new --out')
    data=np.load(args.evidence,allow_pickle=False)
    C,W=data['joints_incam'],data['joints_global']
    fit=np.ones(len(C),dtype=bool)
    if args.holdout_every:
        if args.holdout_every<2:p.error('--holdout-every must be zero or >=2')
        fit[args.holdout_every//2::args.holdout_every]=False
    corrected,report=fit_pose(C,data['keypoints_2d'],data['K'],args.fps,
                              fit_frames=fit,config=RefinementConfig(max_nfev=args.max_nfev))
    world=camera_to_world(C,W,corrected)
    out.mkdir(parents=True,exist_ok=True)
    np.save(out/'gf_joints.npy',world)
    np.save(out/'gf_joints_incam.npy',corrected)
    np.save(out/'gx_joints.npy',W)
    report.update({'evidence_sha256':hashlib.sha256(Path(args.evidence).read_bytes()).hexdigest(),
                   'fps':args.fps,'heldout_frames':np.flatnonzero(~fit).tolist(),
                   'intrinsics_source':str(data['camera_intrinsics_source']),
                   'video_sha256':str(data['video_sha256']),
                   'production_validated':False,
                   'adoption_status':'rejected',
                   'rejection_reason':'Visual review: wrist separation, knee bending and foot height worsened; lower reprojection error is insufficient.'})
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    clips=[]
    for label,J,color in [('公式経路・補正前',W,0xe8a45e),('画像に合わせた補正・不採用',world,0x64c8b2)]:
        clip=analysis.analyze_json(J,args.fps,'golf_swing',anchor=False)
        clip.update({'label':label,'color':color,'fps':args.fps})
        clips.append(clip)
    (out/'review.json').write_text(json.dumps({
        'report':report,'clips':clips,'keypoints_2d':data['keypoints_2d'].tolist(),
        'projection_before':project(C,data['K']).tolist(),
        'projection_after':project(corrected,data['K']).tolist(),
        'image_size_wh':data['image_size_wh'].tolist(),
    },ensure_ascii=False,allow_nan=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
