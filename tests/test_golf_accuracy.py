import json
import numpy as np
import pytest

import analysis
from core.anchor import anchor_feet
from core.refinement import fit_pose, project, camera_to_world, PARENTS, MOVABLE, RefinementConfig
from tests.synth import synth_serve


def skeleton(frames=12):
    # Non-degenerate fixed-length test body in a positive-Z camera.
    J=np.zeros((frames,24,3));J[:,0]=[0,0,4]
    offsets=np.array([[.02, .05, .03]]*24)
    offsets[[1,2]]=[[-.15,.1,0],[.15,.1,0]]
    offsets[[4,5]]=[[-.02,.4,.06],[.02,.4,.06]]
    offsets[[7,8]]=[[.01,.4,-.06],[-.01,.4,-.06]]
    offsets[[16,17]]=[[-.17,-.04,0],[.17,-.04,0]]
    offsets[[18,19]]=[[-.12,.22,.1],[.12,.22,.1]]
    offsets[[20,21]]=[[.18,.19,-.12],[-.18,.19,-.12]]
    for j in range(1,24):J[:,j]=J[:,PARENTS[j]]+offsets[j]
    K=np.array([[1000,0,540],[0,1000,960],[0,0,1.]])
    return J,K


def test_display_anchor_cannot_deform_any_bone():
    J,_=synth_serve(fps=30)
    rng=np.random.default_rng(4)
    J=J+rng.normal(0,.015,J.shape)
    anchored,_=anchor_feet(J)
    np.testing.assert_allclose(anchored-anchored[:,0:1],J-J[:,0:1],atol=1e-12)


def test_display_anchor_cannot_change_measurements():
    J,_=synth_serve(fps=30)
    a=analysis.analyze_json(J,30,"golf_swing",anchor=False)
    b=analysis.analyze_json(J,30,"golf_swing",anchor=True)
    assert json.dumps(a['metrics'],sort_keys=True)==json.dumps(b['metrics'],sort_keys=True)
    assert a['measurement_joints']==b['measurement_joints']


def test_refinement_recovers_image_location_without_stretching_limbs():
    pytest.importorskip("scipy")
    truth,K=skeleton()
    wrong=truth.copy();wrong[:,:,0]+=.10
    observations=np.concatenate([project(truth,K),np.ones((len(truth),24,1))],axis=-1)
    fit=np.arange(len(truth))%4!=2
    refined,report=fit_pose(wrong,observations,K,30,fit_frames=fit,
                            config=RefinementConfig(max_nfev=35))
    assert report['accepted']
    assert report['heldout_median_px_after'] < report['heldout_median_px_before']*.7
    before=np.linalg.norm(wrong[:,MOVABLE]-wrong[:,PARENTS[MOVABLE]],axis=-1)
    after=np.linalg.norm(refined[:,MOVABLE]-refined[:,PARENTS[MOVABLE]],axis=-1)
    np.testing.assert_allclose(after,before,atol=1e-8)
    assert np.linalg.norm(refined-truth,axis=-1).mean()<np.linalg.norm(wrong-truth,axis=-1).mean()


def test_missing_evidence_does_not_invent_correction():
    pytest.importorskip("scipy")
    J,K=skeleton();kp=np.full_like(J,np.nan)
    corrected,report=fit_pose(J,kp,K,30)
    np.testing.assert_array_equal(corrected,J)
    assert report['status']=='no_observations'


def test_camera_world_transfer_rejects_other_pose():
    J,_=skeleton()
    R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    W=J@R.T+[1,2,3]
    changed=J.copy();changed[:,20,0]+=.05
    np.testing.assert_allclose(camera_to_world(J,W,changed),changed@R.T+[1,2,3],atol=1e-8)
    W[:,20,0]+=.2
    with pytest.raises(ValueError,match='same rigid pose'):
        camera_to_world(J,W,changed)
