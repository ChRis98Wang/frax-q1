"""Versioned, full-frame Q1 motion quality checks.

Poses use MuJoCo order ``xyz, quaternion-wxyz, joints``. Missing source tracking
or FRAX sphere evidence is *unverified*, never substituted with zero error.
The default profile reproduces the historical walking-span selection at 30 Hz.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_PROFILE = {
    "schema_version": 1, "name": "legacy_walk_v1", "profile_version": 1,
    "quaternion_norm_tolerance": 2e-6,
    "physical_penetration_m": .002, "sphere_penetration_m": .002,
    "joint_limit_rad": .001, "joint_speed_rad_s": 12.,
    "joint_acceleration_rad_s2": 150., "stance_slip_m_s": .20,
    "root_target_error_m": .08, "foot_target_error_m": .06,
    "min_walking_upright": .85, "min_root_height_m": .28,
    "max_sole_clearance_m": .025, "max_shoulder_roll_rad": .7,
    "calibration_seconds": 1., "edge_buffer_frames": 2,
    "minimum_span_seconds": 3.,
}

PHYSICAL_KEYS = (
    "physical_self_penetration", "physical_floor_penetration", "sole_clearance",
    "stance_slip", "joint_speed", "joint_acceleration", "root_speed",
)


def load_profile(profile=None):
    """Load a complete known schema; reject typos, NaN, infinity and negatives."""
    if profile is None:
        value = dict(DEFAULT_PROFILE)
    elif isinstance(profile, (str, Path)):
        value = json.loads(Path(profile).read_text())
    elif isinstance(profile, dict):
        value = dict(profile)
    else:
        raise TypeError("Quality profile must be a JSON path or dictionary")
    if set(value) != set(DEFAULT_PROFILE):
        raise ValueError(f"Quality profile keys differ: {sorted(set(value)^set(DEFAULT_PROFILE))}")
    if value["schema_version"] != 1 or isinstance(value["schema_version"], bool):
        raise ValueError("Unsupported quality schema_version")
    if not isinstance(value["name"], str) or not value["name"].strip():
        raise ValueError("Quality profile requires a name")
    for key in ("profile_version", "edge_buffer_frames"):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < (1 if key=="profile_version" else 0):
            raise ValueError(f"Invalid integer profile field: {key}")
    for key in set(value)-{"schema_version", "profile_version", "name", "edge_buffer_frames"}:
        if isinstance(value[key], bool) or not isinstance(value[key], (float, int)) or not math.isfinite(value[key]) or value[key] < 0:
            raise ValueError(f"Quality threshold must be finite and nonnegative: {key}")
    if not 0 <= value["min_walking_upright"] <= 1 or value["minimum_span_seconds"] < 3:
        raise ValueError("Upright threshold must lie in [0,1]; accepted spans must be at least 3 seconds")
    return value


def profile_hash(profile=None):
    return hashlib.sha256(json.dumps(load_profile(profile),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def percentiles(values):
    """JSON-safe statistics; empty/nonfinite evidence remains explicit."""
    array=np.asarray(values,dtype=float).reshape(-1)
    finite=array[np.isfinite(array)]
    result={key:None for key in ("p50","p95","p99","max")}
    if len(finite):
        result.update(zip(result,map(float,np.percentile(finite,[50,95,99,100]))))
    result.update(count=int(array.size),nonfinite_count=int(array.size-finite.size))
    return result


def _poses(model,q,fps):
    array=np.asarray(q,dtype=np.float64)
    if array.ndim!=2 or array.shape[1]!=model.nq or len(array)<2:
        raise ValueError(f"Expected at least two poses with shape [frames,{model.nq}]")
    if not math.isfinite(float(fps)) or fps<=0:
        raise ValueError("fps must be finite and positive")
    return array


def physical_metrics(model,q,fps=30.):
    """Actual MuJoCo contact geometry and derivatives at **every** frame.

    Stance slip is XY speed of near-floor sole sphere centers, with neighboring
    contact samples required. It is a geometric proxy, not measured contact force.
    No input pose, model parameter, or source file is modified.
    """
    q=_poses(model,q,fps)
    if not np.isfinite(q).all() or np.any(np.linalg.norm(q[:,3:7],axis=1)<1e-12):
        raise ValueError("Physical replay requires finite poses and nonzero quaternions")
    data=mujoco.MjData(model)
    ids=[g for g in range(model.ngeom) if "_foot_" in (model.geom(g).name or "")]
    if not ids or any(model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_SPHERE for g in ids):
        raise ValueError("Expected named physical sole spheres (*_foot_*)")
    points=np.empty((len(q),len(ids),3)); radii=model.geom_size[ids,0]
    self_depth=np.zeros(len(q)); floor_depth=np.zeros(len(q))
    for i,row in enumerate(q):
        data.qpos[:]=row
        mujoco.mj_forward(model,data)
        points[i]=data.geom_xpos[ids]
        for contact in data.contact:
            b1,b2=model.geom_bodyid[contact.geom1],model.geom_bodyid[contact.geom2]
            if b1 and b2: self_depth[i]=max(self_depth[i],-contact.dist)
            elif bool(b1)!=bool(b2): floor_depth[i]=max(floor_depth[i],-contact.dist)
    z=points[...,2]-radii
    velocity=np.gradient(points,1/fps,axis=0)
    near=z<.008; sustained=near.copy()
    sustained[1:] &= near[:-1]; sustained[:-1] &= near[1:]
    slip=np.max(np.where(sustained,np.linalg.norm(velocity[...,:2],axis=-1),0.),axis=1)
    qd=np.gradient(q[:,7:],1/fps,axis=0); qdd=np.gradient(qd,1/fps,axis=0)
    return {"physical_self_penetration":self_depth,"physical_floor_penetration":floor_depth,
            "sole_clearance":z.min(axis=1),"stance_slip":slip,
            "joint_speed":np.max(np.abs(qd),axis=1) if qd.shape[1] else np.zeros(len(q)),
            "joint_acceleration":np.max(np.abs(qdd),axis=1) if qdd.shape[1] else np.zeros(len(q)),
            "root_speed":np.linalg.norm(np.gradient(q[:,:2],1/fps,axis=0),axis=1)}


def accepted_span_slices(valid,fps=30.,minimum_seconds=3.):
    """Return half-open contiguous (start,end) pairs; never bridge a rejected frame."""
    valid=np.asarray(valid)
    if valid.ndim!=1 or valid.dtype!=np.bool_ or not len(valid):
        raise ValueError("Accepted-frame mask must be a nonempty one-dimensional boolean array")
    if not math.isfinite(float(fps)) or fps<=0 or not math.isfinite(float(minimum_seconds)) or minimum_seconds<3:
        raise ValueError("Invalid sampling rate or span length; minimum is 3 seconds")
    minimum=math.ceil(fps*minimum_seconds-1e-9)
    edges=np.diff(np.r_[False,valid,False].astype(np.int8))
    return [(int(s),int(e)) for s,e in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if e-s>=minimum]


def classify_frames(model,q,physical=None,root_target_error=None,foot_target_error=None,
                    sphere_floor=None,sphere_self_distance=None,walking=True,fps=30.,profile=None):
    """Classify all frames, with rejection masks and explicit verification status.

    Tracking and sphere distances must originate from the same FRAX source/solve.
    Their integrity is recorded by the retarget manifest and checked when sealing;
    this function cannot derive source targets from robot poses alone.
    """
    cfg=load_profile(profile); q=_poses(model,q,fps); n=len(q)
    finite=np.isfinite(q).all(axis=1)
    norms=np.linalg.norm(q[:,3:7],axis=1)
    quaternion_ok=np.isfinite(norms)&(np.abs(norms-1)<=cfg["quaternion_norm_tolerance"])
    reasons={"nonfinite_pose":~finite,"quaternion_norm":~quaternion_ok}
    # Invalid inputs are rejected before replay; a harmless copy permits reporting
    # other frames without feeding NaNs or zero quaternions into MuJoCo.
    safe=q.copy(); usable=finite&(norms>1e-12)
    fallback=q[np.flatnonzero(usable)[0]].copy() if usable.any() else np.r_[np.zeros(3),1.,np.zeros(model.nq-4)]
    safe[~usable]=fallback
    safe[:,3:7]/=np.linalg.norm(safe[:,3:7],axis=1,keepdims=True)
    computed=physical_metrics(model,safe,fps) if physical is None else dict(physical)
    metrics={}; verification={"poses":"verified" if bool(finite.all() and quaternion_ok.all()) else "invalid",
                              "physical":"computed_all_frames" if physical is None else "provided_all_frames",
                              "source_tracking":"verified","sphere_approximation":"verified"}
    for key in PHYSICAL_KEYS:
        if key not in computed: raise ValueError(f"Missing physical metric: {key}")
        value=np.asarray(computed[key],dtype=float)
        if value.shape!=(n,): raise ValueError(f"Physical metric frame-count mismatch: {key}")
        metrics[key]=value
        reasons[f"nonfinite_{key}"]=~np.isfinite(value)
        if not np.isfinite(value).all(): verification["physical"]="invalid"
    evidence={"root_target_error":root_target_error,"foot_target_error":foot_target_error,
              "sphere_floor":sphere_floor,"sphere_self_distance":sphere_self_distance}
    for key,value in evidence.items():
        if value is None:
            verification["source_tracking" if "target" in key else "sphere_approximation"]="unverified"
            reasons[f"unverified_{key}"]=np.ones(n,dtype=bool)
            continue
        array=np.asarray(value,dtype=float)
        if array.shape!=(n,): raise ValueError(f"Evidence frame-count mismatch: {key}")
        metrics[key]=array; reasons[f"nonfinite_{key}"]=~np.isfinite(array)
        if "target" in key: reasons[f"negative_{key}"]=array<0
        if not np.isfinite(array).all() or ("target" in key and np.any(array<0)):
            verification["source_tracking" if "target" in key else "sphere_approximation"]="invalid"
    low,high=model.jnt_range[1:,0],model.jnt_range[1:,1]
    if len(low)!=q.shape[1]-7: raise ValueError("Quality checker requires a free-root Q1 hinge layout")
    violation=np.maximum(low-safe[:,7:],safe[:,7:]-high).max(axis=1) if len(low) else np.zeros(n)
    metrics["joint_limit_violation"]=np.maximum(violation,0.)
    reasons.update(joint_limits=violation>cfg["joint_limit_rad"],joint_speed=metrics["joint_speed"]>cfg["joint_speed_rad_s"],
                   physical_self_collision=metrics["physical_self_penetration"]>cfg["physical_penetration_m"],
                   physical_floor=metrics["physical_floor_penetration"]>cfg["physical_penetration_m"])
    if sphere_floor is not None: reasons["sphere_floor"]=metrics["sphere_floor"] < -cfg["sphere_penetration_m"]
    if sphere_self_distance is not None: reasons["sphere_self_collision"]=metrics["sphere_self_distance"] < -cfg["sphere_penetration_m"]
    base_accepted=~np.logical_or.reduce(list(reasons.values()))
    reasons["stance_slip"]=metrics["stance_slip"]>cfg["stance_slip_m_s"]
    reasons["acceleration"]=metrics["joint_acceleration"]>cfg["joint_acceleration_rad_s2"]
    if root_target_error is not None: reasons["root_target"]=metrics["root_target_error"]>cfg["root_target_error_m"]
    if foot_target_error is not None: reasons["foot_target"]=metrics["foot_target_error"]>cfg["foot_target_error_m"]
    if walking:
        upright=1-2*(safe[:,4]**2+safe[:,5]**2); metrics["upright"]=upright
        reasons.update(aerial_reset=metrics["sole_clearance"]>cfg["max_sole_clearance_m"],
                       low_height=safe[:,2]<cfg["min_root_height_m"],tilt=upright<cfg["min_walking_upright"])
        rolls=[]
        for side in ("left","right"):
            jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f"{side}_shoulder_roll_joint")
            if jid<0: raise ValueError(f"Missing {side} shoulder roll joint for walking calibration QA")
            rolls.append(model.jnt_qposadr[jid])
        calibration=(np.abs(safe[:,rolls])>cfg["max_shoulder_roll_rad"]).any(axis=1)
        calibration[:math.ceil(cfg["calibration_seconds"]*fps-1e-9)]=True
        reasons["calibration"]=calibration
    valid=~np.logical_or.reduce(list(reasons.values()))
    if walking:
        invalid=~valid; buffered=valid.copy()
        for offset in range(1,cfg["edge_buffer_frames"]+1):
            if offset>=n: break
            buffered[offset:] &= ~invalid[:-offset]; buffered[:-offset] &= ~invalid[offset:]
        reasons["boundary_buffer"]=valid&~buffered; valid=buffered
    verified=all(status in ("verified","computed_all_frames","provided_all_frames") for status in verification.values())
    spans=accepted_span_slices(valid,fps,cfg["minimum_span_seconds"])
    summary={"frames":n,"fps":float(fps),"accepted_frames":int(valid.sum()),"accepted_fraction":float(valid.mean()),
             "verified":verified,"passed":bool(verified and valid.all()),"accepted_spans":len(spans),
             "span_frames":sum(e-s for s,e in spans),"rejected_frame_counts":{key:int(mask.sum()) for key,mask in reasons.items()},
             "metrics":{key:percentiles(value) for key,value in metrics.items()},"verification":verification,
             "physical_check_stride":1,"profile_sha256":profile_hash(cfg)}
    return {"accepted":valid,"base_accepted":base_accepted,"reasons":reasons,"metrics":metrics,
            "verification":verification,"summary":summary,"profile":cfg,"profile_sha256":profile_hash(cfg)}


def main(argv=None):
    """Exit 0 only for a fully accepted, verified clip; exit 2 for diagnostic QA."""
    parser=argparse.ArgumentParser(description="Inspect every Q1 motion frame. Without --dataset, source/sphere tracking is UNVERIFIED and exit status is 2.")
    parser.add_argument("--motion",type=Path,required=True,help="FRAX WXYZ NPZ or Q1 XYZW CSV")
    parser.add_argument("--model",type=Path,help="Prepared physical Q1 MJCF; default Q1_FRAX_DATA/q1/q1.xml")
    parser.add_argument("--out",type=Path,help="JSON report; also printed to stdout")
    parser.add_argument("--profile",type=Path,help="Complete versioned quality profile JSON")
    parser.add_argument("--dataset",type=Path,help="Original retarget archive: enables fresh source-target/sphere replay with provenance checks")
    parser.add_argument("--source-dir",type=Path,help="Explicit relocation of original BVH sources for --dataset replay")
    parser.add_argument("--fps",type=float,help="Required for standalone CSV; must agree with NPZ/dataset metadata")
    parser.add_argument("--walking",action=argparse.BooleanOptionalAction,default=True,help="Apply walking/reset/calibration filters (default true)")
    parser.add_argument("--frame-report",type=Path,help="Optional per-frame metric/rejection NPZ")
    args=parser.parse_args(argv)
    from .model import MJCF_PATH
    model_path=(args.model or MJCF_PATH).expanduser().resolve()
    motion_path=args.motion.expanduser().resolve()
    outputs=[path.expanduser().resolve() for path in (args.out,args.frame_report) if path is not None]
    if len(outputs)!=len(set(outputs)): raise ValueError("JSON and per-frame outputs must be different paths")
    dataset_root=args.dataset.expanduser().resolve() if args.dataset else None
    forbidden={motion_path,model_path}
    if args.profile: forbidden.add(args.profile.expanduser().resolve())
    for output in outputs:
        if output in forbidden: raise ValueError("A diagnostic output cannot overwrite a motion/model/profile input")
        if dataset_root is not None and output.is_relative_to(dataset_root):
            raise ValueError("Diagnostic outputs must be outside the source dataset")
        for parent in output.parents:
            manifest_path=parent/"manifest.json"
            if manifest_path.is_file():
                declared=json.loads(manifest_path.read_text())
                if isinstance(declared,dict) and declared.get("seal",{}).get("complete") is True:
                    raise ValueError("Diagnostic outputs cannot be written inside a sealed dataset")
    for output in (args.out,args.frame_report):
        if output is not None and (output.exists() or output.is_symlink()):
            raise FileExistsError(f"Quality reports require a new output path; source artifacts are never overwritten: {output}")
    if args.frame_report and args.frame_report.suffix!=".npz":
        raise ValueError("--frame-report must end in .npz")
    model=mujoco.MjModel.from_xml_path(str(model_path))
    names=[model.joint(i).name for i in range(1,model.njnt)]
    if motion_path.suffix.lower()==".npz":
        with np.load(motion_path,allow_pickle=False) as archive:
            if not {"q","fps","joint_names"}.issubset(archive.files): raise ValueError("NPZ must declare q/fps/joint_names")
            q=np.asarray(archive["q"],dtype=float)
            fps=float(archive["fps"])
            if archive["joint_names"].tolist()!=names: raise ValueError("NPZ joint order differs from the physical model")
    elif motion_path.suffix.lower()==".csv":
        q=np.loadtxt(motion_path,delimiter=",",ndmin=2)
        if q.ndim!=2 or q.shape[1]!=model.nq or not len(q): raise ValueError(f"CSV must have nonempty [frames,{model.nq}] poses")
        q=q[:,np.r_[0:3,6,3:6,7:q.shape[1]]]
        fps=args.fps
    else: raise ValueError("Motion must be a .npz or .csv file")
    evidence={};profile=args.profile;dataset_verified=False
    if args.dataset:
        from .retarget import make_quality_replayer
        from .seal import _json,_artifact,sha256
        root=args.dataset.expanduser().resolve();manifest=_json(root/"manifest.json")
        if manifest.get("validation",{}).get("complete") is not True: raise ValueError("Dataset validation is incomplete")
        if sha256(model_path)!=manifest.get("model_sha256"): raise ValueError("Dataset/model hash mismatch")
        matches=[c for c in manifest.get("clips",[]) if c.get("name")==motion_path.stem]
        if len(matches)!=1: raise ValueError("Motion must identify exactly one original source clip")
        kind=motion_path.suffix.lower()[1:]
        declared,_=_artifact(root,matches[0],kind)
        if declared.resolve()!=motion_path: raise ValueError("Motion path is not the dataset's declared raw artifact")
        if fps is not None and float(fps)!=manifest.get("fps"): raise ValueError("Dataset/motion FPS differs")
        fps=float(manifest["fps"])
        profile=profile or manifest.get("validation",{}).get("quality_profile")
        evidence=make_quality_replayer(manifest,source_dir=args.source_dir)(q,matches[0])
        dataset_verified=True
    if fps is None: raise ValueError("Standalone CSV requires --fps")
    if args.fps is not None and float(args.fps)!=fps: raise ValueError("Explicit FPS differs from motion metadata")
    result=classify_frames(model,q,fps=fps,walking=args.walking,profile=profile,**evidence)
    from .seal import sha256
    report={**result["summary"],"motion":str(motion_path),"model":str(model_path),
            "motion_sha256":sha256(motion_path),"model_sha256":sha256(model_path),
            "source_provenance_verified":dataset_verified,"profile":result["profile"],
            "accepted_spans":[{"start":s,"frames":e-s} for s,e in accepted_span_slices(result["accepted"],fps,result["profile"]["minimum_span_seconds"])]}
    report["passed"]=bool(report["passed"] and dataset_verified)
    text=json.dumps(report,indent=2,allow_nan=False)
    if args.out:
        from .seal import atomic_json
        atomic_json(args.out,report)
    if args.frame_report:
        args.frame_report.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.frame_report,accepted=result["accepted"],**result["metrics"],
                            **{"reject_"+k:v for k,v in result["reasons"].items()})
    print(text)
    return 0 if report["passed"] else 2


if __name__=="__main__": raise SystemExit(main())
