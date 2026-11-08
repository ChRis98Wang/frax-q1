#!/usr/bin/env python3
"""Read-only collision audit of retained motion and interpolated poses.

Checks the configured physical geoms, not disabled visual meshes. Four samples
per source interval are a denser test, not a continuous-time collision proof.
An output report must be new and outside any input/sealed dataset.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from q1_frax.motion_io import guard_output_paths, load_motion, sha256, verify_motion_dependencies
from q1_frax.quality import load_profile, percentiles, physical_metrics, profile_hash


def physical_pairs(model):
    """Q1's enabled geom pairs, respecting weld, parent and body exclusions."""
    if model.npair:
        raise ValueError("Explicit contact pairs require separate audit support")
    excludes=set(map(int,model.exclude_signature))
    parent_filter=not(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    geoms=[i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    pairs=[]
    for i,a in enumerate(geoms):
        for b in geoms[i+1:]:
            ba,bb=int(model.geom_bodyid[a]),int(model.geom_bodyid[b])
            wa,wb=int(model.body_weldid[ba]),int(model.body_weldid[bb])
            if wa==wb: continue
            if parent_filter and wa and wb and (int(model.body_weldid[model.body_parentid[wa]])==wb or int(model.body_weldid[model.body_parentid[wb]])==wa): continue
            x,y=sorted((ba,bb))
            if (x<<16)+y in excludes: continue
            if not ((model.geom_contype[a]&model.geom_conaffinity[b]) or (model.geom_contype[b]&model.geom_conaffinity[a])): continue
            pairs.append((a,b,"self" if ba and bb else "floor"))
    return pairs


def stance_coverage(model,q,fps):
    """Expose when a zero slip statistic has little/no near-ground evidence."""
    data=mujoco.MjData(model)
    ids=[g for g in range(model.ngeom) if "_foot_" in (model.geom(g).name or "")]
    points=[]
    for row in q:
        data.qpos[:]=row;mujoco.mj_forward(model,data);points.append(data.geom_xpos[ids].copy())
    points=np.asarray(points);height=points[...,2]-model.geom_size[ids,0]
    speed=np.linalg.norm(np.gradient(points,1/fps,axis=0)[...,:2],axis=-1)
    result={}
    for threshold in (.008,.02):
        near=height<threshold;sustained=near.copy()
        sustained[1:]&=near[:-1];sustained[:-1]&=near[1:]
        result[str(threshold)]={"sustained_point_samples":int(sustained.sum()),
                               "frames_with_sustained_near_ground_points":int(sustained.any(axis=1).sum()),
                               "frame_coverage_fraction":float(sustained.any(axis=1).mean()),
                               "horizontal_speed_m_s":percentiles(speed[sustained])}
    return result


def audit(model,q,fps,*,start=0,substeps=4,profile=None):
    q=np.asarray(q,dtype=float)
    if q.ndim!=2 or q.shape[1]!=model.nq or len(q)<2 or not np.isfinite(q).all():
        raise ValueError("Audit requires at least two finite full poses")
    if not np.isfinite(fps) or fps<=0 or isinstance(substeps,bool) or not isinstance(substeps,int) or substeps<1:
        raise ValueError("Invalid fps/substeps")
    cfg=load_profile(profile)
    norm_error=np.abs(np.linalg.norm(q[:,3:7],axis=1)-1.)
    if norm_error.max()>cfg["quaternion_norm_tolerance"]: raise ValueError("Nonunit quaternion in motion")
    metrics=physical_metrics(model,q,fps)
    pairs=physical_pairs(model)
    if not any(kind=="self" for _,_,kind in pairs) or not any(kind=="floor" for _,_,kind in pairs):
        raise ValueError("Both self and floor geometry pairs are required; missing evidence cannot pass")
    data=mujoco.MjData(model)
    distances={kind:{"minimum_signed_distance_m":None,"deepest_penetration_m":0.,"negative_samples":0,"worst":None} for kind in ("self","floor")}
    native_max={"self":0.,"floor":0.}
    velocity=np.empty(model.nv); search_cap=.1
    samples=0
    for frame in range(len(q)):
        count=substeps if frame<len(q)-1 else 1
        if frame<len(q)-1: mujoco.mj_differentiatePos(model,velocity,1.,q[frame],q[frame+1])
        for sub in range(count):
            pose=q[frame].copy()
            if sub: mujoco.mj_integratePos(model,pose,velocity,sub/substeps)
            data.qpos[:]=pose;mujoco.mj_forward(model,data)
            samples+=1;mins={"self":search_cap,"floor":search_cap};worst={}
            for a,b,kind in pairs:
                distance=float(mujoco.mj_geomDistance(model,data,a,b,search_cap,None))
                if not np.isfinite(distance): raise ValueError("Nonfinite geometry distance")
                if distance<mins[kind]: mins[kind]=distance;worst[kind]=(a,b)
            for contact in data.contact:
                kind="self" if all(model.geom_bodyid[g] for g in contact.geom) else "floor"
                native_max[kind]=max(native_max[kind],float(-contact.dist))
            for kind,value in distances.items():
                distance=mins[kind]
                value["negative_samples"]+=int(distance<0.)
                value["deepest_penetration_m"]=max(value["deepest_penetration_m"],-distance)
                if value["minimum_signed_distance_m"] is None or distance<value["minimum_signed_distance_m"]:
                    value["minimum_signed_distance_m"]=distance
                    a,b=worst.get(kind,(None,None))
                    value["worst"]={"source_frame":start+frame,"interval_fraction":sub/substeps,
                                    "geoms":None if a is None else [model.geom(a).name,model.geom(b).name],
                                    "bodies":None if a is None else [model.body(model.geom_bodyid[a]).name,model.body(model.geom_bodyid[b]).name]}
    violation=np.maximum(model.jnt_range[1:,0]-q[:,7:],q[:,7:]-model.jnt_range[1:,1])
    joint_violation=float(max(0.,violation.max()))
    gates={"self_no_penetration":distances["self"]["deepest_penetration_m"]<=cfg["physical_penetration_m"] and native_max["self"]<=cfg["physical_penetration_m"],
           "floor_no_penetration":distances["floor"]["deepest_penetration_m"]<=cfg["physical_penetration_m"] and native_max["floor"]<=cfg["physical_penetration_m"],
           "joint_limits":joint_violation<=cfg["joint_limit_rad"],
           "joint_speed":float(metrics["joint_speed"].max())<=cfg["joint_speed_rad_s"],
           "joint_acceleration":float(metrics["joint_acceleration"].max())<=cfg["joint_acceleration_rad_s2"],
           "stance_slip":float(metrics["stance_slip"].max())<=cfg["stance_slip_m_s"]}
    return {"schema_version":1,"scope":"configured physical geoms; dense sampled kinematic replay", "frames":len(q),"fps":float(fps),
            "source_range_half_open":[start,start+len(q)],"duration_seconds":(len(q)-1)/fps,"substeps_per_interval":substeps,
            "checked_samples":samples,"geometry_pair_count":len(pairs),"explicit_excluded_body_pairs":int(model.nexclude),
            "disabled_visual_meshes_checked":False,"continuous_time_guarantee":False,"source_tracking_verified":False,
            "naturalness_review":"requires separate visual review", "distance_search_cap_m":search_cap,
            "collision":distances,"native_contact_deepest_penetration_m":native_max,
            "metrics":{key:percentiles(value) for key,value in metrics.items()},"joint_limit_violation_rad":joint_violation,
            "stance_evidence_by_height_m":stance_coverage(model,q,fps),
            "maximum_joint_step_rad":float(np.abs(np.diff(q[:,7:],axis=0)).max()),
            "quaternion_norm_error_max":float(norm_error.max()),"profile":cfg,"profile_sha256":profile_hash(cfg),
            "walking_filters_applied":False,"gates":gates,"sampled_physical_and_continuity_passed":all(gates.values()),
            "notes":["No pose edits or smoothing; crop indices are explicit.",
                     "Stance slip uses sustained near-ground sole points, not force measurements.",
                     "This report does not certify source tracking, sphere constraints, visual-mesh intersections, or dynamic feasibility."]}


def visual_envelope_diagnostic(model,q,fps,*,start=0,substeps=4,profile=None):
    """Conservative visible-mesh hull diagnostic, never an exact triangle test."""
    derived=deepcopy(model);count=0
    for geom in range(derived.ngeom):
        if not derived.geom_bodyid[geom]: continue
        visual=(derived.geom_type[geom]==mujoco.mjtGeom.mjGEOM_MESH and
                derived.geom_contype[geom]==0 and derived.geom_conaffinity[geom]==0)
        count+=int(visual)
        derived.geom_contype[geom]=2 if visual else 0
        derived.geom_conaffinity[geom]=3 if visual else 0
    if not count: raise ValueError("No disabled visible mesh geometry: visual-envelope evidence is unavailable")
    result=audit(derived,q,fps,start=start,substeps=substeps,profile=profile)
    return {"geometry":"MuJoCo convex envelopes of visible mesh geoms, preserving original body exclusions",
            "derived_in_memory_model_only":True,"visual_mesh_count":count,
            "exact_triangle_surface_intersection_test":False,"continuous_time_guarantee":False,
            "checked_samples":result["checked_samples"],"collision":result["collision"],
            "negative_envelope_distance_detected":any(value["deepest_penetration_m"]>0. for value in result["collision"].values()),
            "note":"Negative convex distance requires investigation and can be conservative for concave surfaces. It is not automatically an exact visible-surface intersection."}


def verify_source_evidence(motion,start,stop,profile,source_dir=None):
    """Fresh whole-source replay; apply the strict gates to the explicit crop."""
    from q1_frax.retarget import make_quality_replayer
    from q1_frax.seal import _artifact,_json
    if motion.kind!="raw" or motion.dataset_root is None or motion.manifest_path is None:
        raise ValueError("Fresh source verification requires an original versioned raw archive")
    manifest=_json(motion.manifest_path)
    if manifest.get("schema_version")!=1 or manifest.get("validation",{}).get("complete") is not True:
        raise ValueError("Fresh source verification requires a completed versioned archive")
    index_hash=sha256(motion.manifest_path)
    record=next((r for r in manifest["clips"] if r["name"]==motion.record["name"]),None)
    if record is None: raise ValueError("Source clip missing from source manifest")
    source_path,_=_artifact(motion.dataset_root,record,"npz")
    if source_path.resolve()!=motion.path.resolve() or sha256(source_path)!=motion.sha256:
        raise ValueError("Fresh source verification must bind the exact archived NPZ")
    qa_path,qa_hash=_artifact(motion.dataset_root,record,"quality")
    replayer=make_quality_replayer(manifest,source_dir=source_dir)
    evidence=replayer(motion.qpos,record)
    arrays={}
    with np.load(qa_path,allow_pickle=False) as archived:
        for key in ("root_target_error","foot_target_error","sphere_floor","sphere_self_distance"):
            values=np.asarray(evidence[key],dtype=float)
            if values.shape!=(len(motion.qpos),) or not np.isfinite(values).all():
                raise ValueError(f"Fresh source evidence invalid: {key}")
            if "target_error" in key and np.any(values<0.):
                raise ValueError(f"Negative source tracking distance: {key}")
            if key not in archived or not np.allclose(values,archived[key],rtol=1e-6,atol=2e-6):
                raise ValueError(f"Fresh whole-source evidence differs from archived QA: {key}")
            arrays[key]=values[start:stop]
    if sha256(motion.manifest_path)!=index_hash or sha256(qa_path)!=qa_hash:
        raise ValueError("Source manifest/QA changed during fresh source replay")
    cfg=load_profile(profile)
    gates={"root_tracking":bool(np.all(arrays["root_target_error"]<=cfg["root_target_error_m"])),
           "foot_tracking":bool(np.all(arrays["foot_target_error"]<=cfg["foot_target_error_m"])),
           "sphere_floor":bool(np.all(arrays["sphere_floor"]>=-cfg["sphere_penetration_m"])),
           "sphere_self":bool(np.all(arrays["sphere_self_distance"]>=-cfg["sphere_penetration_m"]))}
    return {"fresh_source_replay":True,"replayed_source_frames":len(motion.qpos),
            "scope":"Replayed the entire exact NPZ from original BVH/config; thresholds apply only to the declared retained crop",
            "source_bvh_sha256":record["source_sha256"],"source_npz_sha256":motion.sha256,
            "source_manifest_sha256":index_hash,"archived_quality_sha256":qa_hash,
            "mapping_sha256":manifest["mapping_sha256"],"collision_sha256":manifest["collision_sha256"],
            "retarget_parameters":manifest["retarget_parameters"],
            "root_target_error_m":percentiles(arrays["root_target_error"]),
            "foot_target_error_m":percentiles(arrays["foot_target_error"]),
            "minimum_sphere_floor_m":float(arrays["sphere_floor"].min()),
            "minimum_sphere_self_distance_m":float(arrays["sphere_self_distance"].min()),
            "source_gates":gates,"source_and_sphere_gates_passed":all(gates.values())}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--frames",type=int)
    parser.add_argument("--substeps",type=int,default=4)
    parser.add_argument("--visual-envelope",action="store_true",help="Also diagnose visible-mesh convex envelopes in a separate in-memory model; not an exact triangle test")
    parser.add_argument("--verify-source",action="store_true",help="Fresh whole-source FRAX replay; set Q1_FRAX_DATA to the exact model's asset cache before launch")
    parser.add_argument("--source-dir",type=Path,help="Optional relocated original BVH directory for fresh source replay")
    parser.add_argument("--profile",type=Path,default=Path(__file__).resolve().parents[1]/"configs/quality_strict_dynamic_v1.json")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    motion=load_motion(args.motion,model_path=args.model,archive=True)
    guard_output_paths(motion,args.model,args.out)
    if args.out.expanduser().resolve()==args.profile.expanduser().resolve(): raise ValueError("Audit output cannot overwrite the quality profile")
    if args.out.exists() or args.out.is_symlink(): raise FileExistsError("Audit output must be new")
    if args.start<0 or (args.frames is not None and args.frames<2): raise ValueError("Invalid crop")
    stop=len(motion.qpos) if args.frames is None else args.start+args.frames
    if stop>len(motion.qpos) or args.start>=stop: raise ValueError("Crop exceeds source motion")
    deps=motion.model_dependencies
    model=mujoco.MjModel.from_xml_path(str(args.model.absolute()))
    report=audit(model,motion.qpos[args.start:stop],motion.fps,start=args.start,substeps=args.substeps,profile=args.profile)
    if args.visual_envelope:
        report["visual_envelope_diagnostic"]=visual_envelope_diagnostic(model,motion.qpos[args.start:stop],motion.fps,start=args.start,substeps=args.substeps,profile=args.profile)
    if args.verify_source:
        report["source_evidence"]=verify_source_evidence(motion,args.start,stop,args.profile,args.source_dir)
        report["source_tracking_verified"]=True
        report["source_and_sphere_gates_passed"]=report["source_evidence"]["source_and_sphere_gates_passed"]
        report["source_gates"]=report["source_evidence"]["source_gates"]
    report["requested_checks_passed"]=report["sampled_physical_and_continuity_passed"] and not report.get("visual_envelope_diagnostic",{}).get("negative_envelope_distance_detected",False)
    if args.verify_source:report["requested_checks_passed"]&=report["source_and_sphere_gates_passed"]
    report.update(source_file=args.motion.name,source_sha256=sha256(args.motion),model_sha256=sha256(args.model),model_dependencies=deps,
                  model_dependency_identity_verified=motion.dependency_identity_verified)
    if sha256(args.motion)!=motion.sha256 or sha256(args.model)!=motion.model_sha256:
        raise ValueError("Input motion/model changed during collision audit")
    verify_motion_dependencies(motion,args.model)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:report[key] for key in ("source_file","source_range_half_open","checked_samples","collision","gates","sampled_physical_and_continuity_passed","visual_envelope_diagnostic","source_tracking_verified","source_gates","requested_checks_passed") if key in report},indent=2))
    return 0 if report["requested_checks_passed"] else 2


if __name__=="__main__": sys.exit(main())
