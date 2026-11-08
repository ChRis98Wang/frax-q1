"""Batch LAFAN1 -> Q1 through FRAX, with physical QA and immutable provenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

DEFAULT_PARAMETERS = {
    "contact_aware": True, "contact_swing_margin": .015,
    "foot_rotation_weight": 40., "foot_position_weight": 200.,
    "damping": 30., "shoulder_damping": 300., "foot_lift_gain": 0.,
    "self_collision_margin": 0., "self_collision_gain": 30., "self_collision_penalty": 1e6,
    "self_collision_max_rows": 64, "floor_margin": .002, "floor_gain": 30.,
    "floor_penalty": 1e6, "floor_max_rows": 32, "joint_velocity_limit": 12.,
}


def load_parameters(path=None):
    import math
    params=dict(DEFAULT_PARAMETERS)
    if path:
        supplied=dict(path) if isinstance(path,dict) else json.loads(Path(path).read_text())
        if not isinstance(supplied,dict) or set(supplied)-set(params):
            raise ValueError("Unknown retarget configuration fields")
        params.update(supplied)
    if not isinstance(params["contact_aware"],bool):
        raise ValueError("contact_aware must be boolean")
    positive=("self_collision_gain","self_collision_penalty","floor_gain","floor_penalty","joint_velocity_limit")
    for key,value in params.items():
        if key=="contact_aware":
            continue
        if key.endswith("max_rows"):
            if value is not None and (isinstance(value,bool) or not isinstance(value,int) or value<1):
                raise ValueError(f"{key} must be a positive integer or null")
        elif isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or (key in positive and value==0):
            raise ValueError(f"Invalid finite nonnegative retarget parameter: {key}")
    return params


def extract_lafan(archive, destination):
    destination.mkdir(parents=True,exist_ok=True)
    seen=set()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.namelist():
            if Path(member).suffix.lower()!=".bvh":
                continue
            name=Path(member).name
            if name in seen:
                raise ValueError(f"Duplicate BVH filename in source archive: {name}")
            seen.add(name)
            target=destination/name
            payload=zipped.read(member)
            if target.exists() and target.read_bytes()!=payload:
                raise ValueError(f"Extracted BVH differs from pinned source: {target}")
            if not target.exists():
                target.write_bytes(payload)


def verify_frax_installation():
    """Check the actual imported code, not only a separate source checkout."""
    from .paths import SOURCES
    import frax
    expected=SOURCES["frax"]["commit"]
    direct=importlib.metadata.distribution("frax").read_text("direct_url.json")
    if direct and json.loads(direct).get("vcs_info",{}).get("commit_id")==expected:
        return
    package=Path(frax.__file__).resolve().parent
    # An editable checkout has .git beside frax/. Never mistake the enclosing
    # q1-frax repository of a virtualenv for the installed FRAX source checkout.
    checkout=package.parent
    if (checkout/".git").exists():
        try:
            head=subprocess.check_output(["git","-C",str(checkout),"rev-parse","HEAD"],text=True,stderr=subprocess.PIPE).strip()
            dirty=subprocess.check_output(["git","-C",str(checkout),"status","--porcelain","--untracked-files=no"],text=True)
        except subprocess.CalledProcessError:
            pass
        else:
            if head==expected and not dirty:
                return
    raise ValueError("Installed FRAX is not the pinned source revision; install q1-frax[retarget]")



def _quality_evaluator(robot, retargeter):
    import jax
    import jax.numpy as jnp
    @jax.jit
    @jax.vmap
    def metrics(q,source):
        pos,r=robot.link_collision_data(q)
        task=retargeter.solver.tasks[0]
        ref=retargeter.preprocess(source)
        tfs=robot.joint_to_world_transforms(q)[task.joints]@task.offsets
        error=jnp.linalg.norm(tfs[:,:3,3]-ref["frame_pos"],axis=-1)
        return jnp.min(pos[:,2]-r),jnp.min(robot.self_collision_distances(q)),error[0],jnp.max(error[jnp.array([3,6])])
    return metrics


def _evaluate_quality_chunks(evaluator,q,source):
    import jax
    import numpy as np
    values=[[],[],[],[]]
    for i in range(0,len(q),1024):
        chunk=q[i:i+1024];n=len(chunk)
        chunk=np.pad(chunk,((0,1024-n),(0,0)),mode="edge")
        src=jax.tree.map(lambda a:np.pad(a[i:i+n],((0,1024-n),)+((0,0),)*(a.ndim-1),mode="edge"),source)
        for target,value in zip(values,evaluator(chunk,src)):
            target.append(np.asarray(value)[:n])
    return tuple(np.concatenate(v) for v in values)


def make_quality_replayer(manifest,source_dir=None):
    """Build a read-only replay of original source tracking and sphere evidence.

    Raises when exact sources/assets are unavailable or changed. It never accepts
    stored QA arrays as a substitute for recomputing the targets and geometry.
    """
    os.environ.setdefault("JAX_PLATFORMS","cpu")
    os.environ.setdefault("JAX_ENABLE_X64","1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    os.environ.setdefault("XLA_FLAGS","--xla_cpu_multi_thread_eigen=false --xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED")
    import jax
    from frax.ik import smooth_poses
    from .adapter import load_q1,make_retargeter,load_lafan1,contact_aware_inputs,IK_CONFIG,COLLISION
    from .model import MJCF_PATH
    from .paths import sha256,verify_source
    from .integrity import verify_model_dependencies
    manifest=json.loads(Path(manifest).read_text()) if isinstance(manifest,(str,Path)) else dict(manifest)
    for path,key in ((MJCF_PATH,"model_sha256"),(IK_CONFIG,"mapping_sha256"),(COLLISION,"collision_sha256")):
        if sha256(path)!=manifest[key]:
            raise ValueError(f"Quality replay asset changed: {path}")
    verify_model_dependencies(MJCF_PATH,manifest.get("model_dependencies"))
    for name,commit in manifest["commits"].items():
        if verify_source(name)!=commit:
            raise ValueError(f"Quality replay source changed: {name}")
    verify_frax_installation()
    params=load_parameters(manifest["retarget_parameters"])
    contact=params.pop("contact_aware");swing=params.pop("contact_swing_margin")
    robot,model=load_q1(rebuild=False)
    retargeter,bones,scales=make_retargeter(robot,model,manifest["self_collision_avoidance"],**params)
    evaluator=_quality_evaluator(robot,retargeter)
    selection=manifest["input_selection"]
    directory=Path(source_dir or selection["bvh_dir"])
    maximum=selection["max_frames"]
    def replay(q,clip_record):
        source=directory/(clip_record["name"]+".bvh")
        if sha256(source)!=clip_record["source_sha256"]:
            raise ValueError(f"Quality replay BVH changed: {source}")
        poses,names=load_lafan1(source)
        if maximum:poses=poses[:maximum]
        poses=smooth_poses(poses,1/30.,sigma_rot=.05)
        inputs=(contact_aware_inputs(poses,names,bones,scales["leg_scale"],swing_margin=swing)
                if contact else poses[:,[names.index(n) for n in bones]])
        if len(q)!=len(jax.tree.leaves(inputs)[0]):
            raise ValueError(f"Quality replay frame mismatch: {source}")
        floor,dist,root,foot=_evaluate_quality_chunks(evaluator,q,inputs)
        return {"sphere_floor":floor,"sphere_self_distance":dist,"root_target_error":root,"foot_target_error":foot}
    return replay



def configure_backend(platform="cpu", devices=None):
    """Initialize the explicitly requested backend, with no GPU-to-CPU fallback."""
    if platform not in ("cpu", "cuda"):
        raise ValueError("platform must be cpu or cuda")
    if platform == "cuda" and devices is not None:
        raise ValueError("--devices configures virtual CPU devices and is not valid with --platform cuda")
    if devices is not None and (isinstance(devices,bool) or not isinstance(devices,int) or devices<1):
        raise ValueError("--devices must be a positive integer")
    os.environ["JAX_PLATFORMS"] = platform
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if platform == "cpu":
        os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false --xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED")
    import jax
    jax.config.update("jax_enable_x64", True)
    if platform == "cpu":
        jax.config.update("jax_num_cpu_devices", devices or 1)
    actual = jax.default_backend()  # Raises when requested CUDA cannot initialize.
    expected = "gpu" if platform == "cuda" else "cpu"
    if actual != expected:
        raise RuntimeError(f"Requested {platform}, but JAX initialized {actual}; fallback is forbidden")
    active = jax.devices()
    if not active or any(device.platform != expected for device in active):
        raise RuntimeError(f"Unexpected JAX device set for {platform}")
    return {"requested_platform": platform, "jax_backend": actual, "jax_enable_x64":bool(jax.config.jax_enable_x64),
            "devices":[{"id":device.id,"platform":device.platform,"kind":device.device_kind} for device in active],
            "cpu_virtual_devices": (devices or 1) if platform == "cpu" else None}



def guard_output_directory(path):
    """Refuse sealed ancestors or output symlinks before creating/writing files."""
    path=Path(os.path.abspath(Path(path).expanduser()))
    for parent in (path,*path.parents):
        if parent.is_symlink():
            raise ValueError(f"Output path traverses a symlink: {parent}")
        manifest=parent/"manifest.json"
        if manifest.is_symlink():
            raise ValueError(f"Output manifest is a symlink: {manifest}")
        if manifest.is_file():
            data=json.loads(manifest.read_text())
            if isinstance(data,dict) and data.get("seal",{}).get("complete"):
                raise ValueError(f"Dataset is sealed and cannot be retargeted or revalidated: {parent}. Use --out with a new output directory.")
    if path.exists():
        for base,dirs,files in os.walk(path,followlinks=False):
            for name in dirs+files:
                target=Path(base)/name
                if target.is_symlink():
                    raise ValueError(f"Output contains a symlink: {target}")
                if target.is_file() and target.stat().st_nlink > 1:
                    raise ValueError(f"Output contains a multiply hard-linked file: {target}")
    return path


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,help="Asset cache (Q1_FRAX_DATA)")
    p.add_argument("--bvh-dir",type=Path,help="Existing directory of LAFAN-format BVH inputs")
    p.add_argument("--pattern",help="BVH glob, default *.bvh")
    p.add_argument("--platform",choices=("cpu","cuda"),default="cpu",help="Explicit JAX backend; CUDA never falls back to CPU")
    p.add_argument("--devices",type=int,default=None,help="CPU-only virtual device count; default 1")
    p.add_argument("--max-frames",type=int,help="0 = full clips; minimum 2 otherwise")
    p.add_argument("--out",type=Path,default=Path("output/lafan1_q1"))
    p.add_argument("--self-collision",action=argparse.BooleanOptionalAction,default=None)
    p.add_argument("--validate-only",action="store_true",help="Revalidate an unsealed archive using its generation settings")
    p.add_argument("--config",type=Path,help="JSON override of the tuned default retarget parameters")
    p.add_argument("--quality-profile",type=Path,help="Versioned quality profile JSON (default legacy_walk_v1)")
    return p


def main(argv=None):
    p=parser()
    args=p.parse_args(argv)
    if (args.devices is not None and args.devices<1) or (args.max_frames is not None and args.max_frames!=0 and args.max_frames<2):
        p.error("--devices must be >=1; --max-frames must be 0 or >=2")
    if args.platform=="cuda" and args.devices is not None:
        p.error("--devices is only valid with --platform cpu")
    if args.data_dir:
        os.environ["Q1_FRAX_DATA"]=str(args.data_dir.expanduser().resolve())
    args.out=guard_output_directory(args.out)
    runtime=configure_backend(args.platform,args.devices)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from frax.ik import OfflineRetargeter,smooth_poses
    from . import adapter
    from .adapter import load_q1,load_q1_motion,make_retargeter,contact_aware_inputs,load_lafan1,IK_CONFIG,COLLISION
    from .model import DATA,JOINT_NAMES,MJCF_PATH
    from .paths import verify_source,sha256,write_json
    from .integrity import dependency_fingerprints,verify_model_dependencies
    from .quality import physical_metrics,percentiles,classify_frames,load_profile,profile_hash,accepted_span_slices

    out = args.out.expanduser().resolve()
    existing_manifest=out/"manifest.json"
    if existing_manifest.is_file() and json.loads(existing_manifest.read_text()).get("seal",{}).get("complete"):
        raise ValueError(
            f"Dataset is sealed and cannot be retargeted or revalidated: {out}. "
            "Use --out with a new output directory to preserve training data and split hashes."
        )
    if existing_manifest.exists() and not args.validate_only:
        raise ValueError("Output already has a manifest; choose a new --out directory or use --validate-only")
    if args.validate_only and not existing_manifest.is_file():
        raise ValueError("--validate-only requires an existing manifest")
    out.mkdir(parents=True, exist_ok=True)
    rawdir, csvdir = out / "npz", out / "csv"
    rawdir.mkdir(exist_ok=True); csvdir.mkdir(exist_ok=True)
    previous = json.loads(existing_manifest.read_text()) if args.validate_only else None
    if previous and args.config:
        raise ValueError("--validate-only preserves the generation configuration; omit --config")
    if previous and args.self_collision is not None and args.self_collision != previous["self_collision_avoidance"]:
        raise ValueError("--validate-only cannot change self-collision generation settings")
    args.self_collision = previous["self_collision_avoidance"] if previous else (True if args.self_collision is None else args.self_collision)
    params = dict(previous["retarget_parameters"]) if previous else load_parameters(args.config)
    config = dict(params)
    contact_aware = config.pop("contact_aware")
    swing_margin = config.pop("contact_swing_margin")
    selection = previous.get("input_selection", {}) if previous else {}
    pattern = args.pattern or selection.get("pattern", "*.bvh")
    maximum = args.max_frames if args.max_frames is not None else selection.get("max_frames", 0)
    if previous and (pattern != selection.get("pattern") or maximum != selection.get("max_frames")):
        raise ValueError("--validate-only must use the original pattern and frame count")
    bvh_dir = args.bvh_dir or (Path(selection["bvh_dir"]) if previous else DATA/"lafan1_bvh")
    bvh_dir = bvh_dir.expanduser().resolve()
    required_sources = ["frax", "primebot-q1-urdf", "primebot-gmr"]
    if not args.bvh_dir and not previous:
        required_sources.append("lafan1_source")
    commits = {name: verify_source(name) for name in required_sources}
    verify_frax_installation()
    if not args.bvh_dir and not previous:
        extract_lafan(DATA/"lafan1_source/lafan1/lafan1.zip", bvh_dir)
    paths = sorted(bvh_dir.glob(pattern))
    if not paths:
        raise ValueError(f"No clips matching {pattern} in {bvh_dir}")
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("BVH source names must be unique; duplicate stems would overwrite output files")
    if previous:
        old_clips = {c["name"]:c for c in previous["clips"]}
        if [p.stem for p in paths] != list(old_clips):
            raise ValueError("Source clip set differs from generation manifest")
        for path in paths:
            if sha256(path) != old_clips[path.stem]["source_sha256"]:
                raise ValueError(f"Source changed since generation: {path}")
    if previous:
        for asset,key in ((MJCF_PATH,"model_sha256"),(IK_CONFIG,"mapping_sha256"),(COLLISION,"collision_sha256")):
            if sha256(asset)!=previous[key]:
                raise ValueError(f"Generation asset changed; validate-only will not rebuild: {asset}")
    if previous:
        verify_model_dependencies(MJCF_PATH,previous.get("model_dependencies"))
    robot, model = load_q1(rebuild=not args.validate_only)
    retargeter, bones, scales = make_retargeter(robot,model,args.self_collision,**config)
    print(f"Q1: {robot.nq} qpos, {robot.nv} velocities, {len(robot.flat_collision_radii)} spheres; scales {scales}",flush=True)
    def load(path):
        poses,names = load_lafan1(path)
        if maximum:
            poses = poses[:maximum]
        if len(poses) < 2:
            raise ValueError(f"At least two source frames are required: {path}")
        poses=smooth_poses(poses,1/30.,sigma_rot=.05)
        if contact_aware:
            return contact_aware_inputs(poses,names,bones,scales["leg_scale"],swing_margin=swing_margin)
        return poses[:,[names.index(n) for n in bones]]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(8,len(runtime["devices"]))) as pool:
        clips = list(pool.map(load,paths))
    print(f"Loaded {len(paths)} clips / {sum(len(jax.tree.leaves(c)[0]) for c in clips)} frames in {time.perf_counter()-start:.1f}s",flush=True)
    if args.validate_only:
        if previous["model_sha256"] != hashlib.sha256(MJCF_PATH.read_bytes()).hexdigest():
            raise ValueError("Physical model changed: run retargeting again")
        qs=[load_q1_motion(rawdir/(path.stem+".npz")) for path in paths]
        for path,q,source in zip(paths,qs,clips):
            if len(q) != len(jax.tree.leaves(source)[0]):
                raise ValueError(f"Frame count mismatch: {path}")
            if sha256(rawdir/(path.stem+".npz")) != old_clips[path.stem]["npz_sha256"]:
                raise ValueError(f"Archive changed since generation: {path}")
        solve_seconds,compile_seconds=previous["solve_seconds"],previous["compile_seconds"]
    else:
        batch = OfflineRetargeter(retargeter,1/30.)
        start = time.perf_counter()
        qs = batch(clips)
        solve_seconds,compile_seconds = time.perf_counter()-start,batch.compile_time
        print(f"FRAX complete: {solve_seconds:.2f}s including {compile_seconds:.2f}s compilation",flush=True)
    sphere_metrics = _quality_evaluator(robot, retargeter)
    manifest = {"robot":"q1", "fps":30., "joint_names":JOINT_NAMES,"csv_quaternion":"xyzw", "npz_quaternion":"wxyz", "npz_dtype":"float64", "csv_float_format":"%.17g",
                "self_collision_avoidance":args.self_collision, "scales":scales, "solve_seconds":solve_seconds,
                "compile_seconds":compile_seconds,"devices":len(runtime["devices"]),"runtime":runtime,"clips":[],
                "commits":commits,
                "model_dependencies":dependency_fingerprints(MJCF_PATH),
                "model_sha256":hashlib.sha256(MJCF_PATH.read_bytes()).hexdigest(),
                "mapping_sha256":hashlib.sha256(IK_CONFIG.read_bytes()).hexdigest(),
                "collision_sha256":hashlib.sha256(COLLISION.read_bytes()).hexdigest()}
    if args.validate_only:
        # Old trajectories keep the metadata of the solve that produced them.
        manifest = {**previous, "clips": []}
    else:
        manifest["retarget_parameters"] = params
        manifest["retarget_adapter_sha256"] = hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest()
    manifest["input_selection"] = {"bvh_dir":str(bvh_dir),"pattern":pattern,"max_frames":maximum}
    manifest["schema_version"] = 1
    manifest["validation"] = {
        "complete": False, "started_at": datetime.now(timezone.utc).isoformat(), "runtime":runtime,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "collision_sha256": hashlib.sha256(COLLISION.read_bytes()).hexdigest(),
        "pattern": pattern, "max_frames": maximum,
        "quality_profile": load_profile(args.quality_profile),
        "quality_profile_sha256": profile_hash(load_profile(args.quality_profile)),
    }
    for path,q,source in zip(paths,qs,clips):
        q = q.copy()  # JAX's host views are read-only.
        if not np.isfinite(q).all():
            raise ValueError(f"Non-finite FRAX result: {path.name}")
        norms=np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
        if np.any(norms < 1e-6):
            raise ValueError(f"Zero quaternion in FRAX result: {path}")
        if not args.validate_only:
            q[:,3:7] /= norms
            # Keep the exact float64 poses used by QA. FRAX save_motion casts
            # to float32, which can change decisions at a quality threshold.
            np.savez(rawdir/(path.stem+".npz"),q=np.asarray(q,dtype=np.float64),
                     fps=30.,joint_names=np.asarray(JOINT_NAMES))
        csv = np.c_[q[:,:3],q[:,4:7],q[:,3],q[:,7:]]
        np.savetxt(csvdir / (path.stem+".csv"),csv,delimiter=",",fmt="%.17g")
        floor,dist,root_error,foot_error = _evaluate_quality_chunks(sphere_metrics,q,source)
        violation = np.maximum(model.jnt_range[1:,0]-q[:,7:],q[:,7:]-model.jnt_range[1:,1]).max(axis=1)
        physical=physical_metrics(model,q)
        qa=classify_frames(model,q,physical=physical,root_target_error=root_error,foot_target_error=foot_error,
                           sphere_floor=floor,sphere_self_distance=dist,walking=path.stem.startswith("walk"),
                           profile=args.quality_profile)
        reasons,valid=qa["reasons"],qa["accepted"]
        qspeed=physical["joint_speed"]
        physical_depth=physical["physical_self_penetration"]
        record={"name":path.stem,"frames":len(q),"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "min_sphere_floor_m":float(floor.min()),"min_self_sphere_distance_m":float(dist.min()),
                "joint_limit_violation_rad":float(max(0.,violation.max())),"max_joint_speed_rad_s":float(qspeed.max()),
                "physical_self_penetration_m":float(physical_depth.max()),"physical_check_stride":1,
                "root_target_error_m":percentiles(root_error),"foot_target_error_m":percentiles(foot_error),
                "physical_metrics":{key:percentiles(value) for key,value in physical.items()},
                "valid_fraction":float(qa["base_accepted"].mean()),"training_spans":[],
                "npz_sha256":sha256(rawdir/(path.stem+".npz")),"csv_sha256":sha256(csvdir/(path.stem+".csv")),
                "quality":qa["summary"],"verification":qa["verification"]}
        if path.stem.startswith("walk"):
            record["walking_valid_fraction"] = float(valid.mean())
            train_dir=out/"walk_train"; train_dir.mkdir(exist_ok=True)
            for k,(s,e) in enumerate(accepted_span_slices(valid,minimum_seconds=qa["profile"]["minimum_span_seconds"])):
                name=f"{path.stem}_{k:03d}.csv"
                np.savetxt(train_dir/name,csv[s:e],delimiter=",",fmt="%.17g")
                record["training_spans"].append({"file":name,"start":int(s),"frames":int(e-s),
                    "sha256":hashlib.sha256((train_dir/name).read_bytes()).hexdigest()})
        record["rejected_frame_counts"]={reason:int(mask.sum()) for reason,mask in reasons.items()}
        quality_dir=out/"quality";quality_dir.mkdir(exist_ok=True)
        np.savez_compressed(quality_dir/(path.stem+".npz"),accepted=valid,**qa["metrics"],
                            **{f"reject_{key}":value for key,value in reasons.items()})
        record["quality_sha256"] = sha256(quality_dir/(path.stem+".npz"))
        manifest["clips"].append(record)
        write_json(out/"manifest.json",manifest)
        print(f"{path.stem}: {len(q)} frames, valid {record['valid_fraction']:.1%}, physical self penetration {record['physical_self_penetration_m']*1000:.2f}mm, walk spans {len(record['training_spans'])}",flush=True)
    manifest["validation"]["complete"] = True
    manifest["validation"]["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(out/"manifest.json",manifest)
    train_manifest={"robot":"q1","fps":30.,"joint_names":JOINT_NAMES,"root_quaternion":"xyzw",
                    "source_manifest":"../manifest.json","clips":[span for c in manifest['clips'] for span in c['training_spans']]}
    (out/"walk_train").mkdir(exist_ok=True)
    write_json(out/"walk_train/manifest.json",train_manifest)
    print(f"Saved {len(qs)} clips; {len(train_manifest['clips'])} validated walking spans. Manifest: {out/'manifest.json'}",flush=True)


if __name__=="__main__":
    main()
