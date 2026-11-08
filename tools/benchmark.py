#!/usr/bin/env python3
"""Reproduce CPU/CUDA full-CLI and synchronized warm FRAX benchmarks.

Run from the repository root with prepared Q1_FRAX_DATA. Each backend gets a
fresh CLI process and its own output directory. Warm samples are measured in a
second process after one untimed-for-throughput compilation/warmup call. The
same OfflineRetargeter instance is reused for all warm samples. Source motions
and generated artifacts remain in --work-dir; the public JSON contains only
names, hashes, settings, hardware, metrics, and timings.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def hardware():
    cpu="unknown"
    info=Path("/proc/cpuinfo")
    if info.is_file():
        cpu=next((line.split(":",1)[1].strip() for line in info.read_text().splitlines() if line.startswith("model name")),cpu)
    gpu=[]
    try:
        output=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader,nounits"],text=True,stderr=subprocess.DEVNULL)
        for line in output.strip().splitlines():
            name,driver,memory=map(str.strip,line.split(","))
            gpu.append({"name":name,"driver_version":driver,"memory_mib":int(memory)})
    except (FileNotFoundError,subprocess.CalledProcessError):
        pass
    return {"cpu_model":cpu,"logical_cpu_count":os.cpu_count(),"architecture":platform.machine(),"gpus":gpu}


def worker(args):
    os.environ["Q1_FRAX_DATA"]=str(args.data_dir.expanduser().resolve())
    from q1_frax.retarget import configure_backend,load_parameters,verify_frax_installation
    runtime=configure_backend(args.worker,1 if args.worker=="cpu" else None)
    import numpy as np
    from frax.ik import OfflineRetargeter,smooth_poses
    from q1_frax.adapter import load_q1,load_lafan1,make_retargeter,contact_aware_inputs
    from q1_frax.paths import sha256
    from q1_frax.integrity import verify_model_dependencies
    from q1_frax.model import MJCF_PATH
    verify_frax_installation()
    manifest=json.loads((args.work_dir/"manifest.json").read_text())
    verify_model_dependencies(MJCF_PATH,manifest.get("model_dependencies"))
    params=load_parameters(manifest["retarget_parameters"])
    contact=params.pop("contact_aware");swing=params.pop("contact_swing_margin")
    robot,model=load_q1(rebuild=False)
    retargeter,bones,scales=make_retargeter(robot,model,manifest["self_collision_avoidance"],**params)
    directory=Path(manifest["input_selection"]["bvh_dir"])
    maximum=manifest["input_selection"]["max_frames"]
    clips=[]
    for record in manifest["clips"]:
        path=directory/(record["name"]+".bvh")
        if sha256(path)!=record["source_sha256"]:raise ValueError("Benchmark source hash changed")
        poses,names=load_lafan1(path)
        if maximum:poses=poses[:maximum]
        poses=smooth_poses(poses,1/30.,sigma_rot=.05)
        clips.append(contact_aware_inputs(poses,names,bones,scales["leg_scale"],swing_margin=swing)
                     if contact else poses[:,[names.index(n) for n in bones]])
    batch=OfflineRetargeter(retargeter,1/30.)
    begin=time.perf_counter();baseline=batch(clips);warmup=time.perf_counter()-begin
    # Pinned FRAX materializes every dispatched result with np.asarray before
    # returning. Assert this blocking host boundary instead of timing dispatch.
    if not all(isinstance(q,np.ndarray) and np.isfinite(q).all() for q in baseline):
        raise ValueError("FRAX did not return finite synchronized NumPy host results")
    compiled=batch.compile_time
    measurements=[];repeat_differences=[]
    for _ in range(args.warm_repeats):
        begin=time.perf_counter();qs=batch(clips);elapsed=time.perf_counter()-begin
        if batch.compile_time!=compiled:raise ValueError("Warm sample unexpectedly recompiled")
        if not all(isinstance(q,np.ndarray) and np.isfinite(q).all() for q in qs):raise ValueError("Invalid warm result")
        measurements.append(elapsed)
        repeat_differences.append(max(float(np.max(np.abs(q-reference))) for q,reference in zip(qs,baseline)))
    native_versions={}
    if args.worker=="cuda":
        from jax_cuda13_plugin import _versions
        for name in ("cuda_runtime_get_version","cuda_driver_get_version","cudnn_get_version","cublas_get_version",
                     "cufft_get_version","cusolver_get_version","cusparse_get_version"):
            function=getattr(_versions,name,None)
            if function is not None:native_versions[name]=int(function())
    result={"runtime":runtime,"native_cuda_versions":native_versions,"cuda_installation_mode":args.cuda_mode,"python":platform.python_version(),
            "versions":{name:importlib.metadata.version(name) for name in ("q1-frax","frax","jax","jaxlib","numpy","scipy","mujoco")},
            "warmup_seconds_including_compile":warmup,"warmup_compile_seconds":compiled,
            "warm_seconds":measurements,"warm_median_seconds":statistics.median(measurements),
            "repeat_max_abs_q_difference":repeat_differences,
            "synchronization":"OfflineRetargeter np.asarray device-to-host materialization; all return values asserted NumPy arrays",
            "precision":"JAX float64 enabled; source/solver/settings identical to full CLI; physical QA was performed by the preceding CLI"}
    write_json(args.output,result)



def probe_environment(args):
    os.environ["Q1_FRAX_DATA"]=str(args.data_dir.expanduser().resolve())
    from q1_frax.retarget import configure_backend,check_solver_backend,UnsupportedSolverBackend
    runtime=configure_backend(args.probe,1 if args.probe=="cpu" else None)
    import jax.numpy as jnp
    value=float((jnp.eye(2,dtype=jnp.float64)@jnp.eye(2,dtype=jnp.float64)).sum().block_until_ready())
    result={"runtime":runtime,"float64_device_computation_passed":value==2.,"cuda_installation_mode":args.cuda_mode,
            "versions":{name:importlib.metadata.version(name) for name in ("q1-frax","frax","elastiqp","jax","jaxlib")}}
    if args.probe=="cuda":
        from jax_cuda13_plugin import _versions
        result["native_cuda_versions"]={name:int(getattr(_versions,name)()) for name in
                ("cuda_runtime_get_version","cudnn_get_version","cublas_get_version","cusolver_get_version")}
    try:
        result["qp_solver"]=check_solver_backend(runtime)
        result["qp_solver_status"]="supported"
    except UnsupportedSolverBackend as exc:
        result["qp_solver_status"]="unsupported"
        result["reason"]=str(exc)
        result["original_ffi_error"]=str(exc.__cause__)
    write_json(args.output,result)


def compare_artifacts(cpu,gpu):
    import numpy as np
    cm=json.loads((cpu/"manifest.json").read_text());gm=json.loads((gpu/"manifest.json").read_text())
    for key in ("commits","model_sha256","model_dependencies","mapping_sha256","collision_sha256","retarget_parameters","self_collision_avoidance"):
        if cm[key]!=gm[key]:raise ValueError(f"CPU/GPU provenance differs: {key}")
    if cm["validation"]["quality_profile"]!=gm["validation"]["quality_profile"]:raise ValueError("Quality profiles differ")
    if [(c["name"],c["source_sha256"],c["frames"]) for c in cm["clips"]]!=[(c["name"],c["source_sha256"],c["frames"]) for c in gm["clips"]]:
        raise ValueError("CPU/GPU inputs differ")
    differences=[]
    for left,right in zip(cm["clips"],gm["clips"]):
        name=left["name"]
        with np.load(cpu/"npz"/(name+".npz"),allow_pickle=False) as z:a=z["q"].copy()
        with np.load(gpu/"npz"/(name+".npz"),allow_pickle=False) as z:b=z["q"].copy()
        with np.load(cpu/"quality"/(name+".npz"),allow_pickle=False) as z:qa=z["accepted"].copy()
        with np.load(gpu/"quality"/(name+".npz"),allow_pickle=False) as z:qb=z["accepted"].copy()
        cosine=np.abs(np.sum(a[:,3:7]*b[:,3:7],axis=1))/(np.linalg.norm(a[:,3:7],axis=1)*np.linalg.norm(b[:,3:7],axis=1))
        differences.append({"source_clip":name,"max_abs_joint_difference_rad":float(np.max(np.abs(a[:,7:]-b[:,7:]))),
                            "max_root_position_difference_m":float(np.max(np.linalg.norm(a[:,:3]-b[:,:3],axis=1))),
                            "max_root_orientation_difference_rad":float(np.max(2*np.arccos(np.clip(cosine,0,1)))),
                            "accepted_mask_difference_frames":int(np.count_nonzero(qa!=qb)),
                            "cpu_accepted_frames":int(qa.sum()),"gpu_accepted_frames":int(qb.sum())})
    return differences


def run(args):
    cache=args.data_dir.expanduser().resolve();work=args.work_dir.expanduser().resolve()
    if work.exists() and any(work.iterdir()):raise ValueError("Benchmark --work-dir must be new or empty")
    work.mkdir(parents=True,exist_ok=True)
    env={**os.environ,"Q1_FRAX_DATA":str(cache),"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","JAX_ENABLE_COMPILATION_CACHE":"false"}
    source_preexisting=(cache/"lafan1_bvh").is_dir()
    runs={};records=None;metadata=None
    script=str(Path(__file__).resolve())
    for backend,python in (("cpu",args.cpu_python),("cuda",args.gpu_python)):
        probe_path=work/(backend+"_probe.json")
        subprocess.run([str(python),script,"--probe",backend,"--data-dir",str(cache),"--work-dir",str(work),
                        "--output",str(probe_path),"--cuda-mode",args.cuda_mode],env=env,check=True)
        probe=json.loads(probe_path.read_text())
        out=work/backend
        command=[str(python),"-m","q1_frax.retarget","--platform",backend,"--pattern",args.pattern,
                 "--max-frames",str(args.max_frames),"--out",str(out)]
        if backend=="cpu":command += ["--devices","1"]
        if args.config:command += ["--config",str(args.config.resolve())]
        begin=time.perf_counter()
        completed=subprocess.run(command,env=env,capture_output=True,text=True)
        wall=time.perf_counter()-begin
        (work/(backend+"_stdout.log")).write_text(completed.stdout)
        (work/(backend+"_stderr.log")).write_text(completed.stderr)
        print(completed.stdout,end="",flush=True)
        if completed.returncode:
            if probe["qp_solver_status"]!="unsupported":
                raise RuntimeError(f"Unexpected {backend} benchmark failure; inspect private stderr log")
            if out.exists():raise RuntimeError("Unsupported CUDA solver created output before failing")
            runs[backend]={"status":"unsupported_by_locked_qp_backend","environment_probe":probe,"runtime":probe["runtime"],
                           "retarget_completed":False,"solver_total_seconds_including_compile":None,"compile_seconds":None,
                           "solver_frames_per_second_including_compile":None,"whole_cli_frames_per_second":None,
                           "warm":None,"quality":None,"failed_preflight_wall_seconds":wall,"returncode":completed.returncode,
                           "output_created":False,"reason":probe["reason"]}
            print(probe["reason"],flush=True)
            continue
        manifest=json.loads((out/"manifest.json").read_text());frames=sum(c["frames"] for c in manifest["clips"])
        warm_path=work/(backend+"_warm.json")
        subprocess.run([str(python),script,"--worker",backend,"--data-dir",str(cache),
                        "--work-dir",str(out),"--warm-repeats",str(args.warm_repeats),"--output",str(warm_path),
                        "--cuda-mode",args.cuda_mode],env=env,check=True)
        warm=json.loads(warm_path.read_text())
        warm["warm_frames_per_second"]=[frames/t for t in warm["warm_seconds"]]
        warm["warm_median_frames_per_second"]=frames/warm["warm_median_seconds"]
        runs[backend]={"status":"completed","environment_probe":probe,"runtime":manifest["runtime"],
                       "retarget_completed":True,"solver_total_seconds_including_compile":manifest["solve_seconds"],
                       "compile_seconds":manifest["compile_seconds"],"whole_cli_wall_seconds":wall,
                       "solver_frames_per_second_including_compile":frames/manifest["solve_seconds"],
                       "whole_cli_frames_per_second":frames/wall,"warm":warm,
                       "quality":{"physical_check_stride":1,
                                  "accepted_span_count":sum(len(c["training_spans"]) for c in manifest["clips"]),
                                  "accepted_span_frames":sum(s["frames"] for c in manifest["clips"] for s in c["training_spans"]),
                                  "max_physical_self_penetration_m":max(c["physical_self_penetration_m"] for c in manifest["clips"]),
                                  "max_physical_floor_penetration_m":max(c["physical_metrics"]["physical_floor_penetration"]["max"] for c in manifest["clips"]),
                                  "source_clips":[{"name":c["name"],"accepted_frames":c["quality"]["accepted_frames"],
                                                   "spans":[{"start":t["start"],"frames":t["frames"]} for t in c["training_spans"]]} for c in manifest["clips"]]}}
        records=[{key:c[key] for key in ("name","frames","source_sha256")} for c in manifest["clips"]]
        metadata={key:manifest[key] for key in ("commits","model_sha256","model_dependencies","mapping_sha256","collision_sha256","retarget_adapter_sha256","retarget_parameters","self_collision_avoidance")}
        metadata["quality_profile"]=manifest["validation"]["quality_profile"]
    comparison=(compare_artifacts(work/"cpu",work/"cuda") if all(r["retarget_completed"] for r in runs.values()) else
                {"status":"not_comparable","reason":"The locked QP backend did not produce a CUDA trajectory; no GPU quality or FPS is claimed."})
    result={"schema_version":1,"recorded_at_utc":datetime.now(timezone.utc).isoformat(),"hardware":hardware(),
            "inputs":{"source_clips":records,"total_frames":sum(c["frames"] for c in records),"fps":30.,"pattern":args.pattern,
                      "max_frames_per_clip":args.max_frames,"extracted_BVH_preexisting":source_preexisting},
            "configuration_and_provenance":metadata,"results":runs,"cpu_gpu_comparison":comparison,
            "protocol":{"full_cli":"Fresh process per platform; includes imports, source checks, model generation, BVH read, IK, every-frame physical/source QA and file export; excludes download/install, capability probe, seal and rendering.",
                        "warm":f"Separate process per successful platform: first solve compiles and warms; {args.warm_repeats} subsequent solves use the same OfflineRetargeter instance and shapes. GPU-to-host NumPy conversion synchronizes each sample.",
                        "warm_repeats":args.warm_repeats,"precision":"float64 for both; identical constraints, contact-aware targets and QA profile",
                        "cuda_installation_mode":args.cuda_mode,
                        "cuda_local_note":"Local CUDA mode uses CUDA13 ptxas/libdevice and compatible CUDA13/cuDNN/NCCL shared libraries through PATH/LD_LIBRARY_PATH and XLA_FLAGS --xla_gpu_cuda_data_dir. Native libraries are reused read-only; Python environments are separate.",
                        "jax_persistent_compilation_cache_enabled":False,"cpu_devices":1,
                        "threads":{"OPENBLAS_NUM_THREADS":1,"OMP_NUM_THREADS":1},
                        "reproduce":"python tools/benchmark.py --cpu-python .venv-gpu/bin/python --gpu-python .venv-gpu/bin/python --data-dir \"$Q1_FRAX_DATA\" --work-dir .local-data/benchmark-new --output benchmarks/comparison-new.json --warm-repeats 3 --cuda-mode local",
                        "cli_pattern_note":"The declared glob matches the exact source clip names above; do not infer the count from character-class combinations."},
            "upstream_evidence":{"elastiqp_cpu_ffi":"https://github.com/StanfordASL/elastiqp/blob/v0.2.1/python/elastiqp/jax.py#L61-L63",
                                 "frax_qp_call":"https://github.com/StanfordASL/frax/blob/f5e87ab59a0b82ba43095e882c1b88c9b0dacf07/frax/ik/solver.py#L183-L194"},
            "limitations":["One machine and two truncated clips, not a full-dataset or cross-hardware benchmark.",
                           "Warm measurements cover IK only; they do not repeat full QA/file export.",
                           "No float32, collision disabling, changed quality threshold, alternate solver, or CPU fallback for CUDA was used."]}
    write_json(args.output,result)
    print(json.dumps({"output":str(args.output),"results":{key:{"status":r["status"],"runtime":r["runtime"],
                     "full_cli_s":r.get("whole_cli_wall_seconds"),"solver_s":r["solver_total_seconds_including_compile"],
                     "warm_median_s":r["warm"]["warm_median_seconds"] if r["warm"] else None} for key,r in runs.items()}},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-python",type=Path,default=Path(sys.executable))
    parser.add_argument("--gpu-python",type=Path,default=Path(sys.executable))
    parser.add_argument("--data-dir",type=Path,default=Path(os.environ.get("Q1_FRAX_DATA",Path.home()/".cache/q1-frax")))
    parser.add_argument("--work-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pattern",default="walk[12]_subject[23].bvh")
    parser.add_argument("--max-frames",type=int,default=1200)
    parser.add_argument("--warm-repeats",type=int,default=3)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--cuda-mode",choices=("pip","local","unspecified"),default="unspecified",help="Record how CUDA native libraries were provided")
    parser.add_argument("--worker",choices=("cpu","cuda"),help=argparse.SUPPRESS)
    parser.add_argument("--probe",choices=("cpu","cuda"),help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.warm_repeats<1:parser.error("--warm-repeats must be positive")
    if args.probe:probe_environment(args)
    elif args.worker:worker(args)
    else:run(args)


if __name__=="__main__":main()
