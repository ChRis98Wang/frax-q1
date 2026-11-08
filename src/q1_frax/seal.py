"""Seal or verify a Q1 archive using full-frame replay and whole-clip splits.

Sealing never edits raw CSV/NPZ/QA evidence. Existing seals are immutable.
Integrity-only verification deliberately reports quality as unverified: hashes
prove byte consistency, while full verification also replays physical geometry
and the original FRAX source-target/sphere measurements.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import mujoco
import numpy as np

from .quality import PHYSICAL_KEYS,accepted_span_slices,classify_frames,load_profile,profile_hash
from .integrity import verify_model_dependencies

HELD_OUT=("walk1_subject2","walk2_subject4")
NAME=re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
HASH=re.compile(r"[0-9a-f]{64}\Z")


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def _json(path):
    def invalid(value): raise ValueError(f"Nonfinite JSON constant: {value}")
    value=json.loads(Path(path).read_text(),parse_constant=invalid)
    if not isinstance(value,dict): raise ValueError("Manifest must be a JSON object")
    return value


def safe_path(root,relative):
    """Dataset-internal paths must be relative and may not traverse symlinks."""
    root=Path(root).resolve()
    if not isinstance(relative,str) or not relative or relative=="." or "\\" in relative:
        raise ValueError("Invalid dataset path")
    path=Path(relative)
    if path.is_absolute() or any(part in ("..", ".") for part in path.parts):
        raise ValueError(f"Unsafe dataset path: {relative}")
    result=root
    for part in path.parts:
        result=result/part
        if result.is_symlink(): raise ValueError(f"Dataset symlink is not allowed: {relative}")
    if not result.resolve().is_relative_to(root): raise ValueError("Dataset path escapes its root")
    return result


def atomic_json(path,value):
    path=Path(path); content=json.dumps(value,indent=2,allow_nan=False)+"\n"
    if path.is_symlink(): raise ValueError("Refusing to replace a symlink")
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,prefix=".seal-",delete=False) as stream:
        stream.write(content);stream.flush();os.fsync(stream.fileno());temporary=Path(stream.name)
    try: temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()


def _hash_check(path,digest,label):
    if not isinstance(digest,str) or not HASH.fullmatch(digest): raise ValueError(f"Missing/invalid {label} SHA256")
    if not path.is_file() or sha256(path)!=digest: raise ValueError(f"{label} SHA256 mismatch: {path}")


def _integer(value,label,minimum=0):
    if isinstance(value,bool) or not isinstance(value,int) or value<minimum: raise ValueError(f"Invalid {label}")
    return value


def _artifact(root,clip,kind):
    suffix="csv" if kind=="csv" else "npz"
    entry=clip.get("files",{}).get(kind,{})
    path=safe_path(root,entry.get("file",f"{kind}/{clip['name']}.{suffix}"))
    digest=entry.get("sha256",clip.get(kind+"_sha256"))
    _hash_check(path,digest,kind)
    return path,digest


def _load_archive(path,shape,joint_names,fps):
    with np.load(path,allow_pickle=False) as archive:
        if not {"q","fps","joint_names"}.issubset(archive.files): raise ValueError("Incomplete FRAX NPZ metadata")
        if archive["q"].dtype!=np.float64: raise ValueError("New quality seals require canonical float64 NPZ poses")
        q=np.asarray(archive["q"],dtype=float)
        if q.shape!=shape or not np.isfinite(q).all(): raise ValueError("Invalid archived poses")
        if archive["fps"].shape!=() or float(archive["fps"])!=fps: raise ValueError("Archived FPS mismatch")
        if archive["joint_names"].tolist()!=joint_names: raise ValueError("Archived joint order mismatch")
    return q


def _audit(dataset,manifest,model,model_path,heldout,replayer,replay):
    if manifest.get("schema_version")!=1:
        raise ValueError("Legacy archive lacks versioned/hash evidence; inspect it as RAW or regenerate into a new directory; never rewrite an old seal")
    if manifest.get("npz_dtype")!="float64": raise ValueError("New seals require explicit npz_dtype=float64 provenance")
    if manifest.get("csv_float_format")!="%.17g": raise ValueError("New seals require lossless csv_float_format=%.17g provenance")
    if manifest.get("validation",{}).get("complete") is not True: raise ValueError("Retarget validation is incomplete")
    _hash_check(model_path,manifest.get("model_sha256"),"physical model")
    verify_model_dependencies(model_path,manifest.get("model_dependencies"))
    cfg=load_profile(manifest["validation"].get("quality_profile"))
    if manifest["validation"].get("quality_profile_sha256")!=profile_hash(cfg): raise ValueError("Quality profile hash mismatch")
    joint_names=[model.joint(i).name for i in range(1,model.njnt)]
    if manifest.get("joint_names")!=joint_names or model.nq!=7+len(joint_names): raise ValueError("Physical joint order mismatch")
    fps=manifest.get("fps")
    if isinstance(fps,bool) or not isinstance(fps,(int,float)) or not math.isfinite(fps) or fps<=0: raise ValueError("Invalid FPS")
    if manifest.get("csv_quaternion")!="xyzw" or manifest.get("npz_quaternion")!="wxyz": raise ValueError("Explicit quaternion conventions are required")
    clips=manifest.get("clips")
    if not isinstance(clips,list) or not clips: raise ValueError("Empty archives cannot pass validation")
    records=[]; names=set(); filenames=set(); rows=[]
    for clip in clips:
        name=clip.get("name","")
        if not isinstance(name,str) or not NAME.fullmatch(name) or name in names: raise ValueError("Duplicate/unsafe source clip name")
        names.add(name); n=_integer(clip.get("frames"),"source frame count",2)
        if not isinstance(clip.get("source_sha256"),str) or not HASH.fullmatch(clip["source_sha256"]): raise ValueError("Missing source clip hash")
        paths={kind:_artifact(dataset,clip,kind)[0] for kind in ("csv","npz","quality")}
        csv=np.loadtxt(paths["csv"],delimiter=",",ndmin=2)
        if csv.shape!=(n,model.nq) or not np.isfinite(csv).all(): raise ValueError(f"Invalid raw CSV: {name}")
        q=csv.copy();q[:,3:7]=csv[:,[6,3,4,5]]
        if not np.allclose(np.linalg.norm(q[:,3:7],axis=1),1.,rtol=0.,atol=cfg["quaternion_norm_tolerance"]):
            raise ValueError(f"Nonunit root quaternion: {name}")
        archived=_load_archive(paths["npz"],q.shape,joint_names,fps)
        if not np.array_equal(q,archived): raise ValueError(f"NPZ and lossless CSV pose mismatch: {name}")
        with np.load(paths["quality"],allow_pickle=False) as source:
            quality={key:source[key].copy() for key in source.files}
        if quality.get("accepted",np.empty(0)).shape!=(n,) or quality["accepted"].dtype!=np.bool_:
            raise ValueError(f"Invalid accepted-frame mask: {name}")
        accepted=quality["accepted"]
        if replay:
            if replayer is None: raise ValueError("Full quality replay requires original source-target evidence")
            evidence=replayer(q,clip)
            required=("root_target_error","foot_target_error","sphere_floor","sphere_self_distance")
            if not isinstance(evidence,dict) or not set(required).issubset(evidence): raise ValueError("Incomplete replay evidence")
            qa=classify_frames(model,q,walking=name.startswith("walk"),fps=fps,profile=cfg,
                               **{key:evidence[key] for key in required})
            if not qa["summary"]["verified"]: raise ValueError(f"Unverified quality evidence: {name}")
            if not np.array_equal(accepted,qa["accepted"]): raise ValueError(f"Accepted mask differs from full replay: {name}")
            for key,value in qa["metrics"].items():
                actual=quality.get(key)
                tolerance=2e-5 if key=="joint_acceleration" else 2e-6
                if actual is None or actual.shape!=value.shape or not np.isfinite(actual).all() or not np.allclose(actual,value,rtol=1e-6,atol=tolerance):
                    raise ValueError(f"Stored quality differs from replay: {name}/{key}")
            for key,value in qa["reasons"].items():
                actual=quality.get("reject_"+key)
                if actual is None or actual.dtype!=np.bool_ or not np.array_equal(actual,value):
                    raise ValueError(f"Stored rejection mask differs from replay: {name}/{key}")
        spans=clip.get("training_spans")
        if not isinstance(spans,list): raise ValueError(f"Missing span list: {name}")
        expected=accepted_span_slices(accepted,fps,cfg["minimum_span_seconds"]) if name.startswith("walk") else []
        actual=[]
        split="eval" if name in heldout else "train"
        for span in spans:
            filename=span.get("file","")
            if not isinstance(filename,str) or not NAME.fullmatch(filename) or not filename.endswith(".csv") or filename in filenames:
                raise ValueError("Duplicate/unsafe span filename")
            filenames.add(filename)
            start=_integer(span.get("start"),"span start")
            count=_integer(span.get("frames"),"span frame count",1)
            if start+count>n or count<math.ceil(fps*cfg["minimum_span_seconds"]-1e-9): raise ValueError(f"Invalid span length: {filename}")
            actual.append((start,start+count))
            candidates=[safe_path(dataset,f"walk_{kind}/{filename}") for kind in (split,"train","eval")]
            source_path=next((path for path in candidates if path.is_file()),None)
            if source_path is None: raise ValueError(f"Missing accepted span: {filename}")
            _hash_check(source_path,span.get("sha256"),"span")
            values=np.loadtxt(source_path,delimiter=",",ndmin=2)
            if values.shape!=(count,model.nq) or not np.isfinite(values).all() or not np.array_equal(values,csv[start:start+count]):
                raise ValueError(f"Span is not its declared source slice: {filename}")
            if not accepted[start:start+count].all(): raise ValueError(f"Rejected frame in exported span: {filename}")
            sealed_span=dict(span,source_clip=name,split=split)
            if "source_clip" in span and span["source_clip"]!=name: raise ValueError("Span source clip mismatch")
            if "split" in span and span["split"]!=split: raise ValueError("Source clip crosses dataset split")
            records.append((split,source_path,sealed_span))
        if actual!=expected: raise ValueError(f"Declared spans differ from all eligible contiguous spans: {name}")
        rows.append({"source_clip":name,"frames":n,"accepted_frames":int(accepted.sum()),"exported_frames":sum(e-s for s,e in actual),
                     "spans":len(spans),"split":split if spans else "unused"})
    counts={split:{"source_clips":sorted({span["source_clip"] for kind,_,span in records if kind==split}),
                   "spans":sum(kind==split for kind,_,_ in records),
                   "frames":sum(span["frames"] for kind,_,span in records if kind==split)} for split in ("train","eval")}
    if any(counts[split]["frames"]==0 for split in counts): raise ValueError("Both train and held-out eval require accepted motion")
    if set(counts["train"]["source_clips"])&set(counts["eval"]["source_clips"]): raise ValueError("Source clip leakage between splits")
    return records,counts,rows,cfg


def seal_dataset(dataset,*,held_out=HELD_OUT,model_path=None,source_dir=None,replayer=None,verify_only=False,integrity_only=False):
    """Seal fresh data or verify an immutable seal. Injected replayers aid tests."""
    dataset=Path(dataset).expanduser().resolve()
    if not verify_only:
        for parent in dataset.parents:
            index=parent/"manifest.json"
            if index.is_file() and _json(index).get("seal",{}).get("complete") is True:
                raise ValueError("Cannot seal or modify data inside an already sealed ancestor")
    manifest_path=safe_path(dataset,"manifest.json")
    manifest=_json(manifest_path); original=deepcopy(manifest)
    sealed=manifest.get("seal",{}).get("complete") is True
    if sealed and not verify_only: raise ValueError("Dataset is already sealed; use verify_only and never rewrite the source")
    if verify_only and not sealed: raise ValueError("Dataset has no completed seal")
    if integrity_only and not verify_only: raise ValueError("Integrity-only checks cannot create a quality seal")
    if sealed:
        held_out=manifest["seal"].get("heldout_source_clips",())
        payload={key:value for key,value in manifest.items() if key!="seal"}
        if hashlib.sha256(canonical(payload)).hexdigest()!=manifest["seal"].get("manifest_payload_sha256"):
            raise ValueError("Sealed manifest payload hash mismatch")
    heldout=tuple(held_out)
    if not heldout or len(set(heldout))!=len(heldout) or any(not isinstance(n,str) or not NAME.fullmatch(n) for n in heldout):
        raise ValueError("Held-out source clip names must be unique and nonempty")
    if model_path is None:
        from .model import MJCF_PATH
        model_path=MJCF_PATH
    model_path=Path(model_path).expanduser().resolve()
    _hash_check(model_path,manifest.get("model_sha256"),"physical model")
    verify_model_dependencies(model_path,manifest.get("model_dependencies"))
    model=mujoco.MjModel.from_xml_path(str(model_path))
    if not integrity_only and replayer is None:
        from .retarget import make_quality_replayer
        replayer=make_quality_replayer(manifest,source_dir=source_dir)
    records,counts,rows,cfg=_audit(dataset,manifest,model,model_path,heldout,replayer,not integrity_only)
    split_manifests={split:{"schema_version":1,"robot":"q1","fps":manifest["fps"],"joint_names":manifest["joint_names"],
                            "root_quaternion":"xyzw","source_manifest":"../manifest.json","split":split,
                            "model_sha256":manifest["model_sha256"],"quality_profile_sha256":profile_hash(cfg),
                            "heldout_source_clips":list(heldout),"clips":[span for kind,_,span in records if kind==split]}
                     for split in ("train","eval")}
    if verify_only:
        for split,expected in split_manifests.items():
            path=safe_path(dataset,f"walk_{split}/manifest.json")
            _hash_check(path,manifest["seal"].get("split_manifest_sha256",{}).get(split),"split manifest")
            if _json(path)!=expected: raise ValueError(f"{split} manifest content mismatch")
            for row in expected["clips"]: _hash_check(safe_path(dataset,f"walk_{split}/{row['file']}"),row["sha256"],"split span")
        if manifest["seal"].get("split")!=counts: raise ValueError("Sealed split counts mismatch")
        return {"verified":not integrity_only,"integrity_verified":True,"quality_replay":"unverified" if integrity_only else "all_frame_evidence_verified_exported_spans_passed",
                "split":counts,"clips":rows,"profile_sha256":profile_hash(cfg)}
    # Validate every eventual target before moving any span or creating output.
    for split in split_manifests:
        safe_path(dataset,f"walk_{split}/manifest.json")
    for split,source,span in records:
        target=safe_path(dataset,f"walk_{split}/{span['file']}")
        if source!=target and target.exists(): raise ValueError(f"Refusing to overwrite an existing split artifact: {target}")
    if _json(manifest_path)!=original: raise ValueError("Source manifest changed during sealing")
    for split,source,span in records:
        target=safe_path(dataset,f"walk_{split}/{span['file']}");target.parent.mkdir(parents=True,exist_ok=True)
        if source!=target:
            if target.exists(): raise ValueError(f"Refusing to overwrite an existing split artifact: {target}")
            source.replace(target)
    for clip in manifest["clips"]:
        clip["split"]=("eval" if clip["name"] in heldout else "train") if clip["training_spans"] else "unused"
        clip["training_spans"]=[span for _,_,span in records if span["source_clip"]==clip["name"]]
    for split,value in split_manifests.items(): atomic_json(safe_path(dataset,f"walk_{split}/manifest.json"),value)
    manifest["seal"]={"schema_version":1,"complete":True,"sealed_at":datetime.now(timezone.utc).isoformat(),
                      "split_policy":"entire_source_clip","heldout_source_clips":list(heldout),"split":counts,
                      "model_sha256":manifest["model_sha256"],"quality_profile_sha256":profile_hash(cfg),
                      "manifest_payload_sha256":hashlib.sha256(canonical({k:v for k,v in manifest.items() if k!="seal"})).hexdigest(),
                      "split_manifest_sha256":{s:sha256(safe_path(dataset,f"walk_{s}/manifest.json")) for s in counts},
                      "quality_replay":"all_frames_physical_and_source_targets",
                      "checks":["finite normalized poses","original source SHA256","CSV/NPZ consistency","all-frame actual geometry replay",
                                "all-frame FRAX sphere and source-target replay","stored QA and rejection-mask equality",
                                "exact source slices","minimum 3 seconds","whole source clip split","artifact SHA256",
                                "complete external model asset SHA256 and size inventory"]}
    # Raw evidence is byte-immutable; only the previously unsealed index is committed.
    if _json(manifest_path)!=original: raise ValueError("Source manifest changed during sealing")
    atomic_json(manifest_path,manifest)
    return {"sealed":True,"verified":True,"split":counts,"clips":rows,"profile_sha256":profile_hash(cfg)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--model",type=Path)
    parser.add_argument("--source-dir",type=Path)
    parser.add_argument("--held-out",nargs="+",default=list(HELD_OUT))
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--integrity-only",action="store_true",help="Check hashes without claiming source/geometry replay verification")
    args=parser.parse_args(argv)
    result=seal_dataset(args.dataset,held_out=args.held_out,model_path=args.model,source_dir=args.source_dir,
                        verify_only=args.verify_only,integrity_only=args.integrity_only)
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0


if __name__=="__main__": raise SystemExit(main())
