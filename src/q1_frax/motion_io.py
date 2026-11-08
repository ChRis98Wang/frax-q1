"""Strict, read-only motion loading for Q1 FRAX reference inspection.

Sealed CSV spans and raw FRAX NPZ files have different provenance. A raw file
never becomes accepted merely because its current pose passes a collision test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from .model import JOINT_NAMES, MJCF_PATH


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    def invalid(value):
        raise ValueError(f"Non-finite manifest constant: {value}")
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError(f"Manifest must be an object: {path}")
    return value


def _verify(path, expected, description):
    actual = sha256(path)
    if expected is not None and (not isinstance(expected, str) or actual != expected):
        raise ValueError(f"{description} SHA256 mismatch: {path}")
    return actual


def _model_digest(model_path):
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Q1 model is missing: {model_path}. Run explicit asset setup/model build, or pass --model.")
    return sha256(model_path)


def _model_dependencies(meta, model_path):
    """Only versioned evidence can establish external asset identity."""
    if "schema_version" not in meta:
        return None
    if type(meta["schema_version"]) is not int or meta["schema_version"] != 1:
        raise ValueError("Unsupported versioned motion schema")
    from .integrity import verify_model_dependencies
    expected = meta.get("model_dependencies")
    verify_model_dependencies(model_path, expected)
    return dict(expected)


def _validate_metadata(meta):
    if list(meta.get("joint_names", [])) != list(JOINT_NAMES):
        raise ValueError("Motion joint order differs from the 22-joint Q1 order")
    value = meta.get("fps")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Motion fps must be a positive finite number")
    fps = float(value)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Motion fps must be positive and finite")
    return fps


def _validate_qpos(qpos, count=None):
    if qpos.ndim != 2 or qpos.shape[1] != 7+len(JOINT_NAMES) or not len(qpos):
        raise ValueError("Expected nonempty Q1 poses with shape [frames,29]")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or len(qpos) != count):
        raise ValueError("Motion frame count differs from its manifest")
    if not np.isfinite(qpos).all():
        raise ValueError("Motion contains non-finite poses")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1., rtol=0., atol=1e-5):
        raise ValueError("Root quaternions must be normalized; data is never silently repaired")


def _safe_relative(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Declared dataset paths must be nonempty and relative")
    path = (Path(root)/value).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Declared motion path escapes the dataset directory")
    return path


@dataclass(frozen=True)
class Motion:
    path: Path
    qpos: np.ndarray
    fps: float
    kind: str
    sha256: str
    model_sha256: str
    hash_verified: bool
    record: dict = field(default_factory=dict)
    manifest_path: Path | None = None
    rejection_masks: dict = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    source_manifest_path: Path | None = None
    model_dependencies: dict | None = None

    @property
    def duration(self):
        return len(self.qpos)/self.fps

    @property
    def dataset_root(self):
        if self.manifest_path is None:
            return None
        return self.manifest_path.parent.parent if self.kind == "sealed" else self.manifest_path.parent

    @property
    def dependency_identity_verified(self):
        return self.model_dependencies is not None

    def status(self, frame):
        if not 0 <= frame < len(self.qpos):
            raise IndexError("Reference frame is outside the motion")
        reasons = [key for key, value in self.rejection_masks.items() if bool(value[frame])]
        if self.kind == "sealed":
            if not self.dependency_identity_verified:
                return "LEGACY SEALED / ASSET IDENTITY UNVERIFIED", reasons
            return "SEALED / ACCEPTED REFERENCE", reasons
        return ("RAW REJECTED FRAME" if reasons else "RAW / NOT ACCEPTED"), reasons

    def provenance(self):
        return {"motion": str(self.path), "kind": self.kind,
                "frames": len(self.qpos), "fps": self.fps, "quaternion": "wxyz",
                "duration_seconds": self.duration, "source_sha256": self.sha256,
                "model_sha256": self.model_sha256, "source_hash_verified": self.hash_verified,
                "dependency_identity_verified": self.dependency_identity_verified,
                "model_dependencies": self.model_dependencies,
                "manifest": None if self.manifest_path is None else str(self.manifest_path),
                "source_manifest": None if self.source_manifest_path is None else str(self.source_manifest_path),
                "notes": list(self.notes), "mode": "kinematic reference replay, never a policy"}


def guard_output_paths(motion, model_path, *outputs):
    """Preflight every output before writing; reference datasets stay read-only."""
    paths = [Path(path).expanduser() for path in outputs if path is not None]
    if any(path.is_symlink() for path in paths):
        raise ValueError("Output paths must not be symbolic links")
    targets = [path.resolve() for path in paths]
    if len(set(targets)) != len(targets):
        raise ValueError("Render image/video, sidecar, and collision report outputs must be distinct")
    inputs = {motion.path.resolve(), Path(model_path).expanduser().resolve()}
    if motion.source_manifest_path is not None:
        inputs.add(motion.source_manifest_path.resolve())
    if motion.model_dependencies is not None:
        inputs.update((Path(model_path).expanduser().absolute().parent/name).resolve()
                      for name in motion.model_dependencies["files"])
    dataset = motion.dataset_root
    for target in targets:
        if dataset is not None and target.is_relative_to(dataset.resolve()):
            raise ValueError("Outputs must be outside the read-only reference dataset directory")
        if target in inputs:
            raise ValueError("Output cannot overwrite an input motion, model, dependency, or manifest")
        for parent in target.parents:
            manifest = parent/"manifest.json"
            if manifest.is_file() and _json(manifest).get("seal", {}).get("complete") is True:
                raise ValueError("Output cannot write into any sealed reference dataset")
    return targets


def verify_motion_dependencies(motion, model_path):
    """Recheck validated assets at render boundaries, including post-load edits."""
    if motion.model_dependencies is not None:
        from .integrity import verify_model_dependencies
        verify_model_dependencies(model_path, motion.model_dependencies)


def load_motion(path, *, model_path=MJCF_PATH, archive=False, expected_sha256=None):
    """Load a sealed CSV or raw FRAX NPZ, validating all declared metadata.

    Bare FRAX NPZ uses the FRAX-defined WXYZ q format and must contain fps and
    joint_names. Its digest is reported but is unverified without a trusted
    --sha256 or a manifest file digest. Known mismatches are always rejected.
    --archive allows declared raw CSV inspection; it never bypasses validation.
    """
    path = Path(path).expanduser().resolve()
    model_hash = _model_digest(model_path)
    digest = _verify(path, expected_sha256, "Motion")
    if path.suffix.lower() == ".csv" and not archive:
        return _load_sealed_csv(path, model_path, model_hash, digest)
    if path.suffix.lower() not in (".npz", ".csv"):
        raise ValueError("Supported inputs are sealed Q1 CSV and raw FRAX NPZ/CSV")
    root = path.parent.parent if path.parent.name in ("csv", "npz") else path.parent
    manifest_path = root/"manifest.json"
    meta, record, notes, masks = {}, {}, [], {}
    verified = expected_sha256 is not None
    dependencies = None
    if manifest_path.exists():
        meta = _json(manifest_path)
        manifest_fps = _validate_metadata(meta)
        if meta.get("model_sha256") != model_hash:
            raise ValueError("Raw dataset and physical Q1 model SHA256 differ")
        dependencies = _model_dependencies(meta, model_path)
        records = [row for row in meta.get("clips", []) if row.get("name") == path.stem]
        if len(records) != 1:
            raise ValueError("Raw file must be uniquely declared in its dataset manifest")
        record = dict(records[0])
        kind = path.suffix.lower()[1:]
        file_record = record.get("files", {}).get(kind, {})
        declared = file_record.get("sha256", record.get(kind+"_sha256"))
        if file_record.get("file") and _safe_relative(root, file_record["file"]) != path:
            raise ValueError("Raw file path differs from its manifest declaration")
        _verify(path, declared, "Raw motion")
        verified |= declared is not None
        quality_record = record.get("files", {}).get("quality", {})
        quality = _safe_relative(root, quality_record.get("file", f"quality/{path.stem}.npz"))
        if quality.is_file():
            quality_hash = quality_record.get("sha256", record.get("quality_sha256"))
            _verify(quality, quality_hash, "Quality archive")
            if quality_hash is None:
                notes.append("Legacy quality masks have no declared digest; diagnostic only")
            with np.load(quality, allow_pickle=False) as quality_data:
                for key in quality_data.files:
                    if key.startswith("reject_"):
                        masks[key.removeprefix("reject_")] = quality_data[key].astype(bool)
    else:
        manifest_path = None
        manifest_fps = None
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as source:
            if not {"q", "fps", "joint_names"} <= set(source.files):
                raise ValueError("FRAX NPZ requires q, fps, and joint_names")
            fps = _validate_metadata({"fps": source["fps"].item(), "joint_names": source["joint_names"].tolist()})
            convention = str(source["root_quaternion"].item()) if "root_quaternion" in source else "wxyz"
            if convention != "wxyz" or meta.get("npz_quaternion", "wxyz") != "wxyz":
                raise ValueError("FRAX NPZ q must use WXYZ root quaternions")
            qpos = np.array(source["q"], dtype=np.float64, copy=True)
    else:
        if not meta or meta.get("csv_quaternion") != "xyzw":
            raise ValueError("Raw CSV requires a dataset manifest declaring XYZW and joint order")
        fps = manifest_fps
        raw = np.loadtxt(path, delimiter=",", ndmin=2)
        if raw.shape[1] != 29:
            raise ValueError("Raw CSV must have 29 columns")
        qpos = np.c_[raw[:, :3], raw[:, 6], raw[:, 3:6], raw[:, 7:]]
    if manifest_fps is not None and fps != manifest_fps:
        raise ValueError("NPZ fps differs from its manifest")
    _validate_qpos(qpos, record.get("frames"))
    if any(value.shape != (len(qpos),) for value in masks.values()):
        raise ValueError("Quality mask frame count differs from raw motion")
    if not verified:
        notes.append("No trusted file digest supplied: raw motion integrity is unverified")
    if dependencies is None:
        notes.append("Model dependency identity is unverified: no versioned dependency evidence")
    notes.append("Raw motion is not an accepted sealed training/evaluation span")
    qpos.setflags(write=False)
    return Motion(path, qpos, fps, "raw", digest, model_hash, verified,
                  record, manifest_path, masks, tuple(notes), model_dependencies=dependencies)


def _load_sealed_csv(path, model_path, model_hash, digest):
    manifest_path = path.parent/"manifest.json"
    meta = _json(manifest_path)
    fps = _validate_metadata(meta)
    if meta.get("root_quaternion") != "xyzw":
        raise ValueError("Sealed CSV manifest must declare XYZW root quaternions")
    records = [row for row in meta.get("clips", []) if isinstance(row, dict) and row.get("file") == path.name]
    if len(records) != 1:
        raise ValueError("CSV is not uniquely declared in the active split manifest")
    record = dict(records[0])
    if not record.get("sha256"):
        raise ValueError("Accepted CSV requires a declared SHA256")
    _verify(path, record["sha256"], "Sealed CSV")
    source_value = meta.get("source_manifest")
    if not isinstance(source_value, str):
        raise ValueError("Split manifest must identify its sealed source manifest")
    source_path = Path(source_value).expanduser()
    source_path = source_path if source_path.is_absolute() else (path.parent/source_path).resolve()
    source = _json(source_path)
    seal, validation = source.get("seal", {}), source.get("validation", {})
    if seal.get("complete") is not True or validation.get("complete") is not True:
        raise ValueError("Source dataset is not completely validated and sealed")
    _validate_metadata(source)
    if source.get("model_sha256") != model_hash or meta.get("model_sha256", model_hash) != model_hash:
        raise ValueError("Sealed reference and current physical model SHA256 differ")
    dependencies = _model_dependencies(source, model_path)
    if float(source["fps"]) != fps:
        raise ValueError("Split/source frame rates differ")
    declared = [(clip, span) for clip in source.get("clips", []) for span in clip.get("training_spans", [])
                if span.get("file") == path.name]
    if len(declared) != 1 or any(declared[0][1].get(key) != record.get(key) for key in ("frames", "start", "sha256")):
        raise ValueError("Accepted span does not match the sealed source record")
    if "schema_version" in source:
        if source["schema_version"] != 1 or meta.get("schema_version") != 1 or seal.get("schema_version") != 1:
            raise ValueError("Unsupported or incomplete versioned seal schema")
        if source_value != "../manifest.json":
            raise ValueError("Versioned split requires a portable ../manifest.json source path")
        if seal.get("model_sha256") != model_hash or meta.get("model_sha256") != model_hash:
            raise ValueError("Versioned seal model SHA256 differs")
        def canonical_hash(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        profile = validation.get("quality_profile")
        if not isinstance(profile, dict):
            raise ValueError("Versioned seal requires its quality profile")
        profile_hash = canonical_hash(profile)
        if any(row.get("quality_profile_sha256") != profile_hash for row in (seal, validation, meta)):
            raise ValueError("Quality profile SHA256 differs across sealed manifests")
        if canonical_hash({key: value for key, value in source.items() if key != "seal"}) != seal.get("manifest_payload_sha256"):
            raise ValueError("Sealed source manifest payload SHA256 mismatch")
        split = meta.get("split")
        if split not in ("train", "eval") or path.parent.name != "walk_"+split:
            raise ValueError("Versioned span split location differs")
        split_hash = seal.get("split_manifest_sha256", {}).get(split)
        if not split_hash:
            raise ValueError("Versioned seal requires a split manifest SHA256")
        _verify(manifest_path, split_hash, "Split manifest")
        clip, span = declared[0]
        if record.get("source_clip") != clip.get("name") or span.get("source_clip") != clip.get("name"):
            raise ValueError("Accepted span source clip differs from its sealed source")
        if span.get("split") != split or record.get("split") != split:
            raise ValueError("Accepted span split differs from its sealed source")
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    if raw.shape[1] != 29:
        raise ValueError("Sealed CSV must have 29 columns")
    qpos = np.c_[raw[:, :3], raw[:, 6], raw[:, 3:6], raw[:, 7:]]
    _validate_qpos(qpos, record.get("frames"))
    qpos.setflags(write=False)
    notes = () if source.get("schema_version") else ("Legacy sealed schema: span/XML hashes verified; model dependency identity unverified",)
    return Motion(path, qpos, fps, "sealed", digest, model_hash, True, record, manifest_path, {}, notes,
                  source_manifest_path=source_path, model_dependencies=dependencies)
