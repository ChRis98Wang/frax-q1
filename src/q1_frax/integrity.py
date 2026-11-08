"""Portable fingerprints for the external assets consumed by a Q1 MJCF.

Only the standard library is imported, so model identity can be checked before
loading MuJoCo. Asset paths follow MuJoCo's meshdir/texturedir/assetdir rules:
https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler

Includes, attached models, plugins and unknown file-bearing elements are
deliberately unsupported and rejected. They must never silently produce an
incomplete dependency inventory. The generated Q1 model uses none of them.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import xml.etree.ElementTree as ET

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_TEXTURE_FILES = {"file", "fileleft", "fileright", "fileup", "filedown", "filefront", "fileback"}
_SUPPORTED_FILES = {"mesh": {"file"}, "hfield": {"file"}, "skin": {"file"}, "texture": _TEXTURE_FILES}


def _relative(value: str, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not empty):
        raise ValueError(f"Invalid {label}: expected a relative file path")
    if "\\" in value or ":" in value or "\x00" in value or PurePosixPath(value).is_absolute():
        raise ValueError(f"Nonportable {label}: {value!r}")
    return value


def _fingerprint(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != after.st_size:
            raise ValueError(f"Model dependency changed while hashing: {path}")
    except OSError as exc:
        raise ValueError(f"Cannot read model dependency: {path}") from exc
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def dependency_fingerprints(model_path: str | Path) -> dict:
    """Reparse MJCF and hash every supported external asset.

    Keys are normalized POSIX paths relative to the XML's directory, not to a
    resolved symlink target. Thus sibling source checkouts and copied caches
    retain the same identity. Relative ``..`` components are valid for assets;
    this is a read-only inventory, unlike dataset output path validation.
    Absolute paths and resource-provider URIs are rejected as nonportable.
    The caller separately checks the main XML's SHA256.
    """
    model_path = Path(os.path.abspath(Path(model_path).expanduser()))
    try:
        content = model_path.read_bytes()
        if b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
            raise ValueError("MJCF DTD/entity declarations are not supported for dependency verification")
        root = ET.fromstring(content)
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"Cannot parse model dependency XML: {model_path}") from exc
    if root.tag != "mujoco":
        raise ValueError("Dependency verification requires a MJCF mujoco root")
    compilers = root.findall("compiler")
    if len(compilers) > 1:
        raise ValueError("Multiple compiler elements are unsupported for dependency verification")
    compiler = compilers[0].attrib if compilers else {}
    directories = {
        "mesh": compiler.get("meshdir", compiler.get("assetdir", "")),
        "texture": compiler.get("texturedir", compiler.get("assetdir", "")),
    }
    for label, value in directories.items():
        _relative(value, label + "dir", empty=True)
    strip = compiler.get("strippath", "false")
    if strip not in ("true", "false"):
        raise ValueError("Invalid compiler strippath value")
    files = {}
    for node in root.iter():
        if node.tag in ("include", "extension", "plugin", "model"):
            raise ValueError(f"Unsupported MJCF dependency element: {node.tag}")
        file_attributes = {key for key in node.attrib if "file" in key.lower()}
        allowed = _SUPPORTED_FILES.get(node.tag, set())
        if file_attributes - allowed:
            raise ValueError(f"Unsupported external file attribute on {node.tag}: {sorted(file_attributes - allowed)}")
        for key in sorted(file_attributes):
            name = _relative(node.attrib[key], node.tag + "/" + key)
            if strip == "true":
                name = PurePosixPath(name).name
            directory = directories["texture" if node.tag == "texture" else "mesh"]
            relative = os.path.normpath(os.path.join(directory, name)).replace(os.sep, "/")
            _relative(relative, "model dependency")
            path = model_path.parent / relative
            if path == model_path:
                raise ValueError("Model cannot reference itself as an external asset")
            files[relative] = _fingerprint(path)
    return {"schema_version": 1, "files": dict(sorted(files.items()))}


def verify_model_dependencies(model_path: str | Path, expected: dict) -> dict:
    """Require complete, exact asset identity; raise ValueError on any mismatch."""
    if not isinstance(expected, dict) or set(expected) != {"schema_version", "files"} or type(expected["schema_version"]) is not int or expected["schema_version"] != 1:
        raise ValueError("Missing or invalid model_dependencies schema")
    files = expected["files"]
    if not isinstance(files, dict):
        raise ValueError("Invalid model_dependencies files mapping")
    for name, entry in files.items():
        _relative(name, "dependency inventory path")
        if os.path.normpath(name).replace(os.sep, "/") != name or name == ".":
            raise ValueError(f"Noncanonical dependency inventory path: {name}")
        if not isinstance(entry, dict) or set(entry) != {"sha256", "size_bytes"}:
            raise ValueError(f"Invalid dependency fingerprint: {name}")
        if not isinstance(entry["sha256"], str) or not _HASH.fullmatch(entry["sha256"]):
            raise ValueError(f"Invalid dependency SHA256: {name}")
        if type(entry["size_bytes"]) is not int or entry["size_bytes"] < 0:
            raise ValueError(f"Invalid dependency size_bytes: {name}")
    actual = dependency_fingerprints(model_path)
    if set(actual["files"]) != set(files):
        missing = sorted(set(actual["files"]) - set(files))
        extra = sorted(set(files) - set(actual["files"]))
        raise ValueError(f"Model dependency inventory mismatch: missing={missing}, extra={extra}")
    for name, entry in actual["files"].items():
        if entry != files[name]:
            raise ValueError(f"Model dependency SHA256/size mismatch: {name}")
    return actual
