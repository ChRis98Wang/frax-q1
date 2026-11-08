"""Portable asset locations and locked upstream revisions (no download on import)."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

DATA = Path(os.environ.get("Q1_FRAX_DATA", Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))/"q1-frax")).expanduser().resolve()
SOURCES = {
    "frax": {"url":"https://github.com/StanfordASL/frax.git", "commit":"f5e87ab59a0b82ba43095e882c1b88c9b0dacf07", "license":"MIT", "license_file":"LICENSE", "sparse":["frax","pyproject.toml","LICENSE"]},
    "primebot-q1-urdf": {"url":"https://github.com/PrimebotTech/primebot-q1-urdf.git", "commit":"9657817ba375f41b0e5d2b62d8d6b72b757cdf4f", "license":"package.xml declares BSD; full license text not found in pinned repository", "license_file":"package.xml", "sparse":["xml","urdf","meshes","package.xml","README.md"]},
    "primebot-gmr": {"url":"https://github.com/PrimebotTech/GMR.git", "commit":"a7d0682f1bdd10bdb0b11910cf786d86e1423522", "license":"MIT (mapping/code; unrelated bundled robot assets have separate terms)", "license_file":"LICENSE", "sparse":["general_motion_retargeting/ik_configs/bvh_to_q1_lafan.json","LICENSE"]},
    "lafan1_source": {"url":"https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git", "commit":"94084601bacdf9cc3764b5c73daaeccae6035fac", "license":"CC-BY-NC-ND-4.0 (do not redistribute adapted motion without permission)", "license_file":"license.txt", "sparse":["lafan1/lafan1.zip","license.txt","README.md"]},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def verify_source(name, data=DATA):
    """Fail closed on missing, different, or locally modified tracked sources."""
    path = Path(data)/name
    source = SOURCES[name]
    try:
        head = subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True,stderr=subprocess.PIPE).strip()
        dirty = subprocess.check_output(["git","-C",str(path),"status","--porcelain","--untracked-files=no"],text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise ValueError(f"Missing source {name} in {data}; run q1-frax-setup") from exc
    if head != source["commit"] or dirty:
        raise ValueError(f"Source {name} must be clean at pinned commit {source['commit']}; use a separate asset directory")
    return head
