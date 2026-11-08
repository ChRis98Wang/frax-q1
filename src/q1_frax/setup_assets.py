"""Explicitly download pinned upstream sources into a user-owned cache."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def fetch_source(name, data):
    from .paths import SOURCES, verify_source
    target = data/name
    spec = SOURCES[name]
    if target.exists():
        verify_source(name, data)
        return
    # Fetch one immutable revision and only required files. GMR's many unrelated
    # robot meshes are never checked out. Servers may ignore partial clone filters.
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}-",dir=data))
    try:
        def git(*args):
            subprocess.run(["git","-C",str(temporary),*args],check=True)
        git("init","--quiet")
        git("remote","add","origin",spec["url"])
        git("config","remote.origin.promisor","true")
        git("config","remote.origin.partialclonefilter","blob:none")
        git("sparse-checkout","init","--no-cone")
        (temporary/".git/info/sparse-checkout").write_text("\n".join("/"+p for p in spec["sparse"])+"\n")
        git("-c","protocol.version=2","fetch","--depth=1","--filter=blob:none","origin",spec["commit"])
        git("checkout","--detach","FETCH_HEAD")
        temporary.rename(target)
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True)
        raise
    verify_source(name,data)


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir",type=Path,help="Asset cache (default Q1_FRAX_DATA or ~/.cache/q1-frax)")
    result.add_argument("--with-lafan",action="store_true",help="Explicitly fetch LAFAN1 under CC BY-NC-ND 4.0; adapted motions are not redistributable by default")
    result.add_argument("--check",action="store_true",help="Verify pinned source revisions without downloading or changing assets")
    result.add_argument("--skip-build",action="store_true",help="Download sources without generating the MuJoCo model")
    return result


def main(argv=None):
    args=parser().parse_args(argv)
    if args.data_dir:
        os.environ["Q1_FRAX_DATA"]=str(args.data_dir.expanduser().resolve())
    from .paths import DATA,SOURCES,verify_source,sha256,write_json
    names=[n for n in SOURCES if n!="lafan1_source" or args.with_lafan]
    for name in names:
        print(f"{name}: {SOURCES[name]['commit']}\n  {SOURCES[name]['license']}",flush=True)
    if not args.check:
        DATA.mkdir(parents=True,exist_ok=True)
    for name in names:
        if args.check:
            verify_source(name,DATA)
        else:
            fetch_source(name,DATA)
    records={name:{**SOURCES[name],"license_sha256":sha256(DATA/name/SOURCES[name]["license_file"])} for name in names}
    if not args.check:
        if not args.skip_build:
            from .model import build_model,MJCF_PATH
            model=build_model()
            print(f"Built Q1: {model.nq} qpos, {model.nu} actuators at {MJCF_PATH}")
        write_json(DATA/"sources.json",{"schema_version":1,"sources":records})
    print(f"Assets ready: {DATA}\nSet Q1_FRAX_DATA to this directory when using a custom location.")
    return 0


if __name__=="__main__":
    main()
