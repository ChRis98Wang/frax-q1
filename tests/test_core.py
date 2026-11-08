"""Portable configuration, strict archive IO, and inert import boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from q1_frax.model import DEFAULT_Q,JOINT_NAMES
from q1_frax.retarget import DEFAULT_PARAMETERS,extract_lafan,load_parameters


def test_import_and_help_do_not_initialize_jax_or_download(tmp_path):
    code='''import sys
import q1_frax, q1_frax.model, q1_frax.retarget, q1_frax.setup_assets
assert not any(n == "jax" or n.startswith("jax.") for n in sys.modules)
assert not any(n == "mujoco" or n.startswith("mujoco.") for n in sys.modules)
from q1_frax.paths import DATA
assert not DATA.exists()
'''
    env={**os.environ,"Q1_FRAX_DATA":str(tmp_path/"absent")}
    subprocess.run([sys.executable,"-c",code],check=True,env=env,cwd=tmp_path)
    for module in ("q1_frax.retarget","q1_frax.setup_assets"):
        result=subprocess.run([sys.executable,"-m",module,"--help"],env=env,cwd=tmp_path,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
    assert not (tmp_path/"absent").exists()


def test_tuned_defaults_and_invalid_configuration_fail_closed(tmp_path):
    assert load_parameters()==DEFAULT_PARAMETERS
    assert load_parameters()["contact_aware"] is True
    assert load_parameters()["foot_position_weight"]==200
    for bad in ({"typo":1},{"contact_aware":1},{"contact_swing_margin":float("nan")},
                {"self_collision_max_rows":True},{"floor_max_rows":0},{"damping":-1},
                {"joint_velocity_limit":0},{"floor_gain":False}):
        path=tmp_path/"bad.json";path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_parameters(path)


def test_motion_io_preserves_joint_order_and_quaternion(tmp_path):
    from q1_frax.adapter import load_q1_motion
    q=np.tile(np.r_[0.,0.,.42,1.,0.,0.,0.,DEFAULT_Q],(4,1))
    path=tmp_path/"motion.npz"
    np.savez(path,q=q,fps=30.,joint_names=JOINT_NAMES)
    np.testing.assert_array_equal(load_q1_motion(path),q)
    invalid=q.copy();invalid[:,3:7]*=2
    for payload in ({"q":q,"fps":60.,"joint_names":JOINT_NAMES},
                    {"q":q,"fps":30.,"joint_names":JOINT_NAMES[::-1]},
                    {"q":q[:,:-1],"fps":30.,"joint_names":JOINT_NAMES},
                    {"q":invalid,"fps":30.,"joint_names":JOINT_NAMES}):
        np.savez(path,**payload)
        with pytest.raises(ValueError):load_q1_motion(path)


def test_extract_refuses_changed_inputs_and_flattens_archive(tmp_path):
    archive=tmp_path/"source.zip";out=tmp_path/"bvh"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("../../walk.bvh","example source")
        z.writestr("README.md","ignored")
    extract_lafan(archive,out)
    assert (out/"walk.bvh").read_text()=="example source"
    assert not (tmp_path/"walk.bvh").exists()
    (out/"walk.bvh").write_text("modified")
    with pytest.raises(ValueError,match="differs"):
        extract_lafan(archive,out)


def test_manifest_seal_blocks_all_retarget_mutation(tmp_path):
    pytest.importorskip("frax")
    out=tmp_path/"sealed";out.mkdir()
    manifest=out/"manifest.json";manifest.write_text(json.dumps({"seal":{"complete":True}}))
    before=manifest.read_bytes()
    for flags in ([],["--validate-only"]):
        result=subprocess.run([sys.executable,"-m","q1_frax.retarget","--out",str(out),*flags],capture_output=True,text=True)
        assert result.returncode!=0
        assert "Dataset is sealed" in result.stderr
        assert manifest.read_bytes()==before
        assert sorted(p.name for p in out.iterdir())==["manifest.json"]


def test_bvh_rejects_unknown_or_wrong_sample_rate_before_frax_import(tmp_path):
    from q1_frax.adapter import load_lafan1
    path=tmp_path/"invalid.bvh"
    for content in ("Frame Time: 0.01666667\n", "HIERARCHY\n", "Frame Time: nan\n"):
        path.write_text(content)
        with pytest.raises(ValueError,match="30 Hz"):
            load_lafan1(path)


def test_platform_selection_rejects_cpu_devices_for_cuda_without_initializing(tmp_path):
    env={**os.environ,"Q1_FRAX_DATA":str(tmp_path/"absent")}
    result=subprocess.run([sys.executable,"-m","q1_frax.retarget","--platform","cuda","--devices","2"],env=env,text=True,capture_output=True)
    assert result.returncode!=0 and "only valid" in result.stderr
    assert not (tmp_path/"absent").exists()


def test_cpu_backend_is_explicit_and_cuda_never_silently_falls_back():
    pytest.importorskip("jax")
    code="from q1_frax.retarget import configure_backend; import json; print(json.dumps(configure_backend('cpu',1)))"
    result=subprocess.run([sys.executable,"-c",code],text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    metadata=json.loads(result.stdout)
    assert metadata["jax_backend"]=="cpu" and metadata["jax_enable_x64"] is True
    code="from q1_frax.retarget import configure_backend; import json; print(json.dumps(configure_backend('cuda')))"
    result=subprocess.run([sys.executable,"-c",code],env={**os.environ,"CUDA_VISIBLE_DEVICES":""},text=True,capture_output=True)
    assert result.returncode!=0,"CUDA request silently succeeded without a visible GPU"


def test_output_guard_refuses_nested_seals_and_symlinks_before_writes(tmp_path):
    from q1_frax.retarget import guard_output_directory
    sealed=tmp_path/"sealed";sealed.mkdir()
    (sealed/"manifest.json").write_text(json.dumps({"seal":{"complete":True}}))
    with pytest.raises(ValueError,match="sealed"):
        guard_output_directory(sealed/"nested/new-output")
    assert not (sealed/"nested").exists()
    external=tmp_path/"external";external.mkdir();target=external/"preserve.npz";target.write_bytes(b"preserve")
    sealed_manifest=external/"manifest.json"
    sealed_manifest.write_text(json.dumps({"seal":{"complete":True}}))
    sealed_before=sealed_manifest.read_bytes()
    output=tmp_path/"output";output.mkdir()
    (output/"npz").symlink_to(external,target_is_directory=True)
    with pytest.raises(ValueError,match="symlink"):
        guard_output_directory(output)
    assert target.read_bytes()==b"preserve"
    (output/"npz").unlink();(output/"npz").mkdir()
    (output/"npz/motion.npz").symlink_to(target)
    with pytest.raises(ValueError,match="symlink"):
        guard_output_directory(output)
    result=subprocess.run([sys.executable,"-m","q1_frax.retarget","--out",str(output)],text=True,capture_output=True)
    assert result.returncode!=0 and "symlink" in result.stderr
    assert target.read_bytes()==b"preserve" and sealed_manifest.read_bytes()==sealed_before


def test_output_guard_refuses_hardlinks_to_sealed_bytes(tmp_path):
    sealed=tmp_path/"sealed";sealed.mkdir()
    (sealed/"manifest.json").write_text(json.dumps({"seal":{"complete":True}}))
    original=sealed/"keep.csv";original.write_bytes(b"sealed bytes")
    output=tmp_path/"output/csv";output.mkdir(parents=True)
    os.link(original,output/"walk.csv")
    result=subprocess.run([sys.executable,"-m","q1_frax.retarget","--out",str(output.parent)],capture_output=True,text=True)
    assert result.returncode!=0 and "hard-linked" in result.stderr
    assert original.read_bytes()==b"sealed bytes"
