"""Independent FRAX-Q1 coordinate, collision and exported-data checks.

Run pytest tests/test_adapter_integration.py with prepared Q1_FRAX_DATA.
The adapter tests build into a temporary directory; existing training assets are
never rewritten by this test suite. Dataset checks are skipped until exported.
"""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false --xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED")

import pytest
jax=pytest.importorskip("jax")
import mujoco
import numpy as np
import yaml

import pytest
pytest.importorskip("frax")
from q1_frax import adapter as frax_q1, model as q1




@pytest.mark.integration
@unittest.skipUnless((q1.DATA/"primebot-q1-urdf/xml/q1_v3.xml").is_file(), "Run q1-frax-setup before integration checks")
class FraxQ1KinematicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.xml, cls.urdf, cls.spheres = root/"q1.xml", root/"q1.urdf", root/"spheres.yaml"
        with ExitStack() as stack:
            for module, name, value in ((q1,"MJCF_PATH",cls.xml), (frax_q1,"MJCF_PATH",cls.xml),
                                        (frax_q1,"URDF",cls.urdf), (frax_q1,"COLLISION",cls.spheres)):
                stack.enter_context(patch.object(module,name,value))
            cls.robot, cls.model = frax_q1.load_q1()
        cls.fk = staticmethod(jax.jit(cls.robot.joint_to_world_transforms))
        cls.collision_fk = staticmethod(jax.jit(cls.robot.link_collision_data))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_random_root_and_joint_frames_match_physical_mujoco(self):
        robot, model = self.robot, self.model
        self.assertEqual(list(robot.joint_names[6:]),q1.JOINT_NAMES)
        np.testing.assert_array_equal(robot.actuated_joint_lower_limits,model.jnt_range[1:,0])
        np.testing.assert_array_equal(robot.actuated_joint_upper_limits,model.jnt_range[1:,1])
        rng=np.random.default_rng(847)
        data=mujoco.MjData(model)
        for _ in range(16):
            q=np.asarray(robot.default_configuration).copy()
            q[:3]=rng.normal(size=3)
            quat=rng.normal(size=4); q[3:7]=quat/np.linalg.norm(quat)
            q[7:]=rng.uniform(model.jnt_range[1:,0],model.jnt_range[1:,1])
            data.qpos[:]=q; mujoco.mj_forward(model,data)
            transforms=np.asarray(self.fk(q))
            pairs=[("yaw_joint",model.body("pelvis").id)] + [
                (name,int(model.jnt_bodyid[model.joint(name).id])) for name in q1.JOINT_NAMES]
            for name,body in pairs:
                actual=transforms[robot.joint_name_to_index[name]]
                np.testing.assert_allclose(actual[:3,3],data.xpos[body],atol=1e-6,rtol=1e-6,
                                           err_msg=f"Position mismatch at {name}")
                np.testing.assert_allclose(actual[:3,:3],data.xmat[body].reshape(3,3),atol=3e-6,rtol=3e-6,
                                           err_msg=f"Rotation mismatch at {name}")
            # Q1 wrists are fixed: an elbow joint index alone loses the hand offset.
            for side in ("left","right"):
                actual=np.asarray(getattr(robot,f"{side}_hand_transform")(q))
                body=model.body(f"{side}_wrist_yaw_link").id
                np.testing.assert_allclose(actual[:3,3],data.xpos[body],atol=1e-6,rtol=1e-6)
                np.testing.assert_allclose(actual[:3,:3],data.xmat[body].reshape(3,3),atol=3e-6,rtol=3e-6)

    def test_foot_sphere_frames_match_actual_contact_geoms(self):
        robot,model=self.robot,self.model
        q=np.asarray(robot.default_configuration).copy()
        q[3:7]=np.array([.93,.1,-.12,.2]); q[3:7]/=np.linalg.norm(q[3:7])
        q[7:]+=np.linspace(-.04,.04,22)
        data=mujoco.MjData(model); data.qpos[:]=q; mujoco.mj_forward(model,data)
        centers,radii=map(np.asarray,self.collision_fk(q))
        for side in ("left","right"):
            joint=robot.joint_name_to_index[f"{side}_ankle_roll_joint"]
            idx=np.flatnonzero((robot.flat_collision_parent_idxs==joint)&np.isclose(radii,.002))
            self.assertEqual(len(idx),5)
            physical=[model.geom(f"{side}_ankle_roll_link_foot_{i}").id for i in range(5)]
            np.testing.assert_allclose(centers[idx],data.geom_xpos[physical],atol=1e-6,rtol=1e-6)
            np.testing.assert_allclose(centers[idx,2]-radii[idx],
                                       data.geom_xpos[physical,2]-model.geom_size[physical,0],atol=1e-6,rtol=1e-6)
        floor=model.geom("floor").id
        for g in range(model.ngeom):
            if "_foot_" in (model.geom(g).name or ""):
                self.assertTrue((model.geom_contype[g]&model.geom_conaffinity[floor]) or
                                (model.geom_contype[floor]&model.geom_conaffinity[g]))
            elif model.geom_group[g]==1:
                self.assertEqual(int(model.geom_contype[g]|model.geom_conaffinity[g]),0)

    def test_fixed_hand_and_head_collision_geometry_is_retained(self):
        spheres=yaml.safe_load(self.spheres.read_text())["collision_spheres"]
        # Fixed children must be merged into the parent joint frame, not discarded.
        self.assertGreater(max(s["center"][0] for s in spheres["left_elbow_link"]),.16)
        self.assertGreater(max(s["center"][2] for s in spheres["torso_link"]),.25)
        self.assertGreater(len(self.robot.body_sc_pairs),0)

    def test_contact_targets_preserve_stance_and_clear_low_fast_swings(self):
        ret,bones,scales=frax_q1.make_retargeter(self.robot,self.model)
        names=list(bones)+["LeftToe","RightToe"]
        poses=np.zeros((120,len(names),7)); poses[...,3]=1.
        poses[:,:,2]=.8
        poses[:,[names.index("LeftToe"),names.index("RightToe")],2]=.03
        # A low, fast swing is exactly the failure case in the old ankle targets.
        poses[40:81,names.index("LeftToe"),0]=np.arange(41)*.04
        poses[81:,names.index("LeftToe"),0]=1.6
        inputs=frax_q1.contact_aware_inputs(poses,names,bones,scales["leg_scale"])
        np.testing.assert_allclose(inputs["clearance"][10:25],.002,atol=1e-9)
        np.testing.assert_allclose(inputs["clearance"][50:70,0],.017,atol=1e-9)
        np.testing.assert_allclose(inputs["clearance"][:,1],.002,atol=1e-9)
        # Recording origin must not affect clearance inferred from human toes.
        moved=poses.copy();moved[:,:,:3]+=np.array([3.,-2.,.37])
        shifted=frax_q1.contact_aware_inputs(moved,names,bones,scales["leg_scale"])
        np.testing.assert_allclose(inputs["clearance"],shifted["clearance"],atol=1e-9)
        np.testing.assert_array_equal(inputs["poses"],poses[:,:len(bones)])

        # Verify against physical local sphere offsets, including pitched feet.
        footrows=[bones.index("LeftFootMod"),bones.index("RightFootMod")]
        rng=np.random.default_rng(109)
        rotations=rng.normal(size=(120,2,4));rotations/=np.linalg.norm(rotations,axis=-1,keepdims=True)
        inputs["poses"][:,footrows,3:]=rotations
        target=jax.jit(jax.vmap(ret.preprocess))(inputs)
        raw=jax.jit(jax.vmap(ret.preprocess))(inputs["poses"])
        np.testing.assert_allclose(np.asarray(target["frame_pos"])[:,footrows,:2],
                                   np.asarray(raw["frame_pos"])[:,footrows,:2],atol=1e-9)
        for side,row,col in zip(("left","right"),footrows,(0,1)):
            ids=[self.model.geom(f"{side}_ankle_roll_link_foot_{i}").id for i in range(5)]
            local=self.model.geom_pos[ids]
            points=np.asarray(target["frame_pos"])[:,row,None,:]+np.einsum(
                "tij,pj->tpi",np.asarray(target["frame_rmat"])[:,row],local)
            bottom=(points[:,:,2]-self.model.geom_size[ids,0]).min(axis=1)
            np.testing.assert_allclose(bottom,inputs["clearance"][:,col],atol=1e-9)

@pytest.mark.integration
def test_short_retarget_replays_evidence_and_revalidation_is_read_only(tmp_path):
    """Exercise the installed CLI, source evidence, and refusal of modified assets."""
    import shutil
    archive=q1.DATA/"lafan1_source/lafan1/lafan1.zip"
    if not archive.is_file():
        pytest.skip("Explicit LAFAN setup is required for retarget integration")
    from q1_frax.retarget import make_quality_replayer
    out=tmp_path/"retarget"
    command=[sys.executable,"-m","q1_frax.retarget","--out",str(out),"--pattern","walk2_subject3.bvh","--max-frames","120"]
    result=subprocess.run(command,text=True,capture_output=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    manifest=json.loads((out/"manifest.json").read_text())
    record=manifest["clips"][0]
    assert record["frames"]==120 and manifest["validation"]["complete"]
    q=frax_q1.load_q1_motion(out/"npz/walk2_subject3.npz")
    assets=(q1.MJCF_PATH,frax_q1.URDF,frax_q1.COLLISION)
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in assets}
    replay=make_quality_replayer(manifest)
    actual=replay(q,record)
    with np.load(out/"quality/walk2_subject3.npz",allow_pickle=False) as stored:
        for key,value in actual.items():
            np.testing.assert_allclose(value,stored[key],atol=1e-10,rtol=1e-10)
    result=subprocess.run([sys.executable,"-m","q1_frax.retarget","--out",str(out),"--validate-only"],text=True,capture_output=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert before=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in assets}
    # A modified model must be rejected before any derived asset is rebuilt.
    cache=tmp_path/"altered-cache";cache.mkdir()
    for name in ("frax","primebot-q1-urdf","primebot-gmr","lafan1_source"):
        (cache/name).symlink_to(q1.DATA/name,target_is_directory=True)
    shutil.copytree(q1.DATA/"q1",cache/"q1")
    model=cache/"q1/q1.xml";model.write_bytes(model.read_bytes()+b"\n<!-- changed -->\n")
    altered=model.read_bytes();original_manifest=(out/"manifest.json").read_bytes()
    result=subprocess.run([sys.executable,"-m","q1_frax.retarget","--out",str(out),"--validate-only"],
                          env={**os.environ,"Q1_FRAX_DATA":str(cache)},text=True,capture_output=True,timeout=90)
    assert result.returncode!=0 and "asset changed" in result.stderr,result.stderr
    assert model.read_bytes()==altered
    assert (out/"manifest.json").read_bytes()==original_manifest

@pytest.mark.integration
def test_asset_bytes_are_identical_across_cache_roots(tmp_path):
    """Generated models retain relative mesh paths even with source symlinks."""
    source=q1.DATA/"primebot-q1-urdf"
    if not (source/"xml/q1_v3.xml").is_file():
        pytest.skip("Pinned Q1 source assets required")
    payloads=[]
    for dirname in ("first-cache","unrelated/deeper/second-cache"):
        cache=tmp_path/dirname;cache.mkdir(parents=True)
        (cache/"primebot-q1-urdf").symlink_to(source,target_is_directory=True)
        xml=cache/"q1/q1.xml";urdf=cache/"q1/q1.urdf";spheres=cache/"q1/frax_spheres.yaml"
        with ExitStack() as stack:
            for module,name,value in ((q1,"DATA",cache),(q1,"MJCF_PATH",xml),
                                      (frax_q1,"MJCF_PATH",xml),(frax_q1,"URDF",urdf),
                                      (frax_q1,"URDF_SOURCE",cache/"primebot-q1-urdf/urdf/q1_v3.urdf"),
                                      (frax_q1,"COLLISION",spheres)):
                stack.enter_context(patch.object(module,name,value))
            robot,model=frax_q1.load_q1()
            assert robot.nq==29 and model.nu==22
        assert mujoco.MjModel.from_xml_path(str(xml)).nq==29
        # Also check that MuJoCo's URDF mesh resolver can use these filenames.
        assert mujoco.MjModel.from_xml_path(str(urdf)).nmesh>0
        import xml.etree.ElementTree as ET
        assert ET.parse(xml).find("compiler").get("meshdir")=="../primebot-q1-urdf/meshes"
        for mesh in ET.parse(urdf).findall(".//mesh"):
            assert mesh.get("filename").startswith("../primebot-q1-urdf/meshes/")
        files=[path.read_bytes() for path in (xml,urdf,spheres)]
        assert all(str(tmp_path).encode() not in content for content in files)
        assert all(str(source.resolve()).encode() not in content for content in files)
        payloads.append(files)
    assert payloads[0]==payloads[1]
