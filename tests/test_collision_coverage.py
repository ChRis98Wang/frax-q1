"""Actual-contact and geometry regressions for complete Q1 self-collision coverage."""
from contextlib import ExitStack
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.setdefault("JAX_ENABLE_X64","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("XLA_FLAGS","--xla_cpu_multi_thread_eigen=false --xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED")
import pytest
pytest.importorskip("frax")
import mujoco
import numpy as np
import yaml
from q1_frax import adapter,model as spec

pytestmark=pytest.mark.integration


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    source=spec.DATA/"primebot-q1-urdf"
    if not (source/"xml/q1_v3.xml").is_file():pytest.skip("Pinned Q1 model sources required")
    cache=tmp_path_factory.mktemp("coverage")
    (cache/"primebot-q1-urdf").symlink_to(source,target_is_directory=True)
    xml,urdf,spheres=(cache/"q1"/name for name in ("q1.xml","q1.urdf","spheres.yaml"))
    with ExitStack() as stack:
        for module,name,value in ((spec,"DATA",cache),(spec,"MJCF_PATH",xml),(adapter,"MJCF_PATH",xml),
                                  (adapter,"URDF",urdf),(adapter,"COLLISION",spheres),
                                  (adapter,"URDF_SOURCE",cache/"primebot-q1-urdf/urdf/q1_v3.urdf")):
            stack.enter_context(patch.object(module,name,value))
        robot,model=adapter.load_q1()
    return robot,model,yaml.safe_load(spheres.read_text()),spheres


def test_every_actual_contact_is_covered_including_known_missed_links(assets):
    robot,model,payload,_=assets
    assert payload["coverage_policy"]=="physical_geom_pairs_v3"
    covered=set()
    for a,i,b,j,_ in payload["self_collision_pairs"]:
        x=payload["collision_spheres"][a][i];y=payload["collision_spheres"][b][j]
        covered.add(tuple(sorted((x["source_geom"],y["source_geom"]))))
    physical={tuple(sorted(row)) for row in payload["physical_geom_pairs"]}
    assert covered==physical
    body_pairs={tuple(sorted((model.body(int(model.geom_bodyid[model.geom(a).id])).name,
                              model.body(int(model.geom_bodyid[model.geom(b).id])).name))) for a,b in physical}
    for a,b in (("right_ankle_roll_link","left_hip_pitch_link"),("right_ankle_roll_link","pelvis"),
                ("left_knee_link","left_shoulder_yaw_link"),("left_ankle_roll_link","left_wrist_yaw_link")):
        assert tuple(sorted((a,b))) in body_pairs
    rng=np.random.default_rng(8831);data=mujoco.MjData(model);contacts=0
    for _ in range(128):
        q=np.asarray(robot.default_configuration).copy()
        q[7:]=rng.uniform(model.jnt_range[1:,0],model.jnt_range[1:,1])
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        for c in data.contact:
            if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]:
                contacts+=1
                assert tuple(sorted((model.geom(c.geom1).name,model.geom(c.geom2).name))) in covered
    assert contacts>20  # This is an actual MuJoCo-contact oracle, not an empty test.


def test_self_only_foot_hulls_are_covered_but_never_floor_candidates(assets):
    robot,model,payload,_=assets
    q=np.asarray(robot.default_configuration).copy();data=mujoco.MjData(model)
    data.qpos[:]=q;mujoco.mj_forward(model,data)
    assert not any(model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2] for c in data.contact)
    assert float(np.min(robot.self_collision_distances(q)))>0
    flat=[sphere for spheres in payload["collision_spheres"].values() for sphere in spheres]
    centers,radii=map(np.asarray,robot.link_collision_data(q))
    floor=model.geom("floor").id;selected=set(robot.q1_floor_sphere_indices.tolist())
    for side in ("left","right"):
        name=f"{side}_ankle_roll_link_self_volume";geom=model.geom(name).id
        assert (int(model.geom_contype[geom]),int(model.geom_conaffinity[geom]))==(4,6)
        assert not ((model.geom_contype[geom]&model.geom_conaffinity[floor]) or
                    (model.geom_contype[floor]&model.geom_conaffinity[geom]))
        indices=[i for i,s in enumerate(flat) if s["source_geom"]==name]
        assert indices and not (set(indices)&selected)
        # Check all mesh vertices and a dense deterministic OBB grid against the
        # sphere union, so adding a visual foot cannot leave unrepresented toes.
        mesh=int(model.geom_dataid[geom]);start=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
        vertices=model.mesh_vert[start:start+count]
        grid=np.stack(np.meshgrid(*[np.linspace(-v,v,7) for v in model.geom_size[geom]],indexing="ij"),axis=-1).reshape(-1,3)
        points=np.concatenate((vertices,grid))@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
        gaps=np.linalg.norm(points[:,None,:]-centers[indices][None,:,:],axis=-1)-radii[indices]
        assert np.max(np.min(gaps,axis=1))<1e-7
    # Even when deliberately moved through the floor, these new geoms must not
    # add ground constraints. Sole spheres and other body geoms still collide.
    data.qpos[:]=q;data.qpos[2]-=.15;mujoco.mj_forward(model,data)
    floor_contacts=[c for c in data.contact if c.geom1==floor or c.geom2==floor]
    assert floor_contacts
    assert all("self_volume" not in model.geom(c.geom1).name and "self_volume" not in model.geom(c.geom2).name for c in floor_contacts)


def test_ik_floor_and_independent_quality_use_the_same_stored_selection(assets):
    from q1_frax.retarget import _quality_evaluator
    robot,model,payload,_=assets
    retargeter,bones,_=adapter.make_retargeter(robot,model)
    floor=retargeter.solver.constraints[1]
    np.testing.assert_array_equal(floor.sphere_idxs,robot.q1_floor_sphere_indices)
    np.testing.assert_array_equal(floor.sphere_idxs,payload["floor_sphere_indices"])
    q=np.asarray(robot.default_configuration);centers,radii=map(np.asarray,robot.link_collision_data(q))
    expected=(centers[floor.sphere_idxs,2]-radii[floor.sphere_idxs]).min()
    assert (centers[:,2]-radii).min()<expected-.001
    poses=np.zeros((1,len(bones),7));poses[:,:,3]=1.
    actual=_quality_evaluator(robot,retargeter)(q[None,:],poses)[0]
    np.testing.assert_allclose(actual,[expected],atol=1e-10,rtol=1e-10)


def test_read_only_legacy_and_v3_load_preserve_stored_policy_and_bytes(assets,tmp_path):
    robot,model,payload,path=assets
    before=path.read_bytes()
    with patch.object(adapter,"COLLISION",path):
        _,stored=adapter.collision_model(model,write=False)
    assert len(stored["body_sc_pairs"])==len(robot.sc_pairs) and path.read_bytes()==before
    legacy=tmp_path/"legacy.yaml"
    legacy.write_text(yaml.safe_dump({"collision_spheres":payload["collision_spheres"]}))
    before=legacy.read_bytes()
    with patch.object(adapter,"COLLISION",legacy):
        _,old=adapter.collision_model(model,write=False)
    assert len(old["body_sc_pairs"])<len(stored["body_sc_pairs"])
    assert legacy.read_bytes()==before  # Never upgrade an old seal under its old hash.
