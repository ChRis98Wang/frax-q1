"""FRAX adapter for Primebot Q1. Uses the official URDF and physical limits.

The LAFAN correspondence/rotation offsets come from PrimebotTech/GMR's
bvh_to_q1_lafan.json (MIT). Retargeting is performed by StanfordASL/frax.
"""
import json
import os
from itertools import product
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
import yaml

from .model import DATA, DEFAULT_Q, JOINT_NAMES, MJCF_PATH, build_model

URDF_SOURCE = DATA / "primebot-q1-urdf/urdf/q1_v3.urdf"
URDF = DATA / "q1/q1.urdf"
COLLISION = DATA / "q1/frax_spheres.yaml"
IK_CONFIG = DATA / "primebot-gmr/general_motion_retargeting/ik_configs/bvh_to_q1_lafan.json"


def load_q1_motion(path):
    """Read a FRAX NPZ without silently relabeling another robot or frame rate."""
    with np.load(path, allow_pickle=False) as data:
        if not {"q", "fps", "joint_names"}.issubset(data.files):
            raise ValueError(f"Missing Q1 archive metadata: {path}")
        if data["fps"].shape != ():
            raise ValueError(f"Expected scalar archive fps: {path}")
        if data["joint_names"].tolist() != JOINT_NAMES:
            raise ValueError(f"Q1 joint order mismatch: {path}")
        if not np.isclose(float(data["fps"]), 30., atol=1e-6):
            raise ValueError(f"Expected 30 Hz LAFAN motion: {path}")
        q = data["q"].copy()
    if q.ndim != 2 or q.shape[1] != 29 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError(f"Invalid Q1 qpos array: {path}")
    if not np.allclose(np.linalg.norm(q[:,3:7],axis=1), 1., atol=1e-4, rtol=0.):
        raise ValueError(f"Non-unit quaternion in Q1 motion: {path}")
    return q


def contact_aware_inputs(all_poses, all_names, bones, leg_scale, swing_margin=.015):
    """Map human toe clearance to Q1's full sole instead of ankle height alone.

Q1's long sole otherwise scrapes the floor during human swing even when ankle
position tracking is accurate. Clearance is smoothed offline and kept positive
for a fast-moving foot; stance positions/orientations remain source-driven.
"""
    feet = np.stack([all_poses[:,all_names.index(f"{s}Toe"),:3] for s in ("Left","Right")],axis=1)
    floor_z = float(np.percentile(feet[...,2].min(axis=1),5))
    heights = np.maximum(feet[...,2]-floor_z,0.) * leg_scale
    speed = np.linalg.norm(np.gradient(feet[...,:2],1/30.,axis=0),axis=-1)
    moving = np.clip((speed-.15)/.65,0.,1.)
    moving = moving*moving*(3-2*moving)
    clearance = .002 + np.maximum(heights, swing_margin*moving)
    clearance = gaussian_filter1d(clearance,1.,axis=0,mode="nearest")
    return {"poses":all_poses[:,[all_names.index(b) for b in bones]], "clearance":clearance}



def physical_self_collision_pairs(model):
    """All configuration-independent Q1 geom pairs eligible in MuJoCo.

    Mirrors collision selection, not broad-phase proximity: welded bodies,
    parent filtering, explicit body exclusions and contact bitmasks. Explicit
    geom pairs bypass parent/bitmask filters, as in MuJoCo. The world is handled
    separately by FloorConstraint. Geometry approximation remains independent.
    """
    flags=int(model.opt.disableflags)
    if flags & (int(mujoco.mjtDisableBit.mjDSBL_CONTACT)|int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)):
        raise ValueError("Physical contact must be enabled for collision coverage")
    explicit={tuple(sorted((int(a),int(b)))) for a,b in zip(model.pair_geom1,model.pair_geom2)}
    excludes={tuple((int(value)>>16,int(value)&65535)) for value in model.exclude_signature}
    filter_parent=not bool(flags&int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    active=[g for g in range(model.ngeom) if model.geom_bodyid[g] and
            (model.geom_contype[g] or model.geom_conaffinity[g] or any(g in pair for pair in explicit))]
    pairs=[]
    for ia,g1 in enumerate(active):
        for g2 in active[ia+1:]:
            b1,b2=int(model.geom_bodyid[g1]),int(model.geom_bodyid[g2])
            if (g1,g2) in explicit:
                pairs.append((g1,g2));continue
            if not ((int(model.geom_contype[g1])&int(model.geom_conaffinity[g2])) or
                    (int(model.geom_contype[g2])&int(model.geom_conaffinity[g1]))):continue
            w1,w2=int(model.body_weldid[b1]),int(model.body_weldid[b2])
            if w1==w2:continue
            p1=int(model.body_weldid[model.body_parentid[w1]])
            p2=int(model.body_weldid[model.body_parentid[w2]])
            if filter_parent and w1 and w2 and (w1==p2 or w2==p1):continue
            if tuple(sorted((b1,b2))) in excludes:continue
            pairs.append((g1,g2))
    return tuple(pairs)


def _legacy_selected_link_pairs():
    """Version-1 policy, only for exact read-only replay of legacy sphere files."""
    pairs = set()
    def add(a,b):
        if a != b:
            pairs.add(tuple(sorted((a,b))))
    for a,b in product(("hip_yaw", "knee", "ankle_roll"), repeat=2):
        add(f"left_{a}_link", f"right_{b}_link")
    for side,other in (("left","right"),("right","left")):
        arm = f"{side}_elbow_link"  # includes the fixed hand
        for link in ("pelvis", "torso_link", f"{other}_elbow_link", f"{other}_shoulder_yaw_link"):
            add(arm,link)
        for s,part in product((side,other),("hip_pitch","hip_roll","hip_yaw","knee")):
            add(arm,f"{s}_{part}_link")
        add(f"{side}_shoulder_yaw_link", "torso_link")
    return sorted(pairs)



def collision_model(model, *, write=True):
    """Spherical envelopes of the physical primitive geoms, fixed links merged.

The sole uses the official contact points plus a rounded volume for foot/foot
avoidance. Spheres are an IK approximation; MuJoCo validates actual collisions.
"""
    from frax.utils.collision_utils import bubblify_to_mine, link_pairs_to_sc_data

    child = {n:model.body(int(model.jnt_bodyid[model.joint(n).id])).name for n in JOINT_NAMES}
    if not write:
        payload=yaml.safe_load(COLLISION.read_text())
        if (payload.get("schema_version"),payload.get("coverage_policy")) in ((2,"physical_geom_pairs_v2"),(3,"physical_geom_pairs_v3")):
            sc=tuple(tuple(row) for row in payload["self_collision_pairs"])
            if len(sc)!=payload["coverage"]["sphere_pair_count"]:
                raise ValueError("Stored collision pair count differs from collision metadata")
        elif "schema_version" not in payload and "self_collision_pairs" not in payload:
            sc=link_pairs_to_sc_data(COLLISION,_legacy_selected_link_pairs(),tol=.002)
        else:
            raise ValueError("Unsupported stored Q1 collision coverage policy")
        return child,bubblify_to_mine(COLLISION,child,"pelvis",True,sc,verbose=False)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    child = {n: model.body(int(model.jnt_bodyid[model.joint(n).id])).name for n in JOINT_NAMES}
    spheres = {name: [] for name in ["pelvis", *child.values()]}
    floor_id=model.geom("floor").id
    def floor_eligible(g):
        return bool((int(model.geom_contype[g])&int(model.geom_conaffinity[floor_id])) or
                    (int(model.geom_contype[floor_id])&int(model.geom_conaffinity[g])))
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]) or model.geom_bodyid[g] == 0:
            continue
        b = int(model.geom_bodyid[g])
        while model.body(b).name not in spheres:
            b = int(model.body_parentid[b])
        # Express primitive sphere centers in the retained joint's frame.
        Rp = data.xmat[b].reshape(3,3).T
        Rg = data.geom_xmat[g].reshape(3,3)
        origin = Rp @ (data.geom_xpos[g] - data.xpos[b])
        rot = Rp @ Rg
        typ, size = model.geom_type[g], model.geom_size[g]
        if typ == mujoco.mjtGeom.mjGEOM_SPHERE:
            parts = [(np.zeros(3), size[0])]
        elif typ == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, h = size[:2]
            n = max(2, int(np.ceil(2*h / r)) + 1)
            zs = np.linspace(-h, h, n)
            radius = np.sqrt(r*r + ((zs[1]-zs[0])/2)**2)
            parts = [(np.array([0.,0.,z]), radius) for z in zs]
        elif typ == mujoco.mjtGeom.mjGEOM_BOX:
            parts = [(np.zeros(3), np.linalg.norm(size))]
        elif typ == mujoco.mjtGeom.mjGEOM_MESH:
            # Three spheres cover the complete MuJoCo mesh OBB, including
            # head and self-only feet. These are conservative IK envelopes.
            axis = int(size.argmax())
            h = size[axis]
            r = np.linalg.norm(np.delete(size, axis))
            parts = []
            for z in np.linspace(-h*.6, h*.6, 3):
                c = np.zeros(3); c[axis] = z
                parts.append((c, np.sqrt(r*r + (.4*h)**2)))
        else:
            raise ValueError(f"Unsupported Q1 collision geom {typ}")
        for c, r in parts:
            spheres[model.body(b).name].append({"center": (origin+rot@c).tolist(), "radius": float(r),
                "source_geom":model.geom(g).name,"source_body":model.body(int(model.geom_bodyid[g])).name,
                "extra_sole_volume":False,"floor_eligible":floor_eligible(g)})
    for side in ("left", "right"):
        for x in (-.03, .01, .05, .09):
            spheres[f"{side}_ankle_roll_link"].append({"center": [x,0.,-.0191], "radius": .022,
                "source_geom":f"{side}_ankle_roll_link_foot_0","source_body":f"{side}_ankle_roll_link",
                "extra_sole_volume":True,"floor_eligible":True})
    allowed=set(physical_self_collision_pairs(model))
    flat=[(link,i,sphere) for link,entries in spheres.items() for i,sphere in enumerate(entries)]
    sc=[]
    for ia,(link_a,i,a) in enumerate(flat):
        for link_b,j,b in flat[ia+1:]:
            ga=model.geom(a["source_geom"]).id;gb=model.geom(b["source_geom"]).id
            if tuple(sorted((ga,gb))) not in allowed:continue
            if link_a==link_b:
                raise ValueError("An active physical collision pair lies within one fixed joint frame")
            sc.append((link_a,i,link_b,j,.002))
    physical_covered={tuple(sorted((model.geom(flat_a["source_geom"]).id,model.geom(flat_b["source_geom"]).id)))
                      for la,i,lb,j,_ in sc
                      for flat_a,flat_b in [(spheres[la][i],spheres[lb][j])]
                      if not flat_a["extra_sole_volume"] and not flat_b["extra_sole_volume"]}
    if physical_covered!=allowed:
        raise ValueError("Incomplete physical geom pair coverage in Q1 sphere model")
    payload={"schema_version":3,"coverage_policy":"physical_geom_pairs_v3", "collision_spheres":spheres,
             "floor_sphere_indices":[k for k,(_,_,sphere) in enumerate(flat) if sphere["floor_eligible"]],
             "self_collision_pairs":[list(row) for row in sc],
             "physical_geom_pairs":[[model.geom(a).name,model.geom(b).name] for a,b in sorted(allowed)],
             "coverage":{"physical_geom_pair_count":len(allowed),"sphere_pair_count":len(sc),
                         "parent_filter_enabled":not bool(int(model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)),
                         "excluded_body_pairs":[[model.body(int(v)>>16).name,model.body(int(v)&65535).name] for v in model.exclude_signature],
                         "note":"Complete coverage of enabled physical geometry pairs; spheres and added sole volume remain conservative IK approximations, not visual mesh certification."}}
    COLLISION.parent.mkdir(parents=True,exist_ok=True)
    temporary=COLLISION.with_suffix(COLLISION.suffix+".tmp")
    temporary.write_text(yaml.safe_dump(payload,sort_keys=False));temporary.replace(COLLISION)
    return child,bubblify_to_mine(COLLISION,child,"pelvis",True,tuple(sc),verbose=False)


def load_q1(*, rebuild=True):
    """Build derived assets in DATA/q1 and load the pinned Q1 robot."""
    from frax.core.humanoid import Humanoid

    model = build_model() if rebuild else mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    # Official URDF has wider limits than q1_v3.xml (notably shoulder roll).
    # Retarget into exactly the conservative limits used by the physical model.
    tree = ET.parse(URDF_SOURCE)
    for joint in tree.findall("joint"):
        if joint.get("name") in JOINT_NAMES:
            idx = model.joint(joint.get("name")).id
            joint.find("limit").set("lower", str(model.jnt_range[idx,0]))
            joint.find("limit").set("upper", str(model.jnt_range[idx,1]))
    for mesh in tree.findall(".//mesh"):
        mesh.set("filename", os.path.relpath(URDF_SOURCE.parent / mesh.get("filename"), URDF.parent))
    compiler = tree.find("./mujoco/compiler")
    if compiler is not None:
        compiler.set("meshdir", ".")
        compiler.set("strippath", "false")
    if rebuild:
        tree.write(URDF)
    child, collision = collision_model(model, write=rebuild)
    hand = np.eye(4); hand[:3,3] = [.126588,0.,-.01]
    foot = np.eye(4); foot[:3,3] = [.0225,0.,-.0411]
    robot = Humanoid(str(URDF), "left_elbow_joint", "right_elbow_joint",
                     "left_ankle_roll_joint", "right_ankle_roll_joint",
                     left_hand_ee_offset=hand, right_hand_ee_offset=hand,
                     left_foot_ee_offset=foot, right_foot_ee_offset=foot,
                     joint_ordering=tuple(JOINT_NAMES), floating_base="quaternion",
                     default_configuration=np.r_[0.,0.,.42,1.,0.,0.,0.,DEFAULT_Q], collision_data=collision)
    stored=yaml.safe_load(COLLISION.read_text())
    robot.q1_floor_sphere_indices=np.asarray(stored.get("floor_sphere_indices",np.arange(len(robot.flat_collision_radii))),dtype=int)
    if robot.q1_floor_sphere_indices.ndim!=1 or not len(robot.q1_floor_sphere_indices) or np.any(robot.q1_floor_sphere_indices<0) or np.any(robot.q1_floor_sphere_indices>=len(robot.flat_collision_radii)):
        raise ValueError("Invalid stored floor-sphere selection")
    robot.q1_collision_coverage_policy=stored.get("coverage_policy","legacy_selected_pairs_v1")
    np.testing.assert_allclose(robot.actuated_joint_lower_limits, model.jnt_range[1:,0], atol=1e-5)
    np.testing.assert_allclose(robot.actuated_joint_upper_limits, model.jnt_range[1:,1], atol=1e-5)
    return robot, model


def make_retargeter(robot, model, self_collision=True, foot_rotation_weight=10.,
                    foot_position_weight=100., damping=10., shoulder_damping=300., foot_lift_gain=0.,
                    self_collision_margin=.003, self_collision_gain=30., self_collision_penalty=1e6,
                    self_collision_max_rows=512, floor_margin=.002, floor_gain=30.,
                    floor_penalty=1e6, floor_max_rows=32, joint_velocity_limit=12.):
    import jax
    import jax.numpy as jnp
    from frax.ik import DiffIKSolver, FloorConstraint, FrameTask, JointLimits, Retargeter, SelfCollisionConstraint
    from frax.utils.rotation_utils import quat_wxyz_to_rmat, rmat_to_quat_wxyz

    if not isinstance(self_collision, bool):
        raise ValueError("self_collision must be boolean")
    positive = (self_collision_gain,self_collision_penalty,floor_gain,floor_penalty,joint_velocity_limit)
    nonnegative = (self_collision_margin,floor_margin,foot_rotation_weight,foot_position_weight,
                   damping,shoulder_damping,foot_lift_gain)
    if not all(np.isfinite(x) and x>0 for x in positive) or not all(np.isfinite(x) and x>=0 for x in nonnegative):
        raise ValueError("FRAX gains/penalties/speed must be positive; margins/weights/damping nonnegative")
    for rows in (self_collision_max_rows,floor_max_rows):
        if rows is not None and (isinstance(rows,bool) or not isinstance(rows,int) or rows<1):
            raise ValueError("Constraint max_rows must be a positive integer or null (all rows)")
    cfg = json.loads(IK_CONFIG.read_text())
    table = cfg["ik_match_table1"]
    names = tuple(row[0] for row in table.values())
    bodies = tuple(table)
    rotations = Rotation.from_quat([row[4] for row in table.values()], scalar_first=True).as_matrix()
    offsets = np.broadcast_to(np.eye(4), (len(bodies),4,4)).copy()
    joint_idxs = []
    for i, bname in enumerate(bodies):
        b = model.body(bname).id
        p = np.zeros(3)
        while model.body_jntnum[b] == 0:
            p += model.body_pos[b]  # fixed wrist rotations are identity in Q1 v3
            b = model.body_parentid[b]
        j = model.body_jntadr[b]
        jname = "yaw_joint" if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else model.joint(j).name
        joint_idxs.append(robot.joint_name_to_index[jname])
        offsets[i,:3,3] = p
    # Derive body proportions from Q1 geometry, rather than reusing G1's scale.
    data = mujoco.MjData(model); mujoco.mj_forward(model,data)
    pos = lambda name: data.xpos[model.body(name).id]
    hip = (pos("left_hip_pitch_link") + pos("right_hip_pitch_link"))/2
    leg = np.linalg.norm(pos("left_hip_pitch_link")-pos("left_knee_link")) + np.linalg.norm(pos("left_knee_link")-pos("left_ankle_roll_link"))
    upper = np.linalg.norm(pos("left_shoulder_roll_link")-hip) + np.linalg.norm(pos("left_shoulder_roll_link")-pos("left_elbow_link")) + np.linalg.norm(pos("left_elbow_link")-pos("left_wrist_yaw_link"))
    leg_scale, upper_scale = leg/(.4350+.4237), upper/(.5375+.3300+.2520)
    pelvis_offset = data.xpos[model.body("pelvis").id] - hip
    floor_offset = np.array([0.,0.,leg_scale*.077-.0411])
    hips = np.array([names.index(n) for n in ("LeftUpLeg","RightUpLeg")])
    feet = np.array([names.index(n) for n in ("LeftFootMod","RightFootMod")])

    sole=np.array([[-.05,.023,-.0391],[-.05,-.023,-.0391],[.095,.023,-.0391],[.095,-.023,-.0391],[.12,0.,-.035]])
    def preprocess(inputs):
        poses=inputs["poses"] if isinstance(inputs,dict) else inputs
        R = jax.vmap(quat_wxyz_to_rmat)(poses[:,3:]) @ rotations
        h = jnp.mean(poses[hips,:3],axis=0)
        targets = leg_scale*h + upper_scale*(poses[:,:3]-h)
        targets = targets.at[0].set(leg_scale*h + R[0]@pelvis_offset)
        targets = targets.at[feet].set(leg_scale*poses[feet,:3])
        targets = targets-floor_offset
        lift = foot_lift_gain*jnp.maximum(targets[feet,2]-.0431,0.)
        targets = targets.at[feet,2].add(lift)
        if isinstance(inputs,dict):
            sole_low=jnp.min(jnp.einsum("bij,kj->bki",R[feet],sole)[...,2]-.002,axis=-1)
            targets=targets.at[feet,2].set(inputs["clearance"]-sole_low)
        return {"frame_pos": targets, "frame_rmat": R}

    def initial_q(poses):
        ref = preprocess(poses)
        return jnp.asarray(robot.default_configuration).at[:3].set(ref["frame_pos"][0]).at[3:7].set(rmat_to_quat_wxyz(ref["frame_rmat"][0]))

    wp=np.array([r[1] for r in table.values()],dtype=float)
    wr=np.array([r[2] for r in table.values()],dtype=float)
    wp[feet],wr[feet]=foot_position_weight,foot_rotation_weight
    task = FrameTask(robot, joint_idxs, gain_pos=30., gain_rot=30., offsets=offsets,
                     weight_pos=wp, weight_rot=wr)
    limits = JointLimits(robot, q_min=robot.actuated_joint_lower_limits+.003,
                         q_max=robot.actuated_joint_upper_limits-.003,
                         v_max=np.minimum(robot.actuated_joint_max_velocities, joint_velocity_limit), gain=30.)
    constraints = [limits, FloorConstraint(robot, sphere_idxs=getattr(robot,"q1_floor_sphere_indices",None),
                                           max_rows=floor_max_rows, gain=floor_gain,
                                           margin=floor_margin,penalty=floor_penalty)]
    if self_collision:
        constraints.append(SelfCollisionConstraint(robot,max_rows=self_collision_max_rows,
                           gain=self_collision_gain,margin=self_collision_margin,penalty=self_collision_penalty))
    reg = np.full(robot.nv,damping)
    for name in JOINT_NAMES:
        if "shoulder" in name:
            reg[robot.joint_name_to_index[name]] = shoulder_damping
    solver = DiffIKSolver(robot,[task],constraints,regularization=reg)
    retargeter = Retargeter(solver, preprocess, initial_q)
    return retargeter, names, {"leg_scale": float(leg_scale), "upper_scale": float(upper_scale),
                              "pelvis_offset": pelvis_offset.tolist(), "floor_offset": floor_offset.tolist()}


def load_lafan1(path):
    """Read LAFAN BVH as global Z-up poses in meters with WXYZ rotations.

    FootMod uses ankle position and toe orientation, matching FRAX's pinned
    LAFAN example and the GMR Q1 mapping. No example-script import is needed.
    """
    frame_time = None
    with Path(path).open() as stream:
        for line in stream:
            if line.strip().startswith("Frame Time:"):
                frame_time = float(line.split(":", 1)[1])
                break
    if frame_time is None or not np.isclose(frame_time, 1/30., rtol=0., atol=1e-6):
        raise ValueError(f"Expected 30 Hz LAFAN BVH input: {path}")
    from frax.ik import load_bvh

    poses, names = load_bvh(path, np.array([[1,0,0],[0,0,-1],[0,1,0]]))
    index = {name: i for i, name in enumerate(names)}
    feet = [np.concatenate((poses[:,index[f"{s}Foot"],:3], poses[:,index[f"{s}Toe"],3:]),axis=-1)
            for s in ("Left", "Right")]
    return np.concatenate((poses,np.stack(feet,axis=1)),axis=1), names + ["LeftFootMod","RightFootMod"]
