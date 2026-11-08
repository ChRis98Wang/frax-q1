"""Primebot Q1 v3 specification, independent of the simulator frontend.

Gains/home posture: PrimebotTech/primebot_rl_mjlab q1_v3_constants.py.
Joint limits/inertias: PrimebotTech/primebot-q1-urdf, q1_v3.xml.
"""
from .paths import DATA
MJCF_PATH = DATA / "q1" / "q1.xml"
JOINT_NAMES = [
    f"{side}_{part}_joint"
    for side in ("left", "right")
    for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
] + ["waist_roll_joint", "waist_yaw_joint"] + [
    f"{side}_{part}_joint"
    for side in ("left", "right")
    for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
]
TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]
DEFAULT_Q = [-.09, -.02, -.02, .22, -.13, 0., -.09, .02, .02, .22, -.13, 0.,
             0., 0., 0., .18, 0., 1.4, 0., -.18, 0., 1.4]
NUM_J = len(JOINT_NAMES)
# friction, 22 encoder offsets, torso CoM xyz, global mass/inertia, Kp, Kd scales
ACTION_SCALE = .25
ACTION_LIMIT = 13.  # covers every physical joint range relative to DEFAULT_Q; targets/torques remain limited


def motor(name):
    if name == "waist_roll_joint":
        return 80., 3.5, 36.
    if any(s in name for s in ("ankle", "shoulder", "elbow")):
        return 20., .8, 21.  # official RL config conservatively caps 22 Nm motors at 21 Nm
    return 30., 1.5, 36.


def build_model():
    """Build a reproducible physical model with foot points and self collision.

Visual meshes never collide. Adjacent and next-adjacent links are excluded because
their simplified collision cylinders overlap at joints. All other body pairs collide.
"""
    import copy
    import os
    import xml.etree.ElementTree as ET
    import mujoco
    import numpy as np

    source = DATA / "primebot-q1-urdf" / "xml" / "q1_v3.xml"
    if not source.is_file():
        raise FileNotFoundError(f"Q1 model sources not found at {source}. Run q1-frax-setup first.")
    root = ET.parse(source).getroot()
    # Keep the lexical cache layout: resolving a source symlink would embed
    # another machine's checkout location and change the model checksum.
    meshdir = os.path.relpath(source.parent.parent / "meshes", MJCF_PATH.parent)
    root.find("compiler").set("meshdir", meshdir)
    root.find("compiler").set("autolimits", "true")
    for tag in ("option", "actuator", "sensor", "keyframe", "contact"):
        for elem in root.findall(tag):
            root.remove(elem)
    ET.SubElement(root, "option", timestep="0.005", integrator="implicitfast",
                  solver="Newton", iterations="20", ls_iterations="10", cone="pyramidal",
                  jacobian="dense", gravity="0 0 -9.81")
    world = root.find("worldbody")
    # Remove upstream floor(s), replacing with a pair-symmetric material. No priority override.
    for wb in root.findall("worldbody"):
        for g in wb.findall("geom"):
            wb.remove(g)
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 .1",
                  contype="1", conaffinity="2", friction=".6 .005 .0001", condim="3", solref=".01 1")
    bodies = {b.get("name"): b for b in root.findall(".//body")}
    parents = {child.get("name"): b.get("name") for b in bodies.values() for child in b.findall("body")}
    for name, b in bodies.items():
        for i, g in enumerate(list(b.findall("geom"))):
            visual = g.get("contype") == "0" and g.get("conaffinity") == "0"
            if name.endswith("ankle_roll_link") and not visual:
                b.remove(g)
                continue
            if not visual:
                g.set("name", f"{name}_collision_{i}")
                g.set("contype", "2")
                g.set("conaffinity", "3")
                g.set("condim", "3")
                g.set("friction", ".6 .005 .0001")
                g.set("group", "4")
                g.set("solref", ".01 1")
        if name.endswith("ankle_roll_link"):
            # The complete official foot convex hull participates in self
            # collision only. Bit 4 never matches the floor's masks (1/2).
            visual_mesh=next(g for g in b.findall("geom") if g.get("type")=="mesh" and g.get("contype")=="0")
            volume=copy.deepcopy(visual_mesh)
            volume.set("name",f"{name}_self_volume")
            volume.set("contype","4");volume.set("conaffinity","6")
            volume.set("group","4");volume.set("condim","3")
            volume.set("solref",".01 1");volume.set("density","0")
            b.append(volume)
            # Official Q1 RL five-point sole geometry, including toe point.
            for i, pos in enumerate(((-.05,.023,-.0391),(-.05,-.023,-.0391),
                                     (.095,.023,-.0391),(.095,-.023,-.0391),(.12,0.,-.035))):
                ET.SubElement(b, "geom", name=f"{name}_foot_{i}", type="sphere", size=".002",
                              pos=" ".join(map(str,pos)), contype="2", conaffinity="3",
                              friction=".6 .005 .0001", condim="3", group="4", solref=".01 1")
    contact = ET.SubElement(root, "contact")
    for name in bodies:
        ancestor = parents.get(name)
        for _ in range(2):
            if ancestor is None:
                break
            ET.SubElement(contact, "exclude", body1=name, body2=ancestor)
            ancestor = parents.get(ancestor)
    actuator = ET.SubElement(root, "actuator")
    for name in JOINT_NAMES:
        kp, kd, limit = motor(name)
        j = next(j for j in root.findall(".//joint") if j.get("name") == name)
        j.set("armature", ".002")
        j.set("damping", "0")
        j.set("actuatorfrcrange", f"{-limit} {limit}")
        ET.SubElement(actuator, "position", name=name, joint=name, kp=str(kp), kv=str(kd),
                      forcerange=f"{-limit} {limit}", ctrlrange=j.get("range"))
    MJCF_PATH.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    # Atomic replacement, and no timestamp churn when the model is unchanged.
    payload = ET.tostring(root)
    if not MJCF_PATH.exists() or MJCF_PATH.read_bytes() != payload:
        temporary = MJCF_PATH.with_suffix(".xml.tmp")
        temporary.write_bytes(payload)
        temporary.replace(MJCF_PATH)
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    assert [model.joint(i).name for i in range(1, model.njnt)] == JOINT_NAMES
    assert model.nq == 29 and model.nu == 22
    np.testing.assert_array_less(model.jnt_range[1:, 0] - 1e-6, np.array(DEFAULT_Q))
    np.testing.assert_array_less(np.array(DEFAULT_Q), model.jnt_range[1:, 1] + 1e-6)
    return model
