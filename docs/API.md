# Python API

`import q1_frax` has no network, asset-generation or JAX initialization side
effects. The package uses submodule APIs so inspection remains independent of
optional retargeting dependencies. Set `Q1_FRAX_DATA` before importing modules
that resolve model paths.

## Read and inspect a motion

```python
from q1_frax.motion_io import load_motion
from q1_frax.model import MJCF_PATH

motion = load_motion("outputs/walk-smoke/npz/walk2_subject3.npz", model_path=MJCF_PATH)
print(motion.qpos.shape, motion.fps, motion.kind)
print(motion.provenance())
print(motion.status(100))
```

`Motion.qpos` uses MuJoCo order `[x,y,z,qw,qx,qy,qz,22 joints]`. Source CSV files
declare XYZW and are converted explicitly. `JOINT_NAMES` in `q1_frax.model`
defines ordering; files with a different ordering are rejected rather than
silently relabeled. A raw NPZ is not accepted data, even with a valid checksum.

## Render and report collisions

```python
from q1_frax.render import render_motion

report = render_motion(motion, MJCF_PATH, "artifacts/reference.png",
                       start_frame=100, show_link_labels=True,
                       mark_self_collisions=True)
```

The output suffix selects a still image or supported video/animation format.
Rendering operates on a visual copy. Contact reports use the original model
and distinguish self, floor and other environment contacts.

## Quality checks

```python
import mujoco
from q1_frax.quality import classify_frames, load_profile

model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
qa = classify_frames(model, motion.qpos, fps=motion.fps,
                     profile=load_profile())
print(qa["summary"])
```

This example has no original human targets or FRAX sphere evidence, so those
checks are **unverified** and it cannot pass the complete quality profile.
The retargeting pipeline supplies `root_target_error`, `foot_target_error`,
`sphere_floor` and `sphere_self_distance` from the same source and solver.
`accepted_span_slices` extracts uninterrupted accepted regions without bridging
rejected frames. Sealing independently recomputes the evidence.

## Retargeting components

Install the `retarget` extra before using `q1_frax.adapter`. `load_q1()` returns
the FRAX robot and physical MuJoCo model. `make_retargeter()` returns the
retargeter, source bone names and geometric scaling metadata.
`load_parameters()` in `q1_frax.retarget` validates tuned parameters; unknown
keys, invalid row counts and nonfinite values are errors. The command entry
point `q1_frax.retarget.main(argv)` manages CPU/JAX configuration and provenance.

## Seal a dataset

```python
from q1_frax.seal import seal_dataset

report = seal_dataset("outputs/walking",
                      held_out=("walk1_subject2", "walk2_subject4"))
report = seal_dataset("outputs/walking", verify_only=True)
```

Full verification needs original BVH inputs and pinned retarget assets. The
`source_dir` override locates moved BVHs, whose bytes must still match recorded
hashes. `integrity_only=True` is a separate non-replay check on an already
sealed dataset; it does not certify source tracking or physical quality.
