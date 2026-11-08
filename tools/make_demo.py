#!/usr/bin/env python3
"""Generate an original, analytic Q1 viewer demo; never reads motion datasets.

The trajectory is authored here with sine functions, joint limits and a fixed
seed contact-pose search. It is not LAFAN, retargeting output or a trained policy.
Only the local Q1 model/mesh assets are read. Public outputs contain no local
paths; the uncompressed synthetic trajectory and QA images stay in artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from q1_frax.integrity import dependency_fingerprints, verify_model_dependencies
from q1_frax.model import DEFAULT_Q, JOINT_NAMES, MJCF_PATH
from q1_frax.paths import SOURCES
from q1_frax.render import load_model
from q1_frax.visuals import CollisionColors, contact_lines, excluded_body_pairs, font, make_render_model, physical_contacts


DURATION = 24.
FPS = 24
WIDTH, HEIGHT = 960, 720
CHAPTERS = [
    {"start": 0., "end": 7., "title": "01 / ORIGINAL ANALYTIC JOINT MOTION", "mode": "Programmed continuous replay / 1x source clock"},
    {"start": 7., "end": 13., "title": "02 / PROGRAMMED SLOW REPLAY", "mode": "0.25x source clock / same authored trajectory"},
    {"start": 13., "end": 18., "title": "03 / PROGRAMMED SINGLE-FRAME ADVANCE", "mode": "One source frame every 0.5 s / no keyboard interaction recorded"},
    {"start": 18., "end": 24., "title": "04 / INJECTED COLLISION / DEMO", "mode": "Authored test pose / actual MuJoCo geom-pair contacts"},
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analytic_joints(t):
    """A small alternating joint sweep, authored without an input trajectory."""
    q = np.array(DEFAULT_Q, dtype=float)
    phase = 2*math.pi*.55*t
    for side, offset in ((0, 0.), (6, math.pi)):
        swing = max(0., math.sin(phase+offset))
        q[side] -= .20*swing
        q[side+3] += .34*swing
        q[side+4] -= .14*swing
    q[14] = .23*math.sin(phase)
    q[18] = -.23*math.sin(phase)
    q[17] = 1.12+.18*math.sin(phase+.7)
    q[21] = 1.12-.18*math.sin(phase+.7)
    q[13] = .10*math.sin(phase*.5)
    return q


def pose(model, data, joints):
    q = np.zeros(29); q[3] = 1.; q[2] = .7
    limits = model.jnt_range[1:]
    q[7:] = np.clip(joints, limits[:, 0], limits[:, 1])
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    soles = [g for g in range(model.ngeom) if "_foot_" in (model.geom(g).name or "")]
    lowest = min(data.geom_xpos[g, 2]-model.geom_size[g, 0] for g in soles)
    # Kinematic display placement, not a simulation equilibrium or gait result.
    q[2] += .003-lowest
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    return q


def injected_contact_pose(model, data, seed=731):
    """Choose an original in-limit arm test pose with a visible actual contact."""
    rng = np.random.default_rng(seed)
    candidates = []
    for _ in range(1600):
        q = np.array(DEFAULT_Q, dtype=float)
        q[14:18] = rng.uniform([-2.7, .0, -1.3, -.6], [1.2, .8, 1.3, 1.57])
        pose(model, data, q)
        contacts = physical_contacts(model, data)["self"]
        if contacts and 8. <= contacts[0]["penetration_mm"] <= 25. and len(contacts) <= 3:
            score = abs(contacts[0]["penetration_mm"]-14.)+.4*np.linalg.norm(q[14:18])
            candidates.append((score, q.copy()))
    if not candidates:
        raise RuntimeError("No visible synthetic collision pose found; do not fabricate a contact")
    return min(candidates, key=lambda row: row[0])[1]


def timeline(t, injected):
    if t < 7.:
        chapter, source_t = 0, t
    elif t < 13.:
        chapter, source_t = 1, 7.+(t-7.)*.25
    elif t < 18.:
        chapter, source_t = 2, 8.5+math.floor((t-13.)*2)/30.
    else:
        chapter, source_t = 3, 8.5+9/30.
    q = analytic_joints(source_t)
    if chapter == 3:
        a = min(1., (t-18.)/1.5)
        a = a*a*(3.-2.*a)
        q = (1.-a)*q+a*injected
    return chapter, source_t, q


def overlay(rgb, t, source_t, chapter, contacts):
    image = Image.fromarray(rgb).convert("RGBA")
    layer = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(layer)
    def text(x, y, value, size=16, color=(220, 234, 245), bold=False):
        draw.text((x, y), value, font=font(size, bold), fill=color)
    draw.rounded_rectangle((14, 12, 946, 137), radius=11, fill=(12, 22, 33, 242))
    text(29, 22, "ORIGINAL SYNTHETIC MOTION", 25, (135, 224, 205), True)
    text(29, 57, "VIEWER DEMO / NOT LAFAN / NOT A TRAINED POLICY", 18, (255, 202, 132), True)
    text(29, 87, CHAPTERS[chapter]["title"], 18, bold=True)
    text(29, 113, f"Video {t:05.2f} / 24.00 s   |   source frame {int(source_t*30):03d}   |   {CHAPTERS[chapter]['mode']}", 12)
    draw.rounded_rectangle((14, 548, 946, 705), radius=11, fill=(12, 22, 33, 242))
    for n, line in enumerate(contact_lines(contacts, limit=2)[:6]):
        text(29, 558+n*20, line, 13 if line.startswith("  ") else 15,
             (255, 160, 131) if contacts["self"] and not line.startswith("FLOOR") else (218, 232, 242), n == 0)
    text(29, 683, "Kinematic FK only; not a retargeting-quality benchmark. Collision exclusions remain active.", 12, (174, 196, 213))
    return Image.alpha_composite(image, layer).convert("RGB")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MJCF_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/original_demo"))
    args = parser.parse_args(argv)
    model_hash = digest(args.model)
    dependencies = dependency_fingerprints(args.model)
    physical = load_model(args.model); physical_data = mujoco.MjData(physical)
    injected = injected_contact_pose(physical, physical_data)
    render = make_render_model(physical); data = mujoco.MjData(render)
    render.vis.global_.offwidth, render.vis.global_.offheight = WIDTH, HEIGHT
    colors = CollisionColors(render); opt = mujoco.MjvOption()
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.95, 135., -17.
    args.output_dir.mkdir(parents=True, exist_ok=True); args.artifacts.mkdir(parents=True, exist_ok=True)
    video = args.output_dir/"original_q1_demo.mp4"
    preview = args.output_dir/"original_q1_demo.gif"
    frames, poses, chapters, source_frames, contact_records = [], [], [], [], []
    check_indices = {0, 95, 185, 285, 335, 415, 479, 527, 575}
    with mujoco.Renderer(render, width=WIDTH, height=HEIGHT) as renderer:
        with imageio.get_writer(video, fps=FPS, codec="libx264", quality=7, macro_block_size=1,
                                ffmpeg_params=["-movflags", "+faststart"]) as writer:
            for index in range(int(DURATION*FPS)):
                t = index/FPS
                chapter, source_t, joints = timeline(t, injected)
                qpos = pose(physical, physical_data, joints)
                contacts = physical_contacts(physical, physical_data)
                poses.append(qpos); chapters.append(chapter); source_frames.append(int(source_t*30))
                contact_records.append(contacts)
                data.qpos[:] = qpos; mujoco.mj_forward(render, data)
                collision = chapter == 3
                colors.update(contacts, mark=True, show_geoms=collision)
                opt.geomgroup[4] = int(collision)
                camera.azimuth = 135. if not collision else 225.
                camera.lookat[:] = (physical_data.xpos[1:].min(axis=0)+physical_data.xpos[1:].max(axis=0))/2
                renderer.update_scene(data, camera=camera, scene_option=opt)
                frame = overlay(renderer.render().copy(), t, source_t, chapter, contacts)
                writer.append_data(np.asarray(frame))
                if index % 4 == 0:
                    frames.append(frame.resize((480, 360), Image.Resampling.LANCZOS))
                if index in check_indices:
                    frame.save(args.artifacts/f"frame_{index:03d}.png")
    palette = [frame.quantize(colors=96, method=Image.Quantize.FASTOCTREE) for frame in frames]
    def gif_durations(count, fps):
        # GIF stores centiseconds; distribute rounding to retain exact 24 s.
        return [10*(round((i+1)*100/fps)-round(i*100/fps)) for i in range(count)]
    palette[0].save(preview, save_all=True, append_images=palette[1:], duration=gif_durations(len(palette), 6), loop=0, optimize=True, disposal=1)
    gif_fps, gif_size = 6, [480, 360]
    if preview.stat().st_size > 3_000_000:
        palette = [frame.resize((384, 288)).quantize(colors=64, method=Image.Quantize.FASTOCTREE) for frame in frames[::2]]
        palette[0].save(preview, save_all=True, append_images=palette[1:], duration=gif_durations(len(palette), 3), loop=0, optimize=True, disposal=1)
        gif_fps, gif_size = 3, [384, 288]
    if video.stat().st_size > 10_000_000 or preview.stat().st_size > 3_000_000:
        raise RuntimeError("Demo exceeds the requested media size budget")
    with Image.open(preview) as gif:
        gif_duration = 0.
        for index in range(gif.n_frames):
            gif.seek(index); gif_duration += gif.info["duration"]/1000.
        if abs(gif_duration-DURATION) > .01:
            raise RuntimeError("GIF playback duration differs from its source")
    verify_model_dependencies(args.model, dependencies)
    if digest(args.model) != model_hash:
        raise RuntimeError("Physical model changed during demo generation")
    reader = imageio.get_reader(video)
    video_meta, count = reader.get_meta_data(), reader.count_frames(); reader.close()
    if count != 576 or video_meta["size"] != (WIDTH, HEIGHT) or abs(video_meta["duration"]-DURATION) > 1/FPS:
        raise RuntimeError("Encoded MP4 metadata mismatch")
    np.savez_compressed(args.artifacts/"original_trajectory.npz", q=np.asarray(poses), fps=FPS,
                        joint_names=JOINT_NAMES, root_quaternion="wxyz", chapter=chapters, source_frame=source_frames)
    (args.artifacts/"contacts.json").write_text(json.dumps(contact_records, indent=2)+"\n")
    model_source = SOURCES["primebot-q1-urdf"]
    metadata = {
        "schema_version": 1, "origin": "original_synthetic", "uses_lafan": False,
        "generator": "tools/make_demo.py", "generator_sha256": digest(__file__),
        "files": {path.name: {"sha256": digest(path), "size_bytes": path.stat().st_size} for path in (video, preview)},
        "fps": FPS, "frames": count, "duration_seconds": DURATION, "width": WIDTH, "height": HEIGHT,
        "preview": {"fps": gif_fps, "size": gif_size, "duration_seconds": DURATION},
        "chapters": CHAPTERS, "interaction": "Programmed source-clock and frame scheduling; no keyboard interaction is claimed",
        "source": "Original analytic sine joint trajectories and deterministic seed-731 arm-pose search authored in the generator",
        "not_a_retargeting_benchmark": True, "not_a_trained_policy": True,
        "root_placement": "Each FK pose is vertically placed 3 mm above its lowest physical sole sphere; no dynamics are simulated",
        "collision_definition": "Actual MuJoCo mj_forward penetrating geom pairs; self and floor separated; no contacts removed",
        "injected_pose_joint_positions_rad": injected.tolist(),
        "max_injected_self_penetration_mm": max((r["penetration_mm"] for c in contact_records for r in c["self"]), default=0.),
        "excluded_body_pairs": excluded_body_pairs(physical),
        "model_sha256": model_hash, "model_dependencies": dependencies,
        "model_upstream": {"repository": model_source["url"], "revision": model_source["commit"]},
        "source_credits": [{"name": "PrimebotTech Q1 robot geometry", "repository": model_source["url"], "revision": model_source["commit"]},
                           {"name": "MuJoCo", "repository": "https://github.com/google-deepmind/mujoco", "version": mujoco.__version__}],
        "copyright": "Original trajectory, presentation and generator by the frax-q1 contributors. Robot geometry remains copyright its upstream authors; the upstream package declares BSD and does not supply a full license text. No LAFAN motion or adaptation is used or distributed. See THIRD_PARTY_NOTICES.md.",
        "audio": False,
    }
    (args.output_dir/"demo.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"video": str(video), "gif": str(preview), "frames": count, "duration_seconds": DURATION,
                      "sizes_bytes": {p.name: p.stat().st_size for p in (video, preview)},
                      "max_injected_self_penetration_mm": metadata["max_injected_self_penetration_mm"]}, indent=2))


if __name__ == "__main__":
    main()
