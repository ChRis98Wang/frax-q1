"""Headless annotated PNG/MP4 rendering of Q1 kinematic reference motion."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from .model import JOINT_NAMES, MJCF_PATH
from .motion_io import guard_output_paths, load_motion, sha256, verify_motion_dependencies


def _load_render_runtime():
    """Initialize graphics only when rendering is requested, never for --help."""
    global mujoco, CollisionColors, annotate, collision_summary
    global excluded_body_pairs, make_render_model, physical_contacts
    # Respect a backend already selected by native play or an embedding program.
    if "mujoco" not in sys.modules and sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from .visuals import (CollisionColors, annotate, collision_summary, excluded_body_pairs,
                          make_render_model, physical_contacts)


def load_model(path):
    _load_render_runtime()
    model = mujoco.MjModel.from_xml_path(str(Path(path).expanduser().absolute()))
    names = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    if model.nq != 29 or names != list(JOINT_NAMES):
        raise ValueError("Physical model must contain the exact Q1 free root and 22-joint ordering")
    return model


class ReferenceRenderer:
    """Keep an untouched physical model separate from all visual diagnostics."""
    def __init__(self, physical_model, motion, *, width=1280, height=900,
                 show_collision_geoms=False, show_link_labels=False, mark_self_collisions=True):
        _load_render_runtime()
        self.physical_model, self.motion = physical_model, motion
        self.physical_data = mujoco.MjData(physical_model)
        self.model = make_render_model(physical_model)
        self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)
        self.data = mujoco.MjData(self.model)
        self.colors = CollisionColors(self.model)
        self.show_geoms, self.mark = show_collision_geoms, mark_self_collisions
        self.option = mujoco.MjvOption()
        self.option.geomgroup[4] = int(show_collision_geoms)
        self.option.label = mujoco.mjtLabel.mjLABEL_BODY if show_link_labels else mujoco.mjtLabel.mjLABEL_NONE
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.95, 135., -17.
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)

    def frame(self, index, speed=1.):
        if not 0 <= index < len(self.motion.qpos):
            raise IndexError("Frame is outside the motion")
        self.physical_data.qpos[:] = self.motion.qpos[index]
        mujoco.mj_forward(self.physical_model, self.physical_data)
        contacts = physical_contacts(self.physical_model, self.physical_data)
        self.colors.update(contacts, mark=self.mark, show_geoms=self.show_geoms)
        self.data.qpos[:] = self.motion.qpos[index]
        self.data.time = index/self.motion.fps
        mujoco.mj_forward(self.model, self.data)
        body = self.physical_data.xpos[1:]
        self.camera.lookat[:] = (body.min(axis=0)+body.max(axis=0))/2
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        image = annotate(self.renderer.render().copy(), self.motion, index, contacts, speed=speed)
        return image, contacts

    def close(self):
        self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def render_motion(motion, model_path, output, *, start_frame=0, seconds=None, speed=1.,
                  output_fps=None, width=1280, height=900, show_collision_geoms=False,
                  show_link_labels=False, mark_self_collisions=True, collision_report=None):
    if not 0 <= start_frame < len(motion.qpos):
        raise ValueError("start_frame is outside the selected motion")
    fps = motion.fps if output_fps is None else float(output_fps)
    if not all(np.isfinite(x) and x > 0 for x in (fps, speed, width, height)):
        raise ValueError("Frame rates, speed and dimensions must be positive and finite")
    if seconds is not None and (not np.isfinite(seconds) or seconds <= 0):
        raise ValueError("seconds must be positive and finite")
    target = Path(output).expanduser().resolve()
    if target.suffix.lower() not in (".png", ".mp4"):
        raise ValueError("Output must be PNG or MP4")
    report_path = None if collision_report is None else Path(collision_report).expanduser().resolve()
    sidecar = target.with_name(target.name+".render.json")
    guard_output_paths(motion, model_path, target, sidecar, report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    before = sha256(model_path)
    if before != motion.model_sha256:
        raise ValueError("Physical model changed since motion validation")
    verify_motion_dependencies(motion, model_path)
    model = load_model(model_path)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(collision_summary(model, motion), indent=2)+"\n")
    available = (len(motion.qpos)-start_frame)/motion.fps
    duration = min(available, available if seconds is None else seconds)
    count = 1 if target.suffix.lower() == ".png" else max(1, int(np.ceil(duration/speed*fps-1e-10)))
    indices = np.minimum(len(motion.qpos)-1, start_frame+np.floor(np.arange(count)*speed*motion.fps/fps+1e-9).astype(int))
    captures = {}
    with ReferenceRenderer(model, motion, width=int(width), height=int(height),
                           show_collision_geoms=show_collision_geoms, show_link_labels=show_link_labels,
                           mark_self_collisions=mark_self_collisions) as renderer:
        if target.suffix.lower() == ".png":
            image, contacts = renderer.frame(start_frame, speed)
            image.save(target)
            captures[str(start_frame)] = contacts
        else:
            if width % 2 or height % 2:
                raise ValueError("H.264 output dimensions must be even")
            try:
                import imageio.v2 as imageio
            except ImportError as exc:
                raise ImportError("MP4 output requires the visualization extra: pip install 'q1-frax[visualization]'") from exc
            with imageio.get_writer(target, fps=fps, codec="libx264", quality=8, macro_block_size=1,
                                    ffmpeg_params=["-movflags", "+faststart"]) as writer:
                for number, index in enumerate(indices):
                    image, contacts = renderer.frame(int(index), speed)
                    writer.append_data(np.asarray(image))
                    if number in (0, count//2, count-1):
                        captures[str(int(index))] = contacts
    if sha256(model_path) != before or sha256(motion.path) != motion.sha256:
        raise RuntimeError("An input changed during rendering")
    verify_motion_dependencies(motion, model_path)
    report = {**motion.provenance(), "output": str(target), "output_frames": count,
              "output_fps": fps, "output_duration_seconds": 0. if target.suffix.lower() == ".png" else count/fps,
              "width": width, "height": height, "speed": speed, "start_frame": start_frame,
              "diagnostic_frames": captures, "excluded_body_pairs": excluded_body_pairs(model),
              "physical_model_or_motion_modified": False}
    sidecar.write_text(json.dumps(report, indent=2)+"\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--motion", dest="motion_option")
    parser.add_argument("--output", dest="output_option")
    parser.add_argument("--model", type=Path, default=MJCF_PATH)
    parser.add_argument("--archive", action="store_true", help="Allow raw archive CSV; raw input is never labelled accepted")
    parser.add_argument("--sha256", help="Trusted expected input-file checksum")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--seconds", type=float, help="Source-motion duration to render; default is the remaining clip")
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--fps", type=float, help="Output frame rate; default is original source fps")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--show-collision-geoms", action="store_true")
    parser.add_argument("--show-link-labels", action="store_true")
    parser.add_argument("--mark-self-collisions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collision-report")
    args = parser.parse_args(argv)
    source, output = args.motion_option or args.motion, args.output_option or args.output
    if source is None or output is None:
        parser.error("motion and output paths are required")
    motion = load_motion(source, model_path=args.model, archive=args.archive, expected_sha256=args.sha256)
    report = render_motion(motion, args.model, output, start_frame=args.start_frame, seconds=args.seconds,
                           speed=args.speed, output_fps=args.fps, width=args.width, height=args.height,
                           show_collision_geoms=args.show_collision_geoms, show_link_labels=args.show_link_labels,
                           mark_self_collisions=args.mark_self_collisions, collision_report=args.collision_report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
