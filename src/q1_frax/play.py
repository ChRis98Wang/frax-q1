"""Interactive native MuJoCo inspection of Q1 FRAX reference poses.

P/Space pause; Left/Right or comma/period step one frame; +/- change speed;
R restart; C collision geometry; L link labels; M self-collision highlights.
Kinematic reference replay only: no policy, training runtime, or physics steps.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np

from .model import MJCF_PATH
from .motion_io import guard_output_paths, load_motion, sha256, verify_motion_dependencies


@contextmanager
def native_viewer(model, data, **kwargs):
    """Finish native render threads before Python/GLFW interpreter teardown.

    MuJoCo's passive handle.close() requests exit without joining its daemon
    render thread. The standalone CLI owns the threads created by this launch;
    joining them avoids a race between GL context destruction and glfw.terminate.
    """
    from mujoco import viewer as mj_viewer
    existing = set(threading.enumerate())
    handle = mj_viewer.launch_passive(model, data, **kwargs)
    created = set(threading.enumerate())-existing
    try:
        with handle:
            yield handle
    finally:
        for thread in created:
            thread.join(timeout=5.)
        if any(thread.is_alive() for thread in created):
            raise RuntimeError("Native MuJoCo viewer did not finish closing")


@dataclass
class PlaybackClock:
    frames: int
    fps: float
    speed: float = 1.
    loop: bool = True
    start_frame: int = 0
    paused: bool = False
    position: float = 0.

    def __post_init__(self):
        if self.frames < 1 or not math.isfinite(self.fps) or self.fps <= 0 or not math.isfinite(self.speed) or self.speed <= 0:
            raise ValueError("Playback requires nonempty motion and positive finite fps/speed")
        if not 0 <= self.start_frame < self.frames:
            raise ValueError("start_frame is outside the motion")
        self.position = float(self.start_frame)

    @property
    def frame(self):
        return min(self.frames-1, max(0, int(math.floor(self.position+1e-9))))

    def advance(self, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Elapsed time must be finite and nonnegative")
        if self.paused:
            return
        self.position += elapsed*self.fps*self.speed
        if self.loop:
            self.position %= self.frames
        elif self.position >= self.frames:
            self.position = float(self.frames-1)
            self.paused = True

    def step(self, direction):
        self.position = float(min(self.frames-1, max(0, self.frame+direction)))
        self.paused = True

    def key(self, key):
        if key in (32, ord("P"), ord("p")):
            self.paused = not self.paused
        elif key in (262, ord(".")):
            self.step(1)
        elif key in (263, ord(",")):
            self.step(-1)
        elif key in (ord("R"), ord("r")):
            self.position, self.paused = float(self.start_frame), False
        elif key in (ord("+"), ord("="), 265):
            self.speed = min(8., self.speed*2)
        elif key in (ord("-"), 264):
            self.speed = max(.125, self.speed/2)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", nargs="?")
    parser.add_argument("--motion", dest="motion_option")
    parser.add_argument("--model", type=Path, default=MJCF_PATH)
    parser.add_argument("--archive", action="store_true", help="Inspect declared raw CSV; never label raw data accepted")
    parser.add_argument("--sha256", help="Trusted expected motion-file digest")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--paused", action="store_true", help="Open at the selected frame without advancing")
    parser.add_argument("--show-collision-geoms", action="store_true")
    parser.add_argument("--show-link-labels", action="store_true")
    parser.add_argument("--mark-self-collisions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--snapshot", help="Write an annotated PNG instead of opening a window")
    parser.add_argument("--collision-report", help="Export all actual geom/contact pairs and excluded body pairs")
    parser.add_argument("--check-only", action="store_true", help="Validate provenance and FK without opening a window")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    source = args.motion_option or args.motion
    if source is None:
        parser.error("A motion path is required")
    motion = load_motion(source, model_path=args.model, archive=args.archive, expected_sha256=args.sha256)
    if args.collision_report and not args.snapshot:
        guard_output_paths(motion, args.model, args.collision_report)
    clock = PlaybackClock(len(motion.qpos), motion.fps, args.speed, args.loop, args.start_frame, args.paused)
    if args.snapshot:
        from .render import render_motion
        report = render_motion(motion, args.model, args.snapshot, start_frame=args.start_frame, speed=args.speed,
                               show_collision_geoms=args.show_collision_geoms, show_link_labels=args.show_link_labels,
                               mark_self_collisions=args.mark_self_collisions, collision_report=args.collision_report)
        print(json.dumps(report, indent=2))
        return 0
    os.environ.setdefault("MUJOCO_GL", "glfw")
    import mujoco
    from .render import load_model
    from .visuals import CollisionColors, collision_summary, contact_lines, make_render_model, physical_contacts
    verify_motion_dependencies(motion, args.model)
    physical_model = load_model(args.model)
    physical_data = mujoco.MjData(physical_model)
    for index in (0, len(motion.qpos)//2, len(motion.qpos)-1):
        physical_data.qpos[:] = motion.qpos[index]
        mujoco.mj_forward(physical_model, physical_data)
        if not np.isfinite(physical_data.xpos).all():
            raise ValueError("Reference forward kinematics produced non-finite positions")
    print(json.dumps({**motion.provenance(), "speed": args.speed, "loop": args.loop}, indent=2), flush=True)
    if args.collision_report:
        target = Path(args.collision_report).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(collision_summary(physical_model, motion), indent=2)+"\n")
    if args.check_only:
        verify_motion_dependencies(motion, args.model)
        return 0
    model = make_render_model(physical_model)
    data = mujoco.MjData(model)
    data.qpos[:] = motion.qpos[clock.frame]
    data.time = clock.frame/motion.fps
    mujoco.mj_forward(model, data)
    colors = CollisionColors(model)
    keys = queue.SimpleQueue()
    with native_viewer(model, data, key_callback=keys.put,
                       show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = model.body("pelvis").id
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.8, 135., -17.
        previous, last_frame, style_dirty = time.perf_counter(), -1, True
        contacts = {"self": [], "floor": [], "environment": []}
        while viewer.is_running():
            now = time.perf_counter()
            clock.advance(now-previous)
            previous = now
            while not keys.empty():
                key = keys.get()
                clock.key(key)
                if key in (ord("C"), ord("c")):
                    args.show_collision_geoms = not args.show_collision_geoms
                    style_dirty = True
                elif key in (ord("L"), ord("l")):
                    args.show_link_labels = not args.show_link_labels
                    style_dirty = True
                elif key in (ord("M"), ord("m")):
                    args.mark_self_collisions = not args.mark_self_collisions
                    style_dirty = True
            if clock.frame != last_frame:
                physical_data.qpos[:] = motion.qpos[clock.frame]
                mujoco.mj_forward(physical_model, physical_data)
                contacts = physical_contacts(physical_model, physical_data)
                with viewer.lock():
                    data.qpos[:] = motion.qpos[clock.frame]
                    data.qvel[:] = 0.
                    data.time = clock.frame/motion.fps
                    mujoco.mj_forward(model, data)
                last_frame, style_dirty = clock.frame, True
            if style_dirty:
                with viewer.lock():
                    colors.update(contacts, mark=args.mark_self_collisions, show_geoms=args.show_collision_geoms)
                    viewer.opt.geomgroup[4] = int(args.show_collision_geoms)
                    viewer.opt.label = mujoco.mjtLabel.mjLABEL_BODY if args.show_link_labels else mujoco.mjtLabel.mjLABEL_NONE
                style_dirty = False
            status, reasons = motion.status(clock.frame)
            header = (f"Q1 FRAX / LAFAN - KINEMATIC REFERENCE, NOT A POLICY\n{status}\n{motion.path.name}\n"
                      f"{'PAUSED' if clock.paused else 'PLAYING'}  {clock.speed:g}x  frame {clock.frame}/{clock.frames-1}  {clock.frame/clock.fps:.3f} s\n"
                      "P/Space pause  Left/Right step  +/- speed  R restart\nC collision shapes  L link labels  M self-collision marks")
            footer = "\n".join(contact_lines(contacts, limit=2))+"\nRed >=2 mm; orange <2 mm; body exclusions remain active."
            if reasons:
                footer += "\nRejected: "+", ".join(reasons)
            if not motion.hash_verified:
                footer += "\nRAW HASH UNVERIFIED: no trusted motion-file digest was supplied."
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, header, ""),
                              (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, footer, "")])
            viewer.sync()
            time.sleep(max(0., 1/60.-(time.perf_counter()-now)))
    if sha256(args.model) != motion.model_sha256 or sha256(motion.path) != motion.sha256:
        raise RuntimeError("An input changed during playback")
    verify_motion_dependencies(motion, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
