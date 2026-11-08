"""Playback timing, physical-contact separation, and render-copy isolation."""
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from q1_frax.play import PlaybackClock, native_viewer
from q1_frax.visuals import CollisionColors, excluded_body_pairs, make_render_model, physical_contacts


class PlaybackTest(unittest.TestCase):
    def test_native_close_waits_for_its_asynchronous_render_thread(self):
        stop, closed = threading.Event(), threading.Event()
        def launch(*args, **kwargs):
            def render_thread():
                stop.wait(5.)
                closed.set()
            thread = threading.Thread(target=render_thread, daemon=True)
            thread.start()
            class Handle:
                def __enter__(self): return self
                def __exit__(self, *args): stop.set()
            return Handle()
        with patch("mujoco.viewer.launch_passive", launch):
            with native_viewer(None, None):
                self.assertFalse(closed.is_set())
        self.assertTrue(closed.is_set())

    def test_pause_step_speed_restart_and_loop(self):
        clock = PlaybackClock(30, 30., start_frame=3)
        clock.advance(.1); self.assertEqual(clock.frame, 6)
        clock.key(ord("P")); clock.advance(10.); self.assertEqual(clock.frame, 6)
        clock.key(262); self.assertEqual(clock.frame, 7); self.assertTrue(clock.paused)
        clock.key(263); self.assertEqual(clock.frame, 6)
        clock.key(ord("=")); self.assertEqual(clock.speed, 2.)
        clock.key(ord("R")); self.assertEqual(clock.frame, 3); self.assertFalse(clock.paused)
        clock.advance(.5); self.assertEqual(clock.frame, 3)
        clock.key(ord("-")); self.assertEqual(clock.speed, 1.)
        clock.step(-100); self.assertEqual(clock.frame, 0)
        clock.step(100); self.assertEqual(clock.frame, 29)

    def test_no_loop_stops_and_invalid_time_is_rejected(self):
        clock = PlaybackClock(30, 30., loop=False)
        clock.advance(2.); self.assertEqual(clock.frame, 29); self.assertTrue(clock.paused)
        for value in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError): clock.advance(value)
        with self.assertRaises(ValueError): PlaybackClock(0, 30.)
        with self.assertRaises(ValueError): PlaybackClock(30, 0.)
        with self.assertRaises(ValueError): PlaybackClock(30, 30., start_frame=30)

    def physical_fixture(self):
        return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="floor" type="plane" size="3 3 .1"/>
          <body name="left" pos="0 0 .08"><freejoint/><geom name="left_sphere" type="sphere" size=".1" group="4"/></body>
          <body name="right" pos=".15 0 .08"><freejoint/><geom name="right_sphere" type="sphere" size=".1" group="4"/></body>
          <body name="excluded" pos="0 0 .08"><freejoint/><geom name="excluded_sphere" type="sphere" size=".1" group="4"/></body>
          </worldbody><contact><exclude body1="left" body2="excluded"/></contact></mujoco>''')

    def test_actual_geom_pairs_floor_and_exclusions(self):
        model = self.physical_fixture(); data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        contacts = physical_contacts(model, data)
        pair = next(row for row in contacts["self"] if row["links"] == ["left", "right"])
        self.assertEqual(pair["geoms"], ["left_sphere", "right_sphere"])
        self.assertAlmostEqual(pair["penetration_mm"], 50., places=8)
        self.assertTrue(contacts["floor"])
        self.assertAlmostEqual(contacts["floor"][0]["penetration_mm"], 20., places=8)
        self.assertFalse(any(set(row["links"]) == {"left", "excluded"} for row in contacts["self"]))
        self.assertIn(["left", "excluded"], excluded_body_pairs(model))
        self.assertFalse(contacts["environment"])

    def test_render_copy_and_highlight_restore_never_change_physics(self):
        physical = self.physical_fixture(); data = mujoco.MjData(physical)
        mujoco.mj_forward(physical, data)
        rgba, friction, mask = physical.geom_rgba.copy(), physical.geom_friction.copy(), physical.geom_contype.copy()
        render = make_render_model(physical)
        colors = CollisionColors(render)
        render_rgba = render.geom_rgba.copy()
        colors.update(physical_contacts(physical, data), show_geoms=True)
        self.assertFalse(np.array_equal(render.geom_rgba, render_rgba))
        for actual, saved in ((physical.geom_rgba, rgba), (physical.geom_friction, friction), (physical.geom_contype, mask)):
            np.testing.assert_array_equal(actual, saved)
        colors.update({"self": [], "floor": [], "environment": []})
        np.testing.assert_array_equal(render.geom_rgba, render_rgba)

    def test_playback_import_does_not_load_training_or_retarget_runtime(self):
        source = Path(__file__).resolve().parents[1]/"src"
        code = f'''import importlib.abc, sys
sys.path.insert(0, {str(source)!r})
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch','supermimic','frax','jax'):
   raise RuntimeError('Unexpected playback dependency: '+fullname)
sys.meta_path.insert(0,Block())
import q1_frax.play, q1_frax.render, q1_frax.motion_io
print('independent playback PASS')'''
        result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn("independent playback PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
