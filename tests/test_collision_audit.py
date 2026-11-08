from importlib.util import module_from_spec,spec_from_file_location
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np
from test_quality import fixture_model,fixture_poses

SPEC=spec_from_file_location("collision_audit",Path(__file__).resolve().parents[1]/"tools/audit_q1_collision.py")
MODULE=module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)
PROFILE=Path(__file__).resolve().parents[1]/"configs/quality_strict_dynamic_v1.json"


class CollisionAuditTest(unittest.TestCase):
    def test_all_samples_and_crop_are_reported_without_mutation(self):
        model=fixture_model();q=fixture_poses(model,4);q[:,2]=.401;original=q.copy()
        result=MODULE.audit(model,q,30.,start=90,substeps=4,profile=PROFILE)
        self.assertEqual(result["checked_samples"],13)
        self.assertEqual(result["source_range_half_open"],[90,94])
        self.assertTrue(result["sampled_physical_and_continuity_passed"])
        self.assertFalse(result["continuous_time_guarantee"])
        self.assertFalse(result["disabled_visual_meshes_checked"])
        np.testing.assert_array_equal(q,original)
        json.dumps(result,allow_nan=False)

    def test_between_frame_collision_is_not_hidden_by_clean_endpoints(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
        <geom name="floor" type="plane" size="3 3 .1"/>
        <body name="base"><freejoint/><geom type="sphere" size=".04" mass="1"/>
        <geom name="left_foot_0" type="sphere" pos="0 .08 -.38" size=".02"/>
        <geom name="right_foot_0" type="sphere" pos="0 -.08 -.38" size=".02"/>
        <body name="moving" pos="0 0 .2"><joint name="left_shoulder_roll_joint" axis="0 0 1" range="-2 2"/><geom name="moving_sphere" type="sphere" pos=".2 0 0" size=".02"/></body>
        <body name="target" pos="0 0 .2"><joint name="right_shoulder_roll_joint" axis="0 0 1" range="-2 2"/><geom name="target_sphere" type="sphere" pos=".2 0 0" size=".02"/></body>
        </body></worldbody></mujoco>''')
        q=fixture_poses(model,2);q[:,2]=.401;q[:,7]=[-1.,1.]
        endpoints=MODULE.audit(model,q,1.,substeps=1,profile=PROFILE)
        dense=MODULE.audit(model,q,1.,substeps=4,profile=PROFILE)
        self.assertTrue(endpoints["gates"]["self_no_penetration"])
        self.assertFalse(dense["gates"]["self_no_penetration"])
        self.assertAlmostEqual(dense["collision"]["self"]["deepest_penetration_m"],.04)
        self.assertEqual(dense["collision"]["self"]["worst"]["interval_fraction"],.5)

    def test_submillimeter_penetration_is_reported_and_fails(self):
        model=fixture_model();q=fixture_poses(model,3);q[:,2]-=.0005
        result=MODULE.audit(model,q,30.,profile=PROFILE)
        self.assertFalse(result["gates"]["floor_no_penetration"])
        self.assertAlmostEqual(result["collision"]["floor"]["deepest_penetration_m"],.0005)

    def test_no_visual_mesh_evidence_cannot_be_called_a_pass(self):
        model=fixture_model();q=fixture_poses(model,3)
        with self.assertRaisesRegex(ValueError,"evidence is unavailable"):
            MODULE.visual_envelope_diagnostic(model,q,30.,profile=PROFILE)

    def test_disabled_pair_evidence_cannot_be_called_collision_free(self):
        model=fixture_model();q=fixture_poses(model,3)
        model.geom_contype[:]=0;model.geom_conaffinity[:]=0
        with self.assertRaisesRegex(ValueError,"missing evidence cannot pass"):
            MODULE.audit(model,q,30.,profile=PROFILE)


if __name__=="__main__": unittest.main()
