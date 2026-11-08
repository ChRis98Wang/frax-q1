import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from q1_frax.quality import (DEFAULT_PROFILE,accepted_span_slices,classify_frames,
                             load_profile,main,percentiles,physical_metrics,profile_hash)


def fixture_model():
    return mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
      <geom name="floor" type="plane" size="3 3 .1"/>
      <body name="base"><freejoint/><geom type="sphere" size=".04" mass="1"/>
        <geom name="left_foot_0" type="sphere" pos="0 .08 -.38" size=".02"/>
        <geom name="right_foot_0" type="sphere" pos="0 -.08 -.38" size=".02"/>
        <body pos="0 .2 .1"><joint name="left_shoulder_roll_joint" range="-1 1"/><geom type="sphere" size=".02"/></body>
        <body pos="0 -.2 .1"><joint name="right_shoulder_roll_joint" range="-1 1"/><geom type="sphere" size=".02"/></body>
      </body></worldbody></mujoco>''')


def fixture_poses(model,frames=150):
    q=np.zeros((frames,model.nq)); q[:,2]=.4; q[:,3]=1.
    return q


def evidence(frames):
    return dict(root_target_error=np.zeros(frames),foot_target_error=np.zeros(frames),
                sphere_floor=np.zeros(frames),sphere_self_distance=np.ones(frames))


class QualityTest(unittest.TestCase):
    def setUp(self): self.model=fixture_model(); self.q=fixture_poses(self.model)

    def test_default_filter_preserves_calibration_buffer_and_three_second_spans(self):
        result=classify_frames(self.model,self.q,**evidence(len(self.q)))
        self.assertFalse(result["accepted"][:32].any())
        self.assertTrue(result["accepted"][32:].all())
        self.assertEqual(accepted_span_slices(result["accepted"]),[(32,150)])
        self.assertTrue(result["summary"]["verified"])
        json.dumps(result["summary"],allow_nan=False)

    def test_missing_source_evidence_is_unverified_and_cannot_pass(self):
        result=classify_frames(self.model,self.q)
        self.assertFalse(result["accepted"].any())
        self.assertEqual(result["verification"]["source_tracking"],"unverified")
        self.assertNotIn("root_target_error",result["metrics"])
        self.assertFalse(result["summary"]["passed"])

    def test_nonfinite_and_nonunit_poses_rejected_without_mutating_input(self):
        self.q[60,0]=np.nan; self.q[90,3]=2.
        original=self.q.copy()
        result=classify_frames(self.model,self.q,**evidence(len(self.q)))
        self.assertTrue(result["reasons"]["nonfinite_pose"][60])
        self.assertTrue(result["reasons"]["quaternion_norm"][90])
        self.assertFalse(result["accepted"][58:63].any())
        self.assertFalse(result["summary"]["verified"])
        np.testing.assert_equal(self.q,original)

    def test_actual_floor_contact_and_stance_point_slip(self):
        self.q[:,0]=np.arange(len(self.q))*.03
        metrics=physical_metrics(self.model,self.q)
        self.assertGreater(metrics["stance_slip"][10],.89)
        self.q[:,2]=.39
        metrics=physical_metrics(self.model,self.q)
        self.assertGreater(metrics["physical_floor_penetration"].max(),.009)

    def test_frame_count_and_empty_inputs_fail_closed(self):
        with self.assertRaises(ValueError): classify_frames(self.model,self.q[:0])
        with self.assertRaises(ValueError): accepted_span_slices(np.array([],dtype=bool))
        with self.assertRaises(ValueError): classify_frames(self.model,self.q,root_target_error=np.zeros(2))
        self.assertIsNone(percentiles([])["max"])

    def test_profiles_are_complete_versioned_and_finite(self):
        self.assertEqual(profile_hash(),profile_hash(dict(DEFAULT_PROFILE)))
        for key,value in (("stance_slip_m_s",float("nan")),("minimum_span_seconds",2.),
                          ("edge_buffer_frames",1.5),("schema_version",2)):
            cfg=dict(DEFAULT_PROFILE); cfg[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): load_profile(cfg)
        cfg=dict(DEFAULT_PROFILE); cfg["unknown_threshold"]=1
        with self.assertRaises(ValueError): load_profile(cfg)

    def test_short_valid_islands_do_not_join_across_rejection(self):
        valid=np.ones(200,dtype=bool);valid[80:100]=False
        self.assertEqual(accepted_span_slices(valid),[(100,200)])

    def test_cli_output_conflicts_never_modify_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/"robot.xml";mujoco.mj_saveLastXML(str(model),self.model)
            motion=root/"motion.npz";np.savez(motion,q=self.q,fps=30.,joint_names=np.array([self.model.joint(i).name for i in range(1,self.model.njnt)]))
            manifest=root/"manifest.json";manifest.write_text(json.dumps({"seal":{"complete":True}}))
            original={p:p.read_bytes() for p in (model,motion,manifest)}
            base=["--motion",str(motion),"--model",str(model)]
            tests=[["--out",str(motion)], ["--frame-report",str(motion)], ["--out",str(model)],
                   ["--out",str(root/"new_report.json")], ["--dataset",str(root),"--out",str(root/"new.json")],
                   ["--out",str(root/"same.npz"),"--frame-report",str(root/"same.npz")]]
            for extra in tests:
                with self.subTest(extra=extra),self.assertRaises((ValueError,FileExistsError)): main(base+extra)
            for path,content in original.items(): self.assertEqual(path.read_bytes(),content)
            self.assertFalse((root/"new_report.json").exists())

    def test_physical_only_cli_returns_unverified_status_and_bad_csv_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/"robot.xml";mujoco.mj_saveLastXML(str(model),self.model)
            motion=root/"motion.npz";np.savez(motion,q=self.q,fps=30.,joint_names=np.array([self.model.joint(i).name for i in range(1,self.model.njnt)]))
            output=io.StringIO()
            with redirect_stdout(output): code=main(["--motion",str(motion),"--model",str(model),"--no-walking"])
            self.assertEqual(code,2)
            report=json.loads(output.getvalue());self.assertFalse(report["passed"])
            self.assertEqual(report["verification"]["source_tracking"],"unverified")
            bad=root/"bad.csv";bad.write_text("1,2\n3,4\n")
            with self.assertRaisesRegex(ValueError,"CSV must have"):
                main(["--motion",str(bad),"--model",str(model),"--fps","30"])


if __name__=="__main__": unittest.main()
