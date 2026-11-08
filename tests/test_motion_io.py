"""Provenance failures must never silently become accepted reference data."""
from pathlib import Path
import json
import hashlib
import shutil
import tempfile
import unittest

import numpy as np

from q1_frax.model import JOINT_NAMES
from q1_frax.motion_io import guard_output_paths, load_motion, sha256


class MotionIOTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root/"physical.xml"
        self.model.write_text('<mujoco model="hash fixture"/>')
        self.data = self.root/"dataset"
        self.data.mkdir()
        self.split = self.data/"walk_train"
        self.split.mkdir()
        self.q = np.zeros((4, 29)); self.q[:, 2] = .45; self.q[:, 3] = 1.
        self.csv = self.split/"walk_000.csv"
        raw = np.c_[self.q[:, :3], self.q[:, 4:7], self.q[:, 3], self.q[:, 7:]]
        np.savetxt(self.csv, raw, delimiter=",")
        self.record = {"file": self.csv.name, "frames": 4, "start": 0,
                       "source_clip": "walk", "split": "train", "sha256": sha256(self.csv)}
        profile = {"schema_version": 1, "name": "fixture"}
        self.profile_hash = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.source = {"schema_version": 1, "fps": 30., "joint_names": JOINT_NAMES,
                       "csv_quaternion": "xyzw", "npz_quaternion": "wxyz",
                       "model_dependencies": {"schema_version": 1, "files": {}},
                       "model_sha256": sha256(self.model), "seal": {"schema_version": 1, "complete": True,
                           "model_sha256": sha256(self.model), "quality_profile_sha256": self.profile_hash},
                       "validation": {"complete": True, "quality_profile": profile, "quality_profile_sha256": self.profile_hash},
                       "clips": [{"name": "walk", "frames": 4, "training_spans": [dict(self.record)]}]}
        self.manifest = {"schema_version": 1, "split": "train", "fps": 30., "joint_names": JOINT_NAMES, "root_quaternion": "xyzw",
                         "source_manifest": "../manifest.json", "model_sha256": sha256(self.model),
                         "quality_profile_sha256": self.profile_hash,
                         "clips": [dict(self.record)]}
        self.write_manifests()

    def tearDown(self):
        self.temp.cleanup()

    def add_model_mesh(self):
        mesh = self.root/"assets"/"part.stl"
        mesh.parent.mkdir()
        mesh.write_bytes(b"synthetic asset bytes; parsed only for fingerprinting")
        self.model.write_text('<mujoco><compiler meshdir="assets"/><asset><mesh name="part" file="part.stl"/></asset></mujoco>')
        digest = sha256(self.model)
        self.source["model_sha256"] = self.source["seal"]["model_sha256"] = self.manifest["model_sha256"] = digest
        self.source["model_dependencies"] = {"schema_version": 1, "files": {
            "assets/part.stl": {"sha256": sha256(mesh), "size_bytes": mesh.stat().st_size}}}
        self.write_manifests()
        return mesh

    def add_raw_archive(self):
        folder = self.data/"npz"; folder.mkdir()
        path = folder/"walk.npz"
        np.savez(path, q=self.q, fps=30., joint_names=JOINT_NAMES)
        self.source["clips"][0]["files"] = {"npz": {"file": "npz/walk.npz", "sha256": sha256(path)}}
        self.write_manifests()
        return path

    def test_external_mesh_tamper_rejected_for_sealed_and_versioned_raw(self):
        mesh = self.add_model_mesh()
        raw = self.add_raw_archive()
        for path in (self.csv, raw):
            motion = load_motion(path, model_path=self.model)
            self.assertTrue(motion.dependency_identity_verified)
            self.assertEqual(motion.provenance()["model_dependencies"], self.source["model_dependencies"])
        xml_hash = sha256(self.model)
        mesh.write_bytes(b"modified external mesh, same unchanged XML")
        self.assertEqual(sha256(self.model), xml_hash)
        for path in (self.csv, raw):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "(?i)dependenc"):
                load_motion(path, model_path=self.model)

    def test_versioned_dependencies_mapping_must_be_complete(self):
        self.add_model_mesh()
        raw = self.add_raw_archive()
        correct = dict(self.source["model_dependencies"])
        extra = {"schema_version": 1, "files": {**correct["files"], "undeclared.stl": {"sha256": "0"*64, "size_bytes": 0}}}
        for expected in (None, {"schema_version": 1, "files": {}}, extra):
            if expected is None:
                self.source.pop("model_dependencies", None)
            else:
                self.source["model_dependencies"] = expected
            self.write_manifests()
            for path in (self.csv, raw):
                with self.subTest(expected=expected, path=path), self.assertRaisesRegex(ValueError, "(?i)dependenc"):
                    load_motion(path, model_path=self.model)

    def test_legacy_sealed_identity_is_explicitly_unverified(self):
        mesh = self.add_model_mesh()
        self.source.pop("schema_version")
        self.source.pop("model_dependencies")
        self.write_manifests()
        mesh.write_bytes(b"legacy cannot attest this asset")
        motion = load_motion(self.csv, model_path=self.model)
        self.assertEqual(motion.kind, "sealed")
        self.assertFalse(motion.dependency_identity_verified)
        self.assertIn("LEGACY SEALED", motion.status(0)[0])
        self.assertIn("ASSET IDENTITY UNVERIFIED", motion.status(0)[0])
        self.assertFalse(motion.provenance()["dependency_identity_verified"])
        self.assertTrue(any("dependency identity unverified" in note for note in motion.notes))

    def test_render_rechecks_mesh_after_loading_and_protects_dependencies(self):
        from q1_frax.render import render_motion
        mesh = self.add_model_mesh()
        motion = load_motion(self.csv, model_path=self.model)
        output = self.root/"artifacts"/"image.png"
        with self.assertRaisesRegex(ValueError, "overwrite an input"):
            render_motion(motion, self.model, output, collision_report=mesh)
        self.assertFalse(output.parent.exists())
        mesh.write_bytes(b"edited between load and rendering")
        with self.assertRaisesRegex(ValueError, "(?i)dependenc"):
            render_motion(motion, self.model, output)
        self.assertFalse(output.exists())

    def write_manifests(self):
        (self.split/"manifest.json").write_text(json.dumps(self.manifest))
        self.source["seal"]["split_manifest_sha256"] = {"train": sha256(self.split/"manifest.json")}
        payload = {key: value for key, value in self.source.items() if key != "seal"}
        self.source["seal"]["manifest_payload_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        (self.data/"manifest.json").write_text(json.dumps(self.source))

    def test_sealed_manifest_integrity_profile_and_source_clip(self):
        self.manifest["unexpected"] = True
        (self.split/"manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "Split manifest SHA256"):
            load_motion(self.csv, model_path=self.model)
        self.write_manifests()
        self.source["unexpected"] = True
        (self.data/"manifest.json").write_text(json.dumps(self.source))
        with self.assertRaisesRegex(ValueError, "payload SHA256"):
            load_motion(self.csv, model_path=self.model)
        self.write_manifests()
        self.manifest["quality_profile_sha256"] = "0"*64
        self.write_manifests()
        with self.assertRaisesRegex(ValueError, "Quality profile"):
            load_motion(self.csv, model_path=self.model)
        self.manifest["quality_profile_sha256"] = self.profile_hash
        self.manifest["clips"][0]["source_clip"] = "other"
        self.write_manifests()
        with self.assertRaisesRegex(ValueError, "source clip"):
            load_motion(self.csv, model_path=self.model)

    def test_output_guards_protect_dataset_and_preflight_conflicts(self):
        motion = load_motion(self.csv, model_path=self.model)
        before = {path: sha256(path) for path in self.data.rglob("*") if path.is_file()}
        for target in (self.data/"manifest.json", self.split/"manifest.json",
                       self.data/"quality"/"walk.npz", self.data/"new_image.png"):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "read-only reference dataset"):
                    guard_output_paths(motion, self.model, target)
        with self.assertRaisesRegex(ValueError, "overwrite an input"):
            guard_output_paths(motion, self.model, self.model)
        with self.assertRaisesRegex(ValueError, "distinct"):
            guard_output_paths(motion, self.model, self.root/"report.json", self.root/"report.json")
        guard_output_paths(motion, self.model, self.root/"artifacts"/"image.png")
        self.assertEqual(before, {path: sha256(path) for path in self.data.rglob("*") if path.is_file()})

    def test_render_checks_all_outputs_before_creating_any_files(self):
        from q1_frax.render import render_motion
        motion = load_motion(self.csv, model_path=self.model)
        target = self.root/"new_artifacts"/"image.png"
        for report in (self.data/"manifest.json", target, target.with_name(target.name+".render.json")):
            with self.assertRaises(ValueError):
                render_motion(motion, self.model, target, collision_report=report)
            self.assertFalse(target.parent.exists())

    def test_output_guards_also_protect_other_sealed_datasets(self):
        motion = load_motion(self.csv, model_path=self.model)
        other = self.root/"other_sealed_dataset"; other.mkdir()
        (other/"manifest.json").write_text(json.dumps({"seal": {"complete": True}}))
        with self.assertRaisesRegex(ValueError, "any sealed reference dataset"):
            guard_output_paths(motion, self.model, other/"nested"/"report.json")
        alias = self.root/"alias"; alias.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "any sealed reference dataset"):
            guard_output_paths(motion, self.model, alias/"report.json")
        link = self.root/"report.json"; link.symlink_to(other/"report.json")
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            guard_output_paths(motion, self.model, link)

    def test_portable_sealed_csv_is_verified_and_read_only(self):
        moved = self.root/"relocated"
        shutil.copytree(self.data, moved)
        motion = load_motion(moved/"walk_train"/self.csv.name, model_path=self.model)
        np.testing.assert_array_equal(motion.qpos, self.q)
        self.assertEqual(motion.status(0)[0], "SEALED / ACCEPTED REFERENCE")
        self.assertTrue(motion.hash_verified)
        self.assertFalse(motion.qpos.flags.writeable)
        self.assertEqual(motion.fps, 30.)

    def test_tampered_motion_and_model_are_rejected(self):
        self.csv.write_text(self.csv.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_motion(self.csv, model_path=self.model)
        self.record["sha256"] = sha256(self.csv)
        self.manifest["clips"] = [dict(self.record)]
        self.source["clips"][0]["training_spans"] = [dict(self.record)]
        self.write_manifests()
        self.model.write_text('<mujoco model="changed"/>')
        with self.assertRaisesRegex(ValueError, "model SHA256"):
            load_motion(self.csv, model_path=self.model)

    def test_false_split_acceptance_without_sealed_source_is_rejected(self):
        self.source["seal"]["complete"] = False
        self.write_manifests()
        with self.assertRaisesRegex(ValueError, "not completely"):
            load_motion(self.csv, model_path=self.model)
        self.source["seal"]["complete"] = True
        self.source["clips"][0]["training_spans"][0]["start"] = 1
        self.write_manifests()
        with self.assertRaisesRegex(ValueError, "sealed source record"):
            load_motion(self.csv, model_path=self.model)

    def test_joint_order_quaternion_and_fps_are_strict(self):
        for key, value in (("joint_names", JOINT_NAMES[::-1]), ("root_quaternion", "wxyz"), ("fps", float("nan"))):
            with self.subTest(field=key):
                old = self.manifest[key]
                self.manifest[key] = value; self.write_manifests()
                with self.assertRaises(ValueError):
                    load_motion(self.csv, model_path=self.model)
                self.manifest[key] = old

    def test_standard_frax_npz_is_raw_even_when_its_hash_matches(self):
        path = self.root/"bare.npz"
        np.savez(path, q=self.q, fps=60., joint_names=np.array(JOINT_NAMES))
        raw = load_motion(path, model_path=self.model)
        self.assertEqual(raw.kind, "raw")
        self.assertFalse(raw.hash_verified)
        self.assertFalse(raw.dependency_identity_verified)
        self.assertEqual(raw.status(0)[0], "RAW / NOT ACCEPTED")
        trusted = load_motion(path, model_path=self.model, expected_sha256=sha256(path))
        self.assertTrue(trusted.hash_verified)
        self.assertEqual(trusted.kind, "raw")
        self.assertFalse(trusted.dependency_identity_verified)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_motion(path, model_path=self.model, expected_sha256="0"*64)

    def test_nonfinite_zero_quaternion_and_missing_joint_order_rejected(self):
        path = self.root/"bad.npz"
        for bad in ("nan", "quaternion", "metadata"):
            with self.subTest(kind=bad):
                q = self.q.copy()
                if bad == "nan": q[1, 8] = np.nan
                if bad == "quaternion": q[1, 3:7] = 0
                fields = dict(q=q, fps=30.)
                if bad != "metadata": fields["joint_names"] = JOINT_NAMES
                np.savez(path, **fields)
                with self.assertRaises(ValueError):
                    load_motion(path, model_path=self.model)

    def test_raw_manifest_hash_is_enforced_and_rejection_never_accepts(self):
        folder = self.data/"npz"; folder.mkdir()
        quality = self.data/"quality"; quality.mkdir()
        path = folder/"walk.npz"
        np.savez(path, q=self.q, fps=30., joint_names=JOINT_NAMES)
        np.savez(quality/"walk.npz", accepted=np.ones(4, dtype=bool), reject_physical_self_collision=[False, True, False, False])
        self.source["clips"][0]["files"] = {
            "npz": {"file": "npz/walk.npz", "sha256": sha256(path)},
            "quality": {"file": "quality/walk.npz", "sha256": sha256(quality/"walk.npz")}}
        self.write_manifests()
        motion = load_motion(path, model_path=self.model)
        self.assertEqual(motion.status(1), ("RAW REJECTED FRAME", ["physical_self_collision"]))
        self.assertEqual(motion.status(0)[0], "RAW / NOT ACCEPTED")
        np.savez(path, q=self.q, fps=20., joint_names=JOINT_NAMES)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_motion(path, model_path=self.model)


if __name__ == "__main__":
    unittest.main()
