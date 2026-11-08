"""Asset inventories are portable, complete and checked before model loading."""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

from q1_frax.integrity import dependency_fingerprints, verify_model_dependencies


class ModelIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def model(self, xml, files=None, directory="cache"):
        root = self.root / directory
        root.mkdir(parents=True, exist_ok=True)
        for name, content in (files or {}).items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        path = root / "model.xml"
        path.write_text(xml)
        return path

    def test_portable_across_roots_and_sibling_symlink(self):
        xml = '<mujoco><compiler meshdir="../source/meshes"/><asset><mesh file="part.stl"/></asset></mujoco>'
        one = self.model(xml, {"../source/meshes/part.stl": b"mesh-bytes"})
        expected = dependency_fingerprints(one)
        self.assertEqual(set(expected["files"]), {"../source/meshes/part.stl"})
        target = self.root / "copy"
        shutil.copytree(self.root / "cache", target / "cache")
        (target / "source").symlink_to(self.root / "source", target_is_directory=True)
        self.assertEqual(expected, verify_model_dependencies(target / "cache/model.xml", expected))

    def test_compiler_directories_texture_faces_and_strip_path(self):
        xml = '''<mujoco><compiler assetdir="assets" meshdir="mesh" texturedir="tex" strippath="true"/>
        <asset><mesh file="discard/sub/a.stl"/><hfield file="terrain.bin"/>
        <skin file="body.skn"/><texture fileleft="faces/left.png" fileright="right.png"/>
        <texture builtin="checker"/></asset></mujoco>'''
        files = {"mesh/a.stl": b"a", "mesh/terrain.bin": b"h", "mesh/body.skn": b"s",
                 "tex/left.png": b"l", "tex/right.png": b"r"}
        model = self.model(xml, files)
        expected = dependency_fingerprints(model)
        self.assertEqual(set(expected["files"]), set(files))
        for name, content in files.items():
            self.assertEqual(expected["files"][name], {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)})
        inherited = self.model('<mujoco><compiler assetdir="assets"/><asset><mesh file="a.stl"/><texture file="b.png"/></asset></mujoco>',
                               {"assets/a.stl": b"a", "assets/b.png": b"b"}, "inherited")
        self.assertEqual(set(dependency_fingerprints(inherited)["files"]), {"assets/a.stl", "assets/b.png"})

    def test_reparse_detects_missing_extra_changed_and_deleted_files(self):
        model = self.model('<mujoco><asset><mesh file="a.stl"/></asset></mujoco>', {"a.stl": b"abc", "b.stl": b"def"})
        expected = dependency_fingerprints(model)
        omitted = deepcopy(expected); omitted["files"].clear()
        with self.assertRaisesRegex(ValueError, "inventory mismatch"): verify_model_dependencies(model, omitted)
        extra = deepcopy(expected); extra["files"]["b.stl"] = extra["files"]["a.stl"]
        with self.assertRaisesRegex(ValueError, "inventory mismatch"): verify_model_dependencies(model, extra)
        (model.parent / "a.stl").write_bytes(b"abd")  # same size still detected
        with self.assertRaisesRegex(ValueError, "SHA256/size mismatch"): verify_model_dependencies(model, expected)
        (model.parent / "a.stl").write_bytes(b"abc")
        model.write_text('<mujoco><asset><mesh file="a.stl"/><mesh file="b.stl"/></asset></mujoco>')
        with self.assertRaisesRegex(ValueError, "inventory mismatch"): verify_model_dependencies(model, expected)
        (model.parent / "a.stl").unlink()
        with self.assertRaisesRegex(ValueError, "Cannot read"): dependency_fingerprints(model)

    def test_empty_is_valid_only_for_no_dependencies(self):
        model = self.model('<mujoco><asset><texture builtin="checker"/></asset></mujoco>')
        self.assertEqual(verify_model_dependencies(model, {"schema_version": 1, "files": {}}), {"schema_version": 1, "files": {}})
        for value in (None, {}, {"schema_version": True, "files": {}}, {"schema_version": 1, "files": []}):
            with self.subTest(value=value), self.assertRaises(ValueError): verify_model_dependencies(model, value)

    def test_unsupported_references_and_nonportable_paths_fail_closed(self):
        cases = ['<include file="part.xml"/>', '<asset><model file="child.xml"/></asset>',
                 '<extension><plugin plugin="anything"/></extension>', '<flexcomp file="x.msh"/>',
                 '<asset><mesh file="https://example.org/a.stl"/></asset>',
                 '<compiler meshdir="/absolute"/>', '<asset><mesh file="C:\\a.stl"/></asset>',
                 '<compiler/><compiler/>']
        for i, fragment in enumerate(cases):
            model = self.model('<mujoco>' + fragment + '</mujoco>', directory=str(i))
            with self.subTest(fragment=fragment), self.assertRaises(ValueError): dependency_fingerprints(model)
        dtd = self.model('<!DOCTYPE mujoco [<!ENTITY ext "x">]><mujoco/>', directory="dtd")
        with self.assertRaisesRegex(ValueError, "DTD/entity"): dependency_fingerprints(dtd)

    def test_manifest_inventory_fields_are_strict(self):
        model = self.model('<mujoco><asset><mesh file="a.stl"/></asset></mujoco>', {"a.stl": b"abc"})
        expected = dependency_fingerprints(model)
        bad_values = [dict(sha256="Z" * 64, size_bytes=3), dict(sha256="0" * 64, size_bytes=True),
                      dict(sha256="0" * 64, size_bytes=-1), dict(sha256="0" * 64)]
        for entry in bad_values:
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                verify_model_dependencies(model, {"schema_version": 1, "files": {"a.stl": entry}})
        for name in ("./a.stl", "foo/../a.stl", "/a.stl", "a\\b.stl"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_model_dependencies(model, {"schema_version": 1, "files": {name: expected["files"]["a.stl"]}})


if __name__ == "__main__":
    unittest.main()
