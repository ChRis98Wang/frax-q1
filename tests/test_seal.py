from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from q1_frax.quality import classify_frames,load_profile,profile_hash
from q1_frax.seal import atomic_json,seal_dataset,sha256,safe_path
from q1_frax.integrity import dependency_fingerprints
from test_quality import evidence,fixture_model,fixture_poses


class SealTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name); self.model=fixture_model()
        self.model_path=self.root/"robot.xml"
        mujoco.mj_saveLastXML(str(self.model_path),self.model)
        self.names=[self.model.joint(i).name for i in range(1,self.model.njnt)]
        for kind in ("csv","npz","quality","walk_train","sources"): (self.root/kind).mkdir()
        self.manifest={"schema_version":1,"npz_dtype":"float64","csv_float_format":"%.17g","robot":"q1","fps":30.,"joint_names":self.names,
                       "csv_quaternion":"xyzw","npz_quaternion":"wxyz","model_sha256":sha256(self.model_path),
                       "model_dependencies":dependency_fingerprints(self.model_path),
                       "validation":{"complete":True,"quality_profile":load_profile(),"quality_profile_sha256":profile_hash()},"clips":[]}
        for name in ("walk_train_source","walk_eval_source"):
            q=fixture_poses(self.model); csv=q.copy();csv[:,3:7]=q[:,[4,5,6,3]]
            source=self.root/"sources"/(name+".bvh");source.write_text("original source "+name)
            csv_path=self.root/"csv"/(name+".csv");np.savetxt(csv_path,csv,delimiter=",",fmt="%.8f")
            npz_path=self.root/"npz"/(name+".npz");np.savez(npz_path,q=q,fps=30.,joint_names=np.array(self.names))
            qa=classify_frames(self.model,q,**evidence(len(q)))
            quality_path=self.root/"quality"/(name+".npz")
            np.savez(quality_path,accepted=qa["accepted"],**qa["metrics"],**{"reject_"+k:v for k,v in qa["reasons"].items()})
            span_path=self.root/"walk_train"/(name+"_000.csv")
            np.savetxt(span_path,csv[32:],delimiter=",",fmt="%.8f")
            self.manifest["clips"].append({"name":name,"frames":len(q),"source_sha256":sha256(source),
                "csv_sha256":sha256(csv_path),"npz_sha256":sha256(npz_path),"quality_sha256":sha256(quality_path),
                "training_spans":[{"file":span_path.name,"start":32,"frames":len(q)-32,"sha256":sha256(span_path)}]})
        atomic_json(self.root/"manifest.json",self.manifest)

    def replayer(self,q,clip):
        path=self.root/"sources"/(clip["name"]+".bvh")
        if sha256(path)!=clip["source_sha256"]: raise ValueError("Source SHA256 mismatch")
        return evidence(len(q))

    def seal(self,**kwargs):
        return seal_dataset(self.root,model_path=self.model_path,held_out=["walk_eval_source"],replayer=self.replayer,**kwargs)

    def test_two_source_split_full_replay_and_immutable_evidence(self):
        before={str(p.relative_to(self.root)):sha256(p) for kind in ("csv","npz","quality") for p in (self.root/kind).iterdir()}
        result=self.seal()
        self.assertTrue(result["sealed"])
        self.assertEqual(result["split"]["train"]["source_clips"],["walk_train_source"])
        self.assertEqual(result["split"]["eval"]["source_clips"],["walk_eval_source"])
        self.assertTrue(self.seal(verify_only=True)["verified"])
        for name,digest in before.items(): self.assertEqual(sha256(self.root/name),digest)
        manifest=(self.root/"manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError,"already sealed"): self.seal()
        self.assertEqual((self.root/"manifest.json").read_bytes(),manifest)

    def test_integrity_only_never_claims_quality_replay_pass(self):
        self.seal()
        result=self.seal(verify_only=True,integrity_only=True)
        self.assertTrue(result["integrity_verified"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["quality_replay"],"unverified")

    def test_changed_source_and_replay_metrics_rejected(self):
        (self.root/"sources/walk_train_source.bvh").write_text("tampered")
        with self.assertRaisesRegex(ValueError,"Source SHA256"): self.seal()
        self.assertNotIn("seal",json.loads((self.root/"manifest.json").read_text()))

    def test_forged_accepted_mask_rejected_even_if_hash_is_updated(self):
        path=self.root/"quality/walk_train_source.npz"
        with np.load(path) as archive: qa={k:archive[k] for k in archive.files}
        qa["accepted"]=np.ones(150,dtype=bool);np.savez(path,**qa)
        self.manifest["clips"][0]["quality_sha256"]=sha256(path)
        atomic_json(self.root/"manifest.json",self.manifest)
        with self.assertRaisesRegex(ValueError,"mask differs from full replay"): self.seal()

    def test_sealed_span_and_manifest_tamper_rejected(self):
        self.seal()
        path=self.root/"walk_eval/walk_eval_source_000.csv"
        path.write_text(path.read_text()+"\n")
        with self.assertRaisesRegex(ValueError,"SHA256 mismatch"): self.seal(verify_only=True)

    def test_empty_and_single_split_outputs_fail_closed(self):
        original=deepcopy(self.manifest)
        self.manifest["clips"]=[];atomic_json(self.root/"manifest.json",self.manifest)
        with self.assertRaisesRegex(ValueError,"Empty archives"): self.seal()
        original["clips"]=original["clips"][:1];atomic_json(self.root/"manifest.json",original)
        with self.assertRaisesRegex(ValueError,"Both train and held-out"): self.seal()

    def test_sealed_manifest_metadata_tamper_is_detected(self):
        self.seal()
        path=self.root/"manifest.json";manifest=json.loads(path.read_text());manifest["fps"]=31.
        atomic_json(path,manifest)
        with self.assertRaisesRegex(ValueError,"payload hash mismatch"): self.seal(verify_only=True)

    def test_fake_physical_measurements_do_not_pass_with_rehashed_file(self):
        path=self.root/"quality/walk_train_source.npz"
        with np.load(path) as archive: qa={k:archive[k] for k in archive.files}
        qa["root_speed"]=np.ones(150);np.savez(path,**qa)
        self.manifest["clips"][0]["quality_sha256"]=sha256(path)
        atomic_json(self.root/"manifest.json",self.manifest)
        with self.assertRaisesRegex(ValueError,"Stored quality differs from replay"): self.seal()

    def test_unsafe_paths_and_symlinks_rejected(self):
        for path in ("../outside","/etc/passwd","a/../../outside","a\\b"):
            with self.subTest(path=path),self.assertRaises(ValueError): safe_path(self.root,path)
        (self.root/"alias").symlink_to(self.root/"csv",target_is_directory=True)
        with self.assertRaisesRegex(ValueError,"symlink"): safe_path(self.root,"alias/a.csv")

    def test_dependency_inventory_required_and_exact_even_integrity_only(self):
        original=deepcopy(self.manifest)
        del self.manifest["model_dependencies"]
        atomic_json(self.root/"manifest.json",self.manifest)
        with self.assertRaisesRegex(ValueError,"model_dependencies"): self.seal()
        atomic_json(self.root/"manifest.json",original)
        self.seal()
        # Main XML hash alone must not bless an incomplete/forged asset inventory.
        path=self.root/"manifest.json"; sealed=json.loads(path.read_text())
        sealed["model_dependencies"]["files"]["unused.stl"]={"sha256":"0"*64,"size_bytes":0}
        import hashlib
        from q1_frax.seal import canonical
        sealed["seal"]["manifest_payload_sha256"]=hashlib.sha256(canonical({k:v for k,v in sealed.items() if k!="seal"})).hexdigest()
        atomic_json(path,sealed)
        with self.assertRaisesRegex(ValueError,"inventory mismatch"): self.seal(verify_only=True,integrity_only=True)

    def test_sealed_ancestor_cannot_be_modified_by_new_seal(self):
        self.seal()
        before=(self.root/"manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError,"sealed ancestor"):
            seal_dataset(self.root/"nested",model_path=self.model_path,replayer=self.replayer)
        self.assertEqual((self.root/"manifest.json").read_bytes(),before)
        self.assertFalse((self.root/"nested").exists())

    def test_output_symlink_is_rejected_before_any_span_move(self):
        (self.root/"walk_eval").mkdir()
        (self.root/"walk_eval/manifest.json").symlink_to(self.root/"manifest.json")
        before=(self.root/"manifest.json").read_bytes()
        span=self.root/"walk_train/walk_eval_source_000.csv"
        with self.assertRaisesRegex(ValueError,"symlink"): self.seal()
        self.assertTrue(span.is_file())
        self.assertEqual((self.root/"manifest.json").read_bytes(),before)


if __name__=="__main__": unittest.main()
