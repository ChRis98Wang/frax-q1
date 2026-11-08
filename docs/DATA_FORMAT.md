# Motion and dataset format

## Robot coordinates

There are 22 actuated joints and a free root: 29 position coordinates, 28
generalized velocities. Positions are metres, joint angles radians, and sampling
frequency hertz. The internal world is Z-up. Root quaternions must be finite and
normalized. The ordered joint names are explicit in every NPZ and manifest:

1. Left leg: hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll.
2. Right leg: hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll.
3. Waist roll, waist yaw.
4. Left arm: shoulder pitch, shoulder roll, shoulder yaw, elbow.
5. Right arm: shoulder pitch, shoulder roll, shoulder yaw, elbow.

The two wrist links are fixed. A different humanoid's joint columns cannot be
silently reused. The exact identifiers are exposed as `q1_frax.model.JOINT_NAMES`.

## Raw NPZ

Required keys are `q`, `fps`, and `joint_names`. `q` has shape `[frames,29]` and
uses `[x,y,z,qw,qx,qy,qz,joints...]`. New adapter output preserves float64 data;
the manifest records `npz_dtype`. Loading uses `allow_pickle=False`.
Bare FRAX NPZ can be inspected, but has unverified provenance unless matched
against a trusted checksum or declared dataset manifest. It is always raw data.

## CSV

CSV is a headerless numeric matrix with shape `[frames,29]`, ordered
`[x,y,z,qx,qy,qz,qw,joints...]`. Joint order, frame rate and quaternion convention
come from the manifest, not guesses about the filename. New output uses
17 significant digits for lossless float64 round trips. The NPZ, raw CSV,
exported spans and quality checks therefore represent the same trajectory.

## Dataset tree

```text
dataset/
  manifest.json             generation settings, source hashes, QA, seal
  npz/<source>.npz          raw WXYZ motions
  csv/<source>.csv          raw XYZW motions
  quality/<source>.npz      all-frame metrics, accepted mask, reject_* masks
  walk_train/manifest.json  authoritative accepted-span index
  walk_train/<span>.csv
  walk_eval/manifest.json   created by source-level split/sealing
  walk_eval/<span>.csv
```

Quality masks have one entry per original frame. Spans are half-open slices
`[start,start+frames)`, with no interpolation, skipped rejected frames or stitching.
Only the authoritative split manifest determines which files are dataset members.
Unreferenced stale files cannot silently join training.

## Integrity and portability

Root manifests record upstream commits, BVH input hashes, retarget parameters,
adapter/script hashes, model/mapping/sphere hashes, profile hash and raw artifact
hashes. `model_dependencies` records every referenced external mesh/texture by
relative path and content SHA256; verification checks the complete mapping, so
an unchanged XML file cannot hide changed mesh bytes. Unsupported external model
includes or plugins are rejected rather than silently omitted from this audit.
Seals additionally protect split manifests and all exported spans.
Manifest paths are constrained to the dataset and source clips do not cross splits.

Generated MJCF and URDF use relative mesh paths within the asset-cache layout,
so equivalent pinned assets in different cache roots produce identical model
bytes. Original BVHs can be relocated with the verification CLI's `--source-dir`;
their contents must still match recorded hashes. Replacing assets, motion bytes,
quality masks or profiles requires a new dataset, not an edited old seal.
