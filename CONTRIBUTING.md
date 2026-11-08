# Contributing

Keep changes reproducible. Include the command and configuration used, upstream
revisions, model/motion hashes, frame rate and frame numbers for reported defects.
Attach data only when you have permission to redistribute it; small synthetic
reproductions are preferred for regression tests.

Run the unit tests, source release audit and an appropriate integration test.
Changes to joint ordering, quaternion conventions, collision filtering, quality
thresholds or file schemas require a regression that fails for the old behavior.
Do not replace sealed manifests or reuse an output directory for a changed run.

Report rejected frames as failures, including collision link pairs and depths.
Keep raw archives separate from accepted spans. Do not describe a kinematic
reference as a dynamically feasible motion or a learned controller.

Python changes should avoid importing JAX or a GPU runtime in lightweight CLI
help, motion inspection and playback. Keep optional dependencies optional and
avoid machine-specific absolute paths. Do not commit datasets, meshes, local
videos, credentials or generated dependency environments.
