# Upstream sources and asset licensing

The adapter code is MIT licensed. This license does not relicense downloaded
robot models, human motion data, or their derived outputs. Upstream datasets,
meshes, model archives and motion arrays are downloaded separately. The README
includes two explicitly listed rendered LAFAN demonstration files; they are not
licensed as MIT adapter code.

| Component | Pinned revision | License evidence | Role |
|---|---|---|---|
| [FRAX](https://github.com/StanfordASL/frax) | `f5e87ab59a0b82ba43095e882c1b88c9b0dacf07` | [MIT](https://github.com/StanfordASL/frax/blob/f5e87ab59a0b82ba43095e882c1b88c9b0dacf07/LICENSE), Copyright 2026 Daniel Morton | IK solver and upstream LAFAN loading convention |
| [Primebot GMR](https://github.com/PrimebotTech/GMR) | `a7d0682f1bdd10bdb0b11910cf786d86e1423522` | [MIT](https://github.com/PrimebotTech/GMR/blob/a7d0682f1bdd10bdb0b11910cf786d86e1423522/LICENSE), Copyright 2025 Yanjieze | Q1 correspondence and rotation-offset mapping |
| [Primebot Q1 model](https://github.com/PrimebotTech/primebot-q1-urdf) | `9657817ba375f41b0e5d2b62d8d6b72b757cdf4f` | [package.xml](https://github.com/PrimebotTech/primebot-q1-urdf/blob/9657817ba375f41b0e5d2b62d8d6b72b757cdf4f/package.xml) declares BSD; a full license text was not found at this revision | Locally downloaded geometry and physical limits |
| [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) | `94084601bacdf9cc3764b5c73daaeccae6035fac` | [CC BY-NC-ND 4.0](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/94084601bacdf9cc3764b5c73daaeccae6035fac/license.txt) | Optional local source motion |
| [SuperMimic](https://github.com/zita-ch/supermimic_vibed) | Origin of this adapter's local development | MIT, Copyright 2026 ETH Zurich, Chong Zhang | Original adaptation workspace; training code is not a runtime dependency |

Setup downloads the selected sources into a separate local data directory and
retains their upstream notices. LAFAN download is explicit. Its license includes
noncommercial and no-derivatives conditions; this project does not grant rights
to publish retargeted LAFAN motions, rendered derivatives, or training datasets.
Publish only data for which you have the necessary rights. Confirm model asset
redistribution terms with the model owner before bundling those assets.

The complete upstream MIT notices are retained in `licenses/`. Any code adapted
from these projects remains subject to the corresponding notice. Optional
Python dependencies retain their own distribution licenses. This project is an
independent Q1 adapter and does not imply endorsement by the upstream authors.

## README demonstration

`docs/assets/lafan_q1_dynamic.mp4` and its GIF preview show selected LAFAN1
motions retargeted to Q1. Motion attribution: Ubisoft La Forge / LAFAN1;
robot appearance: Primebot Q1. Exact source commits, clip names, frame ranges,
source/output hashes and the generating script are recorded in
[`docs/assets/lafan_demo.json`](docs/assets/lafan_demo.json).
The LAFAN1 upstream license is CC BY-NC-ND 4.0; the code's MIT license does not
grant additional rights to these motion-derived media or waive the upstream
noncommercial/no-derivatives conditions. No source BVH or retargeted motion
array is bundled with the demonstration.
