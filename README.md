# Q1 FRAX

Reproducible LAFAN1 motion retargeting for the **22-joint Primebot Q1**, using
[Stanford FRAX](https://github.com/StanfordASL/frax). Includes interactive motion
playback, actual MuJoCo collision inspection, and explicit frame-level quality
gates. This is an independent adapter, not an official Primebot or Stanford release.

The output is a **kinematic reference motion**, not a learned policy or a proof
that a physical robot can execute it. No SuperMimic, PyTorch, Isaac Lab or GPU is
required for CPU retargeting and reference playback.

## Video

[![Original Q1 playback and collision-inspection demo](docs/assets/original_q1_demo.gif)](docs/assets/original_q1_demo.mp4)

**[Watch the 24-second MP4](docs/assets/original_q1_demo.mp4)** — original synthetic
joint motion, slow replay, single-frame inspection and an injected collision
test. This demonstrates the viewer; it is not LAFAN retargeting quality evidence
or a trained walking policy. Frame scheduling is programmed, not a recording of
keyboard input. [Source, generator and media hashes](docs/assets/demo.json).

## What is included

- Q1 joint order, physical joint limits, body mapping and contact-aware foot targets.
- Batched FRAX IK with configurable floor and self-collision constraints.
- Real-time reference play with pause, frame inspection, speed control, link names
  and collision markers; offscreen images and video export.
- Physical geometry checks in addition to FRAX's approximate collision spheres.
- Per-frame rejection reasons, contiguous accepted spans, source-clip train/eval
  separation, checksums and immutable dataset sealing.
- Unit tests, a CPU CI workflow, pinned upstream revisions and source-only releases.

```mermaid
flowchart LR
    A[LAFAN BVH] --> B[Axis conversion and Q1 scale]
    B --> C[FRAX constrained IK]
    C --> D[Raw motion and provenance]
    D --> E[Every-frame physical and tracking QA]
    E --> F[Rejected frames with reasons]
    E --> G[Contiguous accepted spans]
    G --> H[Replay, source split, immutable seal]
    D --> I[Interactive play and collision inspection]
    H --> I
```

## Install

Linux and Python 3.12 are the primary tested environment. Python 3.11 or later is
declared supported; use the tested environment lock for reproducible benchmarks.
Install into a fresh virtual environment:

```bash
git clone https://github.com/ChRis98Wang/frax-q1.git
cd frax-q1
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[retarget,visualization,test]'
export PATH="$PWD/.venv/bin:$PATH"
export Q1_FRAX_DATA="$PWD/.local-data"
q1-frax-setup --help
q1-frax-retarget --help
q1-frax-play --help
```

For the tested Python 3.12 dependency versions, add `-c requirements-lock.txt`
to the pip install command. Playback alone only needs `.[visualization]`;
JAX and FRAX are optional until retargeting or source-target replay is requested.

**Retargeting currently runs on CPU.** We verified CUDA 13 and float64 JAX on
an RTX 5080, but the pinned `elastiqp==0.2.1` QP solver only registers its FFI
handler for CPU. A real FRAX CUDA solve fails; installing the `cuda13` extra
alone does not enable GPU retargeting. `--platform cuda` is a diagnostic option
and fails explicitly before creating output when the QP backend is unsupported.
`--devices` selects virtual **CPU** devices, not GPUs.
See the [backend evidence and timing protocol](docs/BENCHMARKING.md).

The data directory defaults to `~/.cache/q1-frax`; it is separate from installed
code. Setup retrieves pinned upstream assets. LAFAN is an explicit opt-in download:

```bash
q1-frax-setup --with-lafan
```

Review [upstream licenses](THIRD_PARTY_NOTICES.md) before downloading or sharing
data. The source distribution contains no robot meshes, LAFAN motions, generated
datasets, learned weights, or LAFAN-derived media. The viewer demo uses original
synthetic joint motion and identifies its source explicitly.

## Retarget, inspect, accept

Start with a short clip before processing the complete corpus:

```bash
q1-frax-retarget --pattern 'walk2_subject3.bvh' --max-frames 300 \
  --devices 1 --out outputs/walk-smoke
```

Use `--max-frames 1200` for a longer first walking check; a short clip may contain
only calibration or fail to provide a full three-second accepted span. Settings
are explicit in [retarget_default.json](configs/retarget_default.json).
The [collision review preset](configs/retarget_collision_review.json) requests
extra clearance; it is an experimental comparison, not a certified improvement.

Raw output and accepted data have different meanings. A completed solve is not a
quality pass. Inspect actual link collisions, review the report and rejection
reasons, and seal only verified contiguous spans. Use each command's `--help` for
the installed CLI schema:

```bash
q1-frax-play outputs/walk-smoke/npz/walk2_subject3.npz --show-link-labels
q1-frax-render outputs/walk-smoke/npz/walk2_subject3.npz artifacts/reference.mp4
q1-frax-play outputs/walk-smoke/npz/walk2_subject3.npz --check-only \
  --collision-report artifacts/collisions.json
# Process enough source clips to populate both splits before sealing.
q1-frax-retarget --pattern 'walk*.bvh' --devices 4 --out outputs/walking
q1-frax-seal --dataset outputs/walking --held-out walk1_subject2 walk2_subject4
q1-frax-seal --dataset outputs/walking --verify-only
```

Play controls: **P / Space** pause, **Left / Right** step one frame,
**+ / -** change speed, **R** restart, **C** collision geometry,
**L** link labels, **M** self-collision highlights.

See the [Python API](docs/API.md), [data format](docs/DATA_FORMAT.md),
[quality standards](docs/QUALITY.md), [collision semantics](docs/COLLISIONS.md),
the [benchmark protocol](docs/BENCHMARKING.md), and the
[Chinese quick start](docs/QUICKSTART.zh-CN.md). Playback never hides bad
frames by labeling a raw archive as accepted data. Kinematic playback assigns
poses and computes geometry; it does not step dynamics or run a policy.

## Quality evidence

The [matched backend test](benchmarks/gpu_cpu_comparison_20261108.json) uses
2,400 source frames on a Core Ultra 7 270K Plus, float64 and unchanged collision
settings. This is a short workload, not full-corpus throughput. The RTX 5080
passes the CUDA/JAX device test but cannot complete FRAX IK with the pinned
CPU-only QP backend, so **GPU retargeting FPS is unavailable**.

| Hardware / backend | First IK call, including JIT | Warm IK, 3-run median | Full CLI, including QA/export |
|---|---:|---:|---:|
| Core Ultra 7 270K Plus / CPU, 1 virtual device | 2.279 s / 1,053 fps | 0.885 s / 2,712 fps | 4.973 s / 483 fps |
| RTX 5080 / CUDA 13 | Unsupported QP backend | No valid FPS | Did not complete |

JAX device availability is not evidence that the complete IK solver runs on GPU.
The timings above precede the collision-coverage revision; see the recorded
adapter/configuration hashes. An [earlier CPU smoke run](benchmarks/cpu_smoke_20261108.json)
is retained separately, not mixed into the three-run warm measurement.

The original local Q1 adaptation processed **77 clips / 496,672 frames at 30 Hz**.
That batch used the CPU backend with eight virtual CPU devices; it was not a GPU run.
Its original log recorded **18.3 s for BVH loading** and **14.000 s for IK,
including 8.044 s compilation**. Loading plus IK was approximately 32.3 s.
The log-file creation-to-final-write interval was **72.862 s (approximately
73 s end to end)**, consistent with the recorded 39.936 s QA/export phase.
This is a filesystem-timestamp approximation, not a dedicated process timer;
14 s is only IK. These historical settings predate the collision-coverage fix.
[Historical timing and provenance](benchmarks/predecessor_summary.json).
That is a raw archive, not an all-pass benchmark. The historical walking profile
selected **130 spans / 37,765 frames** (20.98 minutes), split by entire source clip:

| Split | Spans | Source frames | Duration |
|---|---:|---:|---:|
| Train | 94 | 29,618 | 16.45 min |
| Held out | 36 | 8,147 | 4.53 min |

Only 10 of the 12 walking source clips contributed retained long spans. In the
full raw archive, 23 clips had at least one frame with physical self-penetration
over 2 mm. Those raw failure cases are not accepted training motions. These
numbers describe the predecessor dataset; they do **not** claim that a new
version, machine or parameter profile reproduces the same acceptance rate.

Reports distinguish missing evidence from passing evidence. In particular,
physical collision clearance does not establish source-pose tracking accuracy,
dynamic feasibility, torque feasibility, or real-hardware safety.

## Development

```bash
python -m pytest -q
python tools/check_release.py
```

Asset-dependent integration runs are separate from synthetic unit tests. Record
the exact command, environment, source commits, configuration and model hashes
when reporting results. Do not modify accepted manifests to make changed assets
appear compatible. See [contributing](CONTRIBUTING.md).

## Attribution

FRAX supplies the solver; Primebot's Q1 description supplies geometry and limits;
Primebot GMR supplies the LAFAN correspondence; Ubisoft supplies optional LAFAN1
motions. This adapter originated in a SuperMimic development workspace. Full
notices and data restrictions are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Adapter code: [MIT](LICENSE).
