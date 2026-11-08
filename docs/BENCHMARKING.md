# Reproducible CPU / GPU timing

Use the same source clips, frame counts, model, float64 precision, retarget
parameters, self-collision settings and quality profile. Verify the actual JAX
backend and device; GPU requests must not silently fall back to CPU.

```bash
python tools/benchmark.py \
  --cpu-python .venv-gpu/bin/python \
  --gpu-python .venv-gpu/bin/python \
  --data-dir "$Q1_FRAX_DATA" \
  --work-dir .local-data/benchmark-new \
  --output benchmarks/comparison-new.json \
  --warm-repeats 3
```

The CUDA interpreter needs a supported GPU JAX installation. Use the same
interpreter and dependency versions for both backends when comparing them.
Existing outputs are preserved; use a new work directory. Downloading and
installing dependencies happen before timing.

The default workload is `walk[12]_subject[23].bvh`, capped at 1,200 frames per
source. At the pinned LAFAN revision this selects `walk1_subject2` and
`walk2_subject3`: **2,400 frames, 80 seconds of source motion at 30 Hz**. The
report records actual selected clips, hashes and frame counts rather than
assuming that every combinatorial filename exists.

## Current GPU limitation

The tested RTX 5080, driver 580.178.04 and CUDA 13 successfully execute a
synchronized float64 JAX matrix operation. The actual FRAX workload fails with
`No FFI handler registered for elastiqp_solve on a platform CUDA`.
The pinned [elastiqp v0.2.1 registration](https://github.com/StanfordASL/elastiqp/blob/v0.2.1/python/elastiqp/jax.py#L61-L63)
only targets CPU; its [optional dependencies](https://github.com/StanfordASL/elastiqp/blob/v0.2.1/pyproject.toml#L13-L19)
do not provide a CUDA solver extra. Consequently GPU IK/CLI throughput is
unavailable, not zero, and CPU/GPU numerical equivalence cannot be claimed.
Installing JAX CUDA packages or successfully probing a GPU is insufficient.
The benchmark records this failure explicitly rather than substituting CPU work.

## Timing categories

- **First IK call:** `OfflineRetargeter`, including reported JIT compilation.
  Source loading and independent physical QA are outside this number.
- **Reported compilation:** FRAX's compile time. Subtracting this from the first
  call is not the same as repeatedly measuring a warmed process.
- **Warm IK:** a separate worker compiles/warms once, then reuses the same solver
  instance and shapes for three calls. Individual samples and their median are
  reported. Results must materialize as NumPy host arrays before the timer stops;
  asynchronous GPU dispatch alone is not completed work.
- **Full CLI wall time:** a fresh process including imports, source/asset checks,
  BVH loading, model preparation, IK, every-frame QA and exports. It excludes
  download/install, dataset sealing and visualization. Physical MuJoCo QA remains
  on CPU when IK runs on CUDA.

Backends run serially. The report compares output trajectories, accepted masks
and collision metrics and checks reproducibility across warm repeats when both backends complete. Different
floating-point outputs may have different hashes; quality differences remain
visible. Precision changes or disabled collisions are different benchmarks.

One local smoke run is not a statistical hardware ranking or a full-corpus
throughput result. Batch size, sequence length, virtual CPU device count,
compilation cache, precision and GPU kernel overhead affect the comparison.
Do not extrapolate this short workload into a guaranteed production rate.

The historical dataset in `predecessor_summary.json` used eight CPU virtual
devices. Its production history is separate from the matched CPU/GPU benchmark.

## Historical full-corpus run

The original CPU run used eight virtual devices for 77 clips / 496,672 frames.
Its log records BVH loading at 18.3 s and IK at 13.999900 s including 8.043726 s
JIT compilation (approximately 32.3 s loading + IK). The QA/export phase timestamps span 39.936098 s. The log-file creation time
(`2026-11-08T05:38:48.287300002Z`) to its last write
(`2026-11-08T05:40:01.148954047Z`) spans 72.861654 s: approximately **73 s**
for the original complete batch. This filesystem interval is not a dedicated
process wall timer; it excludes subsequent dataset sealing. Subtracting reported
compilation from IK gives 5.956174 s, not a separately measured warm run.
These historical parameters and collision coverage differ from the revised
adapter, so they are not a speed claim for the current version.

## Full-corpus 0.1.1 defaults

[The new complete run](../benchmarks/full_lafan_v3_20261108.json) retargeted all
77 clips / 496,672 frames with full pair coverage, self-only foot volumes,
512 constraint rows and 3 mm extra margin. On the Core Ultra 7 270K Plus with
eight virtual CPU devices, input loading took 18.5 s, IK took 59.239 s including
8.135 s compilation, and the fresh complete CLI took 148.064 s. The CLI timer
includes every-frame QA and export, but not download, installation, sealing or
rendering. Tests/rendering briefly shared the workstation, so this is a recorded
production-style batch duration, not an isolated hardware comparison.

This workload/configuration differs from the historical 73 s run. Native
self-penetration is zero at the original sampled poses, while floor and
conservative sphere failures remain in the raw archive and are enumerated.
Timing does not imply that every output frame passed all quality criteria.
