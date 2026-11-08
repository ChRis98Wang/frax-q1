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
and collision metrics and checks reproducibility across warm repeats. Different
floating-point outputs may have different hashes; quality differences remain
visible. Precision changes or disabled collisions are different benchmarks.

One local smoke run is not a statistical hardware ranking or a full-corpus
throughput result. Batch size, sequence length, virtual CPU device count,
compilation cache, precision and GPU kernel overhead affect the comparison.
Do not extrapolate this short workload into a guaranteed production rate.

The historical dataset in `predecessor_summary.json` used eight CPU virtual
devices. Its production history is separate from the matched CPU/GPU benchmark.
