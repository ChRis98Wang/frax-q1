# Changelog

## 0.1.1

- Derive sphere-pair coverage from the complete enabled MuJoCo geom-pair set,
  including fixed-body ownership; store the policy and pairs in the hashed YAML.
- Add self-collision foot volumes from the official foot meshes while keeping
  sole-point floor contacts separate. Share floor-sphere eligibility between IK
  and independent source replay.
- Preserve legacy sphere-policy semantics during read-only validation.
- Add strict dynamic-motion review, interpolated-pose checks, optional visual
  convex-envelope diagnostics, and reproducible historical collision witnesses.
- Replace the synthetic README showcase with reviewed LAFAN motion excerpts.

## 0.1.0

- Independent installable Q1 FRAX adapter with six command-line entry points.
- Pinned upstream setup, contact-aware LAFAN targets and constrained batch IK.
- CPU retargeting, truthful CUDA QP capability checks and reproducible timing.
- Lazy rendering initialization so import and CLI help work without EGL.
- Portable model generation with relative mesh paths and stable model hashes.
- Float64 NPZ and lossless CSV export sharing the quality-check trajectory.
- Full-frame geometry/tracking QA, versioned profile, rejection evidence and
  source-separated immutable dataset sealing with independent replay.
- Interactive reference play, frame/speed controls, collision link labels,
  offscreen still/video rendering and clean native viewer shutdown.
- Optional retargeting dependencies; playback works without JAX, FRAX or Torch.
- English/Chinese documentation, tests, CPU CI and source-release audit.
