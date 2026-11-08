# Collision coverage audit

The historical 77-clip archive is a collection of raw outputs, including rejected
frames. It is not a certificate that all 77 motions are collision-free. Its
all-frame QA reports physical self-penetration in 25 clips, including penetration
greater than the historical 2 mm acceptance tolerance in 23 clips. The witnesses
below were rejected by that QA. The raw inputs have not been edited.

The generation bug is specific: the adapter originally selected a hand-written
list of link pairs for FRAX's sphere constraints. Some body pairs that MuJoCo
checks were absent from that list. Raising a collision penalty or increasing the
number of retained constraint rows does not create the missing pairs.

| Original clip | Frame (zero based) | Physical body pair | Signed separation |
| --- | ---: | --- | ---: |
| `walk3_subject4` | 5865 | right ankle/foot — left hip pitch | −13.759 mm |
| `fallAndGetUp2_subject3` | 688 | right ankle/foot — pelvis | −29.910 mm |
| `obstacles6_subject4` | 993 | left knee — left shoulder yaw | −26.853 mm |
| `obstacles6_subject4` | 993 | left ankle/foot — left wrist/hand | −2.786 mm |

These are direct MuJoCo contact distances from the archived CSV poses, not
inferences from the displayed mesh. Exact source, output, model, and historical
adapter hashes are in [the machine-readable witness report](../benchmarks/collision_pair_audit.json).
Source BVH files and motion arrays are not included in that report.

The existing FRAX constraints use a sphere approximation, a finite elastic
penalty, local velocity linearization, and a maximum number of nearest rows.
They reduce collisions; they are not a proof that every integrated pose is
collision-free. The physical model also intentionally excludes adjacent and
next-adjacent bodies whose simplified joint geometry overlaps. Disabled visual
meshes are not physical collision shapes, and the feet use five small sole
spheres. Physical collision checks therefore do not certify that every visible
surface is disjoint. MuJoCo documents these [collision selection and exclusion
rules](https://mujoco.readthedocs.io/en/stable/computation.html#collision-detection).

A second issue was independently observed in the historical jump candidate
`jumps1_subject1[1605:1845]`: all configured physical collisions were clear, but
the left/right **visible foot mesh convex envelopes** overlapped by as much as
14.265 mm at source frame 1779. The negative distance appeared in 202 of 957
densely sampled poses. Five small sole points do not represent the solid foot
volume for foot-to-foot collision. Completing the pair list alone does not fix
that geometric coverage gap. The corrective model must add an appropriate
foot self-collision volume while retaining the intended sole-floor contacts.

This convex-envelope result is a diagnostic rather than an exact triangle-mesh
intersection measurement: concave geometry can produce conservative overlaps.
The candidate was withheld pending visual inspection and fresh validation; the
physical-only pass was not advertised as a visual collision-free result.

## Strict dynamic-motion review

`configs/quality_strict_dynamic_v1.json` requires **zero measured self/floor
penetration** and zero negative sphere distance. It retains the 12 rad/s joint
speed, 150 rad/s² acceleration, 0.20 m/s stance-point slip, and source tracking
limits of the historical profile. Joint-limit comparison explicitly permits
1e-6 rad numerical error. It does not add a collision tolerance.

Use this profile with `--no-walking` in `q1-frax-quality`. Jumping, running, and
dance must not be rejected merely for being airborne, having a lower root, or
using arm poses that are unsuitable for a walking reset. This does not relax
collision, continuity, or source-tracking checks. Calibration poses and crop
boundaries still need explicit review; non-walking mode does not silently trim
them. A profile name alone does not select an action mode.

For independent physical review of a retained interval:

```bash
python tools/audit_q1_collision.py \
  --motion "$ARCHIVE/npz/jumps1_subject1.npz" --model "$MODEL" \
  --start 1605 --frames 240 --substeps 4 \
  --visual-envelope \
  --profile configs/quality_strict_dynamic_v1.json \
  --out audit-jump.json
```

This historical interval is a candidate, not an endorsement of a later
regenerated motion. Always rerun the audit on the exact final output and record
its hash. The tool records the source interval, every original pose plus three
intermediate poses per interval, the smallest signed separation, the deepest
penetration, its body/geom pair, and continuity/slip statistics. It never edits,
smooths, or bridges rejected motion. Exit code 2 means at least one measured
gate failed. A cropped clip must be described as a crop.

`--visual-envelope` separately enables the diagnostic above using only an
in-memory model copy. It never changes the input XML, collision masks on disk,
or original poses. Any negative envelope distance makes the requested-checks
result fail and is reported separately from the original physical geometry.

The audit is a **dense sampled kinematic check**, not a continuous-time or
dynamic-feasibility guarantee. Its separate status fields intentionally leave
source tracking and visual-mesh intersections unverified. Full source-target
replay and visual review are also required for the final demonstration.

For a newly generated, versioned archive, add `--verify-source` and set
`Q1_FRAX_DATA` to that archive's exact asset cache before launching the tool.
This rereads the original BVH, verifies its hash and generation configuration,
recomputes tracking and sphere evidence across the entire exact NPZ, and compares
the result with the archived QA. The retained crop then must satisfy every
strict source/sphere threshold. The report records the source, NPZ, model,
mapping, collision, QA, and manifest hashes, the crop's root/foot errors, minimum
sphere distances, and separate gate results. `source_tracking_verified` and
`source_and_sphere_gates_passed` are distinct: verified measurements may fail
their thresholds. The final `requested_checks_passed` includes these gates when
fresh source replay is requested. A legacy archive cannot acquire this status
merely from a historical hash or a previously successful render.

Stance slip is a geometric proxy based on sustained near-floor sole points,
not measured force. The report includes observation counts and coverage at
8 mm and 20 mm height. Zero slip with no qualifying observations is not evidence
of planted feet, and a physically collision-free motion can still look as if it
is floating or skating. Do not use that statistic alone to approve a video.
