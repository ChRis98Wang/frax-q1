# Quality contract

A trajectory is accepted by a named, versioned profile, not by visual appearance
or IK solver completion. A report must state its profile and configuration hash,
source identity, model hash, frame rate, quaternion convention and joint order.
Every source frame is checked. Rejected frames retain reasons; missing required
evidence cannot silently become zero error.

`q1-frax-quality --motion MOTION --out artifacts/quality.json` performs
physical inspection. Add `--dataset DATASET` to recompute the original tracking
and FRAX sphere evidence. Exit status **0** means the complete clip passed all
required checks with verified provenance; **2** means diagnostic, unverified or
partially rejected output. Individual valid spans may exist even when the whole
clip does not pass. A saved report is not itself a seal.

## Historical walking profile

The predecessor dataset used these numerical limits. New profiles must be
recorded separately; changing a threshold invalidates comparisons of pass rates.

| Check | Historical limit / requirement |
|---|---|
| Shape and finite values | 29 qpos columns: root XYZ, root quaternion, 22 ordered joints; all finite |
| Root orientation | Unit quaternion with declared WXYZ (NPZ) or XYZW (CSV) convention |
| Physical self penetration | At most 0.002 m, on enabled MuJoCo collision pairs |
| Floor penetration | At most 0.002 m, including actual sole geometry |
| Joint-limit excess | At most 0.001 rad |
| Joint speed | At most 12 rad/s |
| Joint acceleration | At most 150 rad/s² |
| Near-ground stance-slip proxy | At most 0.20 m/s |
| Root target position error | At most 0.08 m |
| Foot target position error | At most 0.06 m |
| Walking upright score | At least 0.85 |
| Root height | At least 0.28 m |
| Lowest sole clearance | At most 0.025 m for walking reset suitability |
| Shoulder-roll calibration filter | Absolute shoulder roll at most 0.7 rad |
| Initial calibration exclusion | First 1 s |
| Rejection boundary buffer | 2 frames on either side |
| Exported continuous span | At least 90 consecutive accepted frames at 30 Hz (3 s) |

The source loader must identify metres, radians, world-up direction and frame
rate. Derivatives use the declared sample interval. Smoothing and contact-aware
foot clearance alter the target construction and therefore belong in provenance.
The stance-slip metric is a geometric near-ground estimate, not measured contact
force or a demonstrated friction-cone constraint.

## Collision evidence

FRAX spherical geometry is an optimization approximation. The independent
MuJoCo check inspects actual enabled physical collision geometry. It reports
self and floor penetration separately, with link-pair identities and depths.
Exclusions remain visible in the report. A result only covers enabled collision
pairs and the selected physical model, not every possible visual mesh overlap.

## Missing and failed evidence

Malformed frames, nonfinite values, invalid quaternion norms, unknown units,
missing required target tracking, incompatible model hashes and empty inputs
must not produce an accepted dataset. Physical-only inspection may still be
useful, but must label unavailable source tracking as unverified. An output can
contain valid raw trajectories and still have zero accepted long spans.

## Sealing and splits

Accepted spans must be exact source slices, contain no rejected frames, and
meet the minimum duration. A source clip belongs to only one split. File hashes
protect the actual motion bytes, and root manifests identify the source and
quality report. Paths must remain inside the dataset. Once sealed, a dataset is
immutable; use a new output directory for changed parameters or revalidation.

## What this does not certify

Kinematic QA does not prove balance, torque limits, impact tolerance, actuator
tracking, robustness to delay, or successful real-world execution. Those need
separate dynamic and controller evaluation. The project therefore labels every
viewer and rendered artifact as reference playback.
