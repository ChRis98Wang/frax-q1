# Collision inspection and tuning

There are two geometry layers. FRAX uses spherical envelopes to constrain IK.
The final validator and viewer use the enabled MuJoCo physical collision shapes.
Render colors and labels operate on a visualization copy and do not change the
physical model used for the measured collision report.

Self-collision records contain both link names, geometry identities, frame index
and penetration depth in millimetres. Floor contacts are reported separately.
Adjacent and next-adjacent body exclusions are listed explicitly because the
original simplified shapes overlap around some joints. They are not proof that
those regions can never interfere on hardware.

Orange/red markers distinguish smaller penetration from the 2 mm quality limit.
A raw rejected frame remains visibly marked as raw/rejected during inspection.
A motion that looks plausible can still fail physical validation.

## FRAX parameters

| Parameter | Baseline | Meaning |
|---|---:|---|
| `self_collision_margin` | 0 m | Extra clearance relative to signed sphere-pair distance |
| `self_collision_gain` | 30 /s | Barrier response gain |
| `self_collision_penalty` | 1,000,000 | Penalty for relaxing an infeasible inequality |
| `self_collision_max_rows` | 64 | Nearest constraint rows kept per IK step; `null` keeps all |
| `floor_margin` | 0.002 m | Spherical floor-clearance target |
| `contact_swing_margin` | 0.015 m | Minimum additional swing clearance in contact-aware targets |
| `foot_position_weight` | 200 | Foot target position weight |
| `foot_rotation_weight` | 40 | Foot target orientation weight |
| `damping` | 30 | Joint-velocity regularization |
| `shoulder_damping` | 300 | Shoulder regularization |
| `joint_velocity_limit` | 12 rad/s | IK speed ceiling, also bounded by model limits |

The adapter's sphere pairs already use a 2 mm tolerance. An additional 3 mm
margin therefore requests about 5 mm of separation between sphere surfaces.
Increasing gains, penalties, or margins does not guarantee better final motion:
constraints can conflict, and FRAX uses elastic inequalities when infeasible.
See the upstream [FRAX IK reference](https://stanfordasl.github.io/frax/api/ik/).

Change one parameter group at a time, retain the old configuration and source
hashes, evaluate held-out source clips, and compare tracking errors, slip,
penetration, joint speed/acceleration and retained duration together. Do not
lower quality thresholds merely to increase accepted-frame counts.
