# UR5 golf simulation

`golf_scene.xml` is generated from the supplied UR5 model by
`build_golf_scene.py`. The source robot XML and URDF are unchanged. All
MuJoCo dimensions are in metres.

## Physical layout

| Part | Size or placement |
| --- | --- |
| Lower pedestal | 60.64 x 60.64 x 10 cm |
| Upper pedestal | 15.24 x 15.24 x 70 cm |
| Printed connector | `meshes/mount.STL`, loaded from millimetres at 0.001 scale |
| PVC shaft | 70.96 cm long, 34 mm outside diameter |
| Club head | 12.7 x 3.81 x 4.13 cm |
| Ball | 42.67 mm diameter, 45.93 g |
| Ball to cup center | 104.5 cm along the +X target line |

The connector flange is rigidly attached to the final wrist. The original EE
frame is retained, so the connector nub, PVC shaft, and outward axis of
`link6` are collinear. The PVC begins 35 mm beyond the flange and overlaps the
last 25 mm of the connector nub.

The shaft does not enter the center of the club head. Its center is 51.5 mm
from the head center along the long axis, placing it at the heel with a 12 mm
inset from the end. The shaft terminates at the top of the heel just like a
real club rather than passing through the sweet spot.

## Task setup

All six UR5 joints are free and directly torque controlled. Every episode
uses one deterministic `universal` keyframe:

```text
joint angles: [0, 0, 0, 0, 0, 0] rad
```

This is only a repeatable neutral reset. There is no address keyframe,
inverse-kinematic pose, designated swing joint, timed pulse, follow-through
hold, or valid-pose gate. Any physical club-head contact with the ball counts
as a hit. ProPS must learn how to position the complete arm, orient the face,
and generate a swing.

The robot is placed beside the ball at `(0.0, -0.55)` without solving its
joints toward the ball. The environment exposes joint state and club-to-ball
state but does not expose a target robot configuration.

## Matrix controller

Regular ProPS directly edits one unfactorized `26 x 6` matrix:

```text
action[1 x 6] = features[1 x 26] @ W[26 x 6]
```

Columns 0-5 are torques for joints 0-5. Joints 0-2 are clipped to +/-150 Nm;
joints 3-5 are clipped to +/-28 Nm.

| Rows | Controller features |
| --- | --- |
| 0-4 | Five equal episode-time bins |
| 5-10 | Model gravity/Coriolis/dynamics torque for joints 0-5 |
| 11-16 | Joint angles measured from the neutral reset |
| 17-22 | Joint angular speeds |
| 23-25 | World XYZ vector from the club-face center to the ball |

The time rows allow ProPS to discover an open-loop sequence. Joint and
club-to-ball rows allow it to build feedback and cross-joint coordination.
Every matrix entry is editable, so an observation from any row can influence
any of the six joint torques.

The warmup matrix is deliberately not a golf solution. It contains only:

- diagonal model-dynamics compensation;
- modest diagonal velocity damping; and
- zero positioning, orientation, and swing weights.

It therefore holds the universal pose and does not hit the ball. This gives
the LLM a numerically stable starting controller while leaving all golf
behavior to be learned from reward. Factorization remains disabled.

## Reward

Terminal shot quality has a maximum of 140 points:

```text
+10                              club-head contact
+10 x min(launch_speed / 2, 1)   ball launch speed
+10 x launch_direction_quality   launch direction toward the cup
+40 x impact_centrality          face-center quality
+40 / (1 + 4 x resting_distance) cup proximity after a hit
+30                              sunk ball
```

After a hit, the episode terminates once translational ball speed remains at
or below 0.03 m/s for 0.5 seconds. The distance term therefore uses the
resting result when the ball settles before the five-second timeout.

Incremental terms are:

```text
+8 x best-distance reduction / initial club distance (maximum +8)
-0.20 x squared normalized action changes
-0.005 x squared normalized torque each controller step
-20 per newly colliding unwanted geometry pair
-50 per second of sustained unwanted collision (total collision cap: -250)
```

The approach term is intentionally small. It gives feedback before the first
successful hit without overpowering hit quality or final ball outcome. The
collision cap is larger than the entire 140-point terminal shot reward, so a
collision-heavy trajectory cannot beat a clean shot. Unwanted contacts include
robot-ground, robot-pedestal, club-ground, club-pedestal, cup, and robot
self-collisions. The intended club-head/ball contact and ordinary ball-ground
contact are excluded. `info["active_collision_pairs"]` and
`info["collision_pair_events"]` identify the exact geometries involved. The
complete breakdown is available in `info["reward_components"]`.

## Running

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
cd simulationsetup
python build_golf_scene.py
python -m unittest test_golf_rl.py
```

Preview the neutral stabilizing baseline:

```powershell
python run_golf.py --headless
python run_golf.py --video neutral_preview.mp4
```

Run regular ProPS through the shared runner:

```powershell
python run_golf_rl.py --iterations 10
```

Run three independent sequential tests:

```powershell
python run_three_golf_tests.py
```

The script reads `num_episodes`, `seed`, `llm_model_name`, `logdir`, and all
other training settings directly from `golf_props.yaml`. With the current
YAML, it runs 250 episodes per test with seeds 42, 43, and 44. If the YAML
contains:

```yaml
logdir: logs/golf_props_num_optim
```

the three result directories are:

```text
..\props-factorization\logs\golf_props_num_optim\
..\props-factorization\logs\golf_props_num_optim_1\
..\props-factorization\logs\golf_props_num_optim_2\
```

Existing logs are never overwritten. If any of those directories already
exist, the script automatically advances to the first unused consecutive
sequence. A sibling batch-manifest JSON is updated after every test, including
failure or interruption status. Change episode count, model, base seed, or log
name in the YAML; the only script-level setting is the number of repetitions:

```powershell
python run_three_golf_tests.py --runs 3
```

Replay a learned matrix:

```powershell
python run_golf.py --matrix path\to\best_matrix.json --video learned_shot.mp4
```

The golf configuration is
`props-factorization/configs/golf/golf_props.yaml`. Each shared-runner episode
writes the reward graph and full `26 x 6` heatmap. New best policies receive a
video, and the run root receives an animated heatmap GIF.

Run the shared integration test from `props-factorization`:

```powershell
cd ..\props-factorization
python -m unittest test_golf_props_integration.py
```
