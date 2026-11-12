# UR5 golf simulation

`golf_scene.xml` is a MuJoCo scene built from the supplied
`ur5_arm_only.xml` and mesh files. The original robot model and URDF are
unchanged. All XML dimensions are in metres.

| Part | Size or placement |
| --- | --- |
| Lower pedestal | 60.64 × 60.64 × 10 cm |
| Centered upper pedestal | 15.24 × 15.24 × 70 cm |
| Margin around upper pedestal | 22.70 cm on each side |
| Club shaft | 70.96 cm long, 1.2 cm diameter |
| Club head | 12.7 × 3.81 × 4.13 cm |
| Ball | 42.67 mm diameter, 45.93 g |
| Ball to cup center | 104.5 cm along the +X target line |

**Geometry note:** The specified prism widths give `(60.64 - 15.24) / 2 =
22.70 cm`, rather than 22.54 cm. The scene preserves both requested prism
dimensions and centers the upper one, so its actual side margin is 22.70 cm.
The cup has an 11 cm square opening with a recessed liner and bottom. The
green is level with the top of the cup opening. Its sliding, torsional, and
rolling friction coefficients are set on the ground geoms and can be tuned
for a particular grass or carpet surface. The ball is a standard sized,
standard mass golf ball. The club is attached to the robot's end effector;
the 12.7 cm long face of the head is the striking face. The upper arm is tilted
50.4° toward the ball, with the forearm perpendicular to it. This is the
reachable pose selected to preserve the 80 cm pedestal and 70.96 cm shaft.
Four robot joints are braked at that pose. The two wrist joints are driven by
direct torque motors: `joint4` turns the club into the address plane and
`joint5` swings. Every reset starts `joint4` at a universal zero-angle pose,
not at the ready angle of `-pi/2`, so a swing-only policy cannot hit the ball.
The demo positions the club first, then applies an active brake and holds a
short follow-through pose after club-to-ball contact.

The green uses fixed MuJoCo contact settings: sliding friction `0.85`,
torsional friction `0.001`, and rolling friction `0.0012` with six-dimensional
contact enabled. A freely rolling ball launched at 1 m/s slowed by about
0.38 m/s² in this model, close to the roughly 0.4 m/s² reported for a ball
rolling on a putting green in [golf-ball motion research](https://www.quinticballroll.com/research-papers/Phases-of-Motion-of-a-Golf-Ball.pdf).
The shaft mass is 0.10 kg, the head mass is 0.25 kg, and the wrist damping is
0.05 N m s/rad. These values stay fixed for every action.

Run with Python 3.10 or newer and MuJoCo installed:

```bash
python -m pip install mujoco pillow imageio imageio-ffmpeg
cd simulationsetup
python build_golf_scene.py
python run_golf.py
```

With a display, the demo opens the MuJoCo desktop viewer. Without `DISPLAY`
or `WAYLAND_DISPLAY`, it precomputes a 60 fps browser replay using offscreen
rendering and serves each camera as one MP4. This avoids one network request
per frame over a forwarded port.
Open **http://127.0.0.1:8765** in your browser while the command is running.
The page has arm-and-ball and impact cameras, replay controls, and distance,
wrist angle, and torque readouts. If the simulation runs on a remote machine,
forward port 8765 to your computer first (for example, with the VS Code **Ports** panel or an SSH
tunnel). Press Ctrl+C in the terminal to stop the display. Use `--port 8766`
if 8765 is occupied.

Use `--viewer browser` to choose the browser display explicitly,
`--video FILE.mp4` to record an MP4, or `--headless` to print distances
without rendering. The included `golf_hit_demo.mp4` shows the corrected
long-face strike.

`--swing-torque` selects signed swing torque in N m, `--pulse-time` sets how
long to apply it, and `--duration` sets total simulation time. A smooth PD
controller first turns `joint4` from 0 to `-pi/2`; the default demo then
applies +1 N m to `joint5` for 0.15 s starting at 0.55 s. MuJoCo integrates
both wrist angles from torque, inertia, gravity, and contact forces. Once the
head touches the ball, damped controllers hold the address orientation and a
0.15 rad swing follow-through. Both torques are capped at 28 N m, matching
the wrist effort limits in the supplied UR5 URDF.
The example ball stops about 30 cm short of the cup in MuJoCo 3.5, and the
wrist's final speed is effectively zero. The saved MP4 shows the first 2.5
seconds; the browser replay shows the whole 5-second run.

For another controller, import `GolfEnvironment` from `golf_env.py` and use
`reset()`, `step((position_torque, swing_torque), substeps=1)`, and `observe()`.
The returned observation includes both wrist states and torques, ball and cup
distances, first-impact offset and centrality, unwanted collision counts, and
whether the ball was hit or sunk. The model step is 0.001 s, so `substeps=100`
holds both torques for 0.1 s. The raw environment does not brake automatically;
`demo_controller.py` supplies the example two-joint sequence.

The scene uses the USGA's [45.93 g maximum mass](https://www.usga.org/equipment-standards/equipment-rules-2019/equipment-rules/part-4-rule-3.html)
and [42.67 mm minimum diameter](https://www.usga.org/custom-search-pages/rules/equipment-rules/rule-14611.html)
as representative ball values.

## RL task and prompted policy search

`golf_rl_env.py` is a current Gymnasium environment with `reset()` returning
`(observation, info)` and `step()` returning
`(observation, reward, terminated, truncated, info)`. Its action space is a
two-value `Box`: positioning torque for `joint4`, then swing torque for
`joint5`. Each action is held for 20 MuJoCo steps (0.02 s). `render_mode` may
be `None` or `"rgb_array"`, which also makes it compatible with the world and
video interfaces used by `props-factorization`.

Reset is deterministic. `joint4` always begins at the universal `0 rad` pose
while the useful address plane is `-pi/2 rad`. Therefore the matrix must
control both joints; zeroing the first action column makes a swing-only policy
miss.

The episode reward is:

```text
+20                         if the club hits the ball
+30 * impact_centrality     (1 at face center, 0 at a face corner or beyond)
+30 / (1 + 4 * final_distance_m) if the ball was hit
+20                         if the ball is sunk
-0.20 * squared normalized action changes
-0.02 * unwanted robot-contact physics steps
```

The smoothness and collision terms are emitted incrementally; shot-quality
terms are terminal. The sum and all six components are available in
`info["reward_components"]`. First-impact offset, centrality, both wrist
states, and collision counts are also in `info` and the saved history.

`run_golf_rl.py` uses regular ProPS only: `action = state @ W`, with one full
11-by-2 matrix. No LU, QR, SVD, factor matrices, ranks, or factorization
options are used. Rows are named by `FEATURE_NAMES` and cover two positioning
windows, three later swing windows, position error/speed/gravity feedback,
and post-shot swing hold feedback. Columns are always:

1. `position_joint4_torque_nm`
2. `swing_joint5_torque_nm`

Each episode saves:

- `reward_progress.png` plus an episode-specific reward graph snapshot;
- `policy_heatmap_epNNN.png` for the exact 11-by-2 matrix used;
- updated `history.json` and `best_matrix.json`;
- an MP4 under `best_videos/` whenever a new best is found.

The best-video rollout is replayed from reset and checked against the
evaluated reward. `--record-video` additionally records every proposal under
`all_videos/`. Use `--camera overview` or `--camera impact` for other views.

Install and run an offline baseline (no API key or display required):

```bash
python -m pip install gymnasium numpy mujoco matplotlib imageio imageio-ffmpeg
cd simulationsetup
python run_golf_rl.py
```

For LLM revisions through ASU Research Computing:

```bash
python -m pip install openai python-dotenv
# Set OPENAI_API_KEY in simulationsetup/.env or the shell.
python run_golf_rl.py --iterations 10
```

The runner uses `OPENAI_BASE_URL` and `OPENAI_MODEL` when set and never writes
the API key to results. The LLM receives feature names, action columns, recent
reward breakdowns, and an explicit instruction to return a complete regular
11-by-2 matrix without factors. Re-evaluate a saved policy with
`python run_golf_rl.py --matrix rl_runs/DATE/best_matrix.json`.

Use the environment directly with `GolfRLEnv`, or through Gymnasium with
`gym.make("golf_rl_env:GolfSwing-v0")` when `simulationsetup` is on
`PYTHONPATH`. Run `python -m unittest test_golf_rl.py` for the API, reward,
two-joint necessity, determinism, and prompt checks.
