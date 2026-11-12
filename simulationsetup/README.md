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
Five robot joints are braked at that pose. Only the last wrist joint (`joint5`)
swings, driven by a direct torque motor. The demo applies an active brake and
holds a short follow-through pose after club-to-ball contact.

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

`--swing-torque` selects signed wrist torque in N m, `--pulse-time` sets how
long to apply it, and `--duration` sets total simulation time. The default
demo applies +1 N m for 0.15 s starting at 0.05 s. MuJoCo integrates the
resulting swing angle from torque, inertia, gravity, and contact forces. Once
the head touches the ball, a damped controller brakes the wrist and holds it
0.15 rad beyond the contact angle. The brake torque is capped at 28 N m,
matching the wrist effort limit in the supplied UR5 URDF. The wrist starts
0.25 rad behind its vertical position, so gravity contributes to the swing.
The example ball stops about 30 cm short of the cup in MuJoCo 3.5, and the
wrist's final speed is effectively zero. The saved MP4 shows the first 2.5
seconds; the browser replay shows the whole 5-second run.

For another controller, import `GolfEnvironment` from `golf_env.py` and use
`reset()`, `step(torque_nm, substeps=1)`, and `observe()`. The returned
observation gives the ball position, current and closest horizontal distance
to the cup, wrist angle and angular speed, applied torque, contact history,
and whether the ball sank. Actions accept any finite torque. The model step
is 0.001 s, so `substeps=100` applies one torque for 0.1 s. The raw environment
does not brake automatically; `demo_controller.py` supplies the example swing
and braking sequence.

The scene uses the USGA's [45.93 g maximum mass](https://www.usga.org/equipment-standards/equipment-rules-2019/equipment-rules/part-4-rule-3.html)
and [42.67 mm minimum diameter](https://www.usga.org/custom-search-pages/rules/equipment-rules/rule-14611.html)
as representative ball values.

## RL task and prompted policy search

`golf_rl_env.py` wraps the same scene as a Gymnasium environment. Each action
is one wrist motor torque in N m, held for 20 physics steps (0.02 s). The
other five UR5 joints are fixed in the scene. A shot runs for five seconds or
until the ball sinks. The only reward is at the end of the shot:

```text
reward = 1 / (final_horizontal_distance_to_hole_m + 0.02)
```

The maximum is 50 at the hole center. A shot that passes near the hole and
rolls away receives a lower score based on where it finishes. Physics values
and the initial ball/robot positions are identical on every reset. The
environment can be used directly with `GolfRLEnv`, or through Gymnasium with
`gym.make("golf_rl_env:GolfSwing-v0")` when `simulationsetup` is on
`PYTHONPATH`.

`run_golf_rl.py` uses the numerical policy search pattern from the
`props-factorization` project: evaluate a policy matrix, show its matrix and
reward history to an LLM, then evaluate the LLM's revised matrix. It is a
small standalone loop inside `simulationsetup` and does not change the older
project. The policy computes `torque = clip(observation @ matrix, -28, 28)`.
The matrix has seven rows and one column because this scene has one moving
joint. Rows, in order, are:

| Row | Feature | Default weight |
| --- | --- | ---: |
| 0 | Swing indicator from 0.05 to 0.20 s | 1 |
| 1 | Swing indicator from 0.20 to 0.35 s | 0 |
| 2 | Swing indicator from 0.35 to 0.60 s | 0 |
| 3 | Swing indicator from 0.60 to 1.00 s | 0 |
| 4 | Gravity compensation torque after hit | 1 |
| 5 | Follow-through target minus wrist angle after hit | 40 |
| 6 | Wrist angular speed after hit | -8 |

After contact, or after a one-second timeout, the first four features become
zero and the last three turn on. The target is the contact angle plus 0.15 rad.
The default matrix gives the demonstrated swing and active brake. The LLM may
edit every weight, including swing timing and braking, but the 28 N m wrist
torque limit remains fixed. The input and output matrix format is JSON:

```json
{"matrix": [[1], [0], [0], [0], [1], [40], [-8]]}
```

Run an offline baseline (no API key or display required):

```bash
python -m pip install gymnasium numpy mujoco
cd simulationsetup
python run_golf_rl.py
```

To request LLM revisions through ASU Research Computing, install the API
dependencies and fill in `OPENAI_API_KEY` in `simulationsetup/.env`, then run:

```bash
python -m pip install openai python-dotenv
# Edit .env: OPENAI_API_KEY=your_ASU_Research_Computing_key
python run_golf_rl.py --iterations 10
```

The `.env` file already sets `OPENAI_BASE_URL=https://openai.rc.asu.edu/v1`
and `OPENAI_MODEL=gemma4-31b-it`, using your selected model. The
trainer calls `OpenAI(base_url=..., api_key=...)` and uses chat completions.
It requests plain JSON text for the revised matrix, so it does not require
the endpoint to support JSON response mode. `.env.example` is a shareable
template. Shell environment values take precedence over `.env`.
The key is never written to the result files. Each training run saves `history.json` and
`best_matrix.json` under `rl_runs/`. Re-evaluate a saved policy with
`python run_golf_rl.py --matrix rl_runs/DATE/best_matrix.json`. The script
validates matrix dimensions and finite values before applying any proposal.
For a local check, run `python -m unittest test_golf_rl.py`.

### Watch policy rollouts

Record every evaluated matrix, including the baseline and each LLM revision:

```bash
python -m pip install imageio imageio-ffmpeg
python run_golf_rl.py --iterations 1 --record-video
```

The run directory printed by the command contains `trial_000.mp4` for the
baseline and `trial_001.mp4` for the first model proposal, alongside their
scores in `history.json`. Videos use a 30 fps MP4 and can be opened directly
in VS Code's remote file explorer or downloaded. `--camera overview` shows
the full green; `--camera impact` uses the close-up camera. To preview files
in a forwarded browser, run this from `simulationsetup` and open the forwarded
port:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory rl_runs
```

A baseline example is at `rl_runs/baseline/trial_000.mp4` in this workspace.
