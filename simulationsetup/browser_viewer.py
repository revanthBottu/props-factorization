"""Display a MuJoCo golf run in a browser when no X11 display is available."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import time
from urllib.parse import urlparse

import imageio.v2 as imageio
import mujoco

from golf_rl_env import GolfRLEnv, MatrixGolfPolicy


FPS = 50
CAMERAS = ("swing", "impact")

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UR5 golf simulation</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #141914; color: #eef4e9; }
    main { max-width: 1000px; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
    p { color: #b9cbb8; margin: .3rem 0 1rem; }
    .screen { background: #000; border-radius: 10px; overflow: hidden; }
    .screen video { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; }
    .controls { display: flex; gap: .7rem; align-items: center; flex-wrap: wrap; margin: 1rem 0; }
    button, select { padding: .55rem .8rem; border: 1px solid #789477; border-radius: 6px;
      background: #243523; color: white; font: inherit; cursor: pointer; }
    .stats { display: flex; gap: 1.5rem; flex-wrap: wrap; padding: .8rem 0; font-size: 1.05rem; }
    .stats strong { color: #a9df8d; }
  </style>
</head>
<body>
<main>
  <h1>UR5 golf simulation</h1>
  <p>Replay the club strike and ball motion. The long face of the head strikes the ball.</p>
  <div class="screen"><video id="scene" controls autoplay muted playsinline
    src="/video/swing.mp4" aria-label="Robot golf simulation"></video></div>
  <div class="controls">
    <button id="restart" type="button">Replay</button>
    <select id="camera" aria-label="Camera"><option value="swing">Arm and ball</option>
      <option value="impact">Impact close-up</option></select>
  </div>
  <div class="stats">
    <span>Time: <strong id="time">0.00 s</strong></span>
    <span>Distance to hole: <strong id="distance">1.045 m</strong></span>
    <span>Club-to-ball: <strong id="club">0.000 m</strong></span>
    <span>Max joint torque: <strong id="torque">0.0 N m</strong></span>
    <span>Club contact: <strong id="hit">No</strong></span>
    <span>Ball sunk: <strong id="sunk">No</strong></span>
  </div>
</main>
<script>
  const scene = document.getElementById('scene');
  const camera = document.getElementById('camera');
  let samples = [], fps = 50;
  function updateMetrics() {
    const frame = Math.min(samples.length - 1, Math.round(scene.currentTime * fps));
    const observation = samples[frame];
    if (!observation) return;
    document.getElementById('time').textContent = observation.time_s.toFixed(2) + ' s';
    document.getElementById('distance').textContent =
      observation.distance_to_hole_m.toFixed(3) + ' m';
    document.getElementById('club').textContent =
      observation.club_head_distance_to_ball_m.toFixed(3) + ' m';
    document.getElementById('torque').textContent =
      Math.max(...observation.action_torques_nm.map(Math.abs)).toFixed(2) + ' N m';
    document.getElementById('hit').textContent = observation.ball_hit ? 'Yes' : 'No';
    document.getElementById('sunk').textContent = observation.ball_sunk ? 'Yes' : 'No';
  }
  scene.addEventListener('timeupdate', updateMetrics);
  scene.addEventListener('seeked', updateMetrics);
  function tick() {
    if (!scene.paused) updateMetrics();
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
  document.getElementById('restart').onclick = () => {
    scene.currentTime = 0;
    scene.play();
  };
  camera.onchange = () => {
    const previousTime = scene.currentTime;
    const wasPlaying = !scene.paused;
    scene.src = `/video/${camera.value}.mp4`;
    scene.load();
    scene.onloadedmetadata = () => {
      scene.currentTime = previousTime;
      if (wasPlaying) scene.play();
      updateMetrics();
    };
  };
  fetch('/meta').then(response => response.json()).then(meta => {
    samples = meta.samples; fps = meta.fps;
    updateMetrics();
  }).catch(error => { document.querySelector('p').textContent = error.message; });
</script>
</body>
</html>"""


def prepare_videos(matrix, duration):
    env = GolfRLEnv(duration_s=duration)
    policy = MatrixGolfPolicy(matrix)
    state, _ = env.reset()
    renderer = mujoco.Renderer(env.sim.model, height=480, width=640)
    samples = []
    next_frame = 0
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="golf_view_") as directory:
        paths = {camera: Path(directory) / f"{camera}.mp4" for camera in CAMERAS}
        writers = {camera: imageio.get_writer(str(path), fps=FPS, codec="libx264",
                                              quality=8) for camera, path in paths.items()}
        try:
            while True:
                state, _, terminated, truncated, observation = env.step(
                    policy.action(state)
                )
                for camera in CAMERAS:
                    renderer.update_scene(env.sim.data, camera=camera)
                    writers[camera].append_data(renderer.render())
                samples.append(observation)
                next_frame += 1
                if terminated or truncated:
                    break
        finally:
            for writer in writers.values():
                writer.close()
            renderer.close()
        videos = {camera: path.read_bytes() for camera, path in paths.items()}
    print(f"Prepared {len(samples)} video frames in {time.perf_counter() - started:.1f}s")
    result = env.sim.observe()
    print(f"Club hit: {result['ball_hit']}; final distance: "
          f"{result['distance_to_hole_m']:.3f}m; sunk: {result['ball_sunk']}")
    env.close()
    return videos, samples


def serve_demo(matrix, duration, host="127.0.0.1", port=8765):
    videos, samples = prepare_videos(matrix, duration)
    metadata = json.dumps({"fps": FPS, "samples": samples}).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlparse(self.path)
            if request.path in ("/", "/index.html"):
                body, content_type = PAGE.encode("utf-8"), "text/html; charset=utf-8"
            elif request.path == "/meta":
                body, content_type = metadata, "application/json"
            elif request.path.startswith("/video/") and request.path.endswith(".mp4"):
                camera = request.path[len("/video/"):-len(".mp4")]
                body = videos.get(camera)
                if body is None:
                    self.send_error(404, "Video not found")
                    return
                total = len(body)
                range_header = self.headers.get("Range", "")
                if range_header.startswith("bytes="):
                    try:
                        first, last = range_header[6:].split("-", 1)
                        if first:
                            start = int(first)
                            end = int(last) if last else total - 1
                        else:
                            end = total - 1
                            start = max(0, total - int(last))
                        if start < 0 or start >= total or end < start:
                            raise ValueError
                        end = min(end, total - 1)
                    except ValueError:
                        self.send_error(416, "Invalid byte range")
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                    body = body[start:end + 1]
                else:
                    self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            else:
                self.send_error(404, "Not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Open http://127.0.0.1:{server.server_port} in your browser")
    print("If this is a remote machine, forward that port to your computer. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Browser display stopped.")
    finally:
        server.server_close()
