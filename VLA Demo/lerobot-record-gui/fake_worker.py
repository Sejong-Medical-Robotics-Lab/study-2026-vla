# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Fake recording worker -- no robot, no CAN, no cameras.

Speaks the exact tick protocol ``recorder_worker.py`` speaks, by importing its
``Metrics`` and ``Reporter`` rather than reimplementing them. That is the point:
if the two drifted apart, the GUI would be verified against a protocol nothing
real ever sends.

Used by ``server.py --demo`` so the whole GUI -- phase colours, countdown,
buttons, warnings, force-quit -- can be exercised on a laptop. Debugging UI
while someone holds a leader arm is a waste of the robot's time and theirs.

    python fake_worker.py --gui-url http://127.0.0.1:8010 --speed 5 -- --dataset.repo_id=demo/x

Every third episode is deliberately a dead take (arm parked, teleop silent) so
the "팔이 움직이지 않습니다" path gets exercised rather than assumed.
"""

from __future__ import annotations

import math
import sys
import time

from recorder_worker import Metrics, Reporter, parse_args

try:
    import numpy as np
except ImportError:  # numpy is a lerobot dep, but the demo must run without it
    np = None

JOINTS = ["joint_1.pos", "joint_2.pos", "joint_3.pos",
          "joint_4.pos", "joint_5.pos", "joint_6.pos", "gripper.pos"]
CAMERAS = ("front", "wrist")
DEAD_EVERY = 3  # every Nth episode is a dead take


class Sim:
    def __init__(self, fps: float, speed: float):
        self.fps = fps
        self.dt = 1.0 / (fps * speed)
        self.t = 0.0

    def observation(self, moving: bool, frame: int) -> dict:
        self.t += 1.0 / self.fps
        obs: dict = {}
        for i, name in enumerate(JOINTS[:-1]):
            base = 10.0 * i
            obs[name] = base + (25.0 * math.sin(self.t * 0.7 + i) if moving else 0.02 * (i % 3))
        # Gripper opens and closes twice over a 20s episode.
        obs[JOINTS[-1]] = (50.0 + 48.0 * math.sin(self.t * 0.6)) if moving else 0.0
        obs.update(self.frames(moving, frame))
        return obs

    def action(self, obs: dict, moving: bool) -> dict:
        # The follower lags the leader slightly, as it does in reality.
        return {k: v * (1.02 if moving else 1.0) for k, v in obs.items()
                if isinstance(v, float)}

    def frames(self, moving: bool, frame: int) -> dict:
        if np is None:
            return {}
        out = {}
        for i, name in enumerate(CAMERAS):
            if moving:
                shade = (frame * 3 + i * 40) % 255
                img = np.full((480, 640, 3), shade, dtype=np.uint8)
            else:
                img = np.full((480, 640, 3), 120, dtype=np.uint8)  # frozen feed
            out[name] = img
        return out


def phase_loop(sim: Sim, metrics: Metrics, events: dict, seconds: float,
               moving: bool) -> str:
    """Run one phase at fps. Returns why it ended."""
    total = int(seconds * sim.fps)
    for frame in range(total):
        if events["stop_recording"]:
            return "stop"
        if events["rerecord_episode"]:
            return "rerecord"
        if events["exit_early"]:
            events["exit_early"] = False
            return "early"
        obs = sim.observation(moving, frame)
        metrics.on_observation(obs)
        if moving:
            metrics.on_action(sim.action(obs, moving))
        time.sleep(sim.dt)
    return "complete"


def take_speed(argv: list[str]) -> tuple[float, list[str]]:
    """Pull --speed out before handing the rest to the shared parser.

    Keeps ``recorder_worker.parse_args`` free of a flag only the simulator has
    any use for.
    """
    speed, rest = 1.0, []
    i = 0
    while i < len(argv):
        if argv[i] == "--speed" and i + 1 < len(argv):
            speed, i = float(argv[i + 1]), i + 2
            continue
        if argv[i].startswith("--speed="):
            speed, i = float(argv[i].split("=", 1)[1]), i + 1
            continue
        rest.append(argv[i])
        i += 1
    return max(speed, 0.1), rest


def main() -> int:
    speed, argv = take_speed(sys.argv[1:])
    cfg, _rest = parse_args(argv)

    episode_time = cfg.episode_time_s or 20.0
    reset_time = cfg.reset_time_s or 20.0
    target = cfg.num_episodes or 5

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    metrics = Metrics(fps_target=cfg.fps)
    metrics.episode_target = target
    metrics.first_episode_index = 0

    reporter = None
    if cfg.gui_url:
        reporter = Reporter(cfg.gui_url, cfg.token, metrics, events, cfg.tick_hz)
        reporter.start()

    sim = Sim(cfg.fps, speed)
    print(f"[fake] {target} episodes x {episode_time}s (reset {reset_time}s) "
          f"@ {cfg.fps}fps, speed x{speed}", flush=True)

    recorded = 0
    episode_index = 0
    while recorded < target and not events["stop_recording"]:
        moving = (episode_index % DEAD_EVERY) != (DEAD_EVERY - 1)
        metrics.episode_index = episode_index
        metrics.begin_phase("record", f"Recording episode {episode_index}", episode_time)
        print(f"[phase] Recording episode {episode_index}"
              + ("" if moving else "   (dead take, on purpose)"), flush=True)
        why = phase_loop(sim, metrics, events, episode_time, moving)

        if why != "stop" and (recorded < target - 1 or events["rerecord_episode"]):
            metrics.begin_phase("reset", "Reset the environment", reset_time)
            print("[phase] Reset the environment", flush=True)
            why = phase_loop(sim, metrics, events, reset_time, True)

        if events["rerecord_episode"]:
            metrics.rerecords += 1
            metrics.begin_phase("rerecording", "Re-record episode", None)
            print("[phase] Re-record episode", flush=True)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            time.sleep(0.5)
            continue

        if events["stop_recording"]:
            break

        # Stand in for save_episode() + video encoding: the loop stops pulling
        # frames, which is exactly how the real worker infers "saving".
        print(f"[fake] saving episode {episode_index}", flush=True)
        # Not divided by speed: video encoding does not get faster because we
        # fast-forwarded the clock, and this pause is what the "saving" phase
        # detector keys off. Scaling it away would hide that path in demos.
        time.sleep(1.5)
        recorded += 1
        episode_index += 1

    metrics.begin_phase("stopping", "Stop recording", None)
    print("[phase] Stop recording", flush=True)
    time.sleep(0.5)
    metrics.begin_phase("done", "Exiting", None)
    print(f"[fake] finished: {recorded} episode(s), {metrics.rerecords} re-record(s)",
          flush=True)

    if reporter is not None:
        reporter.final("done")
        reporter.stop_flag.set()
        time.sleep(0.2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
