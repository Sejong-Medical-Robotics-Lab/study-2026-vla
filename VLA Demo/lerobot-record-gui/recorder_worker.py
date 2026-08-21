# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
LeRobot Record Worker
=====================

Runs ``lerobot-record`` in-process so a GUI can drive it, and reports what the
control loop is doing back to that GUI.

    python recorder_worker.py --gui-url http://127.0.0.1:8010 -- \
        --robot.type=piper_follower --robot.port=can5 ...

Everything after ``--`` is passed to LeRobot untouched.

How it hooks in
---------------
LeRobot's ``record()`` is one function, and three of the names it uses are bound
in its own module namespace -- so swapping those attributes steers it without
touching a line of LeRobot source:

``init_keyboard_listener``
    Normally returns a pynput listener plus an ``events`` dict with the keys
    ``exit_early`` / ``rerecord_episode`` / ``stop_recording``, which
    ``record_loop`` reads every frame. We return our own dict instead, so the
    GUI's buttons land exactly where the arrow keys used to -- and the pynput,
    sudo and non-headless requirements disappear with it.

``log_say``
    Called at every phase transition ("Recording episode 7", "Reset the
    environment", ...). Intercepting it gives us the phase and episode number
    from an actual call site, not from parsing stdout.

``make_robot_from_config``
    We wrap the robot it returns so ``get_observation`` / ``send_action`` tee
    their values to us. That is where the frame counter, the achieved fps and
    the joint values come from.

The hooks only ever assign to a slot. All HTTP happens on a daemon thread, so
the 30 Hz control loop is never blocked -- that is the hard constraint here; a
GUI that costs recording quality is a failure.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import pkgutil
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

# --------------------------------------------------------------------------
# Thresholds. These mirror lerobot-episode-inspector/server.py so that what the
# GUI flags live matches what the inspector flags afterwards -- an episode that
# looked fine here should not come back "fail" there.
# --------------------------------------------------------------------------

STATIC_JOINT_RANGE = 1.0  # a joint spanning less than this never really moved
NO_MOTION_GRACE_S = 3.0  # do not shout about a still arm before this
PHASE_SETTLE_S = 1.5  # no throughput/camera warnings until the loop settles
CAMERA_FROZEN_DIFF = 0.5  # mean abs frame diff below this reads as a frozen feed
CAMERA_DIFF_EVERY = 6  # only diff every Nth frame; 30fps / 6 = 5 Hz
FPS_LOW_RATIO = 0.9  # achieved fps under this share of target is a warning
LATE_FRAME_RATIO = 1.5  # a loop period over this * (1/fps) counts as late
TRACKING_WARN_RATIO = 0.5  # mean |action-state| over this share of travel
SAVING_STALL_S = 0.5  # no frames for this long mid-run means we left the loop


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


class Metrics:
    """Slots the control-loop hooks write into, and the reporter thread reads.

    Deliberately lock-free: every field is replaced by a single assignment of an
    immutable value (or a list the writer no longer touches), so a reader can
    never see a half-updated structure. Taking a lock in ``get_observation``
    would put GUI latency inside the recording loop.
    """

    def __init__(self, fps_target: float):
        self.fps_target = fps_target

        # phase, set from log_say
        self.phase = "starting"
        self.phase_msg = ""
        self.phase_total_s: float | None = None
        self.phase_started_at = time.monotonic()

        # episode bookkeeping
        self.episode_index: int | None = None  # dataset.num_episodes at phase start
        self.first_episode_index: int | None = None
        self.episode_target: int | None = None
        self.rerecords = 0

        # frames
        self.frames = 0  # frames in the current phase
        self.total_frames = 0
        self.last_frame_at = 0.0
        self.late_frames = 0
        self._recent_periods: list[float] = []

        # joints
        self.joint_names: list[str] = []
        self.state: list[float] = []
        self.action: list[float] = []
        self.state_min: list[float] = []
        self.state_max: list[float] = []
        self.track_err_sum: list[float] = []
        self.track_n = 0
        self.action_seen = False

        # gripper
        self.gripper_index: int | None = None
        self.gripper_cycles = 0
        self._gripper_level: str | None = None

        # cameras
        self.cameras: dict[str, dict] = {}
        self._cam_prev: dict[str, Any] = {}

        self.error: str | None = None

    # -- phase -------------------------------------------------------------

    def begin_phase(self, phase: str, msg: str, total_s: float | None) -> None:
        self.phase = phase
        self.phase_msg = msg
        self.phase_total_s = total_s
        self.phase_started_at = time.monotonic()
        self.frames = 0
        self.late_frames = 0
        self._recent_periods = []
        # Also forget when the last frame arrived. Otherwise the first period of
        # the new phase spans the gap we just sat through (save_episode, video
        # encoding) and drags the mean down far enough to fire a low-fps warning
        # at the top of every single episode.
        self.last_frame_at = 0.0
        if phase == "record":
            # Per-episode stats reset; the "is the arm moving" check is about
            # this episode, not the session.
            self.state_min = []
            self.state_max = []
            self.track_err_sum = []
            self.track_n = 0
            self.action_seen = False
            self.gripper_cycles = 0
            self._gripper_level = None

    # -- hooks (called from the control loop -- keep these cheap) -----------

    def on_observation(self, obs: dict) -> None:
        now = time.monotonic()
        if self.last_frame_at:
            period = now - self.last_frame_at
            self._recent_periods.append(period)
            if len(self._recent_periods) > 60:
                del self._recent_periods[:-60]
            if self.fps_target and period > LATE_FRAME_RATIO / self.fps_target:
                self.late_frames += 1
        self.last_frame_at = now
        self.frames += 1
        self.total_frames += 1

        values, names, frames = _split_observation(obs)

        if not self.joint_names and names:
            self.joint_names = names
            self.gripper_index = _find_gripper(names)
        if values:
            self.state = values
            self._track_range(values)
            self._track_gripper(values)

        if frames and self.total_frames % CAMERA_DIFF_EVERY == 0:
            self._diff_cameras(frames)

    def on_action(self, action: Any) -> None:
        values, _names, _frames = _split_observation(action)
        if not values:
            return
        self.action = values
        self.action_seen = True
        if self.state and len(self.state) == len(values):
            errs = [abs(a - s) for a, s in zip(values, self.state)]
            if not self.track_err_sum:
                self.track_err_sum = errs
            else:
                self.track_err_sum = [t + e for t, e in zip(self.track_err_sum, errs)]
            self.track_n += 1

    # -- derived -----------------------------------------------------------

    def _track_range(self, values: list[float]) -> None:
        if not self.state_min or len(self.state_min) != len(values):
            self.state_min = list(values)
            self.state_max = list(values)
            return
        self.state_min = [min(a, b) for a, b in zip(self.state_min, values)]
        self.state_max = [max(a, b) for a, b in zip(self.state_max, values)]

    def _track_gripper(self, values: list[float]) -> None:
        i = self.gripper_index
        if i is None or not self.state_min or i >= len(self.state_min):
            return
        value = values[i]
        lo, hi = self.state_min[i], self.state_max[i]
        span = hi - lo
        # Below a few units of travel the normalisation is just amplifying
        # encoder noise into imaginary open/close cycles.
        if span < 5.0:
            return
        norm = (value - lo) / span
        if norm > 0.7 and self._gripper_level != "open":
            self.gripper_cycles += self._gripper_level is not None
            self._gripper_level = "open"
        elif norm < 0.3 and self._gripper_level != "closed":
            self.gripper_cycles += self._gripper_level is not None
            self._gripper_level = "closed"

    def _diff_cameras(self, frames: dict) -> None:
        for name, frame in frames.items():
            try:
                thumb = frame[::16, ::16]
                shape = getattr(frame, "shape", None)
                prev = self._cam_prev.get(name)
                self._cam_prev[name] = thumb.copy()
                diff = None
                if prev is not None and prev.shape == thumb.shape:
                    diff = float(abs(thumb.astype("float32") - prev.astype("float32")).mean())
                entry = self.cameras.setdefault(name, {})
                entry["h"] = int(shape[0]) if shape else None
                entry["w"] = int(shape[1]) if shape and len(shape) > 1 else None
                if diff is not None:
                    entry["diff"] = round(diff, 3)
                entry["at"] = time.monotonic()
            except Exception:
                # A metric is never worth taking the recording down for.
                continue

    def fps_actual(self) -> float | None:
        # Half a second of samples. Fewer than that and normal jitter reads as a
        # collapse in throughput.
        if len(self._recent_periods) < max(10, int(self.fps_target // 2)):
            return None
        mean = sum(self._recent_periods) / len(self._recent_periods)
        return round(1.0 / mean, 2) if mean > 0 else None

    def warnings(self, elapsed: float) -> list[dict]:
        out: list[dict] = []

        if self.phase == "record" and elapsed > NO_MOTION_GRACE_S and self.state_min:
            spans = [hi - lo for lo, hi in zip(self.state_min, self.state_max)]
            if spans and max(spans) < STATIC_JOINT_RANGE:
                out.append({
                    "id": "no_motion",
                    "level": "fail",
                    "text": "팔이 움직이지 않습니다 — 텔레옵이 붙어 있는지 확인하세요",
                })
            elif not self.action_seen:
                out.append({
                    "id": "no_action",
                    "level": "fail",
                    "text": "리더에서 명령이 오지 않습니다 (send_action 미호출)",
                })

        # Throughput and camera checks need the loop to have settled. Reporting
        # them from the first frames of a phase produces a burst of warnings at
        # every episode boundary, which teaches people to ignore the panel.
        if elapsed < PHASE_SETTLE_S:
            return out

        fps = self.fps_actual()
        if fps and self.fps_target and fps < self.fps_target * FPS_LOW_RATIO:
            out.append({
                "id": "fps_low",
                "level": "warn",
                "text": f"fps {fps:g} / 목표 {self.fps_target:g} — 프레임이 밀리고 있습니다",
            })

        now = time.monotonic()
        for name, cam in self.cameras.items():
            if now - cam.get("at", 0) > 2.0:
                out.append({"id": f"cam_stale_{name}", "level": "warn",
                            "text": f"{name} 카메라 프레임이 2초 이상 갱신되지 않았습니다"})
            elif cam.get("diff") is not None and cam["diff"] < CAMERA_FROZEN_DIFF:
                out.append({"id": f"cam_frozen_{name}", "level": "warn",
                            "text": f"{name} 카메라 화면이 정지한 것 같습니다 (변화량 {cam['diff']:.2f})"})

        if self.phase == "record" and elapsed > NO_MOTION_GRACE_S and self.track_n > 0:
            spans = [hi - lo for lo, hi in zip(self.state_min, self.state_max)]
            worst, worst_i = 0.0, -1
            for i, (total, span) in enumerate(zip(self.track_err_sum, spans)):
                if span <= STATIC_JOINT_RANGE:
                    continue
                ratio = (total / self.track_n) / span
                if ratio > worst:
                    worst, worst_i = ratio, i
            if worst > TRACKING_WARN_RATIO:
                name = self.joint_names[worst_i] if worst_i < len(self.joint_names) else f"joint {worst_i}"
                out.append({"id": "tracking", "level": "warn",
                            "text": f"{name} 추종 오차가 큽니다 (가동범위의 {worst * 100:.0f}%)"})

        return out

    def snapshot(self) -> dict:
        now = time.monotonic()
        elapsed = now - self.phase_started_at

        # Frames are the honest clock: control_time_s is known and the loop
        # counts frames, so this stays right even if the GUI link stutters.
        if self.fps_target and self.phase in ("record", "reset"):
            elapsed = self.frames / self.fps_target

        phase = self.phase
        stalled = self.last_frame_at and (now - self.last_frame_at) > SAVING_STALL_S
        if phase in ("record", "reset") and stalled:
            # record_loop has returned; we are in save_episode / video encoding.
            phase = "saving"

        spans = [round(hi - lo, 3) for lo, hi in zip(self.state_min, self.state_max)]
        episode_no = None
        if self.episode_index is not None and self.first_episode_index is not None:
            episode_no = self.episode_index - self.first_episode_index + 1

        return {
            "phase": phase,
            "phase_raw": self.phase,
            "phase_msg": self.phase_msg,
            "phase_elapsed_s": round(elapsed, 2),
            "phase_total_s": self.phase_total_s,
            "episode_no": episode_no,
            "episode_index": self.episode_index,
            "episode_target": self.episode_target,
            "rerecords": self.rerecords,
            "frames": self.frames,
            "total_frames": self.total_frames,
            "fps_target": self.fps_target,
            "fps_actual": self.fps_actual(),
            "late_frames": self.late_frames,
            "joint_names": self.joint_names,
            "state": [round(v, 3) for v in self.state],
            "action": [round(v, 3) for v in self.action],
            "state_span": spans,
            "gripper_cycles": self.gripper_cycles,
            "cameras": {
                k: {"w": v.get("w"), "h": v.get("h"), "diff": v.get("diff")}
                for k, v in self.cameras.items()
            },
            "warnings": self.warnings(elapsed),
            "error": self.error,
        }


def _find_gripper(names: list[str]) -> int | None:
    """Locate the gripper by name, never by position.

    The inspector can assume the gripper is the last column because that is the
    order the dataset features are written in. Here the values come from a dict
    that we sort for a stable order -- and on Piper that puts ``gripper.pos``
    *first* ("g" < "j"), so counting the last column would have been counting
    joint 6's open/close cycles.
    """
    for i, name in enumerate(names):
        if "gripper" in name.lower():
            return i
    return len(names) - 1 if names else None


def _split_observation(obs: Any) -> tuple[list[float], list[str], dict]:
    """Separate a LeRobot observation into scalar joint values and image frames.

    Robots return a flat dict mixing ``"joint_1.pos" -> float`` with
    ``"front" -> ndarray``. Sorting keys keeps the joint order stable across
    frames, which the span/tracking maths depends on.
    """
    if not isinstance(obs, dict):
        return [], [], {}
    values: list[float] = []
    names: list[str] = []
    frames: dict = {}
    for key in sorted(obs):
        value = obs[key]
        if hasattr(value, "shape") and getattr(value, "ndim", 0) >= 2:
            frames[key] = value
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            names.append(key)
            values.append(float(value))
    return values, names, frames


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


class Reporter(threading.Thread):
    """Ships snapshots to the GUI and brings commands back on the response.

    The worker has to send status every tick anyway; carrying pending commands
    in the reply means the button path costs no extra requests. Failures are
    swallowed on purpose -- if the GUI dies, the recording keeps going.
    """

    def __init__(self, url: str, token: str, metrics: Metrics, events: dict, hz: float):
        super().__init__(daemon=True)
        self.url = url.rstrip("/") + "/api/worker/tick"
        self.token = token
        self.metrics = metrics
        self.events = events
        self.interval = 1.0 / hz
        self.stop_flag = threading.Event()
        self.seq = 0
        self.failures = 0

    def run(self) -> None:
        while not self.stop_flag.is_set():
            self.tick()
            self.stop_flag.wait(self.interval)

    def tick(self) -> None:
        self.seq += 1
        payload = self.metrics.snapshot()
        payload["seq"] = self.seq
        payload["token"] = self.token
        try:
            body = json.dumps(payload).encode()
        except Exception:
            return
        req = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                reply = json.loads(resp.read() or b"{}")
            self.failures = 0
        except (urllib.error.URLError, OSError, ValueError):
            self.failures += 1
            return

        for cmd in reply.get("cmds", []):
            self.apply(cmd)

    def apply(self, cmd: str) -> None:
        if cmd == "exit_early":
            self.events["exit_early"] = True
        elif cmd == "rerecord":
            self.events["rerecord_episode"] = True
            self.events["exit_early"] = True
        elif cmd == "stop":
            self.events["stop_recording"] = True
            self.events["exit_early"] = True
        else:
            return
        print(f"[worker] command: {cmd}", flush=True)

    def final(self, phase: str, error: str | None = None) -> None:
        self.metrics.phase = phase
        self.metrics.error = error
        self.tick()


# --------------------------------------------------------------------------
# LeRobot plumbing
# --------------------------------------------------------------------------


class NullListener:
    """Stand-in for the pynput listener.

    ``record()`` calls ``listener.stop()`` in its cleanup path, so returning
    ``None`` from the patched ``init_keyboard_listener`` would blow up at the
    very end of a good run -- after the last episode, before the dataset is
    finalised.
    """

    def stop(self) -> None:
        pass

    def join(self, *_args, **_kwargs) -> None:
        pass


def add_plugin_paths(paths: list[str]) -> None:
    """Make uninstalled third-party device packages importable.

    LeRobot 0.6+ registers third-party hardware by scanning sys.path for
    top-level packages named lerobot_robot_* / lerobot_teleoperator_* /
    lerobot_camera_*. That scan only sees what is on sys.path, so a package
    sitting in a working directory is invisible unless we put it there --
    which is why a custom robot type can resolve from one directory and not
    another.
    """
    for raw in paths:
        path = str(Path(raw).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"[worker] sys.path += {path}", flush=True)


PLUGIN_PREFIXES = ("lerobot_robot_", "lerobot_teleoperator_", "lerobot_camera_")


def describe_plugins() -> dict[str, list[str]]:
    """Third-party device packages, both as LeRobot sees them and as they really are.

    LeRobot registers third-party hardware by walking sys.path with
    pkgutil.iter_modules(). That only sees real directories, so a package
    installed with `pip install -e .` (PEP 660, exposed through a meta-path
    finder) is importable but invisible to the scan -- huggingface/lerobot#2460.

    Asking pkgutil alone would just reproduce that bug and report nothing.
    importlib.metadata knows about the editable install, so the gap between the
    two lists is the diagnosis: installed, importable, and still unregistered.
    """
    scanned = {m.name for m in pkgutil.iter_modules() if m.name.startswith(PLUGIN_PREFIXES)}

    installed = set()
    try:
        from importlib import metadata

        for dist in metadata.distributions():
            name = (dist.metadata["Name"] or "").replace("-", "_")
            if name.startswith(PLUGIN_PREFIXES):
                installed.add(name)
    except Exception:
        pass

    return {
        "scanned": sorted(scanned),
        "installed": sorted(installed),
        "hidden": sorted(installed - scanned),
    }


DISCOVER_SUFFIX = "discover_packages_path"


def _is_discover_arg(arg: str) -> bool:
    key, sep, _value = arg.lstrip("-").partition("=")
    return bool(sep) and key.endswith(DISCOVER_SUFFIX)


def _discover_args(argv: list[str]) -> list[str]:
    """Package names given via LeRobot's --<field>.discover_packages_path flags.

    Deduplicated: robot and teleop usually name the same package, and importing
    it twice just prints the plugin's banner twice.
    """
    seen = []
    for arg in argv:
        if _is_discover_arg(arg):
            value = arg.split("=", 1)[1]
            if value and value not in seen:
                seen.append(value)
    return seen


def _known_choices(module_path: str) -> str:
    """The type names LeRobot will accept, straight from its own registry."""
    config_name = "RobotConfig" if "robots" in module_path else "TeleoperatorConfig"
    try:
        module = __import__(module_path, fromlist=[config_name])
        names = sorted(getattr(module, config_name).get_known_choices())
    except Exception as exc:
        return f"(could not list: {type(exc).__name__}: {exc})"
    return ", ".join(names) or "(none)"


def resolve_record_module():
    """Find LeRobot's record module across the layouts we might meet.

    v0.3.x has ``lerobot.record``; main moved it to
    ``lerobot.scripts.lerobot_record``. Trying both means we do not have to know
    the robot PC's version in advance.
    """
    errors = []
    for name in ("lerobot.record", "lerobot.scripts.lerobot_record"):
        try:
            module = __import__(name, fromlist=["record"])
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
            continue
        if hasattr(module, "record") or hasattr(module, "main"):
            return module
        errors.append(f"{name}: no record()/main()")
    raise ImportError("could not locate LeRobot's record module:\n  " + "\n  ".join(errors))


def install_hooks(module, metrics: Metrics, events: dict, cfg: argparse.Namespace) -> None:
    module.init_keyboard_listener = lambda *a, **k: (NullListener(), events)

    original_log_say = getattr(module, "log_say", None)

    def log_say(text, play_sounds=False, blocking=False):
        _on_log_say(str(text), metrics, cfg)
        print(f"[phase] {text}", flush=True)
        if original_log_say is not None and cfg.play_sounds:
            try:
                original_log_say(text, play_sounds, blocking)
            except Exception:
                pass

    module.log_say = log_say

    original_make_robot = getattr(module, "make_robot_from_config", None)
    if original_make_robot is None:
        return

    def make_robot_from_config(robot_cfg):
        robot = original_make_robot(robot_cfg)
        _wrap_robot(robot, metrics)
        return robot

    module.make_robot_from_config = make_robot_from_config


def _on_log_say(text: str, metrics: Metrics, cfg: argparse.Namespace) -> None:
    lowered = text.lower()
    if lowered.startswith("recording episode"):
        index = _trailing_int(text)
        if index is not None:
            metrics.episode_index = index
            if metrics.first_episode_index is None:
                metrics.first_episode_index = index
        metrics.begin_phase("record", text, cfg.episode_time_s)
    elif lowered.startswith("reset the environment"):
        metrics.begin_phase("reset", text, cfg.reset_time_s)
    elif lowered.startswith("re-record"):
        metrics.rerecords += 1
        metrics.begin_phase("rerecording", text, None)
    elif lowered.startswith("stop recording"):
        metrics.begin_phase("stopping", text, None)
    elif lowered.startswith("exiting"):
        metrics.begin_phase("done", text, None)


def _trailing_int(text: str) -> int | None:
    token = text.strip().split()[-1]
    try:
        return int(token)
    except ValueError:
        return None


def _wrap_robot(robot, metrics: Metrics) -> None:
    """Tee get_observation / send_action by shadowing the bound methods.

    Instance attributes win over class methods, so this needs no subclassing and
    works for whatever Piper class the plugin registers. If the class uses
    __slots__ the assignment fails -- recording still runs, we just lose the
    live metrics, which is the right way round to fail.
    """
    for name, wrapper in (("get_observation", _obs_wrapper), ("send_action", _act_wrapper)):
        original = getattr(robot, name, None)
        if original is None:
            continue
        try:
            setattr(robot, name, wrapper(original, metrics))
        except (AttributeError, TypeError) as exc:
            print(f"[worker] could not hook {name}: {exc} (metrics degraded)", flush=True)


def _obs_wrapper(original: Callable, metrics: Metrics) -> Callable:
    def wrapped(*args, **kwargs):
        obs = original(*args, **kwargs)
        try:
            metrics.on_observation(obs)
        except Exception:
            pass
        return obs

    return wrapped


def _act_wrapper(original: Callable, metrics: Metrics) -> Callable:
    def wrapped(action, *args, **kwargs):
        try:
            metrics.on_action(action)
        except Exception:
            pass
        return original(action, *args, **kwargs)

    return wrapped


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(
        description="Run lerobot-record under GUI control",
        epilog="Put LeRobot's own flags after a bare --",
    )
    ap.add_argument("--gui-url", default=None, help="GUI server base URL; omit to run headless")
    ap.add_argument("--token", default="", help="shared secret echoed back to the GUI")
    ap.add_argument("--tick-hz", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=30.0, help="target fps, for the metrics")
    ap.add_argument("--episode-time-s", type=float, default=None)
    ap.add_argument("--reset-time-s", type=float, default=None)
    ap.add_argument("--num-episodes", type=int, default=None)
    ap.add_argument("--play-sounds", action="store_true", help="let LeRobot speak as well")
    ap.add_argument("--plugin-path", action="append", default=[], metavar="DIR",
                    help="prepend DIR to sys.path before importing lerobot; repeatable. "
                         "Use when a third-party robot package (lerobot_robot_*) lives in "
                         "a directory that is not installed, so LeRobot's plugin scan "
                         "cannot see it from here")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the module and validate the flags, then exit")
    known, rest = ap.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    return known, rest


def dry_run(module, lerobot_argv: list[str]) -> int:
    print(f"record module   : {module.__name__}")
    print(f"                  {getattr(module, '__file__', '?')}")
    try:
        import lerobot

        print(f"lerobot version : {getattr(lerobot, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"lerobot version : unavailable ({exc})")

    # The single most useful line when a custom robot will not resolve: whether
    # the type is registered at all, from *this* interpreter and directory.
    plugins = describe_plugins()
    print(f"3rd-party 탐색됨 : {', '.join(plugins['scanned']) or '(없음)'}")
    print(f"3rd-party 설치됨 : {', '.join(plugins['installed']) or '(없음)'}")
    for name in plugins["hidden"]:
        print(f"  ! {name}: 설치돼 있지만 LeRobot 자동 탐색에 안 잡힙니다 "
              f"(editable 설치, lerobot#2460).")
        print(f"    -> --robot.discover_packages_path={name} 를 붙이면 등록됩니다")
    # Load whatever the flags ask for before listing types, the same way
    # record() will -- otherwise the dry run reports the registry as it looks
    # *without* the fix and a working setup still reads as broken.
    for pkg in _discover_args(lerobot_argv):
        try:
            importlib.import_module(pkg)
            print(f"plugin 로드     : {pkg}")
        except Exception as exc:
            print(f"plugin 실패     : {pkg} -- {type(exc).__name__}: {exc}")

    for label, path in (("robot", "lerobot.robots"), ("teleop", "lerobot.teleoperators")):
        print(f"{label + ' types':<16}: {_known_choices(path)}")

    print(f"flags           : {' '.join(lerobot_argv)}")

    config_class = getattr(module, "RecordConfig", None)
    if config_class is None:
        print("config          : RecordConfig not found; cannot validate flags")
        return 0
    try:
        import draccus
    except ImportError:
        # draccus ships with lerobot; without it we can still confirm the module
        # resolved, we just cannot judge the flags. Do not call that a config error.
        print("config          : draccus 없음 — 플래그 검증을 건너뜁니다")
        return 0
    # draccus rejects a bad flag through argparse, which prints a few hundred
    # lines of usage and raises SystemExit -- not an Exception, so it sailed past
    # the handler and buried the one line that says what was wrong. Capture the
    # noise and surface only the verdict.
    # LeRobot's parser.wrap() strips the discover_packages_path flags after
    # loading the plugins and before handing the rest to draccus. Skipping that
    # step here made a correct command look rejected -- a dry run that does not
    # mirror the real path is worse than no dry run.
    parse_argv = [arg for arg in lerobot_argv if not _is_discover_arg(arg)]

    noise = io.StringIO()
    try:
        with contextlib.redirect_stderr(noise), contextlib.redirect_stdout(io.StringIO()):
            cfg = draccus.parse(config_class=config_class, args=parse_argv)
    except SystemExit:
        lines = [line.strip() for line in noise.getvalue().splitlines() if line.strip()]
        verdict = next((line for line in reversed(lines) if "error:" in line),
                       lines[-1] if lines else "reason not reported")
        print(f"config          : REJECTED -- {verdict}")
        return 1
    except Exception as exc:
        print(f"config          : FAILED to parse -- {type(exc).__name__}: {exc}")
        return 1
    print("config          : parsed OK")
    print(f"  robot         : {getattr(cfg.robot, 'type', '?')} @ {getattr(cfg.robot, 'port', '?')}")
    print(f"  teleop        : {getattr(cfg.teleop, 'type', None)} @ {getattr(cfg.teleop, 'port', None)}")
    print(f"  dataset       : {cfg.dataset.repo_id} -> {cfg.dataset.root}")
    print(f"  episodes      : {cfg.dataset.num_episodes} x {cfg.dataset.episode_time_s}s "
          f"(reset {cfg.dataset.reset_time_s}s) @ {cfg.dataset.fps}fps")
    return 0


def main(argv: list[str] | None = None) -> int:
    cfg, lerobot_argv = parse_args(argv if argv is not None else sys.argv[1:])

    # Before anything imports lerobot: its third-party scan reads sys.path once.
    add_plugin_paths(cfg.plugin_path)

    try:
        module = resolve_record_module()
    except ImportError as exc:
        print(f"[worker] {exc}", file=sys.stderr, flush=True)
        return 2

    if cfg.dry_run:
        return dry_run(module, lerobot_argv)

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    metrics = Metrics(fps_target=cfg.fps)
    metrics.episode_target = cfg.num_episodes

    reporter = None
    if cfg.gui_url:
        reporter = Reporter(cfg.gui_url, cfg.token, metrics, events, cfg.tick_hz)
        reporter.start()

    # SIGTERM is the graceful stop, so `kill` from a terminal behaves like the
    # GUI's stop button: finish what is buffered, then wind down cleanly.
    def on_sigterm(_signum, _frame):
        print("[worker] SIGTERM -> stopping after this episode", flush=True)
        events["stop_recording"] = True
        events["exit_early"] = True

    for signame in ("SIGTERM", "SIGINT"):
        handler = getattr(signal, signame, None)
        if handler is not None:
            try:
                signal.signal(handler, on_sigterm)
            except (ValueError, OSError):
                pass

    install_hooks(module, metrics, events, cfg)

    entry = getattr(module, "record", None) or getattr(module, "main")
    saved_argv = sys.argv
    # record() is wrapped in @parser.wrap(), which parses sys.argv itself. Going
    # through the real CLI parser rather than hand-building RecordConfig is what
    # makes custom robot types like piper_follower resolve exactly as they do on
    # the command line.
    sys.argv = ["lerobot-record", *lerobot_argv]
    status, error = "done", None
    try:
        entry()
    except SystemExit as exc:
        if exc.code:
            status, error = "error", f"exited with code {exc.code}"
    except BaseException as exc:  # noqa: BLE001 - report anything before dying
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        sys.argv = saved_argv
        if reporter is not None:
            reporter.final(status, error)
            reporter.stop_flag.set()

    print(f"[worker] finished: {status}" + (f" ({error})" if error else ""), flush=True)
    return 1 if status == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
