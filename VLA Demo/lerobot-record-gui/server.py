# /// script
# requires-python = ">=3.10"
# dependencies = ["fastapi", "uvicorn", "numpy"]
# ///
"""
LeRobot Record GUI
==================

A control panel for Piper teleoperation data collection. Run it on the robot PC
and open the browser anywhere on the LAN:

    python server.py --host 0.0.0.0 --port 8010

Replaces this, typed by hand every session::

    lerobot-record --robot.type=piper_follower --robot.port=can5 ... (12 more flags)

...and, more to the point, replaces *reading a scrolling terminal while holding
the leader arm with both hands* to find out whether you are recording or
resetting.

Shape
-----
The recording itself runs in a child process (``recorder_worker.py``), not a
thread. Real hardware hangs -- a blocked CAN read inside a thread cannot be
killed, and this GUI has a force-quit button that has to mean something. The
child also keeps LeRobot's monkeypatched module state out of the server.

The child POSTs its status to ``/api/worker/tick`` about ten times a second, and
the *response* to that POST carries any pending button presses back down. The
child has to send status anyway, so control costs no extra round trips.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import preflight

HERE = Path(__file__).parent
STATIC = HERE / "static"
PRESETS = HERE / "presets"

LOG_LINES = 500
STALE_TICK_S = 3.0  # no tick for this long means we lost the worker


# --------------------------------------------------------------------------
# Turning the form into a command line
# --------------------------------------------------------------------------

# Only these become flags. The form is structured JSON rather than free text and
# the child is spawned with a list (never a shell), so nothing a browser sends
# can turn into a command -- this server is meant to be bound to 0.0.0.0 on a
# lab network and must not be a remote execution hole.
DATASET_FIELDS = {
    "repo_id": str,
    "root": str,
    "single_task": str,
    "fps": int,
    "episode_time_s": float,
    "reset_time_s": float,
    "num_episodes": int,
    "video": bool,
    "push_to_hub": bool,
    "private": bool,
    "num_image_writer_processes": int,
    "num_image_writer_threads_per_camera": int,
    "video_encoding_batch_size": int,
}
# discover_packages_path is LeRobot's own plugin hook: it imports that package
# before parsing, which is how a third-party robot type like piper_follower gets
# registered. Leaving it out of the whitelist silently dropped the one flag that
# makes a custom robot resolvable.
DEVICE_FIELDS = {"type": str, "port": str, "id": str, "discover_packages_path": str}
CAMERA_FIELDS = {"type": str, "index_or_path": str, "width": int, "height": int, "fps": int}
TOP_FIELDS = {"resume": bool, "play_sounds": bool, "display_data": bool}

IDENT = re.compile(r"^[A-Za-z0-9_.:/\\-]+$")


class ConfigError(ValueError):
    pass


def _coerce(value, kind, where: str):
    if kind is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ConfigError(f"{where}: true/false 여야 합니다")
    if kind in (int, float):
        try:
            return kind(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: 숫자여야 합니다 (받은 값: {value!r})") from None
    text = str(value)
    if "\n" in text or "\x00" in text:
        raise ConfigError(f"{where}: 줄바꿈을 넣을 수 없습니다")
    return text


def build_argv(form: dict) -> list[str]:
    """Form -> the exact flag list ``lerobot-record`` would have been given."""
    argv: list[str] = []

    for section in ("robot", "teleop"):
        block = form.get(section) or {}
        if section == "teleop" and not block.get("type"):
            continue
        if section == "robot" and not block.get("type"):
            raise ConfigError("robot.type 은 필수입니다")
        for key, kind in DEVICE_FIELDS.items():
            if not block.get(key):
                continue
            value = _coerce(block[key], kind, f"{section}.{key}")
            if not IDENT.match(value):
                raise ConfigError(f"{section}.{key}: 허용되지 않는 문자가 있습니다")
            argv.append(f"--{section}.{key}={value}")

    cameras = form.get("cameras") or []
    if cameras:
        argv.append(f"--robot.cameras={_cameras_literal(cameras)}")

    dataset = form.get("dataset") or {}
    for key in ("repo_id", "single_task"):
        if not dataset.get(key):
            raise ConfigError(f"dataset.{key} 은 필수입니다")
    for key, kind in DATASET_FIELDS.items():
        if key not in dataset or dataset[key] in ("", None):
            continue
        value = _coerce(dataset[key], kind, f"dataset.{key}")
        argv.append(f"--dataset.{key}={_render(value)}")

    for key, kind in TOP_FIELDS.items():
        if key not in form or form[key] in ("", None):
            continue
        argv.append(f"--{key}={_render(_coerce(form[key], kind, key))}")

    return argv


def _cameras_literal(cameras: list[dict]) -> str:
    """Render the cameras block the way the CLI takes it.

    LeRobot parses this as inline YAML, e.g.
    ``{front: {type: opencv, index_or_path: 10, width: 640, ...}}``.
    """
    parts = []
    for cam in cameras:
        name = str(cam.get("name") or "").strip()
        if not name or not IDENT.match(name):
            raise ConfigError(f"카메라 이름이 올바르지 않습니다: {name!r}")
        fields = []
        for key, kind in CAMERA_FIELDS.items():
            if cam.get(key) in ("", None):
                continue
            value = _coerce(cam[key], kind, f"cameras.{name}.{key}")
            if key == "index_or_path":
                if not IDENT.match(value):
                    raise ConfigError(f"cameras.{name}.index_or_path: 허용되지 않는 문자")
                # A bare number must stay a number; a device path stays a string.
                fields.append(f"{key}: {value}")
            else:
                fields.append(f"{key}: {value}")
        if not fields:
            raise ConfigError(f"카메라 {name} 에 설정이 없습니다")
        parts.append(f"{name}: {{{', '.join(fields)}}}")
    return "{" + ", ".join(parts) + "}"


def _render(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    # 20, not 20.0. LeRobot types these as `int | float`, so both parse, but the
    # command preview is meant to be something you could paste into a terminal
    # and recognise as the one you have always run.
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def preview_command(argv: list[str]) -> str:
    """The command line as a human would type it, for the form's preview."""
    out = ["lerobot-record"]
    for flag in argv:
        key, _, value = flag.partition("=")
        out.append(f"{key}={value}" if IDENT.match(value) else f"{key}='{value}'")
    return " \\\n    ".join(out)


# --------------------------------------------------------------------------
# Run state
# --------------------------------------------------------------------------


class RunState:
    """Everything about the current (or last) recording run.

    One lock guards the whole thing. Updates arrive from the tick endpoint, the
    log reader thread and the process watcher; readers are the SSE loop and the
    REST handlers.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.proc: subprocess.Popen | None = None
        self.token = ""
        self.form: dict = {}
        self.argv: list[str] = []
        self.command_line = ""
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.exit_code: int | None = None
        self.killed = False
        self.tick: dict = {}
        self.tick_at: float = 0.0
        self.commands: deque[str] = deque()
        self.log: deque[str] = deque(maxlen=LOG_LINES)
        self.log_seq = 0
        self.run_id = 0
        self.version = 0

    # -- lifecycle ---------------------------------------------------------

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, proc: subprocess.Popen, token: str, form: dict, argv: list[str]) -> None:
        with self.lock:
            self.proc = proc
            self.token = token
            self.form = form
            self.argv = argv
            self.command_line = preview_command(argv)
            self.started_at = time.time()
            self.finished_at = None
            self.exit_code = None
            self.killed = False
            self.tick = {"phase": "starting"}
            self.tick_at = time.time()
            self.commands.clear()
            self.log.clear()
            # log_seq deliberately keeps counting. Clients track "lines I have
            # already seen" by sequence number, so restarting it at 0 would make
            # a second run's output look like output they had already read, and
            # the log panel would sit empty exactly when something went wrong.
            # run_id is what tells them a new run began.
            self.run_id += 1
            self.version += 1

    def finish(self, code: int) -> None:
        with self.lock:
            self.exit_code = code
            self.finished_at = time.time()
            self.version += 1

    # -- updates -----------------------------------------------------------

    def add_log(self, line: str) -> None:
        with self.lock:
            self.log_seq += 1
            self.log.append(f"{self.log_seq}\t{line}")
            self.version += 1

    def update_tick(self, payload: dict) -> list[str]:
        with self.lock:
            payload.pop("token", None)
            self.tick = payload
            self.tick_at = time.time()
            self.version += 1
            cmds = list(self.commands)
            self.commands.clear()
            return cmds

    def queue(self, cmd: str) -> None:
        with self.lock:
            if cmd not in self.commands:
                self.commands.append(cmd)
            self.version += 1

    # -- reads -------------------------------------------------------------

    def logs_since(self, cursor: int) -> tuple[list[str], int]:
        with self.lock:
            out = []
            newest = cursor
            for entry in self.log:
                seq_text, _, line = entry.partition("\t")
                seq = int(seq_text)
                if seq > cursor:
                    out.append(line)
                    newest = max(newest, seq)
            return out, newest

    def snapshot(self) -> dict:
        with self.lock:
            running = self.running()
            tick = dict(self.tick)
            stale = running and self.tick_at and (time.time() - self.tick_at) > STALE_TICK_S
            if not running and self.exit_code is not None:
                # A non-zero exit after the user pressed force-quit is the button
                # working, not a fault. Saying "오류" there would send people
                # hunting through the log for a problem they created on purpose.
                if self.killed:
                    tick["phase"] = "killed"
                    tick["error"] = ("강제 종료했습니다 — 저장 중이던 에피소드는 "
                                     "깨졌을 수 있습니다. 검수 후 사용하세요")
                else:
                    tick.setdefault("phase", "error" if self.exit_code else "done")
                    if self.exit_code and not tick.get("error"):
                        tick["error"] = f"워커가 코드 {self.exit_code} 로 종료했습니다"
                    if tick.get("phase") in ("record", "reset", "saving", "starting"):
                        tick["phase"] = "error" if self.exit_code else "done"
            return {
                "running": running,
                "run_id": self.run_id,
                "stale": bool(stale),
                "pid": self.proc.pid if self.proc else None,
                "exit_code": self.exit_code,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "command_line": self.command_line,
                "form": self.form,
                "tick": tick,
                "version": self.version,
            }


# --------------------------------------------------------------------------
# Process control
# --------------------------------------------------------------------------


def spawn(script: Path, gui_url: str, token: str, form: dict, argv: list[str],
          python: str, speed: float = 1.0, workdir: str | None = None,
          plugin_paths: list[str] | None = None) -> subprocess.Popen:
    dataset = form.get("dataset") or {}
    cmd = [
        python, "-u", str(script),
        "--gui-url", gui_url,
        "--token", token,
        "--fps", str(dataset.get("fps", 30)),
        "--episode-time-s", str(dataset.get("episode_time_s", 60)),
        "--reset-time-s", str(dataset.get("reset_time_s", 60)),
        "--num-episodes", str(dataset.get("num_episodes", 50)),
    ]
    if form.get("play_sounds"):
        cmd.append("--play-sounds")
    if speed and speed != 1.0:
        cmd += ["--speed", str(speed)]
    for path in plugin_paths or []:
        cmd += ["--plugin-path", path]
    cmd += ["--", *argv]

    kwargs: dict = {}
    if os.name == "posix":
        # Own process group, so force-quit takes the image-writer subprocesses
        # with it instead of orphaning them onto the CAN bus.
        kwargs["start_new_session"] = True
    else:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    # Run where the operator normally runs lerobot-record. LeRobot's plugin scan
    # walks sys.path, which includes the working directory -- so a custom robot
    # type can resolve from one directory and not another.
    cwd = str(Path(workdir).expanduser()) if workdir else str(HERE)

    return subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, **kwargs,
    )


def kill_tree(proc: subprocess.Popen, sig: int) -> None:
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill() if sig == getattr(signal, "SIGKILL", 9) else proc.terminate()
    except OSError:
        pass


def pump_output(proc: subprocess.Popen, state: RunState) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        state.add_log(line.rstrip("\n"))
    proc.stdout.close()


def watch(proc: subprocess.Popen, state: RunState) -> None:
    code = proc.wait()
    state.add_log(f"[gui] 워커 종료 (exit {code})")
    state.finish(code)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


def build_app(state: RunState, opts: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="LeRobot Record GUI")
    worker = HERE / ("fake_worker.py" if opts.demo else "recorder_worker.py")
    gui_url = f"http://127.0.0.1:{opts.port}"

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    def config():
        return {
            "demo": opts.demo,
            "worker": str(worker),
            "python": opts.python,
            "defaults": _defaults(opts),
            "inspector_url": opts.inspector_url,
        }

    # -- run control -------------------------------------------------------

    @app.post("/api/start")
    async def start(request: Request):
        if state.running():
            raise HTTPException(409, "이미 실행 중입니다")
        form = await request.json()
        try:
            argv = build_argv(form)
        except ConfigError as exc:
            raise HTTPException(400, str(exc)) from None

        token = secrets.token_urlsafe(16)
        try:
            speed = opts.demo_speed if opts.demo else 1.0
            proc = spawn(worker, gui_url, token, form, argv, opts.python, speed,
                         opts.workdir, opts.plugin_path)
        except OSError as exc:
            raise HTTPException(500, f"워커를 실행하지 못했습니다: {exc}") from None

        state.start(proc, token, form, argv)
        state.add_log(f"[gui] {preview_command(argv)}")
        threading.Thread(target=pump_output, args=(proc, state), daemon=True).start()
        threading.Thread(target=watch, args=(proc, state), daemon=True).start()
        return {"ok": True, "pid": proc.pid, "command_line": state.command_line}

    @app.post("/api/command")
    async def command(request: Request):
        body = await request.json()
        cmd = str(body.get("command", ""))
        if cmd not in ("exit_early", "rerecord", "stop"):
            raise HTTPException(400, f"알 수 없는 명령: {cmd}")
        if not state.running():
            raise HTTPException(409, "실행 중이 아닙니다")
        state.queue(cmd)
        state.add_log(f"[gui] 명령 전송: {cmd}")
        return {"ok": True}

    @app.post("/api/kill")
    def kill():
        proc = state.proc
        if proc is None or proc.poll() is not None:
            raise HTTPException(409, "실행 중이 아닙니다")
        state.add_log("[gui] 강제 종료 (SIGKILL)")
        state.killed = True
        kill_tree(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
        return {"ok": True}

    @app.post("/api/preview")
    async def preview(request: Request):
        form = await request.json()
        try:
            return {"command_line": preview_command(build_argv(form))}
        except ConfigError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    # -- worker channel ----------------------------------------------------

    @app.post("/api/worker/tick")
    async def worker_tick(request: Request):
        host = request.client.host if request.client else ""
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise HTTPException(403, "worker channel is loopback only")
        payload = await request.json()
        if not state.token or payload.get("token") != state.token:
            raise HTTPException(403, "bad token")
        return {"cmds": state.update_tick(payload)}

    # -- streaming ---------------------------------------------------------

    @app.get("/api/events")
    async def events(request: Request):
        async def stream():
            cursor = 0
            last_version = -1
            last_send = 0.0
            while True:
                if await request.is_disconnected():
                    return
                lines, cursor = state.logs_since(cursor)
                now = time.time()
                # Push on change, plus a floor of 1 Hz so a paused run still
                # proves the link is alive and clients can detect a dead server.
                if state.version != last_version or lines or now - last_send > 1.0:
                    last_version = state.version
                    last_send = now
                    payload = state.snapshot()
                    payload["log"] = lines
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.1)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })

    @app.get("/api/status")
    def status():
        payload = state.snapshot()
        payload["log"] = state.logs_since(0)[0]
        return payload

    # -- preflight ---------------------------------------------------------

    @app.post("/api/preflight")
    async def run_preflight(request: Request):
        form = await request.json()
        dataset = form.get("dataset") or {}
        cameras = form.get("cameras") or []
        indices = []
        for cam in cameras:
            raw = str(cam.get("index_or_path", ""))
            if raw.isdigit():
                indices.append(int(raw))
        ifaces = [
            (form.get(section) or {}).get("port")
            for section in ("robot", "teleop")
        ]
        ifaces = [i for i in ifaces if i and i.startswith("can")]
        return preflight.run_all(
            can_ifaces=ifaces or ["can4", "can5"],
            camera_indices=indices,
            root=dataset.get("root"),
            resume=bool(form.get("resume")),
            num_episodes=int(dataset.get("num_episodes") or 0),
            episode_time_s=float(dataset.get("episode_time_s") or 0),
            fps=float(dataset.get("fps") or 30),
            set_roles_path=form.get("set_roles_path") or opts.set_roles,
            python=opts.python,
            demo=opts.demo,
        )

    @app.post("/api/preflight/can")
    async def preflight_can(request: Request):
        if state.running():
            raise HTTPException(409, "녹화 중에는 CAN을 건드릴 수 없습니다")
        body = await request.json()
        ifaces = [i for i in (body.get("ifaces") or []) if re.fullmatch(r"can\d+", str(i))]
        if not ifaces:
            raise HTTPException(400, "can0 같은 인터페이스 이름이 필요합니다")
        return preflight.bring_up_can(ifaces)

    @app.post("/api/preflight/roles")
    async def preflight_roles(request: Request):
        if state.running():
            raise HTTPException(409, "녹화 중에는 실행할 수 없습니다")
        body = await request.json()
        return preflight.run_set_roles(body.get("path") or opts.set_roles)

    # -- presets -----------------------------------------------------------

    @app.get("/api/presets")
    def list_presets():
        PRESETS.mkdir(exist_ok=True)
        return sorted(p.stem for p in PRESETS.glob("*.json"))

    @app.get("/api/presets/{name}")
    def read_preset(name: str):
        path = _preset_path(name)
        if not path.exists():
            raise HTTPException(404, "없는 프리셋입니다")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/api/presets/{name}")
    async def write_preset(name: str, request: Request):
        path = _preset_path(name)
        PRESETS.mkdir(exist_ok=True)
        body = await request.json()
        path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"ok": True}

    @app.delete("/api/presets/{name}")
    def delete_preset(name: str):
        _preset_path(name).unlink(missing_ok=True)
        return {"ok": True}

    return app


def _preset_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_\-가-힣 ]{1,60}", name):
        raise HTTPException(400, "프리셋 이름에 쓸 수 없는 문자가 있습니다")
    return PRESETS / f"{name}.json"


def _defaults(opts: argparse.Namespace) -> dict:
    """The session the lab actually runs, pre-filled."""
    return {
        "robot": {"type": "piper_follower", "port": "can5", "id": "piper_follower"},
        "teleop": {"type": "piper_leader", "port": "can4", "id": "piper_leader"},
        "cameras": [
            {"name": "front", "type": "opencv", "index_or_path": "10",
             "width": 640, "height": 480, "fps": 30},
            {"name": "wrist", "type": "opencv", "index_or_path": "4",
             "width": 640, "height": 480, "fps": 30},
        ],
        "dataset": {
            "repo_id": "sejong/red_bowl_1",
            "root": str(Path(opts.dataset_dir).expanduser() / "red_bowl_1"),
            "single_task": "Pick up the red cube and place it in the bowl",
            "fps": 30,
            "num_episodes": 30,
            "episode_time_s": 20,
            "reset_time_s": 20,
            "push_to_hub": False,
        },
        "resume": False,
        "play_sounds": False,
        "set_roles_path": opts.set_roles,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="LeRobot recording control panel")
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 0.0.0.0 to reach it from another machine on the LAN")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--demo", action="store_true",
                    help="drive fake_worker.py instead of real hardware")
    ap.add_argument("--demo-speed", type=float, default=1.0,
                    help="fast-forward the simulated run (demo mode only)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the worker (must have lerobot installed)")
    ap.add_argument("--workdir", default=None, metavar="DIR",
                    help="run the worker from DIR -- use the directory you normally "
                         "run lerobot-record from, since LeRobot's third-party device "
                         "scan reads sys.path and that includes the working directory")
    ap.add_argument("--plugin-path", action="append", default=[], metavar="DIR",
                    help="prepend DIR to the worker's sys.path so an uninstalled "
                         "lerobot_robot_* package becomes importable; repeatable")
    ap.add_argument("--dataset-dir", default="~/lerobot_datasets")
    ap.add_argument("--set-roles",
                    default="~/lerobot_teleop/piper_robot_source/set_roles.py")
    ap.add_argument("--inspector-url", default="",
                    help="episode inspector URL, linked from the summary")
    opts = ap.parse_args()

    state = RunState()
    app = build_app(state, opts)

    shown = "localhost" if opts.host in ("127.0.0.1", "0.0.0.0") else opts.host
    print(f"worker   : {'fake_worker.py (DEMO)' if opts.demo else 'recorder_worker.py'}")
    print(f"python   : {opts.python}")
    print(f"\n  ->  http://{shown}:{opts.port}\n")

    import uvicorn

    uvicorn.run(app, host=opts.host, port=opts.port, log_level="warning")


if __name__ == "__main__":
    main()
