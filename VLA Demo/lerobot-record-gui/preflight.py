# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Pre-flight checks for a Piper teleoperation recording session.

Everything here answers one question: *if I press start now, will the next
twenty minutes of recording be worth keeping?* The failures this catches are
the ones that are invisible until the data is already on disk --

* a CAN interface that is down, or up at the wrong bitrate
* ``/dev/video10`` having become ``/dev/video8`` after a reboot, so the front
  camera records the ceiling
* ``--dataset.root`` already existing, so the run either dies at startup or
  quietly appends to a different task's data
* not enough disk for the episodes that were asked for

Runs standalone for a quick terminal check:

    python preflight.py --can can4 can5 --cameras 10 4 --root ~/lerobot_datasets/red_bowl_1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# A frame of 640x480 h264/AV1 at reasonable quality lands well under this, but
# sizing the warning off an optimistic number is how you run out of disk at
# episode 25. Measured LeRobot output sits around 40-90 KB/frame/camera.
BYTES_PER_FRAME_PER_CAMERA = 90_000
PIPER_BITRATE = 1_000_000

IS_LINUX = sys.platform.startswith("linux")


def _check(check_id: str, label: str, level: str, detail: str, **extra) -> dict:
    return {"id": check_id, "label": label, "level": level, "detail": detail, **extra}


def _run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{cmd[0]}: timed out"
    except OSError as exc:
        return 1, "", str(exc)


# --------------------------------------------------------------------------
# CAN
# --------------------------------------------------------------------------


def can_status(iface: str) -> dict:
    """Read one CAN interface's operational state and bitrate."""
    if not IS_LINUX:
        return _check(f"can_{iface}", f"CAN {iface}", "skip",
                      "Linux가 아니라 확인할 수 없습니다", iface=iface)

    code, out, err = _run(["ip", "-details", "link", "show", iface])
    if code != 0:
        return _check(f"can_{iface}", f"CAN {iface}", "fail",
                      f"인터페이스가 없습니다 ({err.strip() or 'ip link 실패'})", iface=iface)

    state = "UNKNOWN"
    match = re.search(r"state (\w+)", out)
    if match:
        state = match.group(1)
    # `ip -details` prints the CAN line as: "can state ERROR-ACTIVE ... bitrate 1000000 ..."
    bitrate = None
    match = re.search(r"\bbitrate (\d+)", out)
    if match:
        bitrate = int(match.group(1))

    is_up = "<NOARP,UP,LOWER_UP" in out or state == "UP" or "UP" in _flags(out)
    if not is_up:
        return _check(f"can_{iface}", f"CAN {iface}", "fail",
                      f"DOWN 상태입니다 (state {state})", iface=iface,
                      up=False, bitrate=bitrate)
    if bitrate is None:
        return _check(f"can_{iface}", f"CAN {iface}", "warn",
                      "UP 이지만 bitrate를 읽지 못했습니다", iface=iface,
                      up=True, bitrate=None)
    if bitrate != PIPER_BITRATE:
        return _check(f"can_{iface}", f"CAN {iface}", "fail",
                      f"bitrate가 {bitrate:,} 입니다 — Piper는 {PIPER_BITRATE:,} 이어야 합니다",
                      iface=iface, up=True, bitrate=bitrate)
    return _check(f"can_{iface}", f"CAN {iface}", "ok",
                  f"UP, bitrate {bitrate:,}", iface=iface, up=True, bitrate=bitrate)


def _flags(ip_output: str) -> str:
    match = re.search(r"<([^>]*)>", ip_output)
    return match.group(1) if match else ""


SUDOERS_HINT = (
    "# /etc/sudoers.d/lerobot-can  (visudo -f 로 편집)\n"
    "%s ALL=(root) NOPASSWD: /usr/sbin/ip link set can* down, "
    "/usr/sbin/ip link set can* type can bitrate *, /usr/sbin/ip link set can* up\n"
)


def bring_up_can(ifaces: list[str], bitrate: int = PIPER_BITRATE) -> dict:
    """down -> set bitrate -> up, for each interface.

    Needs a NOPASSWD sudoers rule; without one ``sudo`` would sit waiting for a
    password nobody can type into a web page, so we force non-interactive mode
    and hand back the rule to paste.
    """
    if not IS_LINUX:
        return {"ok": False, "log": "Linux가 아닙니다", "hint": ""}

    log: list[str] = []
    ok = True
    for iface in ifaces:
        for args in (
            ["link", "set", iface, "down"],
            ["link", "set", iface, "type", "can", "bitrate", str(bitrate)],
            ["link", "set", iface, "up"],
        ):
            cmd = ["sudo", "-n", "ip", *args]
            code, out, err = _run(cmd)
            log.append(f"$ {' '.join(cmd)}")
            if out.strip():
                log.append(out.strip())
            if code != 0:
                ok = False
                log.append(err.strip() or f"exit {code}")
                break

    hint = ""
    if not ok and any("password" in line or "sudo:" in line for line in log):
        hint = SUDOERS_HINT % os.environ.get("USER", "sejong")
    return {"ok": ok, "log": "\n".join(log), "hint": hint}


def run_set_roles(script: str) -> dict:
    path = Path(script).expanduser()
    if not path.exists():
        return {"ok": False, "log": f"파일이 없습니다: {path}", "hint": ""}
    code, out, err = _run([sys.executable, str(path)], timeout=60.0)
    return {
        "ok": code == 0,
        "log": (out + err).strip() or f"exit {code}",
        "hint": "",
        "cwd": str(path.parent),
    }


def set_roles_status(script: str) -> dict:
    path = Path(script).expanduser()
    if not path.exists():
        return _check("set_roles", "역할 설정", "warn",
                      f"set_roles.py 를 찾지 못했습니다: {path}", path=str(path))
    return _check("set_roles", "역할 설정", "info",
                  f"{path} — 리더/팔로워 역할은 CAN을 올린 뒤 한 번 실행해야 합니다",
                  path=str(path))


# --------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------


def list_cameras() -> list[dict]:
    """Enumerate /dev/video* with their kernel-reported names.

    Reading ``/sys/class/video4linux/videoN/name`` avoids needing v4l2-ctl
    installed, and the name is what actually identifies a camera -- the index is
    just whatever order the kernel probed them in this boot.
    """
    if not IS_LINUX:
        return []
    found = []
    for dev in sorted(glob.glob("/dev/video*"), key=_video_index):
        index = _video_index(dev)
        name = ""
        try:
            name = Path(f"/sys/class/video4linux/video{index}/name").read_text().strip()
        except OSError:
            pass
        found.append({"index": index, "device": dev, "name": name,
                      "capture": _is_capture_device(index)})
    return found


def _video_index(dev: str) -> int:
    match = re.search(r"(\d+)$", dev)
    return int(match.group(1)) if match else -1


def _is_capture_device(index: int) -> bool:
    """Every UVC camera exposes a metadata node alongside the capture node.

    Half the /dev/video* entries on a machine with two webcams cannot produce
    frames at all, and telling the user "index 11 exists" when it is a metadata
    node would be worse than saying nothing.
    """
    try:
        modes = Path(f"/sys/class/video4linux/video{index}/index").read_text().strip()
        return modes == "0"
    except OSError:
        return True


def camera_status(wanted: list[int]) -> dict:
    if not IS_LINUX:
        return _check("cameras", "카메라", "skip", "Linux가 아니라 확인할 수 없습니다",
                      devices=[])
    devices = list_cameras()
    if not devices:
        return _check("cameras", "카메라", "fail", "/dev/video* 장치가 없습니다", devices=[])

    available = {d["index"] for d in devices}
    capture = {d["index"] for d in devices if d["capture"]}
    missing = [i for i in wanted if i not in available]
    not_capture = [i for i in wanted if i in available and i not in capture]

    listing = ", ".join(
        f"{d['index']}:{d['name'] or '?'}" for d in devices if d["capture"]
    )
    if missing:
        return _check("cameras", "카메라", "fail",
                      f"인덱스 {missing} 가 없습니다. 사용 가능: {listing}. "
                      f"재부팅하면 인덱스가 바뀝니다 — 설정을 고치세요",
                      devices=devices)
    if not_capture:
        return _check("cameras", "카메라", "warn",
                      f"인덱스 {not_capture} 는 캡처 장치가 아닐 수 있습니다 (메타데이터 노드). "
                      f"사용 가능: {listing}",
                      devices=devices)
    return _check("cameras", "카메라", "ok",
                  f"요청한 인덱스 {wanted} 모두 존재. 사용 가능: {listing}",
                  devices=devices)


# --------------------------------------------------------------------------
# Dataset destination
# --------------------------------------------------------------------------


def dataset_status(root: str | None, resume: bool) -> dict:
    if not root:
        return _check("dataset", "데이터셋 경로", "warn",
                      "--dataset.root 이 비어 있습니다 (기본 캐시 경로에 저장됩니다)")
    path = Path(root).expanduser()
    if not path.exists():
        parent = path.parent
        if not parent.exists():
            return _check("dataset", "데이터셋 경로", "warn",
                          f"상위 폴더가 없습니다: {parent} (수집 시 생성됩니다)", path=str(path))
        return _check("dataset", "데이터셋 경로", "ok", f"새 경로: {path}", path=str(path))

    episodes = _count_episodes(path)
    if resume:
        return _check("dataset", "데이터셋 경로", "ok",
                      f"이어서 수집합니다 — 기존 {episodes} 에피소드에 추가됩니다",
                      path=str(path), existing=episodes)

    # A failed or cancelled start still creates the root, so an empty directory
    # is almost always leftovers rather than data worth protecting.
    if episodes == 0:
        return _check("dataset", "데이터셋 경로", "warn",
                      f"폴더가 이미 있지만 에피소드가 없습니다 — 이전 시도의 잔여물로 보입니다: {path}",
                      path=str(path), existing=0)

    # Not a hard failure: LeRobot decides what to do with an existing root, and
    # it refuses or resumes on its own. Blocking here on a guess only produced a
    # dialog people learned to click through.
    return _check("dataset", "데이터셋 경로", "warn",
                  f"이미 에피소드 {episodes}개가 있습니다. 이어서 찍으려면 resume 를 켜고, "
                  f"별도로 모으려면 다른 이름을 쓰세요",
                  path=str(path), existing=episodes)


def _count_episodes(root: Path) -> int:
    info = root / "meta" / "info.json"
    try:
        return int(json.loads(info.read_text(encoding="utf-8")).get("total_episodes", 0))
    except (OSError, ValueError, TypeError):
        pass
    return len(list((root / "data").rglob("*.parquet"))) if (root / "data").is_dir() else 0


def disk_status(root: str | None, num_episodes: int, episode_time_s: float,
                fps: float, num_cameras: int) -> dict:
    target = Path(root).expanduser() if root else Path.home()
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return _check("disk", "디스크", "warn", f"확인 실패: {exc}")

    frames = num_episodes * episode_time_s * fps
    need = int(frames * max(num_cameras, 1) * BYTES_PER_FRAME_PER_CAMERA)
    free = usage.free
    detail = (f"여유 {_gb(free)} / 예상 사용량 {_gb(need)} "
              f"({num_episodes}회 x {episode_time_s:g}초 x 카메라 {num_cameras}대)")
    if free < need:
        return _check("disk", "디스크", "fail", detail + " — 공간이 부족합니다",
                      free=free, need=need)
    if free < need * 3:
        return _check("disk", "디스크", "warn", detail + " — 여유가 빠듯합니다",
                      free=free, need=need)
    return _check("disk", "디스크", "ok", detail, free=free, need=need)


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


# --------------------------------------------------------------------------
# LeRobot install
# --------------------------------------------------------------------------

_PROBE = (
    "import json,sys\n"
    "out={'python':sys.executable}\n"
    "try:\n"
    "    import lerobot; out['version']=getattr(lerobot,'__version__','unknown')\n"
    "except Exception as e:\n"
    "    out['error']=f'{type(e).__name__}: {e}'; print(json.dumps(out)); sys.exit(0)\n"
    "for name in ('lerobot.record','lerobot.scripts.lerobot_record'):\n"
    "    try:\n"
    "        m=__import__(name,fromlist=['record'])\n"
    "    except ImportError:\n"
    "        continue\n"
    "    if hasattr(m,'record') or hasattr(m,'main'):\n"
    "        out['module']=name; out['file']=getattr(m,'__file__','?'); break\n"
    "print(json.dumps(out))\n"
)


def lerobot_status(python: str | None = None) -> dict:
    interpreter = python or sys.executable
    code, out, err = _run([interpreter, "-c", _PROBE], timeout=90.0)
    if code != 0:
        return _check("lerobot", "LeRobot", "fail",
                      f"확인 실패: {(err or out).strip()[:300]}")
    try:
        info = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return _check("lerobot", "LeRobot", "fail", f"응답을 읽지 못했습니다: {out[:200]}")

    if "error" in info:
        return _check("lerobot", "LeRobot", "fail",
                      f"import 실패 — {info['error']}. venv 안에서 서버를 띄웠는지 확인하세요",
                      python=interpreter)
    if "module" not in info:
        return _check("lerobot", "LeRobot", "fail",
                      f"v{info.get('version')} 이지만 record 모듈을 찾지 못했습니다",
                      python=interpreter)
    return _check("lerobot", "LeRobot", "ok",
                  f"v{info.get('version')} — {info['module']}",
                  python=interpreter, module=info["module"], file=info.get("file"))


# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------


def run_all(*, can_ifaces: list[str], camera_indices: list[int], root: str | None,
            resume: bool, num_episodes: int, episode_time_s: float, fps: float,
            set_roles_path: str, python: str | None = None,
            demo: bool = False) -> dict:
    # Demo mode drives fake_worker.py, which never imports lerobot -- reporting
    # it missing would train people to click through a red board.
    checks = [] if demo else [lerobot_status(python)]
    checks += [can_status(iface) for iface in can_ifaces]
    checks.append(set_roles_status(set_roles_path))
    checks.append(camera_status(camera_indices))
    checks.append(dataset_status(root, resume))
    checks.append(disk_status(root, num_episodes, episode_time_s, fps, len(camera_indices)))

    worst = "ok"
    for check in checks:
        if check["level"] == "fail":
            worst = "fail"
            break
        if check["level"] == "warn":
            worst = "warn"
    return {"level": worst, "checks": checks, "platform": sys.platform}


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-flight checks for LeRobot recording")
    ap.add_argument("--can", nargs="*", default=["can4", "can5"])
    ap.add_argument("--cameras", nargs="*", type=int, default=[])
    ap.add_argument("--root", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--num-episodes", type=int, default=30)
    ap.add_argument("--episode-time-s", type=float, default=20)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--set-roles",
                    default="~/lerobot_teleop/piper_robot_source/set_roles.py")
    args = ap.parse_args()

    result = run_all(
        can_ifaces=args.can, camera_indices=args.cameras, root=args.root,
        resume=args.resume, num_episodes=args.num_episodes,
        episode_time_s=args.episode_time_s, fps=args.fps,
        set_roles_path=args.set_roles,
    )
    mark = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL", "skip": "--  ", "info": "..  "}
    for check in result["checks"]:
        print(f"[{mark.get(check['level'], '?')}] {check['label']:<14} {check['detail']}")
    print(f"\noverall: {result['level']}")
    return 1 if result["level"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
