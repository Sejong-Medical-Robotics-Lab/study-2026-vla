# /// script
# requires-python = ">=3.9"
# dependencies = ["fastapi", "uvicorn", "pandas", "pyarrow", "numpy"]
# ///
"""
LeRobot Episode Inspector
=========================

A dataset-QC GUI for LeRobot datasets (v2.0 / v2.1 / v3.0 layouts, and bare
parquet files with no meta/ directory).

Run it *on the machine that holds the data* (e.g. the Jetson) and open the
browser on any machine on the same network:

    uv run server.py --data /path/to/dataset --host 0.0.0.0 --port 8000

Design notes
------------
* Videos are streamed with HTTP Range requests, so scrubbing only pulls the
  bytes it needs -- no need to copy episodes off the robot.
* v3.0 packs several episodes into one mp4, so each episode carries a
  (from_timestamp, to_timestamp) window into its video file. The frontend
  offsets playback by that window.
* Review verdicts are written to a JSON sidecar so bad episodes can be
  filtered out before finetuning.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

import mp4probe

# --------------------------------------------------------------------------
# QC thresholds
# --------------------------------------------------------------------------

GAP_TOLERANCE = 1.5  # a frame gap > this * (1/fps) counts as a dropped frame
# Motion thresholds are in the units of observation.state (degrees for Piper).
# They must sit above encoder noise: a 1e-4 tolerance reads sensor jitter on a
# parked arm as movement and lets dead takes through.
STATIC_JOINT_RANGE = 1.0  # a joint spanning less than this never really moved
MOVE_EPS = 0.5  # displacement over MOVE_WINDOW_S that counts as moving
MOVE_WINDOW_S = 0.25  # look at travel over a window, not frame to frame
FROZEN_EDGE_SECONDS = 1.0  # idle head/tail longer than this is worth flagging
ACTIVE_FRACTION_MIN = 0.2  # motion must span at least this share of the episode
TRACKING_ABS_FLOOR = 0.5  # ignore tracking error below this; it is noise
JUMP_FRACTION = 0.15  # a one-frame move over this share of a joint's travel is a glitch
JUMP_P99_FACTOR = 8.0  # ...and it must also stand out from the joint's own p99 speed
SHORT_EPISODE_RATIO = 0.5  # shorter than this * median length -> flagged
LONG_EPISODE_RATIO = 2.0


# --------------------------------------------------------------------------
# Dataset discovery
# --------------------------------------------------------------------------


@dataclass
class VideoRef:
    key: str
    path: Path
    t0: float = 0.0  # start of this episode inside the file
    t1: Optional[float] = None  # end of this episode inside the file


@dataclass
class Episode:
    index: int
    length: int
    parquet: Path
    tasks: List[str] = field(default_factory=list)
    videos: Dict[str, VideoRef] = field(default_factory=dict)

    @property
    def instruction(self) -> str:
        return self.tasks[0] if self.tasks else ""


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    except Exception:
        pass
    return rows


class Dataset:
    """Loads a LeRobot dataset in whatever layout it happens to be in."""

    def __init__(self, target: Path, review_path: Optional[Path] = None):
        target = target.expanduser().resolve()
        if not target.exists():
            raise FileNotFoundError(f"not found: {target}")

        self.target = target
        if target.is_file():
            if target.suffix != ".parquet":
                raise ValueError(f"expected a directory or a .parquet file, got {target}")
            self.data_files = [target]
            self.root = self._find_root_above(target)
        else:
            self.root = target
            self.data_files = self._find_data_files(self.root)

        if not self.data_files:
            raise FileNotFoundError(f"no parquet data files under {self.root}")

        self.info: dict = _read_json(self.root / "meta" / "info.json") or {}
        self.features: dict = self.info.get("features", {}) or {}
        self.codebase_version: str = self.info.get("codebase_version", "unknown")
        self.robot_type: str = self.info.get("robot_type", "unknown")

        self._frame_cache: Dict[Path, pd.DataFrame] = {}
        self._cache_lock = threading.Lock()

        self.tasks_by_index: Dict[int, str] = self._load_tasks()
        self.episodes: Dict[int, Episode] = self._load_episodes()
        self.fps: float = self._resolve_fps()
        self.video_keys: List[str] = self._resolve_video_keys()
        self.joint_names: Dict[str, List[str]] = self._resolve_joint_names()

        self.review_path = (
            review_path.expanduser().resolve()
            if review_path
            else self.root / "episode_review.json"
        )
        self.reviews: Dict[str, dict] = _read_json(self.review_path) or {}

        self._qc_cache: Dict[int, dict] = {}
        self._median_length = float(
            np.median([e.length for e in self.episodes.values()]) if self.episodes else 0.0
        )

    # -- layout ------------------------------------------------------------

    @staticmethod
    def _find_root_above(parquet: Path) -> Path:
        """Walk up from a data file looking for the directory holding meta/."""
        for candidate in list(parquet.parents)[:4]:
            if (candidate / "meta" / "info.json").exists():
                return candidate
        return parquet.parent

    @staticmethod
    def _find_data_files(root: Path) -> List[Path]:
        data_dir = root / "data"
        if data_dir.is_dir():
            files = sorted(data_dir.rglob("*.parquet"))
            if files:
                return files
        # Bare directory of parquet files: take everything except meta/.
        return sorted(p for p in root.rglob("*.parquet") if "meta" not in p.parts)

    # -- meta --------------------------------------------------------------

    def _load_tasks(self) -> Dict[int, str]:
        meta = self.root / "meta"

        # v3.0: meta/tasks.parquet, indexed by the task string.
        tasks_parquet = meta / "tasks.parquet"
        if tasks_parquet.exists():
            try:
                df = pd.read_parquet(tasks_parquet)
                if "task_index" in df.columns:
                    if "task" in df.columns:
                        return {int(r.task_index): str(r.task) for r in df.itertuples()}
                    # The task string lives in the index.
                    return {
                        int(ti): str(task)
                        for task, ti in zip(df.index.tolist(), df["task_index"].tolist())
                    }
            except Exception:
                pass

        # v2.x: meta/tasks.jsonl
        rows = _read_jsonl(meta / "tasks.jsonl")
        if rows:
            return {int(r["task_index"]): str(r.get("task", "")) for r in rows if "task_index" in r}

        return {}

    def _episode_meta_rows(self) -> List[dict]:
        meta = self.root / "meta"

        # v3.0: meta/episodes/chunk-XXX/file-XXX.parquet
        ep_dir = meta / "episodes"
        if ep_dir.is_dir():
            frames = []
            for f in sorted(ep_dir.rglob("*.parquet")):
                try:
                    frames.append(pd.read_parquet(f))
                except Exception:
                    continue
            if frames:
                return pd.concat(frames, ignore_index=True).to_dict("records")

        # v2.x: meta/episodes.jsonl
        return _read_jsonl(meta / "episodes.jsonl")

    def _load_episodes(self) -> Dict[int, Episode]:
        # Ground truth for which episodes exist and how long they are always
        # comes from the data files -- meta can be stale or missing.
        counts: Dict[int, Tuple[Path, int]] = {}
        task_index_by_ep: Dict[int, int] = {}
        for f in self.data_files:
            try:
                df = pd.read_parquet(f, columns=["episode_index", "task_index"])
            except Exception:
                df = pd.read_parquet(f, columns=["episode_index"])
                df["task_index"] = -1
            for ep, group in df.groupby("episode_index"):
                ep = int(ep)
                if ep in counts:
                    # episode_index is global in v3.0, so this means the data
                    # files overlap -- a partial re-record or a bad merge.
                    print(f"WARNING: episode {ep} appears in both "
                          f"{counts[ep][0].name} and {f.name}; using {f.name}")
                counts[ep] = (f, int(len(group)))
                ti = int(group["task_index"].iloc[0])
                if ti >= 0:
                    task_index_by_ep[ep] = ti

        episodes = {
            ep: Episode(index=ep, length=n, parquet=path) for ep, (path, n) in counts.items()
        }

        # Instructions: prefer meta/episodes (authoritative, allows >1 task),
        # fall back to task_index -> meta/tasks.
        meta_rows = {}
        for row in self._episode_meta_rows():
            if "episode_index" in row:
                meta_rows[int(row["episode_index"])] = row

        for ep, episode in episodes.items():
            row = meta_rows.get(ep, {})
            tasks = row.get("tasks")
            if tasks is None:
                tasks = row.get("task")
            if isinstance(tasks, str):
                tasks = [tasks]
            elif isinstance(tasks, np.ndarray):
                tasks = tasks.tolist()
            if tasks:
                episode.tasks = [str(t) for t in tasks]
            elif ep in task_index_by_ep:
                task = self.tasks_by_index.get(task_index_by_ep[ep])
                if task:
                    episode.tasks = [task]

        self._attach_videos(episodes, meta_rows)
        return dict(sorted(episodes.items()))

    # -- videos ------------------------------------------------------------

    def _resolve_video_keys(self) -> List[str]:
        keys = [
            k
            for k, v in self.features.items()
            if isinstance(v, dict) and v.get("dtype") in ("video", "image")
        ]
        if keys:
            return sorted(keys)
        # No info.json: infer from the videos/ directory layout.
        videos_dir = self.root / "videos"
        found = set()
        if videos_dir.is_dir():
            for mp4 in videos_dir.rglob("*.mp4"):
                for part in mp4.relative_to(videos_dir).parts:
                    if part.startswith("observation."):
                        found.add(part)
        return sorted(found)

    def _attach_videos(self, episodes: Dict[int, Episode], meta_rows: Dict[int, dict]) -> None:
        videos_dir = self.root / "videos"
        if not videos_dir.is_dir():
            return

        keys = [
            k
            for k, v in self.features.items()
            if isinstance(v, dict) and v.get("dtype") in ("video", "image")
        ]
        if not keys:
            found = set()
            for mp4 in videos_dir.rglob("*.mp4"):
                for part in mp4.relative_to(videos_dir).parts:
                    if part.startswith("observation."):
                        found.add(part)
            keys = sorted(found)

        template = self.info.get("video_path")
        chunks_size = int(self.info.get("chunks_size", 1000))

        for ep, episode in episodes.items():
            row = meta_rows.get(ep, {})
            for key in keys:
                path: Optional[Path] = None
                t0, t1 = 0.0, None

                # v3.0: per-key chunk/file indices + a timestamp window.
                ci = row.get(f"videos/{key}/chunk_index")
                fi = row.get(f"videos/{key}/file_index")
                if template and ci is not None and fi is not None:
                    path = self.root / template.format(
                        video_key=key, chunk_index=int(ci), file_index=int(fi)
                    )
                    t0 = float(row.get(f"videos/{key}/from_timestamp", 0.0) or 0.0)
                    raw_t1 = row.get(f"videos/{key}/to_timestamp")
                    t1 = float(raw_t1) if raw_t1 is not None and not pd.isna(raw_t1) else None
                elif template and "episode_index" in template:
                    # v2.x: one file per episode.
                    path = self.root / template.format(
                        video_key=key, episode_index=ep, episode_chunk=ep // chunks_size
                    )

                if path is None or not path.exists():
                    path = self._glob_video(videos_dir, key, ep)
                    t0, t1 = 0.0, None

                if path is not None and path.exists():
                    episode.videos[key] = VideoRef(key=key, path=path, t0=t0, t1=t1)

    @staticmethod
    def _glob_video(videos_dir: Path, key: str, ep: int) -> Optional[Path]:
        """Last-resort lookup for datasets whose meta doesn't line up."""
        candidates = [p for p in videos_dir.rglob("*.mp4") if key in p.parts or key in p.name]
        if not candidates:
            return None
        for p in candidates:
            m = re.search(r"episode_(\d+)", p.name)
            if m and int(m.group(1)) == ep:
                return p
        # A single file with a single episode is unambiguous.
        return candidates[0] if len(candidates) == 1 and ep == 0 else None

    # -- misc metadata -----------------------------------------------------

    def _resolve_fps(self) -> float:
        fps = self.info.get("fps")
        if fps:
            return float(fps)
        # Infer from the first episode's timestamps.
        for ep in self.episodes:
            df = self.frames(ep)
            if len(df) > 1 and "timestamp" in df.columns:
                dt = np.diff(df["timestamp"].to_numpy(dtype=np.float64))
                dt = dt[dt > 0]
                if len(dt):
                    return float(round(1.0 / float(np.median(dt)), 3))
            break
        return 30.0

    def _resolve_joint_names(self) -> Dict[str, List[str]]:
        out = {}
        for key in ("observation.state", "action"):
            feat = self.features.get(key, {})
            names = feat.get("names")
            if isinstance(names, dict):  # some datasets nest under {"motors": [...]}
                names = next((v for v in names.values() if isinstance(v, list)), None)
            if names:
                out[key] = [str(n) for n in names]
        return out

    # -- frame access ------------------------------------------------------

    def frames(self, ep: int) -> pd.DataFrame:
        episode = self.episodes.get(ep)
        if episode is None:
            raise KeyError(ep)
        with self._cache_lock:
            df = self._frame_cache.get(episode.parquet)
            if df is None:
                df = pd.read_parquet(episode.parquet)
                # Keep at most 2 data files resident; they can be hundreds of MB.
                if len(self._frame_cache) >= 2:
                    self._frame_cache.pop(next(iter(self._frame_cache)))
                self._frame_cache[episode.parquet] = df
        out = df[df["episode_index"] == ep]
        sort_key = "frame_index" if "frame_index" in out.columns else "index"
        return out.sort_values(sort_key).reset_index(drop=True)

    @staticmethod
    def _matrix(df: pd.DataFrame, column: str) -> Optional[np.ndarray]:
        if column not in df.columns or len(df) == 0:
            return None
        return np.stack(df[column].to_numpy()).astype(np.float64)

    # -- reviews -----------------------------------------------------------

    def set_review(self, ep: int, status: str, note: str) -> dict:
        entry = {"status": status, "note": note}
        self.reviews[str(ep)] = entry
        tmp = self.review_path.with_suffix(".json.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(self.reviews, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.review_path)
        return entry


# --------------------------------------------------------------------------
# QC checks
# --------------------------------------------------------------------------


def _check(check_id: str, label: str, level: str, detail: str) -> dict:
    return {"id": check_id, "label": label, "level": level, "detail": detail}


_PROBE_CACHE: Dict[Path, dict] = {}


def _probe(path: Path) -> dict:
    """Probe an mp4 once per file -- several episodes share one video file."""
    if path not in _PROBE_CACHE:
        try:
            _PROBE_CACHE[path] = mp4probe.probe(path)
        except Exception as exc:
            _PROBE_CACHE[path] = {"error": f"could not read: {exc}"}
    return _PROBE_CACHE[path]


def run_qc(ds: Dataset, ep: int) -> dict:
    episode = ds.episodes[ep]
    df = ds.frames(ep)
    fps = ds.fps
    checks: List[dict] = []

    state = ds._matrix(df, "observation.state")
    action = ds._matrix(df, "action")
    ts = (
        df["timestamp"].to_numpy(dtype=np.float64)
        if "timestamp" in df.columns
        else np.arange(len(df)) / fps
    )

    # 1. Instruction present.
    if episode.instruction.strip():
        checks.append(_check("instruction", "Instruction", "ok", episode.instruction))
    else:
        checks.append(
            _check(
                "instruction",
                "Instruction",
                "fail",
                "No task string found. Pi0 conditions on language -- an empty "
                "instruction makes this episode useless for finetuning. Check "
                "meta/tasks.parquet and meta/episodes/.",
            )
        )

    # 2. Episode length vs. the rest of the dataset.
    med = ds._median_length
    dur = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    if med and episode.length < med * SHORT_EPISODE_RATIO:
        checks.append(_check("length", "Duration", "warn",
                             f"{episode.length} frames ({dur:.1f}s) -- under half the dataset "
                             f"median of {med:.0f}. Truncated recording?"))
    elif med and episode.length > med * LONG_EPISODE_RATIO:
        checks.append(_check("length", "Duration", "warn",
                             f"{episode.length} frames ({dur:.1f}s) -- over 2x the dataset "
                             f"median of {med:.0f}. Idle time at the end?"))
    else:
        checks.append(_check("length", "Duration", "ok",
                             f"{episode.length} frames, {dur:.1f}s @ {fps:g} fps"))

    # 3. Timestamp continuity (dropped frames during teleop).
    if len(ts) > 1:
        dt = np.diff(ts)
        expected = 1.0 / fps
        gaps = np.flatnonzero(dt > expected * GAP_TOLERANCE)
        backwards = int(np.sum(dt <= 0))
        if backwards:
            checks.append(_check("timing", "Frame timing", "fail",
                                 f"{backwards} non-monotonic timestamps -- the recording is corrupt."))
        elif len(gaps):
            worst = float(dt[gaps].max())
            checks.append(_check("timing", "Frame timing", "warn",
                                 f"{len(gaps)} gap(s) over {expected * GAP_TOLERANCE * 1000:.0f}ms "
                                 f"(worst {worst * 1000:.0f}ms, at t={ts[gaps[int(np.argmax(dt[gaps]))]]:.2f}s). "
                                 f"Dropped frames desync video and state."))
        else:
            checks.append(_check("timing", "Frame timing", "ok",
                                 f"No gaps; jitter {float(np.std(dt)) * 1000:.1f}ms"))

    # 4. Finite values.
    bad_dims = []
    for name, mat in (("observation.state", state), ("action", action)):
        if mat is not None and not np.isfinite(mat).all():
            bad_dims.append(name)
    if bad_dims:
        checks.append(_check("finite", "NaN / Inf", "fail",
                             f"Non-finite values in {', '.join(bad_dims)}."))
    else:
        checks.append(_check("finite", "NaN / Inf", "ok", "All values finite"))

    # 5. Dead takes and idle head/tail. An episode where the arm never moves is
    #    worse than useless -- it teaches the policy to sit still on this
    #    instruction. This is the most common way a recording silently fails.
    if state is not None and len(state) > 2:
        span = state.max(axis=0) - state.min(axis=0)
        active = span > STATIC_JOINT_RANGE
        if not active.any():
            checks.append(_check("motion", "Motion", "fail",
                                 f"The arm never moves: every joint spans less than "
                                 f"{STATIC_JOINT_RANGE:g} (largest {span.max():.2f}). "
                                 f"Dead take -- drop it."))
        else:
            # Displacement across a short window, so slow motion still counts
            # and per-frame sensor noise does not.
            w = max(1, int(round(fps * MOVE_WINDOW_S)))
            if len(state) > w:
                disp = np.abs(state[w:, active] - state[:-w, active]).max(axis=1)
                idx = np.flatnonzero(disp > MOVE_EPS)
            else:
                idx = np.array([], dtype=int)

            if len(idx) == 0:
                checks.append(_check("motion", "Motion", "warn",
                                     f"Only {span.max():.1f} of travel on the most active joint, "
                                     f"never faster than {MOVE_EPS:g} per {MOVE_WINDOW_S:g}s. "
                                     f"Barely moves -- check this one."))
            else:
                head = float((idx[0] + w // 2) / fps)
                tail = float((len(disp) - 1 - idx[-1] + w // 2) / fps)
                longest_still = float(np.max(np.diff(idx)) / fps) if len(idx) > 1 else 0.0
                active_s = float((idx[-1] - idx[0]) / fps)
                total_s = len(state) / fps
                if active_s < total_s * ACTIVE_FRACTION_MIN:
                    # A brief twitch followed by a long freeze is a dead take
                    # too, even though something technically moved.
                    checks.append(_check("motion", "Motion", "fail",
                                         f"Moves for only {active_s:.1f}s of {total_s:.1f}s "
                                         f"({int(active.sum())}/{len(span)} joints active, then "
                                         f"idle {tail:.1f}s). Dead take -- drop it."))
                else:
                    level = ("warn" if max(head, tail) > FROZEN_EDGE_SECONDS or longest_still > 2.0
                             else "ok")
                    checks.append(_check("motion", "Motion", level,
                                         f"{int(active.sum())}/{len(span)} joints active; "
                                         f"idle {head:.1f}s at start, {tail:.1f}s at end; "
                                         f"longest mid-episode pause {longest_still:.1f}s"))

    # 6. Gripper actuation -- a manipulation episode where it never closes is
    #    almost always a failed take.
    if state is not None and state.shape[1] >= 1:
        grip = state[:, -1]
        span = float(grip.max() - grip.min())
        if span < 1e-3:
            checks.append(_check("gripper", "Gripper", "warn",
                                 f"Last DoF never changes (constant {grip[0]:.3f}). "
                                 f"Nothing was grasped, or the gripper isn't wired up."))
        else:
            # Count open<->close cycles with hysteresis so slow, smooth travel
            # registers as one transition rather than none.
            norm = (grip - grip.min()) / span
            level_state, transitions = None, 0
            for v in norm:
                if v > 0.7 and level_state != "open":
                    transitions += level_state is not None
                    level_state = "open"
                elif v < 0.3 and level_state != "closed":
                    transitions += level_state is not None
                    level_state = "closed"
            checks.append(_check("gripper", "Gripper", "ok",
                                 f"Range {grip.min():.1f} to {grip.max():.1f} "
                                 f"(travel {span:.1f}), {transitions} open/close transition(s)"))

    # 7. Action/state tracking error -- catches a leader-follower mismatch.
    if state is not None and action is not None and state.shape == action.shape:
        per_joint = np.abs(action - state).mean(axis=0)
        span = state.max(axis=0) - state.min(axis=0)
        # Only rate joints that actually moved and whose error is above noise --
        # dividing a tiny error by a near-zero range yields nonsense ratios.
        rated = (span > STATIC_JOINT_RANGE) & (per_joint > TRACKING_ABS_FLOOR)
        detail = "mean |action-state| per joint: " + ", ".join(f"{v:.2f}" for v in per_joint)
        if rated.any():
            rel = np.where(rated, per_joint / np.maximum(span, 1e-6), 0.0)
            worst = int(np.argmax(rel))
            if rel[worst] > 0.5:
                checks.append(_check("tracking", "Action vs state", "warn",
                                     f"Joint {worst} tracks poorly ({rel[worst] * 100:.0f}% of its "
                                     f"{span[worst]:.1f} travel). " + detail))
            else:
                checks.append(_check("tracking", "Action vs state", "ok", detail))
        else:
            checks.append(_check("tracking", "Action vs state", "ok",
                                 "Tracking error below noise floor. " + detail))

    # 8. Velocity spikes -- teleop glitches / encoder jumps. A MAD-based
    #    z-score is useless here: most frames move very little, so the spread
    #    collapses to ~0 and every ordinary motion looks like an outlier.
    #    Judge against the joint's own travel instead.
    if state is not None and len(state) > 10:
        vel = np.abs(np.diff(state, axis=0))
        rng = state.max(axis=0) - state.min(axis=0)
        thresh = np.maximum(JUMP_FRACTION * rng,
                            JUMP_P99_FACTOR * np.percentile(vel, 99, axis=0))
        # A joint that never moved has no travel to be 15% of -- judging it
        # would just flag its sensor noise.
        spikes = (vel > thresh) & (rng > STATIC_JOINT_RANGE)
        rows = np.flatnonzero(spikes.any(axis=1))
        if len(rows):
            joint = int(np.argmax(spikes.sum(axis=0)))
            checks.append(_check("spikes", "Discontinuities", "warn",
                                 f"{len(rows)} frame(s) jump more than "
                                 f"{JUMP_FRACTION * 100:.0f}% of a joint's travel in one step "
                                 f"(worst joint {joint}, first at frame {int(rows[0])}, "
                                 f"t={ts[int(rows[0])]:.2f}s)."))
        else:
            checks.append(_check("spikes", "Discontinuities", "ok",
                                 "No single-frame jumps"))

    # 9. Cameras.
    expected_keys = ds.video_keys
    if not expected_keys:
        checks.append(_check("video", "Cameras", "warn",
                             "No camera streams in this dataset. Pi0 needs images -- "
                             "this recording is state-only."))
    else:
        missing = [k for k in expected_keys if k not in episode.videos]
        if missing:
            checks.append(_check("video", "Cameras", "fail",
                                 f"Missing video for: {', '.join(missing)}"))
        else:
            bits = []
            level = "ok"
            for key, ref in sorted(episode.videos.items()):
                short = key.split(".")[-1]
                info = _probe(ref.path)
                if "error" in info:
                    level = "fail"
                    bits.append(f"{short}: {info['error']}")
                    continue

                # The episode's slice of the file, which for v3.0 holds several.
                span = (ref.t1 - ref.t0) if ref.t1 is not None else info.get("duration")
                detail = f"{short}: {info.get('width')}x{info.get('height')} {info.get('codec_name', '?')}"
                if span is not None:
                    drift = abs(span - dur)
                    if drift > max(0.5, dur * 0.05):
                        level = "warn"
                        detail += f", {span:.1f}s vs state {dur:.1f}s (drift {drift:.1f}s)"
                    else:
                        detail += f", {span:.1f}s"

                # A file-level fps mismatch desyncs every episode inside it.
                vfps = info.get("fps")
                if vfps and abs(vfps - fps) > 0.5:
                    level = "warn"
                    detail += f", video {vfps:g}fps vs dataset {fps:g}fps"

                if info.get("codec") in mp4probe.SAFARI_UNSUPPORTED:
                    detail += " (no Safari playback)"
                bits.append(detail)
            checks.append(_check("video", "Cameras", level, "; ".join(bits)))

    worst = "ok"
    for c in checks:
        if c["level"] == "fail":
            worst = "fail"
            break
        if c["level"] == "warn":
            worst = "warn"
    return {"level": worst, "checks": checks}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def range_response(path: Path, range_header: Optional[str]) -> Response:
    """Serve a file, honouring a single Range request so <video> can seek."""
    size = path.stat().st_size
    if not range_header:
        return FileResponse(path, media_type="video/mp4")

    m = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not m:
        return FileResponse(path, media_type="video/mp4")

    start_s, end_s = m.group(1), m.group(2)
    if start_s:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    else:  # suffix range: bytes=-N
        start = max(0, size - int(end_s or 0))
        end = size - 1
    end = min(end, size - 1)
    if start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def stream():
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        stream(),
        status_code=206,
        media_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
        },
    )


def build_app(ds: Dataset) -> FastAPI:
    app = FastAPI(title="LeRobot Episode Inspector")
    static_dir = Path(__file__).parent / "static"

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/api/dataset")
    def dataset_info():
        return {
            "root": str(ds.root),
            "target": str(ds.target),
            "codebase_version": ds.codebase_version,
            "robot_type": ds.robot_type,
            "fps": ds.fps,
            "num_episodes": len(ds.episodes),
            "total_frames": int(sum(e.length for e in ds.episodes.values())),
            "video_keys": ds.video_keys,
            "joint_names": ds.joint_names,
            "has_meta": bool(ds.info),
            "review_path": str(ds.review_path),
            "data_files": [str(p) for p in ds.data_files],
        }

    @app.get("/api/episodes")
    def episodes():
        out = []
        for ep, episode in ds.episodes.items():
            if ep not in ds._qc_cache:
                ds._qc_cache[ep] = run_qc(ds, ep)
            qc = ds._qc_cache[ep]
            out.append({
                "index": ep,
                "length": episode.length,
                "duration": round(episode.length / ds.fps, 2),
                "instruction": episode.instruction,
                "tasks": episode.tasks,
                "videos": sorted(episode.videos.keys()),
                "qc_level": qc["level"],
                "review": ds.reviews.get(str(ep), {}).get("status", "unreviewed"),
            })
        return out

    @app.get("/api/episodes/{ep}")
    def episode_detail(ep: int):
        if ep not in ds.episodes:
            raise HTTPException(404, f"no episode {ep}")
        episode = ds.episodes[ep]
        df = ds.frames(ep)

        state = ds._matrix(df, "observation.state")
        action = ds._matrix(df, "action")
        ts = (
            df["timestamp"].to_numpy(dtype=np.float64)
            if "timestamp" in df.columns
            else np.arange(len(df)) / ds.fps
        )

        def series(mat: Optional[np.ndarray]) -> Optional[List[List[float]]]:
            if mat is None:
                return None
            return np.round(mat, 5).T.tolist()  # [dim][t] -- one array per joint

        dim = state.shape[1] if state is not None else (action.shape[1] if action is not None else 0)
        names = ds.joint_names.get("observation.state")
        if (not names or len(names) != dim) and dim:
            # No names in info.json: assume the usual "N joints + gripper".
            names = [f"joint_{i + 1}" for i in range(dim - 1)] + ["gripper"]
        names = names or []

        return {
            "index": ep,
            "length": episode.length,
            "fps": ds.fps,
            "instruction": episode.instruction,
            "tasks": episode.tasks,
            "timestamps": np.round(ts - ts[0] if len(ts) else ts, 4).tolist(),
            "joint_names": names,
            "state": series(state),
            "action": series(action),
            "videos": [
                {
                    "key": k,
                    "url": f"/api/video/{ep}/{k}",
                    "t0": v.t0,
                    "t1": v.t1,
                    "file": str(v.path),
                }
                for k, v in sorted(episode.videos.items())
            ],
            "qc": ds._qc_cache.setdefault(ep, run_qc(ds, ep)),
            "review": ds.reviews.get(str(ep), {"status": "unreviewed", "note": ""}),
        }

    @app.get("/api/video/{ep}/{key}")
    def video(ep: int, key: str, request: Request):
        episode = ds.episodes.get(ep)
        if not episode or key not in episode.videos:
            raise HTTPException(404, "no such video")
        return range_response(episode.videos[key].path, request.headers.get("range"))

    @app.post("/api/review/{ep}")
    async def review(ep: int, request: Request):
        if ep not in ds.episodes:
            raise HTTPException(404, f"no episode {ep}")
        body = await request.json()
        status = str(body.get("status", "unreviewed"))
        if status not in ("good", "bad", "unsure", "unreviewed"):
            raise HTTPException(400, "bad status")
        return ds.set_review(ep, status, str(body.get("note", "")))

    @app.get("/api/export")
    def export():
        """Episode indices to keep / drop, for filtering before finetuning."""
        good, bad = [], []
        for ep in ds.episodes:
            status = ds.reviews.get(str(ep), {}).get("status", "unreviewed")
            (bad if status == "bad" else good).append(ep)
        return JSONResponse(
            {"keep": good, "drop": bad, "reviews": ds.reviews},
            headers={"Content-Disposition": 'attachment; filename="episode_review_export.json"'},
        )

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="LeRobot episode QC viewer")
    ap.add_argument("--data", required=True,
                    help="dataset root directory, or a single .parquet data file")
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 0.0.0.0 to view from another machine on the LAN")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--review", default=None,
                    help="where to store review verdicts (default: <root>/episode_review.json)")
    args = ap.parse_args()

    ds = Dataset(Path(args.data), Path(args.review) if args.review else None)
    print(f"dataset root   : {ds.root}")
    print(f"layout         : {ds.codebase_version}   robot: {ds.robot_type}")
    print(f"episodes       : {len(ds.episodes)}  ({sum(e.length for e in ds.episodes.values())} frames @ {ds.fps:g} fps)")
    print(f"camera streams : {', '.join(ds.video_keys) or '(none)'}")
    print(f"reviews        : {ds.review_path}")
    print(f"\n  ->  http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}\n")

    import uvicorn

    uvicorn.run(build_app(ds), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
