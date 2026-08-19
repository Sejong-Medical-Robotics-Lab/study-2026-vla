# /// script
# requires-python = ">=3.9"
# dependencies = ["pandas", "pyarrow", "numpy"]
# ///
"""
Assemble loose files into a LeRobot v3.0 dataset layout, for previewing the
inspector on a machine that only has a parquet + some mp4s (no meta/).

    uv run make_local_dataset.py --parquet file-000.parquet \
        --video observation.images.front=front.mp4 \
        --video observation.images.wrist=wrist.mp4 \
        --task "pick up the block" --out ./piper_pick

Episode boundaries, lengths and fps come from the parquet. Video timestamp
windows are derived from each episode's own duration, which matches how
LeRobot packs several episodes into one mp4.

This is a preview aid -- the real meta/ on the robot is authoritative.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import mp4probe

# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="assemble a v3.0 dataset from loose files")
    ap.add_argument("--parquet", required=True, help="the data parquet")
    ap.add_argument("--video", action="append", default=[], metavar="KEY=PATH",
                    help="e.g. observation.images.front=front.mp4 (repeatable)")
    ap.add_argument("--task", default="", help="instruction for every episode")
    ap.add_argument("--robot", default="piper")
    ap.add_argument("--out", help="dataset root to create (not needed with --probe-only)")
    ap.add_argument("--probe-only", action="store_true", help="just report on the videos")
    args = ap.parse_args()
    if not args.probe_only and not args.out:
        ap.error("--out is required unless --probe-only is given")

    videos: Dict[str, Path] = {}
    for spec in args.video:
        if "=" not in spec:
            raise SystemExit(f"--video needs KEY=PATH, got {spec!r}")
        key, _, p = spec.partition("=")
        videos[key] = Path(p).expanduser().resolve()

    print("=== videos ===")
    probes = {}
    for key, p in videos.items():
        info = mp4probe.probe(p)
        probes[key] = info
        print(f"{key}\n    {mp4probe.describe(info)}")
        if info.get("codec") in mp4probe.SAFARI_UNSUPPORTED:
            print(f"    note: {info.get('codec_name')} does not play in Safari -- "
                  f"use Chrome, Edge or Firefox")

    src = Path(args.parquet).expanduser().resolve()
    df = pd.read_parquet(src)
    fps = 30.0
    if "timestamp" in df.columns and len(df) > 1:
        first = df[df["episode_index"] == df["episode_index"].iloc[0]]
        dt = np.diff(first["timestamp"].to_numpy(dtype=np.float64))
        dt = dt[dt > 0]
        if len(dt):
            fps = round(1.0 / float(np.median(dt)), 3)

    lengths = df.groupby("episode_index").size().sort_index()
    print(f"\n=== parquet ===\n{len(lengths)} episode(s), {len(df)} frames @ {fps:g} fps")
    for ep, n in lengths.items():
        print(f"    episode {ep}: {n} frames = {n / fps:.2f}s")

    # Cross-check: state duration vs. container duration.
    total = len(df) / fps
    for key, info in probes.items():
        dur = info.get("duration")
        if dur:
            drift = abs(dur - total)
            flag = "OK" if drift <= max(0.5, total * 0.05) else "MISMATCH"
            print(f"    {key}: video {dur:.2f}s vs state {total:.2f}s  ({flag}, drift {drift:.2f}s)")
        n = info.get("frames")
        if n:
            flag = "OK" if n == len(df) else "MISMATCH"
            print(f"    {key}: video {n} frames vs state {len(df)} frames  ({flag})")

    if args.probe_only:
        return

    out = Path(args.out).expanduser().resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists -- remove it or pick another --out")
    (out / "data" / "chunk-000").mkdir(parents=True)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    shutil.copy2(src, out / "data" / "chunk-000" / "file-000.parquet")

    ep_rows, cursor = [], 0.0
    for ep, n in lengths.items():
        row = {
            "episode_index": int(ep),
            "tasks": np.array([args.task]),
            "length": int(n),
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        dur = n / fps
        for key in videos:
            row[f"videos/{key}/chunk_index"] = 0
            row[f"videos/{key}/file_index"] = 0
            row[f"videos/{key}/from_timestamp"] = cursor
            row[f"videos/{key}/to_timestamp"] = cursor + dur
        cursor += dur
        ep_rows.append(row)
    pd.DataFrame(ep_rows).to_parquet(
        out / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)

    pd.DataFrame({"task_index": [0]},
                 index=pd.Index([args.task], name="task")).to_parquet(out / "meta" / "tasks.parquet")

    dim = len(df["observation.state"].iloc[0])
    names = [f"joint_{i + 1}" for i in range(dim - 1)] + ["gripper"]
    features = {
        "action": {"dtype": "float32", "shape": [dim], "names": names},
        "observation.state": {"dtype": "float32", "shape": [dim], "names": names},
    }
    for key, info in probes.items():
        features[key] = {
            "dtype": "video",
            "shape": [info.get("height", 0), info.get("width", 0), 3],
            "info": {"video.fps": info.get("fps")},
        }
        dst = out / "videos" / key / "chunk-000"
        dst.mkdir(parents=True)
        shutil.copy2(videos[key], dst / "file-000.mp4")

    (out / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0",
        "robot_type": args.robot,
        "fps": fps,
        "chunks_size": 1000,
        "total_episodes": len(lengths),
        "total_frames": len(df),
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }, indent=2), encoding="utf-8")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
