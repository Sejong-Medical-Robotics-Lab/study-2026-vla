"""
Minimal MP4 container probe -- duration, frame count, resolution and codec
straight out of the box structure.

ffmpeg is not always installed next to a dataset (and shelling out per episode
is slow), so this reads the handful of boxes we actually care about.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# Codecs a <video> element can decode. LeRobot v3.0 encodes AV1 by default,
# which every current browser handles except Safari on older hardware.
BROWSER_CODECS = {"av01": "AV1", "avc1": "H.264", "hev1": "HEVC", "hvc1": "HEVC", "vp09": "VP9"}
SAFARI_UNSUPPORTED = {"av01", "vp09"}


def _boxes(data: bytes, start: int, end: int) -> Iterator[Tuple[str, int, int]]:
    """Yield (type, payload_start, payload_end) for boxes in [start, end)."""
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        btype = data[pos + 4:pos + 8].decode("latin-1", "replace")
        header = 8
        if size == 1:  # 64-bit extended size
            size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
            header = 16
        elif size == 0:  # box extends to end of file
            size = end - pos
        if size < header:
            return
        yield btype, pos + header, min(pos + size, end)
        pos += size


def _find(data: bytes, path: List[str], start: int, end: int) -> Optional[Tuple[int, int]]:
    for btype, ps, pe in _boxes(data, start, end):
        if btype == path[0]:
            return (ps, pe) if len(path) == 1 else _find(data, path[1:], ps, pe)
    return None


def _timescale_duration(data: bytes, ps: int) -> Tuple[int, int]:
    """Read (timescale, duration) from an mvhd or mdhd payload."""
    if data[ps] == 1:  # version 1 uses 64-bit times
        return struct.unpack(">I", data[ps + 20:ps + 24])[0], struct.unpack(">Q", data[ps + 24:ps + 32])[0]
    return struct.unpack(">I", data[ps + 12:ps + 16])[0], struct.unpack(">I", data[ps + 16:ps + 20])[0]


def probe(path: Path, header_bytes: int = 1 << 20) -> Dict[str, object]:
    """
    Probe an mp4. Only the header is read -- moov usually sits at the front of
    a LeRobot recording, and these files run to hundreds of megabytes.
    """
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as f:
        data = f.read(min(size, header_bytes))

    out: Dict[str, object] = {"file": path.name, "bytes": size}

    moov = _find(data, ["moov"], 0, len(data))
    if not moov:
        # moov is at the end (not "faststart"): pull the tail and retry.
        with path.open("rb") as f:
            f.seek(max(0, size - header_bytes))
            tail = f.read()
        moov = _find(tail, ["moov"], 0, len(tail))
        if not moov:
            out["error"] = "no moov box found (truncated or not an mp4)"
            return out
        data = tail

    mvhd = _find(data, ["mvhd"], *moov)
    if mvhd:
        ts, du = _timescale_duration(data, mvhd[0])
        out["duration"] = round(du / ts, 3) if ts else None

    for btype, ts_, te in _boxes(data, *moov):
        if btype != "trak":
            continue
        hdlr = _find(data, ["mdia", "hdlr"], ts_, te)
        if not hdlr or data[hdlr[0] + 8:hdlr[0] + 12] != b"vide":
            continue  # skip audio / metadata tracks

        stbl_path = ["mdia", "minf", "stbl"]
        stsz = _find(data, stbl_path + ["stsz"], ts_, te)
        if stsz:
            out["frames"] = struct.unpack(">I", data[stsz[0] + 8:stsz[0] + 12])[0]

        stsd = _find(data, stbl_path + ["stsd"], ts_, te)
        if stsd:
            fourcc = data[stsd[0] + 12:stsd[0] + 16].decode("latin-1", "replace")
            out["codec"] = fourcc
            out["codec_name"] = BROWSER_CODECS.get(fourcc, fourcc)

        tkhd = _find(data, ["tkhd"], ts_, te)
        if tkhd:
            tp = tkhd[0]
            # Fixed-point width/height sit right after the 36-byte matrix.
            off = 88 if data[tp] == 1 else 76
            out["width"] = struct.unpack(">I", data[tp + off:tp + off + 4])[0] >> 16
            out["height"] = struct.unpack(">I", data[tp + off + 4:tp + off + 8])[0] >> 16

        mdhd = _find(data, ["mdia", "mdhd"], ts_, te)
        if mdhd:
            mts, mdu = _timescale_duration(data, mdhd[0])
            if mts:
                out["duration"] = round(mdu / mts, 3)

        dur, n = out.get("duration"), out.get("frames")
        if dur and n:
            out["fps"] = round(n / dur, 3)
        break

    return out


def describe(info: Dict[str, object]) -> str:
    if "error" in info:
        return str(info["error"])
    bits = []
    if info.get("width"):
        bits.append(f"{info['width']}x{info['height']}")
    if info.get("codec_name"):
        bits.append(str(info["codec_name"]))
    if info.get("duration"):
        bits.append(f"{info['duration']:.2f}s")
    if info.get("frames"):
        bits.append(f"{info['frames']} frames")
    if info.get("fps"):
        bits.append(f"{info['fps']:g}fps")
    return ", ".join(bits)
