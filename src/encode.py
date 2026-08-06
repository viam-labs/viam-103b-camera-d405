"""Turning arrays into the bytes the Viam camera API carries.

Every method on the camera API returns encoded bytes and the MIME type that
describes them. Three encodings appear in this module:

- ``image/jpeg`` for color, which Pillow writes.
- ``image/vnd.viam.dep`` for depth, which nothing writes for you.
- ``pointcloud/pcd`` for the point cloud, likewise.

The two Viam formats are written here by hand, byte for byte, against the
layouts the RDK reads.
"""

from __future__ import annotations

import io
import struct
from typing import Optional

import numpy as np
from PIL import Image

from frames import Intrinsics

# rimage.DepthMapMagicNumber in the RDK: eight bytes, then width and height as
# big-endian uint64, then one big-endian uint16 per pixel holding millimeters.
DEPTH_MAGIC = b"DEPTHMAP"
DEPTH_HEADER_LEN = 24


def encode_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    """Encode an ``(h, w, 3)`` uint8 RGB array as JPEG."""
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def encode_viam_depth(depth_mm: np.ndarray) -> bytes:
    """Encode an ``(h, w)`` uint16 depth array as ``image/vnd.viam.dep``.

    The header is the eight-byte magic, then the width and the height as
    big-endian 64-bit integers. The payload is the readings in row-major order,
    each a big-endian uint16 of millimeters, with 0 meaning no reading.
    """
    if depth_mm.ndim != 2:
        raise ValueError(f"depth must be 2-D, got shape {depth_mm.shape}")
    height, width = depth_mm.shape
    header = DEPTH_MAGIC + struct.pack(">QQ", width, height)
    return header + depth_mm.astype(">u2").tobytes()


def decode_viam_depth(data: bytes) -> np.ndarray:
    """Read ``image/vnd.viam.dep`` back into an array, to check an encoder."""
    if data[:8] != DEPTH_MAGIC:
        raise ValueError("not a Viam raw depth image: wrong magic number")
    width, height = struct.unpack(">QQ", data[8:DEPTH_HEADER_LEN])
    body = np.frombuffer(data[DEPTH_HEADER_LEN:], dtype=">u2")
    if body.size != width * height:
        raise ValueError(f"expected {width * height} readings, found {body.size}")
    return body.reshape(height, width).astype(np.uint16)


def deproject(depth_mm: np.ndarray, intr: Intrinsics):
    """Turn a depth image into points in camera coordinates, in millimeters.

    The pinhole model backwards: shift by the principal point, divide by the
    focal length, scale by the depth. Pixels with no reading are dropped, so the
    result is a flat list of points rather than a grid.

    Returns the ``(n, 3)`` points and the ``(n,)`` flat indices they came from,
    so a caller can look up each point's color.
    """
    height, width = depth_mm.shape
    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    z = depth_mm.astype(np.float32)

    keep = z > 0                      # 0 is an absent measurement, not a surface
    z = z[keep]
    x = (u[keep] - intr.ppx) / intr.fx * z
    y = (v[keep] - intr.ppy) / intr.fy * z

    points = np.stack([x, y, z], axis=1).astype(np.float32)
    return points, np.flatnonzero(keep)


def encode_pcd(points: np.ndarray, colors: Optional[np.ndarray] = None) -> bytes:
    """Encode ``(n, 3)`` points in millimeters as binary PCD.

    The header is the one the RDK's reader accepts, and the payload is three
    little-endian float32 coordinates per point, followed by a packed RGB
    integer when colors are supplied.
    """
    n = len(points)
    if colors is not None and len(colors) != n:
        raise ValueError(f"{n} points but {len(colors)} colors")

    fields = "x y z rgb" if colors is not None else "x y z"
    sizes = "4 4 4 4" if colors is not None else "4 4 4"
    types = "F F F I" if colors is not None else "F F F"
    counts = "1 1 1 1" if colors is not None else "1 1 1"

    header = (
        "VERSION .7\n"
        f"FIELDS {fields}\n"
        f"SIZE {sizes}\n"
        f"TYPE {types}\n"
        f"COUNT {counts}\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    ).encode("ascii")

    if colors is None:
        body = points.astype("<f4").tobytes()
    else:
        rgb = (colors[:, 0].astype(np.uint32) << 16 |
               colors[:, 1].astype(np.uint32) << 8 |
               colors[:, 2].astype(np.uint32))
        record = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"),
                                    ("z", "<f4"), ("rgb", "<u4")])
        record["x"], record["y"], record["z"] = points.T
        record["rgb"] = rgb
        body = record.tobytes()

    return header + body
