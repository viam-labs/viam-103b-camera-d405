"""Turning arrays into the bytes the Viam camera API carries.

Every method on the camera API returns encoded bytes and the MIME type that
describes them. Three encodings appear in this module:

- ``image/jpeg`` for color, which Pillow writes.
- ``image/vnd.viam.dep`` for depth, which nothing writes for you.

The Viam depth format is written here by hand, byte for byte, against the
layouts the RDK reads.
"""

from __future__ import annotations

import io
import struct

import numpy as np
from PIL import Image


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


