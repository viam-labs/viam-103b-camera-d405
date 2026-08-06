"""Turning arrays into the bytes the Viam camera API carries.

Every method on the camera API returns encoded bytes and the MIME type that
describes them. Color is JPEG, which Pillow writes. The formats that nothing writes for
you arrive over the next few exercises.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def encode_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    """Encode an ``(h, w, 3)`` uint8 RGB array as JPEG."""
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


