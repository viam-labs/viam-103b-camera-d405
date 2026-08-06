"""The seam between the camera API and whatever is producing frames.

Everything above this file speaks the Viam camera API: MIME types, properties,
point clouds. Everything below it speaks to a device, or to a scene renderer
standing in for one. The two sides meet at `FrameSource`, which hands up a
`Frame` and the `Intrinsics` that describe the lens it came through.

That split is the reason this module can carry two models that share one
implementation, and the reason its behavior can be tested with nothing plugged
in.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """The pinhole parameters of one stream, at one resolution.

    ``fx`` and ``fy`` are focal lengths in pixels; ``ppx`` and ``ppy`` are the
    principal point, where the optical axis meets the sensor. They belong to a
    resolution: change the stream size and every one of them changes.
    """

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    # What is left to correct after the stream was rectified. An empty model
    # means nothing is: that is what the shipped RealSense module reports when
    # librealsense says the stream carries no distortion, and it keeps a caller
    # from undistorting an image that is already straight. When there is a
    # model, the RDK reads the coefficients as [k1, k2, k3, p1, p2].
    distortion_model: str = ""
    distortion_coeffs: Tuple[float, ...] = ()

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float) -> "Intrinsics":
        """Square pixels, centered principal point, from a horizontal field of view."""
        fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        return cls(width, height, fx, fx, (width - 1) / 2.0, (height - 1) / 2.0)


@dataclass(frozen=True)
class Frame:
    """One moment, as both streams see it.

    ``color`` is ``(height, width, 3)`` uint8 RGB. ``depth`` is
    ``(height, width)`` uint16 holding millimeters along the optical axis, where
    **0 means no reading** rather than a surface at the lens. ``captured_at`` is
    a Unix timestamp in seconds.
    """

    color: np.ndarray
    depth: np.ndarray
    captured_at: float


class FrameSource(abc.ABC):
    """Where frames come from. One implementation per kind of camera."""

    @property
    @abc.abstractmethod
    def intrinsics(self) -> Intrinsics:
        """The parameters of the stream this source is producing right now."""

    @property
    def frame_rate(self) -> float:
        """Frames per second this source produces, for `get_properties`."""
        return 0.0

    @abc.abstractmethod
    def read(self) -> Frame:
        """Return the next frame, or raise if the device cannot produce one."""

    def close(self) -> None:
        """Release whatever the source is holding. Safe to call twice."""
