"""A depth camera with no hardware behind it, looking at a scene we measured.

This is not a placeholder that returns a picture. It is a camera whose right
answers are written down: the wall is 600 mm away, the near box's front face is
250 mm away and 180 mm across. A driver's depth encoding and point cloud can
therefore be checked against numbers instead of against how the render looks.

Coordinates are the camera's own, in millimeters: +X right, +Y down, +Z along
the optical axis away from the lens. Depth is the Z coordinate of the surface a
pixel sees, which is what a stereo depth camera reports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from frames import Frame, FrameSource, Intrinsics

# The D405 family's published depth field of view, used to give the simulated
# stream a plausible lens. It describes this scene renderer, not a measurement
# of any particular device.
DEFAULT_HFOV_DEG = 87.0


@dataclass(frozen=True)
class Box:
    """An axis-aligned box, in millimeters, with an RGB color."""

    name: str
    x: Tuple[float, float]
    y: Tuple[float, float]
    z: Tuple[float, float]
    color: Tuple[int, int, int]

    @property
    def width_mm(self) -> float:
        return self.x[1] - self.x[0]

    @property
    def front_z_mm(self) -> float:
        return self.z[0]


# --- the scene, as measured -------------------------------------------------
#
# A back wall, a table the camera looks across, and two boxes on the table. The
# course quotes these numbers in its checkpoints, so changing one here changes
# what the learner is told to expect.

WALL_Z_MM = 600.0            # back wall, facing the camera
TABLE_Y_MM = 150.0           # table top, 150 mm below the optical axis
TABLE_NEAR_Z_MM = 150.0      # nearest table surface the camera can see

BOXES: List[Box] = [
    # The near box: front face 250 mm from the lens, 180 mm wide, 120 mm tall.
    Box("near", x=(-150.0, 30.0), y=(30.0, TABLE_Y_MM), z=(250.0, 370.0),
        color=(196, 74, 58)),
    # The far box: front face 420 mm from the lens, 100 mm wide, 100 mm tall.
    Box("far", x=(80.0, 180.0), y=(50.0, TABLE_Y_MM), z=(420.0, 520.0),
        color=(58, 108, 196)),
]

WALL_COLOR = (168, 172, 178)
TABLE_COLOR = (150, 128, 96)

# Every surface in the scene is axis-aligned, so shading only needs one factor
# per orientation. Enough relief to read the render; no bearing on the depth.
_SHADE = {"front": 1.00, "top": 0.86, "side": 0.72, "wall": 0.94}

MAX_RANGE_MM = 1000.0        # beyond this the camera reports no reading


class SceneSource(FrameSource):
    """Renders the scene above at a configured resolution.

    Args:
        width: stream width in pixels.
        height: stream height in pixels.
        hfov_deg: horizontal field of view used to derive the intrinsics.
    """

    def __init__(self, width: int = 640, height: int = 480,
                 hfov_deg: float = DEFAULT_HFOV_DEG) -> None:
        self._intrinsics = Intrinsics.from_fov(width, height, hfov_deg)
        self._color, self._depth = _render(self._intrinsics)

    @property
    def intrinsics(self) -> Intrinsics:
        return self._intrinsics

    def read(self) -> Frame:
        # The scene does not move, so every frame is the same render with a
        # fresh timestamp. A real source produces a new one each call.
        return Frame(color=self._color, depth=self._depth, captured_at=time.time())



def _render(intr: Intrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """Ray-cast the scene once, returning RGB and depth in millimeters.

    One ray per pixel, leaving the lens through ``((u - ppx) / fx, (v - ppy) /
    fy, 1)``. Because the ray's Z component is 1, the parameter along it *is*
    the depth in millimeters, which keeps the intersection arithmetic short.
    """
    v, u = np.meshgrid(np.arange(intr.height), np.arange(intr.width), indexing="ij")
    dx = (u - intr.ppx) / intr.fx
    dy = (v - intr.ppy) / intr.fy

    inf = np.full(dx.shape, np.inf)
    depth = np.full(dx.shape, WALL_Z_MM)
    color = np.empty(dx.shape + (3,), dtype=np.float64)
    color[:] = np.array(WALL_COLOR) * _SHADE["wall"]

    # The table: a horizontal plane at y = TABLE_Y_MM, visible where the ray
    # points downward and lands beyond the camera's near edge.
    with np.errstate(divide="ignore", invalid="ignore"):
        t_table = np.where(dy > 0, TABLE_Y_MM / dy, inf)
    hit = (t_table >= TABLE_NEAR_Z_MM) & (t_table < depth)
    depth = np.where(hit, t_table, depth)
    color[hit] = np.array(TABLE_COLOR) * _SHADE["top"]

    for box in BOXES:
        t_hit, face = _intersect_box(dx, dy, box)
        hit = np.isfinite(t_hit) & (t_hit < depth)
        depth = np.where(hit, t_hit, depth)
        for name, shade in _SHADE.items():
            sel = hit & (face == name)
            if sel.any():
                color[sel] = np.array(box.color) * shade

    depth = np.where(depth <= MAX_RANGE_MM, depth, 0.0)
    return (np.clip(color, 0, 255).astype(np.uint8),
            np.rint(depth).astype(np.uint16))


def _intersect_box(dx: np.ndarray, dy: np.ndarray, box: Box):
    """Slab intersection against an axis-aligned box, parameterized by depth.

    Returns the depth of the nearest hit (``inf`` where the ray misses) and the
    name of the face it landed on, which is what the shading uses.
    """
    inf = np.full(dx.shape, np.inf)
    lo = np.full(dx.shape, box.z[0])
    hi = np.full(dx.shape, box.z[1])

    for d, (a, b) in ((dx, box.x), (dy, box.y)):
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = np.where(d != 0, a / d, np.where(a <= 0, -inf, inf))
            t1 = np.where(d != 0, b / d, np.where(b >= 0, inf, -inf))
        lo = np.maximum(lo, np.minimum(t0, t1))
        hi = np.minimum(hi, np.maximum(t0, t1))

    ok = (lo <= hi) & (lo > 0)
    t = np.where(ok, lo, inf)

    # Which slab produced the entry point tells us which face was hit.
    face = np.full(dx.shape, "front", dtype="<U5")
    face[ok & (np.abs(t - box.z[0]) > 1e-9) & (np.abs(dy * t - box.y[0]) < 1e-6)] = "top"
    on_side = ok & (np.abs(t - box.z[0]) > 1e-9) & ~(np.abs(dy * t - box.y[0]) < 1e-6)
    face[on_side] = "side"
    return t, face
