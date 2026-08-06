"""`viam-labs:camera:d405-sim` — the camera API over a scene we measured.

No hardware. The frames come from `scene.py`, whose geometry is written down,
which is what lets this model's output be checked against numbers.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict

from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

from frames import FrameSource
from models.camera_base import D405CameraBase, resolution
from scene import DEFAULT_HFOV_DEG, SceneSource

class D405Sim(D405CameraBase, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "camera"), "d405-sim")

    def build_source(self, attrs: Dict[str, Any]) -> FrameSource:
        width, height = resolution(attrs)
        return SceneSource(
            width=width,
            height=height,
            hfov_deg=float(attrs.get("hfov_deg", DEFAULT_HFOV_DEG)),
        )
