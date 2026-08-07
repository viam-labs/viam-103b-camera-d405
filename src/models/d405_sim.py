"""`viam-labs:camera:d405-sim` — the camera API over a scene we measured.

No hardware. The frames come from `scene.py`, whose geometry is written down,
which is what lets this model's output be checked against numbers.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Sequence, Tuple

from viam.proto.app.robot import ComponentConfig
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict

from frames import FrameSource
from models.camera_base import D405CameraBase, resolution
from scene import DEFAULT_HFOV_DEG, DEFAULT_LINK, SceneSource

FAIL_MODES = ("", "absent", "dropped", "flaky", "slow")


class D405Sim(D405CameraBase, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "camera"), "d405-sim")

    @classmethod
    def validate_config(cls, config: ComponentConfig
                        ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)

        for key in ("width_px", "height_px"):
            value = attrs.get(key)
            if value is None:
                continue
            if int(value) <= 0:
                raise ValueError(f"{key} must be a positive number of pixels, got {value}")

        hfov = float(attrs.get("hfov_deg", DEFAULT_HFOV_DEG))
        if not 0 < hfov < 180:
            raise ValueError(f"hfov_deg must be between 0 and 180, got {hfov}")

        fail = str(attrs.get("fail", ""))
        if fail not in FAIL_MODES:
            raise ValueError(
                f"fail must be one of {', '.join(repr(f) for f in FAIL_MODES)}, got {fail!r}"
            )

        return [], []

    def build_source(self, attrs: Dict[str, Any]) -> FrameSource:
        width, height = resolution(attrs)
        return SceneSource(
            width=width,
            height=height,
            hfov_deg=float(attrs.get("hfov_deg", DEFAULT_HFOV_DEG)),
            fail=str(attrs.get("fail", "")),
            link=str(attrs.get("link", DEFAULT_LINK)),
            slow_seconds=float(attrs.get("slow_seconds", 6.0)),
        )
