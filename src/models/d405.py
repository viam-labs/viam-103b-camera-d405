"""`viam-labs:camera:d405` — the camera API over a real Intel RealSense D405.

The same class as the simulated model, over a different frame source. Adding
this model cost one source and one subclass; none of the encoding, intrinsics,
or point-cloud code changed.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Sequence, Tuple

from viam.proto.app.robot import ComponentConfig
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict

from frames import FrameSource
from models.camera_base import D405CameraBase, resolution
from realsense import RealSenseSource

DEFAULT_FPS = 30


class D405(D405CameraBase, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "camera"), "d405")

    @classmethod
    def validate_config(cls, config: ComponentConfig
                        ) -> Tuple[Sequence[str], Sequence[str]]:
        """Reject what can be judged without the camera.

        Validation runs wherever the configuration is edited, which is not
        necessarily where the device is. So it checks the settings that are
        wrong on their face — a negative width, an empty serial number — and
        leaves "can this camera stream 999x999" to the moment the stream is
        opened, where the device itself can answer and say what it does support.
        """
        attrs = struct_to_dict(config.attributes)

        for key in ("width_px", "height_px", "fps"):
            value = attrs.get(key)
            if value is None:
                continue
            if float(value) != int(float(value)) or int(value) <= 0:
                raise ValueError(
                    f"{key} must be a positive whole number, got {value}"
                )

        if "serial_number" in attrs and not str(attrs["serial_number"]).strip():
            raise ValueError(
                "serial_number is set but empty; remove it to use the first "
                "camera found, or give the serial of the one you want"
            )

        return [], []

    def build_source(self, attrs: Dict[str, Any]) -> FrameSource:
        width, height = resolution(attrs)
        return RealSenseSource(
            width=width,
            height=height,
            serial_number=str(attrs.get("serial_number", "")),
            fps=int(attrs.get("fps", DEFAULT_FPS)),
        )
