"""The camera API, written once.

Both models in this module use this class. It knows about MIME types,
intrinsics, and point clouds, and it knows nothing about where frames come
from: a subclass supplies a `FrameSource` and inherits every method the Viam
camera API requires.

The camera API requires three methods of a model:

    get_images        the frames, each named, each with a MIME type
    get_point_cloud   the scene as points, with the MIME type describing them
    get_properties    the lens, the encodings, and what else this camera can do

`close` comes from the resource base and is where the device is handed back.
viam-server calls it whenever the resource goes away, which includes every
configuration change: a reconfigured resource is closed and built again rather
than updated in place.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from google.protobuf.timestamp_pb2 import Timestamp
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, NamedImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, ResponseMetadata
from viam.proto.component.camera import DistortionParameters, IntrinsicParameters
from viam.resource.base import ResourceBase
from viam.utils import struct_to_dict

from encode import encode_jpeg, encode_viam_depth
from frames import FrameSource

COLOR_SOURCE = "color"
DEPTH_SOURCE = "depth"

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


class D405CameraBase(Camera):
    """Everything the camera API asks for, over an unspecified frame source."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._source: Optional[FrameSource] = None

    # --- lifecycle ---------------------------------------------------------

    @classmethod
    def new(cls, config: ComponentConfig,
            dependencies: Mapping[ResourceName, ResourceBase]) -> "D405CameraBase":
        """Build the model and open its frame source.

        This is the only place a resource is configured. There is no separate
        update step: on a configuration change viam-server closes this resource
        and calls `new` again, so opening the device here and releasing it in
        `close` covers the whole lifecycle.
        """
        self = cls(config.name)
        self._source = self.build_source(struct_to_dict(config.attributes))
        return self

    async def close(self) -> None:  # type: ignore[override]
        if self._source is not None:
            self._source.close()
            self._source = None

    # --- what a subclass supplies -----------------------------------------

    def build_source(self, attrs: Dict[str, Any]) -> FrameSource:
        raise NotImplementedError

    def source(self) -> FrameSource:
        if self._source is None:
            raise RuntimeError(f"{self.name} has no frame source: it was closed")
        return self._source

    # --- the camera API ----------------------------------------------------

    async def get_images(
        self,
        *,
        filter_source_names: Optional[Sequence[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[Sequence[NamedImage], ResponseMetadata]:
        frame = self.source().read()

        wanted = set(filter_source_names or (COLOR_SOURCE, DEPTH_SOURCE))
        images = []
        if COLOR_SOURCE in wanted:
            images.append(NamedImage(COLOR_SOURCE, encode_jpeg(frame.color),
                                     CameraMimeType.JPEG))
        if DEPTH_SOURCE in wanted:
            images.append(NamedImage(DEPTH_SOURCE, encode_viam_depth(frame.depth),
                                     CameraMimeType.VIAM_RAW_DEPTH))

        # The timestamp says when the camera saw this, not when the request
        # arrived. A caller matching a frame to a robot pose depends on it.
        captured_at = Timestamp()
        captured_at.FromNanoseconds(int(frame.captured_at * 1e9))
        return images, ResponseMetadata(captured_at=captured_at)

    async def get_properties(self, *, timeout: Optional[float] = None,
                             **kwargs) -> Camera.Properties:
        intr = self.source().intrinsics
        return Camera.Properties(
            supports_pcd=False,
            intrinsic_parameters=IntrinsicParameters(
                width_px=intr.width,
                height_px=intr.height,
                focal_x_px=intr.fx,
                focal_y_px=intr.fy,
                center_x_px=intr.ppx,
                center_y_px=intr.ppy,
            ),
            # Both sources hand up rectified frames, so the model is named and
            # its coefficients are zero. Reporting numbers here would have a
            # caller undistort an image that is already straight.
            distortion_parameters=DistortionParameters(model="brown_conrady",
                                                       parameters=[0.0] * 5),
            mime_types=[CameraMimeType.JPEG, CameraMimeType.VIAM_RAW_DEPTH],
        )

def resolution(attrs: Dict[str, Any]) -> Tuple[int, int]:
    """Read the configured stream size, with the module's defaults."""
    return (int(attrs.get("width_px", DEFAULT_WIDTH)),
            int(attrs.get("height_px", DEFAULT_HEIGHT)))
