"""The same frames, off a real Intel RealSense D405.

Everything vendor-specific in this module lives in this file: starting a
pipeline, waiting on frames, reading the device's own calibration, and putting
the device back. The camera API code above it does not change.

`pyrealsense2` is imported when a source is built rather than when the module
loads, so a machine that only runs the simulated model needs no vendor SDK.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from frames import Frame, FrameSource, Intrinsics

# The D405 is a short-range camera. Readings outside its working range are
# reported as no reading rather than as geometry.
MIN_RANGE_MM = 40.0
MAX_RANGE_MM = 1000.0


def _distortion_model(rs, model) -> str:
    """librealsense's distortion enum, named the way the RDK names them.

    An unmapped value, including "no distortion", comes back empty, which is
    how a rectified stream says there is nothing to correct.
    """
    return {
        rs.distortion.brown_conrady: "brown_conrady",
        rs.distortion.modified_brown_conrady: "brown_conrady",
        rs.distortion.inverse_brown_conrady: "inverse_brown_conrady",
    }.get(model, "")


class RealSenseSource(FrameSource):
    """Color and depth from a connected D405, aligned to the color stream.

    Args:
        width: requested stream width.
        height: requested stream height.
        serial_number: which device to open when more than one is plugged in.
            Empty means the first one the SDK lists.
        fps: requested frame rate.
    """

    def __init__(self, width: int = 640, height: int = 480,
                 serial_number: str = "", fps: int = 30) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:                       # pragma: no cover
            raise RuntimeError(
                "pyrealsense2 is not installed. It is declared in "
                "requirements.txt; if you are running from source, install it "
                "into the module's virtual environment."
            ) from exc

        self._rs = rs
        self._width, self._height = width, height
        self._serial = serial_number
        self._fps = fps
        self._pipeline = None
        self._align = rs.align(rs.stream.color)
        self._depth_scale_mm = 1.0
        self._intrinsics: Optional[Intrinsics] = None
        self._frame_rate = float(fps)
        self._start()

    # --- device lifecycle --------------------------------------------------

    def _start(self) -> None:
        rs = self._rs
        config = rs.config()
        if self._serial:
            config.enable_device(self._serial)
        config.enable_stream(rs.stream.color, self._width, self._height,
                             rs.format.rgb8, self._fps)
        config.enable_stream(rs.stream.depth, self._width, self._height,
                             rs.format.z16, self._fps)

        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(config)
        except RuntimeError as exc:
            # The device is the authority on what it can stream. Rather than
            # keep a table of resolutions here that some firmware will
            # eventually disagree with, ask it and put the answer in the error.
            raise RuntimeError(
                f"could not start the camera at {self._width}x{self._height} "
                f"at {self._fps} fps: {exc}. {self._available(rs)}"
            ) from exc

        # The device knows its own calibration. Reading it here is what keeps
        # one implementation correct across units and resolutions; a datasheet
        # describes the model, not the camera on the desk.
        video = profile.get_stream(rs.stream.color).as_video_stream_profile()
        i = video.get_intrinsics()
        self._intrinsics = Intrinsics(
            width=i.width, height=i.height,
            fx=i.fx, fy=i.fy, ppx=i.ppx, ppy=i.ppy,
            distortion_model=_distortion_model(rs, i.model),
            # librealsense reports coefficients in OpenCV order
            # [k1, k2, p1, p2, k3]; the RDK reads all the radial terms first.
            distortion_coeffs=(i.coeffs[0], i.coeffs[1], i.coeffs[4],
                               i.coeffs[2], i.coeffs[3]),
        )
        self._frame_rate = float(video.fps())

        # Raw depth units are device-specific: convert once, here, so every
        # reading above this file is already in millimeters.
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale_mm = depth_sensor.get_depth_scale() * 1000.0
        self._pipeline = pipeline

    def _available(self, rs) -> str:
        """What the camera we were asked for can stream, for an error message.

        Only that camera. With two devices plugged in, listing every mode on
        the bus produces a list the configured camera does not support, which
        is worse than saying nothing.
        """
        try:
            devices = list(rs.context().query_devices())
            if not devices:
                return ("No RealSense camera is connected: check the cable and "
                        "that the device shows up in rs-enumerate-devices.")

            if self._serial:
                devices = [d for d in devices
                           if d.get_info(rs.camera_info.serial_number) == self._serial]
                if not devices:
                    connected = ", ".join(
                        f"{d.get_info(rs.camera_info.name)} ({d.get_info(rs.camera_info.serial_number)})"
                        for d in rs.context().query_devices())
                    return (f"No camera with serial {self._serial} is connected. "
                            f"Connected: {connected}.")
            elif len(devices) > 1:
                names = ", ".join(
                    f"{d.get_info(rs.camera_info.name)} ({d.get_info(rs.camera_info.serial_number)})"
                    for d in devices)
                return (f"{len(devices)} cameras are connected and no "
                        f"serial_number is configured, so the first one was "
                        f"used: {names}.")

            device = devices[0]
            name = device.get_info(rs.camera_info.name)
            modes = set()
            for sensor in device.query_sensors():
                for profile in sensor.get_stream_profiles():
                    video = profile.as_video_stream_profile()
                    if video:
                        modes.add((video.width(), video.height(), video.fps()))
            # If the camera does support what was asked for, the resolution is
            # not the problem and a list of modes is a red herring. The usual
            # cause is that something else already has the device open, which
            # is a different thing for the reader to go and check.
            if (self._width, self._height, self._fps) in modes:
                return (f"{name} does support {self._width}x{self._height} at "
                        f"{self._fps} fps, so something else is likely holding "
                        f"the device open. Check for another viam-server, "
                        f"module, or script using it.")

            # Put the modes at the requested resolution first. Truncating a
            # sorted list can cut off the very mode the reader needs, which is
            # the one nearest what they asked for.
            wanted = [m for m in modes if m[0] == self._width and m[1] == self._height]
            other = [m for m in modes if m not in wanted]
            listed = sorted(wanted, reverse=True) + sorted(other, reverse=True)[:8]
            return f"{name} supports: " + ", ".join(
                f"{w}x{h}@{f}" for w, h, f in listed
            )
        except Exception:                                # pragma: no cover
            return "Could not enumerate the camera's supported stream profiles."

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None

    # --- the source contract ----------------------------------------------

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @property
    def intrinsics(self) -> Intrinsics:
        if self._intrinsics is None:
            raise RuntimeError("the camera has not been started")
        return self._intrinsics

    def read(self) -> Frame:
        if self._pipeline is None:
            # The device went away on an earlier call. Try to pick it up again
            # before failing: a cable knocked loose should not need a
            # reconfiguration to recover from.
            self._start()

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError as exc:
            self.close()
            raise RuntimeError(
                f"the camera stopped delivering frames: {exc}. The next call "
                "will try to reopen it."
            ) from exc

        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("the camera returned an incomplete frame set")

        color = np.asanyarray(color_frame.get_data())
        raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_mm = raw * self._depth_scale_mm
        # Out of range is not a measurement. Zero is how the rest of the module
        # says so, and dropping these keeps them out of the point cloud.
        depth_mm[(depth_mm < MIN_RANGE_MM) | (depth_mm > MAX_RANGE_MM)] = 0.0

        # Host time, deliberately. librealsense's frame timestamps come from a
        # domain that varies with the device and its metadata support, so they
        # are not always a Unix epoch. A caller correlating a frame with a robot
        # pose needs a clock it shares with the machine, and this is that clock.
        return Frame(color=color,
                     depth=np.rint(depth_mm).astype(np.uint16),
                     captured_at=time.time())
