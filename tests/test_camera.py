"""The course's checkpoints, as assertions.

Every check here runs with no machine, no viam-server, and no camera: the model
is constructed directly and its API methods are called. That is possible
because the frame source is a seam, and it is the reason a driver for hardware
you do not have can still be known to work.

The numbers come from `scene.py`, where the simulated scene is measured.
"""

from __future__ import annotations

import asyncio
import struct

import numpy as np
import pytest
from viam.media.video import CameraMimeType

from encode import DEPTH_MAGIC, decode_viam_depth, encode_viam_depth
from models.camera_base import COLOR_SOURCE, DEPTH_SOURCE
from models.d405_sim import D405Sim
from scene import BOXES, TABLE_Y_MM, WALL_Z_MM

WIDTH, HEIGHT = 640, 480
NEAR_BOX = BOXES[0]


def build(**attrs) -> D405Sim:
    """Construct the simulated model the way viam-server would."""
    from viam.proto.app.robot import ComponentConfig
    from viam.utils import dict_to_struct

    config = ComponentConfig(name="camera", attributes=dict_to_struct(
        {"width_px": WIDTH, "height_px": HEIGHT, **attrs}))
    return D405Sim.new(config, {})


def read_depth(camera: D405Sim) -> np.ndarray:
    images, _ = asyncio.run(camera.get_images(filter_source_names=[DEPTH_SOURCE]))
    return decode_viam_depth(images[0].data)


# --- the depth encoding -----------------------------------------------------

def test_depth_header_is_the_layout_the_rdk_reads():
    depth = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    data = encode_viam_depth(depth)

    assert data[:8] == DEPTH_MAGIC
    assert struct.unpack(">QQ", data[8:24]) == (3, 2)      # width, then height
    assert len(data) == 24 + 3 * 2 * 2                     # header + uint16 each
    assert data[24:26] == b"\x00\x01"                      # big-endian, not little


def test_depth_round_trips():
    depth = np.random.default_rng(0).integers(0, 65535, (12, 20), dtype=np.uint16)
    assert np.array_equal(decode_viam_depth(encode_viam_depth(depth)), depth)


def test_depth_rejects_a_wrong_magic():
    data = bytearray(encode_viam_depth(np.zeros((2, 2), dtype=np.uint16)))
    data[:8] = b"NOTDEPTH"
    with pytest.raises(ValueError, match="magic"):
        decode_viam_depth(bytes(data))


# --- what the camera reports ------------------------------------------------

def test_get_images_returns_both_streams_named():
    camera = build()
    images, metadata = asyncio.run(camera.get_images())

    assert [image.name for image in images] == [COLOR_SOURCE, DEPTH_SOURCE]
    assert images[0].mime_type == CameraMimeType.JPEG
    assert images[1].mime_type == CameraMimeType.VIAM_RAW_DEPTH
    assert metadata.captured_at.seconds > 0


def test_filter_source_names_returns_only_what_was_asked_for():
    camera = build()
    images, _ = asyncio.run(camera.get_images(filter_source_names=[COLOR_SOURCE]))
    assert [image.name for image in images] == [COLOR_SOURCE]


def test_properties_describe_the_configured_stream():
    camera = build()
    properties = asyncio.run(camera.get_properties())
    intrinsics = properties.intrinsic_parameters

    assert properties.supports_pcd is True
    assert (intrinsics.width_px, intrinsics.height_px) == (WIDTH, HEIGHT)
    assert intrinsics.center_x_px == pytest.approx((WIDTH - 1) / 2)
    assert intrinsics.center_y_px == pytest.approx((HEIGHT - 1) / 2)
    assert intrinsics.focal_x_px > 0


def test_intrinsics_follow_the_configured_resolution():
    small = asyncio.run(build(width_px=320, height_px=240).get_properties())
    large = asyncio.run(build(width_px=640, height_px=480).get_properties())

    # Same field of view over twice the pixels is twice the focal length.
    assert large.intrinsic_parameters.focal_x_px == pytest.approx(
        2 * small.intrinsic_parameters.focal_x_px, rel=1e-3)


# --- the scene, measured through the driver ---------------------------------

def test_the_wall_reads_its_recorded_distance():
    depth = read_depth(build())
    # The top-left corner looks past the table and the boxes, at the wall.
    assert depth[0, 0] == pytest.approx(WALL_Z_MM, abs=1)


def near_box_face(points: np.ndarray) -> np.ndarray:
    """The points that landed on the near box's front face.

    Depth alone is not enough to pick it out: the table is a plane, so it has
    points at every distance, including this one. The face is what sits at that
    distance *and* above the table.
    """
    return points[(np.abs(points[:, 2] - NEAR_BOX.front_z_mm) < 1.5)
                  & (points[:, 1] < TABLE_Y_MM - 5)]


def test_the_near_box_front_face_reads_its_recorded_distance():
    face = near_box_face(point_cloud_points(build()))

    assert len(face) > 500, "the near box's front face should fill part of the frame"
    assert face[:, 2].mean() == pytest.approx(NEAR_BOX.front_z_mm, abs=1)


def test_the_near_box_measures_its_recorded_width():
    face = near_box_face(point_cloud_points(build()))

    measured = face[:, 0].max() - face[:, 0].min()
    # One pixel at 250 mm is about 0.7 mm across, so the face is measured to
    # within a pixel of the width the scene records.
    assert measured == pytest.approx(NEAR_BOX.width_mm, abs=2)


def test_nothing_in_the_scene_is_nearer_than_the_table_edge():
    depth = read_depth(build())
    readings = depth[depth > 0]
    # The nearest thing the camera sees is the table running under the lens,
    # not either box: a lower minimum than this means a ray went somewhere the
    # scene does not describe.
    assert readings.min() < NEAR_BOX.front_z_mm
    assert readings.max() == pytest.approx(WALL_Z_MM, abs=1)


def test_the_table_sits_where_the_scene_says():
    points = point_cloud_points(build())
    below = points[points[:, 1] > TABLE_Y_MM - 2]
    assert len(below) > 1000
    assert below[:, 1].max() == pytest.approx(TABLE_Y_MM, abs=1)


# --- the point cloud --------------------------------------------------------

def point_cloud_points(camera: D405Sim) -> np.ndarray:
    data, mime_type = asyncio.run(camera.get_point_cloud())
    assert mime_type == CameraMimeType.PCD
    return parse_pcd(data)


def parse_pcd(data: bytes) -> np.ndarray:
    """Read the binary PCD back, the way the RDK's reader does."""
    header, body = data.split(b"DATA binary\n", 1)
    fields = dict(line.split(b" ", 1) for line in header.strip().splitlines())

    assert fields[b"VERSION"] == b".7"
    assert fields[b"FIELDS"] == b"x y z rgb"
    assert fields[b"TYPE"] == b"F F F I"
    assert fields[b"HEIGHT"] == b"1"

    count = int(fields[b"POINTS"])
    assert int(fields[b"WIDTH"]) == count
    record = np.frombuffer(body, dtype=[("x", "<f4"), ("y", "<f4"),
                                        ("z", "<f4"), ("rgb", "<u4")])
    assert len(record) == count
    return np.stack([record["x"], record["y"], record["z"]], axis=1)


def test_the_point_cloud_carries_no_points_without_a_reading():
    points = point_cloud_points(build())
    assert (points[:, 2] > 0).all(), "a zero depth reading is not a point at the lens"


def test_every_pixel_with_a_reading_becomes_one_point():
    camera = build()
    depth = read_depth(camera)
    points = point_cloud_points(camera)
    assert len(points) == int((depth > 0).sum())
