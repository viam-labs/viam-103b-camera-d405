# viam-103b-camera-d405

The reference camera module for **Viam 103B**, the self-paced course on writing
a camera driver. It implements the [Viam camera
API](https://docs.viam.com/dev/reference/apis/components/camera/) for an Intel
RealSense D405, in two models:

| Model | What it is |
| ----- | ---------- |
| `viam-labs:camera:d405-sim` | A depth camera with no hardware behind it. It renders a scene whose dimensions are recorded in [`src/scene.py`](src/scene.py) |
| `viam-labs:camera:d405` | The same camera API over a connected D405, through `pyrealsense2` |

Both models share one implementation. The camera API code — the encodings, the
intrinsics, the point cloud — lives in
[`src/models/camera_base.py`](src/models/camera_base.py) and is written once.
What differs between the models is where frames come from.

> This is course material, not a supported module. For production use of a
> RealSense camera, see
> [`viam-modules/viam-camera-realsense`](https://github.com/viam-modules/viam-camera-realsense).

## Why it is built this way

A driver has two jobs that change for different reasons: speaking the platform's
API, and speaking to a device. Putting a seam between them
([`src/frames.py`](src/frames.py)) buys three things the course depends on:

1. **Every exercise runs with no camera attached.** The simulated source carries
   the course from the first implementation page to the last.
2. **A second model costs a source, not a driver.** Adding
   `viam-labs:camera:d405` changed no camera API code.
3. **Correctness is measurable.** The simulated scene has known geometry, so
   `tests/test_camera.py` can assert that the near box's front face is 250 mm
   away and 180 mm across rather than that the picture looks about right.

## The scene

The simulated model looks at a back wall, a table, and two boxes:

| Feature | Where it is |
| ------- | ----------- |
| Back wall | 600 mm from the lens |
| Table top | 150 mm below the optical axis |
| Near box | front face 250 mm away, 180 mm wide, 120 mm tall |
| Far box | front face 420 mm away, 100 mm wide, 100 mm tall |

Depth is the Z distance along the optical axis in millimeters, and **0 means no
reading**. Points, like everything else on the Viam platform, are millimeters.

## Configure

### Model `viam-labs:camera:d405-sim`

```json
{
  "width_px": 640,
  "height_px": 480,
  "hfov_deg": 87
}
```

| Attribute | Type | Inclusion | Description |
| --------- | ---- | --------- | ----------- |
| `width_px` | int | Optional | Stream width in pixels. Default 640 |
| `height_px` | int | Optional | Stream height in pixels. Default 480 |
| `hfov_deg` | number | Optional | Horizontal field of view used to derive the intrinsics. Default 87 |
| `fail` | string | Optional | Simulate a failure on every read: `absent` for a device that was never there, `dropped` for one that vanished mid-stream. Default is no failure |

### Model `viam-labs:camera:d405`

```json
{
  "width_px": 640,
  "height_px": 480,
  "fps": 30,
  "serial_number": ""
}
```

| Attribute | Type | Inclusion | Description |
| --------- | ---- | --------- | ----------- |
| `width_px` | int | Optional | Requested stream width. Default 640 |
| `height_px` | int | Optional | Requested stream height. Default 480 |
| `fps` | int | Optional | Requested frame rate. Default 30 |
| `serial_number` | string | Optional | Which camera to open when more than one is connected. Empty means the first one found |
| `require_usb3` | bool | Optional | Refuse to build when the camera negotiated a USB 2 link, which puts the resource into UNHEALTHY with an error naming the port. Default false: a camera on a slow link still works for some configurations, and that call belongs to whoever configured it |

Validation rejects the settings that are wrong on their face. Whether this
camera can stream a given size is answered by the device when the stream opens,
and the error lists what it does support.

A camera on a USB 2 link opens and then fails to deliver frames, so the driver
names the link in that error and logs it once at startup. `require_usb3` turns
that from a warning into a refusal to build, which is the only way a module can
put a resource into `UNHEALTHY`: `viam-server` sets that state when
construction fails, and there is no call for it at runtime.

## API

| Method | Returns |
| ------ | ------- |
| `GetImages` | `color` as `image/jpeg` and `depth` as `image/vnd.viam.dep`, plus the capture timestamp. `filter_source_names` selects a subset |
| `GetPointCloud` | Binary PCD (`pointcloud/pcd`), one point per depth reading, colored from the color frame |
| `GetProperties` | Intrinsics for the configured stream, the MIME types above, and `supports_pcd` |

## Develop

```sh
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt pytest
PYTHONPATH=src ./venv/bin/python -m pytest
```

The tests need no machine, no `viam-server`, and no camera.

To run it on a machine, use hot reload from the Viam CLI:

```sh
viam module reload --part-id <your-part-id>
```

## Course steps

The module was built by walking the course's exercises in order, and each
page's end state is a tag:

| Tag | Page |
| --- | ---- |
| `step-03-scaffold` | Generate the camera module |
| `step-04-color` | Return a color image |
| `step-05-depth` | Return depth |
| `step-07-properties` | Report the camera's properties |
| `step-08-point-cloud` | Build the point cloud |
| `step-09-tests` | Test the driver with no camera |
| `step-10-hardware` | Add the real D405 |
| `step-11-failures` | Survive the hardware |

`git diff step-04-color step-05-depth` is the answer to one exercise.
