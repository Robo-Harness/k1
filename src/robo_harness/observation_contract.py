"""Sensor-only references. No object identity, evaluator state or approximate hashes.

Pixels always address the original RGB image. A fingerprint includes the exact
RGB, metric depth and calibration so repeated frame numbers cannot reuse a
measurement from another sensor snapshot. Hashing does not verify calibration.
"""

import hashlib
import json

import numpy as np


def camera_reference(obs, name):
    frame = obs["frame_id"]
    if type(frame) is not int or frame < 0:
        raise ValueError("frame_id must be a nonnegative integer")
    camera = obs["vision"][name]
    rgb = np.asarray(camera["color"])
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("Expected nonempty original HxWx3 RGB image")
    h, w = rgb.shape[:2]
    digest = hashlib.sha256(json.dumps([frame, name]).encode())
    for field in ("color", "depth", "intrinsic_matrix", "extrinsic_matrix"):
        digest.update(field.encode())
        value = camera.get(field)
        if value is None:
            digest.update(b"absent")
            continue
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("Sensor arrays cannot contain Python objects")
        if field == "depth" and array.squeeze().shape != (h, w):
            raise ValueError("Metric depth must be aligned to the original RGB image")
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return dict(
        camera=name,
        frame_id=frame,
        snapshot_id=digest.hexdigest(),
        original_size_wh=[w, h],
        pixel_axes="u right, v down",
    )


def validate_reference(args, obs):
    if "frame_id" in args and (
        type(args["frame_id"]) is not int or args["frame_id"] != obs["frame_id"]
    ):
        raise ValueError("Stale request; use CURRENT frame_id")
    reference = camera_reference(obs, args["camera"])
    if args.get("snapshot_id", reference["snapshot_id"]) != reference["snapshot_id"]:
        raise ValueError("Sensor snapshot changed; inspect CURRENT image before measuring")
    return reference
