import cv2
import numpy as np

from robo_harness.geometry import unproject
from robo_harness.tracked_points import TrackedPoints


def scene(shift=0, camera_x=0, frame=0):
    rgb = np.random.default_rng(42).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    rgb = cv2.warpAffine(rgb, np.float32([[1, 0, shift], [0, 1, 0]]), (100, 100))
    pose = np.eye(4)
    pose[0, 3] = camera_x
    return {
        "frame_id": frame,
        "vision": {
            "cam": {
                "color": rgb,
                "depth": np.ones((100, 100)),
                "intrinsic_matrix": np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]]),
                "extrinsic_matrix": pose,
            }
        },
    }


def start(obs):
    tracks = TrackedPoints()
    point = unproject(obs["vision"]["cam"], [50, 50])
    tracks.start({**point, "id": "P1", "camera": "cam"}, obs)
    return tracks


def test_camera_translation_is_not_reported_as_world_motion():
    tracks = start(scene())
    obs = scene(-2, 0.02, 1)
    tracks.update(obs)
    point = tracks.context()[0]
    assert point["status"] == "tracked_estimate"
    assert point["world_distance_m"] < 0.002
    assert point["pixel"][0] == 48
    raw = obs["vision"]["cam"]["color"].copy()
    tracks.overlay(raw, "cam", 1)
    np.testing.assert_array_equal(raw, obs["vision"]["cam"]["color"])


def test_world_motion_and_loss_are_distinguished():
    tracks = start(scene())
    tracks.update(scene(2, 0, 1))
    assert abs(tracks.context()[0]["world_displacement_m"][0] - 0.02) < 0.002
    assert tracks.context()[0]["motion_evidence"] == "above_5mm_threshold"
    obs = scene(2, 0, 2)
    obs["vision"]["cam"]["depth"][:] = np.nan
    tracks.update(obs)
    assert tracks.context()[0]["motion_evidence"] == "unknown"
    assert "xyz_world_m" not in tracks.context()[0]
    tracks.update(scene(2, 0, 3))
    assert tracks.context()[0]["status"] == "lost_remeasure"
    assert tracks.context()[0]["reason"]


def test_track_cap_and_frame_gap():
    tracks = start(scene())
    tracks.update(scene(frame=20))
    assert tracks.context()[0]["motion_evidence"] == "unknown"


def test_explicit_invalidation_never_reacquires():
    tracks = start(scene())
    tracks.invalidate(["P1"], "contact_observed", 0)
    tracks.update(scene(frame=1))
    point = tracks.context()[0]
    assert point["reason"] == "contact_observed"
    assert point["status"] == "lost_remeasure"
    assert "pixel" not in point and "xyz_world_m" not in point


def test_backend_is_forwarded(monkeypatch):
    import robo_harness.tracked_points as module

    original = module.FeatureTracker
    calls = []

    def factory(backend, device):
        calls.append((backend, device))
        return original("lk")

    monkeypatch.setattr(module, "FeatureTracker", factory)
    obs = scene()
    point = {**unproject(obs["vision"]["cam"], [50, 50]), "id": "P1", "camera": "cam"}
    tracks = TrackedPoints(backend="tapnext", device="cuda:1")
    tracks.start(point, obs)
    assert calls == [("tapnext", "cuda:1")]
    tracks.update(scene(frame=1))
    assert tracks.context()[0]["backend"] == "tapnext"
