import numpy as np
from scipy.spatial.transform import Rotation

from robo_harness.interaction_evidence import compare_anchors


def arms(xyz=(0, 0, 0), rotation=None):
    return {
        "arm": {"xyz_world_m": list(xyz), "axes_world": np.eye(3) if rotation is None else rotation}
    }


def points(offset=(0, 0, 0)):
    return {i: np.array(p) + offset for i, p in enumerate([[0, 0, 1], [0.01, 0, 1], [0, 0.01, 1]])}


def test_carried_points_have_low_residual_but_no_attachment_certificate():
    r = compare_anchors(points(), points((0.03, 0, 0)), arms(), arms((0.03, 0, 0)))
    assert r["relative_to_hands"]["arm"]["rigid_co_motion_residual_m"] < 1e-9
    assert "attachment_verified" not in r


def test_stationary_target_and_moving_hand_warns():
    r = compare_anchors(points(), points(), arms(), arms((0.03, 0, 0)))
    assert r["relative_to_hands"]["arm"]["inconsistent_with_rigid_carry_hypothesis"]


def test_stationary_scene_does_not_assert_attachment_or_failure():
    r = compare_anchors(points(), points(), arms(), arms())
    assert not r["relative_to_hands"]["arm"]["inconsistent_with_rigid_carry_hypothesis"]


def test_rotation_is_accounted_for():
    rot = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    p = points()
    r = compare_anchors(p, {i: rot @ xyz for i, xyz in p.items()}, arms(), arms(rotation=rot))
    assert r["relative_to_hands"]["arm"]["rigid_co_motion_residual_m"] < 1e-9


def test_missing_correspondence_is_unknown():
    assert compare_anchors(points(), {0: np.ones(3)}, arms(), arms())["status"] == "unknown"


def test_manager_resets_on_reinitialization_and_marks_lost_unknown():
    from robo_harness.interaction_evidence import InteractionEvidence
    from test_tracked_points import scene

    obs = scene()
    obs["arms"] = arms()
    entry = {
        "camera": "cam",
        "lost": False,
        "frame": 0,
        "track": {
            "points_uv": [[30, 30], [40, 30], [30, 40]],
            "visible": [True] * 3,
            "needs_revalidation": [False] * 3,
        },
    }
    m = InteractionEvidence()
    m.update({"S1": entry}, obs)
    assert m.context()["S1"]["status"] == "unknown"
    obs["frame_id"] = entry["frame"] = 12
    m.update({"S1": entry}, obs)
    assert m.context()["S1"]["matched_anchor_count"] == 3
    replacement = dict(entry)
    m.update({"S1": replacement}, obs)
    assert m.context()["S1"]["status"] == "unknown"
    replacement["lost"] = True
    m.update({"S1": replacement}, obs)
    assert m.context()["S1"]["reason"] == "tracking_not_current"
