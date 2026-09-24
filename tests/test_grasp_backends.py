import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robo_harness.grasp_backends import contact_aligned_pose, validate_predictions


def test_contact_alignment_uses_robot_pad_offset_and_semantic_axes():
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [20, 30, 40], degrees=True).as_matrix()
    pose[:3, 3] = [0.2, -0.1, 0.8]
    axes = {"approach": [0, 0, 1], "closing": [1, 0, 0], "lateral": [0, -1, 0]}
    local = {"approach": [0, 0, 1], "closing": [0, -1, 0], "lateral": [-1, 0, 0]}
    robot = {
        "axes_world": np.eye(3),
        "geometry": {
            "axes_local": local,
            "finger_pad_centers_local_m": [[0, 0.04, -0.0036], [0, -0.04, -0.0036]],
        },
    }
    result = contact_aligned_pose(pose, robot, axes, [0, 0, 0.1034])
    r = Rotation.from_quat(result["target_quaternion_xyzw"]).as_matrix()
    np.testing.assert_allclose(
        np.asarray(result["candidate_tcp_world_m"]) + r @ [0, 0, -0.0036],
        pose[:3, 3] + pose[:3, 2] * 0.1034,
    )
    np.testing.assert_allclose(r @ local["approach"], pose[:3, 2])
    assert abs((r @ local["closing"]) @ pose[:3, 0]) == pytest.approx(1)
    assert result["collision_certified"] is False


@pytest.mark.parametrize("corruption", ["reflection", "nan", "bottom_row", "scale"])
def test_invalid_provider_transforms_rejected(corruption):
    pose = np.eye(4)
    if corruption == "reflection":
        pose[0, 0] = -1
    elif corruption == "nan":
        pose[0, 3] = np.nan
    elif corruption == "bottom_row":
        pose[3, 0] = 1
    else:
        pose[0, 0] = 2
    with pytest.raises(ValueError):
        validate_predictions(pose[None], [0.9])


def test_provider_size_mismatch():
    with pytest.raises(ValueError):
        validate_predictions(np.eye(4)[None], [0.9, 0.8])


def test_learned_candidates_accept_non_slender_regions_and_keep_scores_separate(monkeypatch):
    import robo_harness.grasp_backends as module
    from robo_harness.grasp_visualization import candidate_panel
    from test_grasp_geometry import fixture

    obs, manager = fixture()
    manager.summary = lambda *args: {
        "status": "current_visible_surface_estimate",
        "major_axis_reliable": False,
    }
    pose = np.eye(4)
    pose[:3, 3] = [0, 0, 0.9]
    monkeypatch.setattr(module, "request_graspgen", lambda *args: (pose[None], [0.9]))
    result = module.learned_candidates(
        manager,
        "S1",
        "hand",
        obs,
        "test",
        {"approach": [0, 0, 1], "closing": [1, 0, 0], "lateral": [0, -1, 0]},
        [0, 0, 0.1034],
    )
    item = result["candidates"][0]
    assert item["learned_confidence_not_success_probability"] == 0.9
    assert "visibility_score" in item
    assert not item["jaw_fit_verified"]
    camera = next(iter(obs["vision"]))
    raw = obs["vision"][camera]["color"].copy()
    assert candidate_panel(result, obs, "hand", camera).shape == raw.shape
    np.testing.assert_array_equal(raw, obs["vision"][camera]["color"])


def test_learned_provider_is_not_called_on_stale_region(monkeypatch):
    import robo_harness.grasp_backends as module
    from test_grasp_geometry import fixture

    obs, manager = fixture()
    manager.summary = lambda *args: {"status": "unknown_reinspect"}
    monkeypatch.setattr(
        module, "request_graspgen", lambda *args: pytest.fail("Must reject stale input first")
    )
    with pytest.raises(ValueError, match="current measured"):
        module.learned_candidates(manager, "S1", "hand", obs, "test", {}, [0, 0, 0])
