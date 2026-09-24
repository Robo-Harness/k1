import itertools
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robo_harness.grasp_geometry import candidates
from test_tracked_points import scene


def fixture():
    obs = scene()
    obs["arms"] = {
        "hand": {
            "xyz_world_m": [0, 0.1, 1.2],
            "axes_world": Rotation.from_euler("x", 180, degrees=True).as_matrix().tolist(),
            "gripper_opening_m": 0.08,
            "geometry": {
                "axes_local": {"approach": [0, 0, 1], "closing": [0, -1, 0], "lateral": [-1, 0, 0]},
                "finger_pad_centers_local_m": [[0, 0.04, -0.0036], [0, -0.04, -0.0036]],
                "sweep_probes_local_m": list(
                    itertools.product([-0.01, 0.01], [-0.04, 0.04], [-0.08, 0.01])
                ),
            },
        }
    }

    class Manager:
        entries = {"S1": {"xyz": np.array([[x, 0, 1] for x in np.linspace(-0.04, 0.04, 30)])}}

        def summary(self, key, obs):
            return {
                "status": "current_visible_surface_estimate",
                "major_axis_reliable": True,
                "surface_reference_world_m": [0, 0, 1],
                "major_axis_world_unsigned": [1, 0, 0],
            }

    return obs, Manager()


def test_candidates_map_calibrated_axes_and_return_hypotheses_only():
    obs, manager = fixture()
    result = candidates(manager, "S1", "hand", obs)
    assert len(result["candidates"]) == 3
    for item in result["candidates"]:
        r = Rotation.from_quat(item["target_quaternion_xyzw"])
        np.testing.assert_allclose(r.apply([0, 0, 1]), item["approach_world"], atol=1e-8)
        np.testing.assert_allclose(r.apply([0, -1, 0]), item["closing_world_unsigned"], atol=1e-8)
        assert abs(np.dot(item["closing_world_unsigned"], [1, 0, 0])) < 1e-8
        assert np.linalg.norm(
            np.array(item["candidate_tcp_world_m"]) - item["pregrasp_tcp_world_m"]
        ) == pytest.approx(0.07)
        assert "grasp_success" not in item and "collision_free" not in item


def test_candidates_require_open_gripper():
    obs, manager = fixture()
    obs["arms"]["hand"]["gripper_opening_m"] = 0
    with pytest.raises(ValueError, match="Open gripper"):
        candidates(manager, "S1", "hand", obs)
