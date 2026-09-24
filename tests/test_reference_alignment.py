from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robo_harness.reference_alignment import measured_region_alignment, translation_hypothesis


def test_tool_schema_avoids_unsupported_unique_items():
    import json
    from robo_harness.runtime import ToolRegistry
    from test_motion import Robot

    robot = Robot()
    registry = ToolRegistry(robot, region_tools=True)
    schema = next(
        s
        for s in registry.schemas(robot.observe())
        if s["function"]["name"] == "align_region_references"
    )
    assert "uniqueItems" not in json.dumps(schema)
    # Uniqueness remains enforced by translation_hypothesis (test_invalid_axes).


def test_offset_compensation_and_clearance():
    result = translation_hypothesis(
        [0.04, -0.02, 0.2], [0.2, 0.3, 0], [0, 0, 0.25], [0, 0, 0], ["x", "y"]
    )
    np.testing.assert_allclose(result["target_tcp_world_m"], [0.16, 0.32, 0.25])
    moved_reference = np.array([0.04, -0.02, 0.2]) + result["translation_world_m"]
    np.testing.assert_allclose(moved_reference, [0.2, 0.3, 0.2])


def test_full_alignment_is_rigid_coordinate_equivariant():
    rng = np.random.default_rng(17)
    for _ in range(20):
        source, target, tcp, offset, origin = rng.normal(size=(5, 3))
        rotation = Rotation.random(random_state=rng)
        first = translation_hypothesis(source, target, tcp, offset)
        second = translation_hypothesis(
            rotation.apply(source) + origin,
            rotation.apply(target) + origin,
            rotation.apply(tcp) + origin,
            rotation.apply(offset),
        )
        np.testing.assert_allclose(
            second["target_tcp_world_m"],
            rotation.apply(first["target_tcp_world_m"]) + origin,
            atol=1e-12,
        )


@pytest.mark.parametrize("axes", [[], ["x", "x"], ["camera_x"], ["z", "oops"]])
def test_invalid_axes(axes):
    with pytest.raises(ValueError):
        translation_hypothesis([0] * 3, [1] * 3, [0] * 3, [0] * 3, axes)


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        translation_hypothesis([np.nan, 0, 0], [1] * 3, [0] * 3, [0] * 3)


def test_current_measurements_required_and_no_motion():
    summaries = {
        key: {
            "status": "current_visible_surface_estimate",
            "frame_id": 7,
            "surface_reference_world_m": xyz,
        }
        for key, xyz in [("a", [0.03, 0, 0.1]), ("b", [0.2, 0.2, 0.1])]
    }
    manager = SimpleNamespace(summary=lambda key, obs: summaries[key])
    obs = {"frame_id": 7, "arms": {"custom_hand": {"xyz_world_m": [0, 0, 0.1]}}}
    args = {
        "frame_id": 7,
        "arm": "custom_hand",
        "source_region_id": "a",
        "target_region_id": "b",
        "target_offset_world_m": [0, 0, 0],
    }
    result = measured_region_alignment(manager, obs, args)
    np.testing.assert_allclose(result["target_tcp_world_m"], [0.17, 0.2, 0.1])
    assert not result["source_identity_verified"]
    assert obs["arms"]["custom_hand"]["xyz_world_m"] == [0, 0, 0.1]
    summaries["a"]["frame_id"] = 6
    with pytest.raises(ValueError, match="currently measured"):
        measured_region_alignment(manager, obs, args)
    summaries["a"]["frame_id"] = 7
    summaries["a"]["status"] = "unknown_reinspect"
    with pytest.raises(ValueError, match="currently measured"):
        measured_region_alignment(manager, obs, args)
    with pytest.raises(ValueError, match="Stale"):
        measured_region_alignment(manager, obs, {**args, "frame_id": 6})
