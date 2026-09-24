from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from robo_harness.self_geometry import RobotSelfGeometry, visible_self_mask


def box():
    model = SimpleNamespace(geom_type=[6], geom_size=np.array([[0.02, 0.03, 0.04]]))
    geometry = RobotSelfGeometry(model, [0])
    data = SimpleNamespace(
        geom_xpos=np.array([[0.1, 0.2, 0.3]]),
        geom_xmat=Rotation.from_euler("z", 37, degrees=True).as_matrix().reshape(1, 9),
    )
    return geometry, data


def test_absent_geometry_preserves_fallback():
    assert visible_self_mask({}, "arm", [[0, 0, 0]]) is None
    assert visible_self_mask({"_robot_self_parts": {"other": []}}, "arm", [[0, 0, 0]]) is None


def test_known_robot_only_and_tolerance():
    geometry, data = box()
    points = np.array([[0, 0, 0], [0.021, 0, 0], [0.023, 0, 0], [0.05, 0, 0]])
    world = points @ data.geom_xmat.reshape(3, 3).T + data.geom_xpos[0]
    mask = visible_self_mask({"_robot_self_parts": {"arm": geometry.snapshot(data)}}, "arm", world)
    assert mask.tolist() == [True, True, False, False]


def test_fast_bounds_match_full_plane_test_and_do_not_mutate_snapshot():
    geometry, data = box()
    parts = geometry.snapshot(data)
    points = np.random.default_rng(7).uniform(-0.08, 0.08, (10000, 3)) + 0.1
    points[:, 1] += 0.1
    points[:, 2] += 0.2
    expected = np.zeros(len(points), bool)
    for p in parts:
        local = (points - p["center"]) @ p["rotation"]
        eq = p["planes"]
        expected |= (local @ eq[:, :3].T + eq[:, 3] <= p["tolerance"]).all(1)
    assert np.array_equal(
        expected, visible_self_mask({"_robot_self_parts": {"arm": parts}}, "arm", points)
    )
    original = parts[0]["center"].copy()
    data.geom_xpos[:] = 9
    assert np.array_equal(parts[0]["center"], original)


def test_invalid_points_never_claimed_as_self():
    geometry, data = box()
    assert not visible_self_mask(
        {"_robot_self_parts": {"arm": geometry.snapshot(data)}},
        "arm",
        [[np.nan, 0, 0], [np.inf, 0, 0]],
    ).any()


def test_unsupported_geometry_is_explicit():
    with pytest.raises(ValueError, match="Unsupported"):
        RobotSelfGeometry(SimpleNamespace(geom_type=[2]), [0])
