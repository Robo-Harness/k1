from types import SimpleNamespace
import numpy as np
import pytest
from robo_harness.point_targets import resolve
from robo_harness.observation_contract import camera_reference


def fixture():
    camera = dict(
        color=np.zeros((8, 8, 3), np.uint8),
        depth=np.ones((8, 8)),
        intrinsic_matrix=np.array([[10, 0, 4], [0, 10, 4], [0, 0, 1]]),
        extrinsic_matrix=np.eye(4),
    )
    obs = dict(frame_id=12, vision={"head": camera})
    entry = dict(
        lost=False,
        camera="head",
        sensor_reference=camera_reference(obs, "head"),
        latest=dict(frame_id=12, status="initialized", xyz_world_m=[2.747, -2.744, 1.505]),
    )
    points = SimpleNamespace(entries={"P5": entry})
    args = dict(target_point=dict(point_id="P5", offset_world_m=dict(dx=-0.047, dy=0, dz=0.02)))
    return obs, entry, points, args


def test_measured_id_avoids_copying_xyz():
    obs, e, p, a = fixture()
    xyz, receipt = resolve(a, obs, p)
    np.testing.assert_allclose(xyz, [2.7, -2.744, 1.525])
    assert receipt["identity_verified"] is False
    assert e["latest"]["xyz_world_m"] == [2.747, -2.744, 1.505]


@pytest.mark.parametrize(
    "kind", ["lost", "stale", "sensor", "unknown", "ambiguous", "missing_offset", "bad_axes"]
)
def test_no_stale_fallback_or_ambiguous_target(kind):
    obs, e, p, a = fixture()
    if kind == "lost":
        e["lost"] = True
    if kind == "stale":
        obs["frame_id"] += 1
    if kind == "sensor":
        obs["vision"]["head"]["color"][0, 0] = 255
    if kind == "unknown":
        a["target_point"]["point_id"] = "P99"
    if kind == "ambiguous":
        a["target_xyz_world_m"] = [1, 2, 3]
    if kind == "missing_offset":
        a["target_point"].pop("offset_world_m")
    if kind == "bad_axes":
        a["target_point"]["offset_world_m"] = {"x": 0, "y": 0, "z": 0}
    with pytest.raises(ValueError):
        resolve(a, obs, p)


def test_named_axes_and_legacy_are_not_permuted():
    obs, e, p, a = fixture()
    for value in ([2.7, 1.505, -2.74], dict(x=2.7, y=1.505, z=-2.74)):
        xyz, _ = resolve(dict(target_xyz_world_m=value), obs, p)
        assert xyz.tolist() == [2.7, 1.505, -2.74]  # must not silently 'fix' intent


def test_current_tracked_estimate_is_used_not_initial_point():
    obs, e, p, a = fixture()
    e["latest"].update(status="tracked_estimate", xyz_world_m=[1, 2, 3])
    xyz, _ = resolve(a, obs, p)
    np.testing.assert_allclose(xyz, [0.953, 2, 3.02])


def test_bad_reference_does_not_remove_old_intent_or_dispatch_motion():
    from robo_harness.runtime import ToolRegistry

    obs, e, p, a = fixture()
    obs["arms"] = {"arm": {}}
    robot = SimpleNamespace()
    registry = ToolRegistry(robot, advisory_controls=True)
    registry.tracked_points = p
    registry.pose_targets.active["arm"] = "existing-intent"
    a.update(arm="arm", frame_id=12)
    e["lost"] = True
    with pytest.raises(ValueError):
        registry.execute("move_toward", a, obs)
    assert registry.pose_targets.active["arm"] == "existing-intent"


def test_uncalibrated_robot_does_not_offer_geometry_candidate_tool():
    from robo_harness.runtime import ToolRegistry
    from test_portable_harness import scene

    obs = scene()
    registry = ToolRegistry(SimpleNamespace(), region_tools=True)
    assert "grasp_candidates" not in [s["function"]["name"] for s in registry.schemas(obs)]


def test_textureless_current_measurement_still_usable_but_explicit_invalidation_is_not():
    obs, e, p, a = fixture()
    fresh = dict(e["latest"], id="P5", camera="head", sensor_reference=e["sensor_reference"])
    e.update(lost=True, loss_reason="initial_correspondence_uncertain")
    e["latest"] = dict(status="lost_remeasure", frame_id=12)
    xyz, _ = resolve(a, obs, p, [fresh])
    np.testing.assert_allclose(xyz, [2.7, -2.744, 1.525])
    e["loss_reason"] = "agent_observed_contact"
    with pytest.raises(ValueError):
        resolve(a, obs, p, [fresh])
