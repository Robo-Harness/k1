import pytest

from robo_harness.reference_alignment import qualify_carry_alignment


@pytest.mark.parametrize("state", [True, False, None])
def test_current_requested_arm_evidence(state):
    original = {"target_tcp_world_m": [1, 2, 3], "translation_world_m": [0, 0, 0.02]}
    evidence = {
        "to_frame_id": 12,
        "relative_to_hands": {"left": {"inconsistent_with_rigid_carry_hypothesis": state}},
    }
    result = qualify_carry_alignment(original, evidence, "left", 12)
    assert ("target_tcp_world_m" not in result) == (state is True)
    assert result["translation_world_m"] == original["translation_world_m"]
    assert not result["carry_applicability"]["attachment_verified"]
    assert "target_tcp_world_m" in original


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        {"status": "unknown"},
        {
            "to_frame_id": 11,
            "relative_to_hands": {"left": {"inconsistent_with_rigid_carry_hypothesis": True}},
        },
        {
            "to_frame_id": 12,
            "relative_to_hands": {"right": {"inconsistent_with_rigid_carry_hypothesis": True}},
        },
    ],
)
def test_unknown_stale_and_other_arm_do_not_reject(evidence):
    result = qualify_carry_alignment({"target_tcp_world_m": [1, 2, 3]}, evidence, "left", 12)
    assert result["target_tcp_world_m"] == [1, 2, 3]
    assert result["carry_applicability"]["status"] == "unverified"


@pytest.mark.parametrize("conflict", [True, False])
def test_registry_qualifies_alignment_without_executing(monkeypatch, conflict):
    import robo_harness.runtime as runtime
    from test_motion import Robot

    robot = Robot()
    registry = runtime.ToolRegistry(robot)
    registry.regions.enabled = True
    registry.interaction_evidence.latest = {
        "source": {
            "to_frame_id": robot.frame,
            "relative_to_hands": {"hand": {"inconsistent_with_rigid_carry_hypothesis": conflict}},
        }
    }
    monkeypatch.setattr(
        runtime,
        "measured_region_alignment",
        lambda *args: {"target_tcp_world_m": [0, 0, 0.5], "translation_world_m": [0, 0, 0.02]},
    )
    result = registry.execute(
        "align_region_references",
        {
            "source_region_id": "source",
            "arm": "hand",
            "frame_id": robot.frame,
        },
        robot.observe(),
    )
    assert ("target_tcp_world_m" not in result) == conflict
    assert not robot.calls
