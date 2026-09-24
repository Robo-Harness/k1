from robo_harness.execution_ledger import ExecutionLedger


def observation(frame, command=1, arm="arbitrary_hand"):
    return {"frame_id": frame, "arms": {arm: {"gripper_command_open": command}}}


def test_candidates_do_not_become_executed_grasps():
    ledger = ExecutionLedger()
    obs = observation(0)
    for i in range(12):
        ledger.record(
            i,
            "grasp_candidates",
            {"arm": "arbitrary_hand", "region_id": "S4"},
            {"candidates": [{}]},
            obs,
            obs,
        )
    c = ledger.context()
    a = c["arms"]["arbitrary_hand"]
    assert a["candidate_requests_since_command_transition"] == 12
    assert a["last_observed_command_transition"] is None
    assert len(c["recent_receipts"]) == 8
    assert a["last_candidate_request"]["returned_candidates"] == 1
    assert c["tool_request_counts"] == {"grasp_candidates": 12}


def test_rejected_close_does_not_reset_preparation_history():
    ledger = ExecutionLedger()
    obs = observation(0)
    ledger.record(0, "grasp_candidates", {"arm": "arbitrary_hand"}, {}, obs, obs)
    ledger.record(
        1,
        "set_gripper",
        {"arm": "arbitrary_hand", "gripper": 0},
        {"error": "Closing not allowed"},
        obs,
        obs,
    )
    a = ledger.context()["arms"]["arbitrary_hand"]
    assert a["candidate_requests_since_command_transition"] == 1
    assert a["last_observed_command_transition"] is None
    after = observation(24, 0)
    ledger.record(2, "set_gripper", {"arm": "arbitrary_hand", "gripper": 0}, {}, obs, after)
    a = ledger.context()["arms"]["arbitrary_hand"]
    assert a["candidate_requests_since_command_transition"] == 0
    assert a["last_observed_command_transition"]["call_id"] == 2
    # Repeating a closed command is not another transition.
    ledger.record(
        3, "move_relative", {"arm": "arbitrary_hand", "gripper": 0}, {}, after, observation(36, 0)
    )
    assert (
        ledger.context()["arms"]["arbitrary_hand"]["last_observed_command_transition"]["call_id"]
        == 2
    )


def test_receipts_are_independent_snapshots_and_do_not_assume_arm_name():
    ledger = ExecutionLedger()
    obs = observation(0, arm="left")
    ledger.record(0, "move_to_pose", {"arm": "left"}, {"pose_goal_id": "T9"}, obs, obs)
    c = ledger.context()
    assert c["recent_receipts"][0]["pose_goal_id"] == "T9"
    c["arms"]["left"]["observed_command"] = "mutated"
    assert ledger.context()["arms"]["left"]["observed_command"] == "open"


def test_unavailable_gripper_observation_remains_unknown():
    ledger = ExecutionLedger()
    obs = {"frame_id": 0, "arms": {"hand": {}}}
    ledger.record(0, "set_gripper", {"arm": "hand", "gripper": 0}, {}, obs, obs)
    assert ledger.context()["arms"]["hand"]["observed_command"] is None
    assert ledger.context()["arms"]["hand"]["last_observed_command_transition"] is None


def test_saved_candidates_are_bounded_historical_and_survive_failed_requery():
    ledger = ExecutionLedger()
    obs = observation(12)
    args = {"arm": "arbitrary_hand", "region_id": "S2"}
    candidates = [{"candidate_id": "S2-G1", "candidate_tcp_world_m": [1, 2, 3]}] * 4
    ledger.record(5, "grasp_candidates", args, {"candidates": candidates}, obs, obs)
    candidates[0]["candidate_tcp_world_m"][0] = 100
    ledger.record(6, "grasp_candidates", args, {"error": "region lost"}, obs, obs)
    a = ledger.context()["arms"]["arbitrary_hand"]
    saved = a["last_successful_candidate_receipt"]
    assert saved["historical_not_current"]
    assert saved["frame_id"] == 12 and saved["call_id"] == 5
    assert len(saved["candidates"]) == 3
    assert saved["candidates"][0]["candidate_tcp_world_m"] == [1, 2, 3]
    assert a["last_candidate_request"]["error"] == "region lost"
