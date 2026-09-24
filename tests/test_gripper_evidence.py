from robo_harness.gripper_evidence import GripperEvidence


def update(monitor, frame, gap, command=0):
    monitor.update(
        {
            "frame_id": frame,
            "arms": {
                "arm": {
                    "gripper_opening_m": gap,
                    "gripper_command_open": command,
                    "gripper_limits_m": [0, 0.08],
                }
            },
        }
    )
    return monitor.context()["arm"]


def test_normal_closing_is_not_contact_loss():
    m = GripperEvidence()
    for i, gap in enumerate([0.08, 0.06, 0.03, 0.017, 0.017, 0.017]):
        assert not update(m, i, gap)["large_gap_decrease_while_command_still_closed"]
    assert update(m, 6, 0.002)["large_gap_decrease_while_command_still_closed"]


def test_opening_resets_reference():
    m = GripperEvidence()
    for i in range(3):
        update(m, i, 0.02)
    r = update(m, 3, 0.01, 1)
    assert r["largest_settled_gap_since_close_m"] is None
    assert not r["large_gap_decrease_while_command_still_closed"]
    assert [x["command"] for x in r["command_events"]] == ["close", "open"]


def test_duplicate_observations_do_not_create_fake_settling():
    m = GripperEvidence()
    for _ in range(3):
        r = update(m, 0, 0.02)
    assert r["largest_settled_gap_since_close_m"] is None


def test_sparse_frames_not_treated_as_settled():
    m = GripperEvidence()
    for f in (0, 12, 24):
        r = update(m, f, 0.02)
    assert r["largest_settled_gap_since_close_m"] is None
