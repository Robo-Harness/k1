from copy import deepcopy

from robo_harness.feedback_presentation import actor_feedback


def test_nested_feedback_removed_without_mutating_diagnostics():
    raw = {
        "motion": {
            "feedback": {
                "arm": {
                    "sweep": {"risk_pixels": [1]},
                    "decision_advisories": [{"kind": "possible_surface_intersection"}],
                    "recovery_hint": "retreat",
                    "low_progress_streak": 4,
                    "actual_translation_m": [0, 0, 0.001],
                    "commanded_translation_m": [0, 0, 0.03],
                }
            }
        },
        "pre_action_target_advisory": {"waypoint": "old"},
    }
    original = deepcopy(raw)
    clean = actor_feedback(raw)
    assert raw == original
    assert clean["motion"]["feedback"]["arm"] == {
        "low_progress_streak": 4,
        "actual_translation_m": [0, 0, 0.001],
        "commanded_translation_m": [0, 0, 0.03],
    }
    assert "pre_action_target_advisory" not in clean


def test_retains_errors_validity_grasp_scores_and_completion():
    raw = [
        {
            "error": "Nonfinite coordinate",
            "status": "unknown",
            "visibility_score": 0.5,
            "near_surface_probe_fraction": 0.2,
            "opening_m": 0.0015,
            "completion_check": {"native_success": False},
            "remaining_distance_m": 0.02,
        }
    ]
    assert actor_feedback(raw) == raw


def test_idempotent():
    raw = {"results": [{"sweep": {}, "frame_id": 12}]}
    assert actor_feedback(actor_feedback(raw)) == actor_feedback(raw)
