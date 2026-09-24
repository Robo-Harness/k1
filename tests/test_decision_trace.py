from robo_harness.decision_trace import decision_trace


def test_model_authored_trace_retained_exactly():
    note = (
        "<THINK>S1 contour matches the bowl; attachment unknown. Lift 1cm to check motion.</THINK>"
    )
    result = decision_trace({}, {"decision_note": note})
    assert result["format_valid"] and result["raw"] == note


def test_missing_invalid_and_empty_are_not_fabricated():
    assert not decision_trace({}, {})["format_valid"]
    assert decision_trace({}, {})["raw"] is None
    assert not decision_trace({}, {"decision_note": "<THINK></THINK>"})["format_valid"]
    assert not decision_trace({}, {"decision_note": {"text": "x"}})["format_valid"]


def test_assistant_content_fallback_is_explicit():
    result = decision_trace(
        {"content": "<THINK>Reobserve S1: current identity uncertain.</THINK>"}, {}
    )
    assert result["format_valid"] and result["source"] == "assistant.content"


def test_metadata_does_not_reach_tool_or_change_original_arguments(monkeypatch):
    from robo_harness.runtime import ToolRegistry

    registry = object.__new__(ToolRegistry)
    registry.advisory_controls = False
    received = []
    monkeypatch.setattr(registry, "_execute", lambda n, a, o: received.append(a) or {"ok": True})
    args = {"delta": [0, 0, 0.01], "decision_note": "<THINK>Move.</THINK>"}
    assert registry.execute("move_relative", args, {}) == {"ok": True}
    assert received == [{"delta": [0, 0, 0.01]}]
    assert "decision_note" in args
