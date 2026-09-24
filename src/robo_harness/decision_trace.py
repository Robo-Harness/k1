"""Archive a model-authored, observable decision rationale, not hidden reasoning."""


def decision_trace(message, arguments):
    text = arguments.get("decision_note")
    source = "tool_arguments.decision_note"
    if text is None:
        text = message.get("content")
        source = "assistant.content"
    valid = (
        isinstance(text, str)
        and text.strip().startswith("<THINK>")
        and text.strip().endswith("</THINK>")
        and text.count("<THINK>") == 1
        and text.count("</THINK>") == 1
        and bool(text.strip()[7:-8].strip())
    )
    return {
        "format": "THINK_v1",
        "source": source,
        "raw": text,
        "format_valid": bool(valid),
        "meaning": "Model-authored decision rationale; not verified evidence or a success label.",
    }
