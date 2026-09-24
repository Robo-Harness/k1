from types import SimpleNamespace

import pytest

from robo_harness.libero_adapter import environment_instruction


def test_actual_instruction_not_filename():
    env = SimpleNamespace(
        language_instruction="open the bottom drawer",
        task=SimpleNamespace(language="open the middle drawer"),
    )
    assert environment_instruction(env) == "open the bottom drawer"


def test_tokenized_instruction():
    assert (
        environment_instruction(SimpleNamespace(language_instruction=["put", "object", "inside"]))
        == "put object inside"
    )


@pytest.mark.parametrize("value", [None, "", "   ", 1, [1]])
def test_missing_instruction_fails_closed(value):
    with pytest.raises(ValueError):
        environment_instruction(SimpleNamespace(language_instruction=value))
