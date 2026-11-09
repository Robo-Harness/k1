"""Bounded recovery from responses truncated before any tool call.

Changes only subsequent logged requests. The high-water mark lasts for this
episode so a successful tool call does not immediately reintroduce truncation.
Never retransmits an action or changes simulation/model-call budgets.
"""


def default_output_budget(model):
    """Explicit model profile; an explicit run_agent budget takes precedence."""
    return 3600 if model.split("/")[-1] == "gemini-3.8-flash" else 1800


class OutputBudget:
    def __init__(self, base=1800, maximum=7200):
        if type(base) is not int or type(maximum) is not int or not 1 <= base <= maximum:
            raise ValueError("Output token budgets must be positive integers with base <= maximum")
        self.base, self.maximum, self.current = base, maximum, base

    def after_response(self, finish_reason, has_tool_call):
        before = self.current
        if finish_reason == "length" and not has_tool_call:
            self.current = min(self.maximum, 2 * self.current)
            return dict(
                reason="truncated_before_tool_call",
                used_max_tokens=before,
                next_max_tokens=self.current,
                at_cap=self.current == before,
                action_replayed=False,
                extra_model_call=False,
            )
        # A successful short answer is not evidence that the next decision
        # needs less reasoning. Reset only when a NEW episode creates a budget.
        return None
