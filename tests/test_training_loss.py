from types import SimpleNamespace

import pytest

from robo_harness.training.sft import encode, target_loss


def test_suffix_loss_matches_full_causal_loss_and_gradient():
    torch = pytest.importorskip("torch")
    torch.manual_seed(11)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(13, 5)
            self.projection = torch.nn.Linear(5, 13)

        def forward(self, input_ids, logits_to_keep=0, **kwargs):
            logits = self.projection(self.embedding(input_ids))
            return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)

    model = TinyModel()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = tokens.clone()
    labels[:, :3] = -100
    batch = {"input_ids": tokens, "labels": labels}
    expected = torch.nn.functional.cross_entropy(
        model(tokens).logits[:, :-1].reshape(-1, 13), labels[:, 1:].reshape(-1)
    )
    actual = target_loss(model, batch, 3)
    torch.testing.assert_close(actual, expected)
    expected_gradient = torch.autograd.grad(expected, tuple(model.parameters()))
    actual_gradient = torch.autograd.grad(actual, tuple(model.parameters()))
    for left, right in zip(expected_gradient, actual_gradient):
        torch.testing.assert_close(left, right)


def test_sft_masks_prompt_and_rejects_truncation():
    torch = pytest.importorskip("torch")

    class Processor:
        def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
            values = [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4, 5]
            return {"input_ids": torch.tensor([values])}

    row = {
        "messages": [{"role": "user", "content": "synthetic"}],
        "tools": [],
        "target": {"role": "assistant", "content": "synthetic target"},
    }
    batch, prefix = encode(Processor(), row, maximum=5)
    assert prefix == 3 and batch["labels"].tolist() == [[-100, -100, -100, 4, 5]]
    with pytest.raises(ValueError, match="oversized"):
        encode(Processor(), row, maximum=4)
