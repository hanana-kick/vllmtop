from types import SimpleNamespace

import pytest
import torch

from autoselect.selector import (
    QwenSingleForwardSelector,
    _isotonic_nonincreasing,
    _mean_log_odds_probability,
)


_BASE = "<assistant>Answer:"


class FakeTokenizer:
    padding_side = "right"
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return "<assistant>"

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        if text == _BASE:
            return [1, 2]
        mapping = {
            "false": 10,
            "true": 11,
            "A": 12,
            "B": 13,
            "C": 14,
        }
        if text.startswith(_BASE):
            suffix = text[len(_BASE) :]
            if suffix in mapping:
                return [1, 2, mapping[suffix]]
            return [1, 2, 99, 98]
        if text in mapping:
            return [mapping[text]]
        return [99, 98]

    def __call__(
        self,
        prompts,
        *,
        return_tensors,
        padding,
        add_special_tokens,
    ):
        assert return_tensors == "pt"
        assert padding is True
        assert add_special_tokens is False
        batch = len(prompts)
        return {
            "input_ids": torch.ones((batch, 2), dtype=torch.long),
            "attention_mask": torch.ones((batch, 2), dtype=torch.long),
        }


class FakeModel:
    def __init__(
        self,
        *,
        true_probabilities,
        b_probabilities=None,
    ):
        self.calls = 0
        self.true_probabilities = list(true_probabilities)
        self.b_probabilities = (
            list(b_probabilities)
            if b_probabilities is not None
            else [0.5] * len(self.true_probabilities)
        )
        self.last_batch_size = None

    def to(self, device):
        assert device == "cpu"
        return self

    def eval(self):
        return self

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        use_cache,
        logits_to_keep,
    ):
        self.calls += 1
        self.last_batch_size = int(input_ids.shape[0])
        assert attention_mask.shape == input_ids.shape
        assert use_cache is False
        assert logits_to_keep == 1
        assert len(self.true_probabilities) == input_ids.shape[0]
        assert len(self.b_probabilities) == input_ids.shape[0]

        logits = torch.full((input_ids.shape[0], 1, 128), -20.0)
        for row, (probability_true, probability_b) in enumerate(
            zip(self.true_probabilities, self.b_probabilities, strict=True)
        ):
            logits[row, 0, 10] = torch.log(torch.tensor(1.0 - probability_true))
            logits[row, 0, 11] = torch.log(torch.tensor(probability_true))
            logits[row, 0, 12] = torch.log(torch.tensor(1.0 - probability_b))
            logits[row, 0, 13] = torch.log(torch.tensor(probability_b))
            logits[row, 0, 14] = -1.0
        return SimpleNamespace(logits=logits)


def _mapped_b_probabilities(semantic_true_probabilities):
    rows = []
    for probability_true in semantic_true_probabilities:
        rows.extend([probability_true, 1.0 - probability_true])
    return rows


def test_dynamic_number_and_boolean_share_one_forward() -> None:
    threshold_probs = [0.9, 0.8, 0.7, 0.6]
    # 4 thresholds * 2 swapped mappings + 1 boolean field = batch 9.
    b_probs = _mapped_b_probabilities(threshold_probs) + [0.5]
    true_probs = [0.5] * 8 + [0.95]
    model = FakeModel(
        true_probabilities=true_probs,
        b_probabilities=b_probs,
    )
    selector = QwenSingleForwardSelector(
        tokenizer=FakeTokenizer(),
        model=model,
        numeric_thresholds=4,
    )

    result = selector.select(
        "The measured temperature is 55 C. Run is required.",
        {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "number",
                    "minimum": -20,
                    "maximum": 80,
                },
                "run": {"type": "boolean"},
            },
        },
    )

    assert model.calls == 1
    assert model.last_batch_size == 9
    assert result.forward_calls == 1
    assert result.batch_size == 9
    assert result.value["temperature"] == pytest.approx(55.0)
    assert result.value["run"] is True
    assert result.fields[0].method == "mapped-thresholds"
    assert result.fields[1].method == "direct-choice"


def test_large_integer_range_uses_fixed_threshold_batch_size() -> None:
    threshold_probs = [0.9, 0.8, 0.7, 0.6]
    model = FakeModel(
        true_probabilities=[0.5] * 8,
        b_probabilities=_mapped_b_probabilities(threshold_probs),
    )
    selector = QwenSingleForwardSelector(
        tokenizer=FakeTokenizer(),
        model=model,
        numeric_thresholds=4,
    )

    result = selector.select(
        "The score is about three quarters of the allowed range.",
        {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "minimum": 1000,
                    "maximum": 50000,
                }
            },
        },
    )

    assert model.calls == 1
    assert result.batch_size == 8
    assert result.value["score"] == 37750


def test_mapping_swap_cancels_constant_label_bias() -> None:
    # Semantic log-odds = logit(0.7). Add +1.0 bias to token B.
    semantic_logit = torch.logit(torch.tensor(0.7)).item()
    label_bias = 1.0

    normal_semantic_true = torch.sigmoid(
        torch.tensor(semantic_logit + label_bias)
    ).item()

    # In the swapped prompt semantic true is A, so semantic P(true)
    # is sigmoid(s - bias).
    swapped_semantic_true = torch.sigmoid(
        torch.tensor(semantic_logit - label_bias)
    ).item()

    calibrated = _mean_log_odds_probability(
        normal_semantic_true,
        swapped_semantic_true,
    )
    assert calibrated == pytest.approx(0.7, abs=1e-6)


def test_isotonic_probability_repair() -> None:
    assert _isotonic_nonincreasing([0.9, 0.3, 0.7, 0.1]) == pytest.approx(
        [0.9, 0.5, 0.5, 0.1]
    )


def test_multitoken_enum_falls_back_to_surrogate_labels() -> None:
    model = FakeModel(
        true_probabilities=[0.5],
        b_probabilities=[0.7310586],
    )
    selector = QwenSingleForwardSelector(
        tokenizer=FakeTokenizer(),
        model=model,
    )

    result = selector.select(
        "Choose hide.",
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["attack now", "hide now", "heal now"],
                }
            },
        },
    )

    assert model.calls == 1
    assert result.fields[0].method == "surrogate-labels"
    assert result.value["action"] == "hide now"
