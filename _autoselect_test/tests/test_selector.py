from types import SimpleNamespace

import pytest
import torch

from autoselect.selector import (
    QwenSingleForwardSelector,
    _isotonic_nonincreasing,
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
    def __init__(self, boolean_log_odds, enum_winner=None):
        self.calls = 0
        self.boolean_log_odds = list(boolean_log_odds)
        self.enum_winner = enum_winner
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

        logits = torch.full((input_ids.shape[0], 1, 128), -20.0)
        bool_row = 0
        for row in range(input_ids.shape[0]):
            if bool_row < len(self.boolean_log_odds):
                lod = float(self.boolean_log_odds[bool_row])
                logits[row, 0, 10] = 0.0
                logits[row, 0, 11] = lod
                bool_row += 1
            elif self.enum_winner is not None:
                logits[row, 0, 12] = 0.0
                logits[row, 0, 13] = 1.0 if self.enum_winner == 1 else 0.0
                logits[row, 0, 14] = 1.0 if self.enum_winner == 2 else -1.0
        return SimpleNamespace(logits=logits)


def logit(probability: float) -> float:
    return float(torch.logit(torch.tensor(probability)).item())


def test_global_calibration_recovers_numeric_probabilities_in_one_forward() -> None:
    semantic = [0.9, 0.8, 0.7, 0.6]
    bias = 0.5
    threshold_log_odds = [logit(p) + bias for p in semantic]
    # task order: thresholds, boolean choice, one shared calibration row
    model = FakeModel(threshold_log_odds + [logit(0.95), bias])
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
    assert model.last_batch_size == 6
    assert result.forward_calls == 1
    assert result.batch_size == 6
    assert result.value["temperature"] == pytest.approx(55.0)
    assert result.value["run"] is True
    assert result.fields[0].method == "globally-calibrated-thresholds"
    assert result.fields[0].details["calibration_log_odds"] == pytest.approx(bias)
    assert result.fields[1].method == "direct-choice"


def test_one_calibration_row_is_shared_across_multiple_numeric_fields() -> None:
    bias = 0.4
    first = [0.9, 0.8, 0.7, 0.6]
    second = [0.8, 0.7, 0.6, 0.5]
    model = FakeModel(
        [logit(p) + bias for p in first + second] + [bias]
    )
    selector = QwenSingleForwardSelector(
        tokenizer=FakeTokenizer(),
        model=model,
        numeric_thresholds=4,
    )

    result = selector.select(
        "Two numeric values.",
        {
            "type": "object",
            "properties": {
                "a": {"type": "number", "minimum": 0, "maximum": 100},
                "b": {"type": "number", "minimum": -10, "maximum": 10},
            },
        },
    )

    assert model.calls == 1
    assert result.batch_size == 9
    assert result.value["a"] == pytest.approx(75.0)
    assert result.value["b"] == pytest.approx(3.0)


def test_large_integer_range_uses_fixed_task_budget_plus_one_calibration() -> None:
    bias = 0.5
    semantic = [0.9, 0.8, 0.7, 0.6]
    model = FakeModel([logit(p) + bias for p in semantic] + [bias])
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
    assert result.batch_size == 5
    assert result.value["score"] == 37750


def test_isotonic_probability_repair() -> None:
    assert _isotonic_nonincreasing([0.9, 0.3, 0.7, 0.1]) == pytest.approx(
        [0.9, 0.5, 0.5, 0.1]
    )


def test_multitoken_enum_falls_back_to_surrogate_labels() -> None:
    class EnumModel(FakeModel):
        def __call__(self, *, input_ids, attention_mask, use_cache, logits_to_keep):
            self.calls += 1
            logits = torch.full((input_ids.shape[0], 1, 128), -20.0)
            logits[:, :, 12] = 0.0
            logits[:, :, 13] = 1.0
            logits[:, :, 14] = -1.0
            return SimpleNamespace(logits=logits)

    model = EnumModel([])
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
