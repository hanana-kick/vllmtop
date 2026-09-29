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
    def __init__(self, true_probabilities):
        self.calls = 0
        self.true_probabilities = list(true_probabilities)
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

        logits = torch.full((input_ids.shape[0], 1, 128), -20.0)
        for row, probability_true in enumerate(self.true_probabilities):
            logits[row, 0, 10] = torch.log(torch.tensor(1.0 - probability_true))
            logits[row, 0, 11] = torch.log(torch.tensor(probability_true))
            logits[row, 0, 12] = 0.0
            logits[row, 0, 13] = 1.0
            logits[row, 0, 14] = -1.0
        return SimpleNamespace(logits=logits)


def test_dynamic_number_and_boolean_share_one_forward() -> None:
    # Four threshold rows estimate u=(.9+.8+.7+.6)/4=.75 => -20 + .75*100 = 55.
    # The final row is the boolean field and strongly favors true.
    model = FakeModel([0.9, 0.8, 0.7, 0.6, 0.95])
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
    assert model.last_batch_size == 5
    assert result.forward_calls == 1
    assert result.batch_size == 5
    assert result.value["temperature"] == pytest.approx(55.0)
    assert result.value["run"] is True
    assert result.fields[0].method == "normalized-thresholds"
    assert result.fields[1].method == "direct-choice"


def test_large_integer_range_uses_fixed_threshold_batch_size() -> None:
    model = FakeModel([0.9, 0.8, 0.7, 0.6])
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
    assert result.batch_size == 4
    assert result.value["score"] == 37750


def test_unique_numeric_literal_skips_model_for_numeric_only_request() -> None:
    model = FakeModel([])
    selector = QwenSingleForwardSelector(
        tokenizer=FakeTokenizer(),
        model=model,
        numeric_thresholds=2,
    )

    result = selector.select(
        "The measured temperature is exactly 55 degrees Celsius.",
        {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "number",
                    "minimum": -20,
                    "maximum": 80,
                }
            },
        },
    )

    assert model.calls == 0
    assert result.forward_calls == 0
    assert result.batch_size == 0
    assert result.value["temperature"] == 55
    assert result.fields[0].method == "literal-fastpath"


def test_literal_fastpath_and_boolean_share_one_remaining_forward() -> None:
    model = FakeModel([0.95])
    selector = QwenSingleForwardSelector(
        tokenizer=FakeTokenizer(),
        model=model,
        numeric_thresholds=2,
    )

    result = selector.select(
        "The measured temperature is exactly 55 degrees Celsius and run is required.",
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
    assert result.forward_calls == 1
    assert result.batch_size == 1
    assert result.value == {"temperature": 55, "run": True}
    assert result.fields[0].method == "literal-fastpath"


def test_literal_fastpath_refuses_ambiguous_numeric_fields() -> None:
    model = FakeModel([0.9, 0.8, 0.9, 0.8])
    selector = QwenSingleForwardSelector(
        tokenizer=FakeTokenizer(),
        model=model,
        numeric_thresholds=2,
    )

    result = selector.select(
        "The observed value is 10.",
        {
            "type": "object",
            "properties": {
                "left": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                },
                "right": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                },
            },
        },
    )

    assert model.calls == 1
    assert result.batch_size == 4
    assert all(field.method == "normalized-thresholds" for field in result.fields)


def test_literal_fastpath_refuses_range_boundary() -> None:
    model = FakeModel([0.9, 0.8])
    selector = QwenSingleForwardSelector(
        tokenizer=FakeTokenizer(),
        model=model,
        numeric_thresholds=2,
    )

    result = selector.select(
        "There is 1 enemy approaching.",
        {
            "type": "object",
            "properties": {
                "danger": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                }
            },
        },
    )

    assert model.calls == 1
    assert result.batch_size == 2
    assert result.fields[0].method == "normalized-thresholds"


def test_literal_fastpath_parses_thousands_separator() -> None:
    model = FakeModel([])
    selector = QwenSingleForwardSelector(
        tokenizer=FakeTokenizer(),
        model=model,
    )

    result = selector.select(
        "The current score is 37,750.",
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

    assert result.forward_calls == 0
    assert result.value["score"] == 37750


def test_isotonic_probability_repair() -> None:
    assert _isotonic_nonincreasing([0.9, 0.3, 0.7, 0.1]) == pytest.approx(
        [0.9, 0.5, 0.5, 0.1]
    )


def test_multitoken_enum_falls_back_to_surrogate_labels() -> None:
    model = FakeModel([0.5])
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
