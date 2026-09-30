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
    def __init__(self, row_specs):
        self.calls = 0
        self.row_specs = list(row_specs)
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
        assert len(self.row_specs) == input_ids.shape[0]

        logits = torch.full((input_ids.shape[0], 1, 128), -20.0)
        for row, spec in enumerate(self.row_specs):
            kind, probability_true = spec
            p = torch.tensor(float(probability_true))
            q = 1.0 - p
            if kind == "ab_normal":
                logits[row, 0, 12] = torch.log(q)
                logits[row, 0, 13] = torch.log(p)
            elif kind == "ab_swapped":
                logits[row, 0, 12] = torch.log(p)
                logits[row, 0, 13] = torch.log(q)
            elif kind == "bool":
                logits[row, 0, 10] = torch.log(q)
                logits[row, 0, 11] = torch.log(p)
            elif kind == "enum":
                logits[row, 0, 12] = 0.0
                logits[row, 0, 13] = 1.0
                logits[row, 0, 14] = -1.0
            else:
                raise AssertionError(kind)
        return SimpleNamespace(logits=logits)


def threshold_rows(probabilities):
    rows = []
    for probability in probabilities:
        rows.append(("ab_normal", probability))
        rows.append(("ab_swapped", probability))
    return rows


def test_dynamic_number_and_boolean_share_one_forward() -> None:
    model = FakeModel(
        threshold_rows([0.9, 0.8, 0.7, 0.6]) + [("bool", 0.95)]
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
    assert result.fields[0].method == "label-swap-thresholds"
    assert result.fields[1].method == "direct-choice"


def test_large_integer_range_uses_fixed_task_budget() -> None:
    model = FakeModel(threshold_rows([0.9, 0.8, 0.7, 0.6]))
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


def test_label_swap_cancels_symmetric_token_bias() -> None:
    # A constant +1 token-bias logit in favor of B affects normal and swapped
    # mappings in opposite semantic directions, so averaging semantic log-odds
    # recovers the unbiased probability.
    semantic_log_odds = 1.2
    token_bias_b_minus_a = 1.0
    normal = torch.sigmoid(torch.tensor(semantic_log_odds + token_bias_b_minus_a)).item()
    swapped = torch.sigmoid(torch.tensor(semantic_log_odds - token_bias_b_minus_a)).item()
    calibrated = _mean_log_odds_probability(normal, swapped)
    expected = torch.sigmoid(torch.tensor(semantic_log_odds)).item()
    assert calibrated == pytest.approx(expected, rel=1e-6)


def test_isotonic_probability_repair() -> None:
    assert _isotonic_nonincreasing([0.9, 0.3, 0.7, 0.1]) == pytest.approx(
        [0.9, 0.5, 0.5, 0.1]
    )


def test_multitoken_enum_falls_back_to_surrogate_labels() -> None:
    model = FakeModel([("enum", 0.5)])
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
