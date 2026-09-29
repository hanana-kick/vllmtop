from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import fmean
import math
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .schema import (
    ChoiceFieldSpec,
    FieldSpec,
    NumericFieldSpec,
    build_object,
    compile_schema,
    numeric_threshold_value,
    quantize_numeric,
)


MODEL_ID = "Qwen/Qwen3.5-0.8B"
_LABELS = tuple(chr(ord("A") + i) for i in range(26))


@dataclass(frozen=True)
class FieldSelection:
    path: str
    value: Any
    method: str
    confidence: float | None
    details: dict[str, Any]


@dataclass(frozen=True)
class SelectionResult:
    value: dict[str, Any]
    fields: tuple[FieldSelection, ...]
    forward_calls: int
    batch_size: int


@dataclass(frozen=True)
class _Task:
    field_index: int
    method: str
    rendered_prompt: str
    token_ids: tuple[int, ...]
    option_values: tuple[Any, ...]
    option_texts: tuple[str, ...]
    normalized_threshold: float | None = None
    threshold_value: int | float | None = None


class QwenSingleForwardSelector:
    """Select structured JSON values using at most one model forward call."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        dtype: str = "bfloat16",
        max_candidates: int = 26,
        numeric_thresholds: int = 8,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if numeric_thresholds < 2:
            raise ValueError("numeric_thresholds must be at least 2")

        self.model_id = model_id
        self.max_candidates = max_candidates
        self.numeric_thresholds = numeric_thresholds

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if model is None:
            torch_dtype = _resolve_dtype(dtype)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
        self.model = model
        self.model.to("cpu")
        self.model.eval()

    def select(self, prompt: str, schema: dict[str, Any]) -> SelectionResult:
        fields = compile_schema(schema, max_candidates=self.max_candidates)
        tasks: list[_Task] = []
        rows_by_field: dict[int, list[int]] = {i: [] for i in range(len(fields))}
        fixed_values: dict[int, int | float] = {}

        for field_index, field in enumerate(fields):
            if isinstance(field, ChoiceFieldSpec):
                task = self._prepare_choice_task(prompt, field_index, field)
                rows_by_field[field_index].append(len(tasks))
                tasks.append(task)
                continue

            if field.minimum == field.maximum:
                fixed_values[field_index] = quantize_numeric(field, 0.0)
                continue

            for normalized_threshold in self._normalized_thresholds():
                for swap_mapping in (False, True):
                    task = self._prepare_mapped_threshold_task(
                        prompt,
                        field_index,
                        field,
                        normalized_threshold,
                        swap_mapping=swap_mapping,
                    )
                    rows_by_field[field_index].append(len(tasks))
                    tasks.append(task)

        task_probabilities: list[list[float]] = []
        forward_calls = 0

        if tasks:
            encoded = self.tokenizer(
                [task.rendered_prompt for task in tasks],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            encoded = {key: value.to("cpu") for key, value in encoded.items()}

            with torch.inference_mode():
                forward_calls = 1
                outputs = self.model(
                    **encoded,
                    use_cache=False,
                    logits_to_keep=1,
                )

            next_token_logits = outputs.logits[:, -1, :].float()
            for row, task in enumerate(tasks):
                ids = torch.tensor(
                    task.token_ids,
                    dtype=torch.long,
                    device=next_token_logits.device,
                )
                candidate_logits = next_token_logits[row, ids]
                probabilities = torch.softmax(candidate_logits, dim=-1)
                task_probabilities.append(
                    [float(probabilities[i].item()) for i in range(len(task.token_ids))]
                )

        selections: list[tuple[FieldSpec, Any]] = []
        details: list[FieldSelection] = []

        for field_index, field in enumerate(fields):
            if field_index in fixed_values:
                value = fixed_values[field_index]
                selections.append((field, value))
                details.append(
                    FieldSelection(
                        path=field.dotted_path,
                        value=value,
                        method="fixed-range",
                        confidence=1.0,
                        details={
                            "minimum": field.minimum,
                            "maximum": field.maximum,
                        },
                    )
                )
                continue

            rows = rows_by_field[field_index]
            if isinstance(field, ChoiceFieldSpec):
                row = rows[0]
                task = tasks[row]
                probabilities = task_probabilities[row]
                winner = max(range(len(probabilities)), key=probabilities.__getitem__)
                value = task.option_values[winner]
                selections.append((field, value))
                details.append(
                    FieldSelection(
                        path=field.dotted_path,
                        value=value,
                        method=task.method,
                        confidence=probabilities[winner],
                        details={
                            "selected_output": task.option_texts[winner],
                            "candidates": [
                                {
                                    "value": candidate,
                                    "output": output_text,
                                    "probability": probabilities[i],
                                    "token_id": task.token_ids[i],
                                }
                                for i, (candidate, output_text) in enumerate(
                                    zip(
                                        task.option_values,
                                        task.option_texts,
                                        strict=True,
                                    )
                                )
                            ],
                        },
                    )
                )
                continue

            raw_probabilities: list[float] = []
            threshold_details: list[dict[str, Any]] = []
            if len(rows) % 2 != 0:
                raise RuntimeError(
                    f"numeric field {field.dotted_path} has an unpaired mapping row"
                )

            for offset in range(0, len(rows), 2):
                normal_row = rows[offset]
                swapped_row = rows[offset + 1]
                normal_task = tasks[normal_row]
                swapped_task = tasks[swapped_row]
                if normal_task.normalized_threshold != swapped_task.normalized_threshold:
                    raise RuntimeError(
                        f"numeric field {field.dotted_path} has mismatched threshold mappings"
                    )

                normal_probs = task_probabilities[normal_row]
                swapped_probs = task_probabilities[swapped_row]
                normal_true = normal_probs[normal_task.option_values.index(True)]
                swapped_true = swapped_probs[swapped_task.option_values.index(True)]
                probability_true = _mean_log_odds_probability(
                    normal_true,
                    swapped_true,
                )
                raw_probabilities.append(probability_true)
                threshold_details.append(
                    {
                        "normalized_threshold": normal_task.normalized_threshold,
                        "threshold": normal_task.threshold_value,
                        "probability_ge": probability_true,
                        "mapping_probabilities": {
                            "A=false,B=true": normal_true,
                            "A=true,B=false": swapped_true,
                        },
                        "output_tokens": {
                            "A": normal_task.token_ids[0],
                            "B": normal_task.token_ids[1],
                        },
                    }
                )

            monotonic_probabilities = _isotonic_nonincreasing(raw_probabilities)
            normalized_value = fmean(monotonic_probabilities)
            value = quantize_numeric(field, normalized_value)
            confidence = fmean(abs(probability - 0.5) * 2 for probability in raw_probabilities)

            for item, adjusted in zip(
                threshold_details,
                monotonic_probabilities,
                strict=True,
            ):
                item["probability_ge_monotonic"] = adjusted

            selections.append((field, value))
            details.append(
                FieldSelection(
                    path=field.dotted_path,
                    value=value,
                    method="mapped-thresholds",
                    confidence=confidence,
                    details={
                        "minimum": field.minimum,
                        "maximum": field.maximum,
                        "multiple_of": field.multiple_of,
                        "integer": field.integer,
                        "threshold_count": len(rows),
                        "normalized_value": normalized_value,
                        "thresholds": threshold_details,
                    },
                )
            )

        return SelectionResult(
            value=build_object(selections),
            fields=tuple(details),
            forward_calls=forward_calls,
            batch_size=len(tasks),
        )

    def _normalized_thresholds(self) -> list[float]:
        count = self.numeric_thresholds
        return [(index + 0.5) / count for index in range(count)]

    def _prepare_choice_task(
        self,
        prompt: str,
        field_index: int,
        field: ChoiceFieldSpec,
    ) -> _Task:
        if field.candidates == (False, True):
            for false_text, true_text in (
                ("false", "true"),
                ("no", "yes"),
                ("0", "1"),
            ):
                output_texts = (false_text, true_text)
                rendered = self._render_choice_prompt(
                    prompt,
                    field,
                    output_texts,
                    use_labels=False,
                )
                false_id = self._single_continuation_token_id(rendered, false_text)
                true_id = self._single_continuation_token_id(rendered, true_text)
                if (
                    false_id is not None
                    and true_id is not None
                    and false_id != true_id
                ):
                    return _Task(
                        field_index=field_index,
                        method="direct-choice",
                        rendered_prompt=rendered,
                        token_ids=(false_id, true_id),
                        option_values=field.candidates,
                        option_texts=output_texts,
                    )

        direct_texts = tuple(_candidate_output_text(value) for value in field.candidates)
        if (
            all(text is not None for text in direct_texts)
            and len(set(direct_texts)) == len(direct_texts)
        ):
            output_texts = tuple(text for text in direct_texts if text is not None)
            rendered = self._render_choice_prompt(
                prompt,
                field,
                output_texts,
                use_labels=False,
            )
            token_ids = tuple(
                self._single_continuation_token_id(rendered, text)
                for text in output_texts
            )
            if all(token_id is not None for token_id in token_ids):
                resolved_ids = tuple(int(token_id) for token_id in token_ids if token_id is not None)
                if len(set(resolved_ids)) == len(resolved_ids):
                    return _Task(
                        field_index=field_index,
                        method="direct-choice",
                        rendered_prompt=rendered,
                        token_ids=resolved_ids,
                        option_values=field.candidates,
                        option_texts=output_texts,
                    )

        labels = _LABELS[: len(field.candidates)]
        rendered = self._render_choice_prompt(
            prompt,
            field,
            labels,
            use_labels=True,
        )
        token_ids = tuple(
            self._single_continuation_token_id(rendered, label) for label in labels
        )
        if any(token_id is None for token_id in token_ids):
            raise RuntimeError(
                f"could not encode one-token surrogate labels for {field.dotted_path}"
            )
        return _Task(
            field_index=field_index,
            method="surrogate-labels",
            rendered_prompt=rendered,
            token_ids=tuple(int(token_id) for token_id in token_ids if token_id is not None),
            option_values=field.candidates,
            option_texts=labels,
        )

    def _prepare_mapped_threshold_task(
        self,
        prompt: str,
        field_index: int,
        field: NumericFieldSpec,
        normalized_threshold: float,
        *,
        swap_mapping: bool,
    ) -> _Task:
        threshold_value = numeric_threshold_value(field, normalized_threshold)
        option_values = (True, False) if swap_mapping else (False, True)
        rendered = self._render_mapped_threshold_prompt(
            prompt,
            field,
            threshold_value,
            option_values,
        )
        token_ids = tuple(
            self._single_continuation_token_id(rendered, label)
            for label in ("A", "B")
        )
        if any(token_id is None for token_id in token_ids):
            raise RuntimeError(
                f"could not encode A/B threshold labels for numeric field {field.dotted_path}"
            )
        return _Task(
            field_index=field_index,
            method="mapped-threshold",
            rendered_prompt=rendered,
            token_ids=tuple(int(token_id) for token_id in token_ids if token_id is not None),
            option_values=option_values,
            option_texts=("A", "B"),
            normalized_threshold=normalized_threshold,
            threshold_value=threshold_value,
        )

    def _render_choice_prompt(
        self,
        prompt: str,
        field: ChoiceFieldSpec,
        output_texts: tuple[str, ...],
        *,
        use_labels: bool,
    ) -> str:
        if use_labels:
            options = "\n".join(
                f"{label} = {json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
                for label, value in zip(output_texts, field.candidates, strict=True)
            )
            instruction = "Return exactly one label from the list."
        else:
            options = "\n".join(
                f"- {json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
                f" -> output {json.dumps(output_text, ensure_ascii=False)}"
                for value, output_text in zip(
                    field.candidates,
                    output_texts,
                    strict=True,
                )
            )
            instruction = "Return exactly one output text from the list."

        description = f"\nField description: {field.description}" if field.description else ""
        messages = [
            {
                "role": "system",
                "content": "You are a deterministic classifier. Output only the requested answer.",
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{prompt}\n\n"
                    f"Field: {field.dotted_path}{description}\n"
                    f"Allowed choices:\n{options}\n\n"
                    f"{instruction}"
                ),
            },
        ]
        return self._render_assistant_prefill(messages)

    def _render_mapped_threshold_prompt(
        self,
        prompt: str,
        field: NumericFieldSpec,
        threshold_value: int | float,
        option_values: tuple[bool, bool],
    ) -> str:
        description = f"\nField description: {field.description}" if field.description else ""
        mapping = "\n".join(
            f"{label} = {'true' if value else 'false'}"
            for label, value in zip(("A", "B"), option_values, strict=True)
        )
        messages = [
            {
                "role": "system",
                "content": "You estimate numeric values from context. Output only A or B.",
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{prompt}\n\n"
                    f"Field: {field.dotted_path}{description}\n"
                    f"Allowed range: [{field.minimum}, {field.maximum}]\n"
                    f"Question: Is the best value for this field greater than or equal to "
                    f"{threshold_value}?\n"
                    f"Answer mapping:\n{mapping}\n"
                    "Return exactly A or B."
                ),
            },
        ]
        return self._render_assistant_prefill(messages)

    def _render_assistant_prefill(self, messages: list[dict[str, str]]) -> str:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return rendered + "Answer:"

    def _single_continuation_token_id(
        self,
        rendered_prompt: str,
        text: str,
    ) -> int | None:
        base_ids = self.tokenizer.encode(rendered_prompt, add_special_tokens=False)
        full_ids = self.tokenizer.encode(rendered_prompt + text, add_special_tokens=False)
        if full_ids[: len(base_ids)] == base_ids and len(full_ids) == len(base_ids) + 1:
            return full_ids[-1]

        # Next-token scoring operates on the already-tokenized prompt. A candidate
        # does not need prompt+text to re-tokenize with an identical prefix; it only
        # needs to exist as one vocabulary token that can be emitted next.
        direct_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(direct_ids) == 1:
            return direct_ids[0]
        return None


def _candidate_output_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value if value else None
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return None


def _mean_log_odds_probability(first: float, second: float) -> float:
    epsilon = 1e-6
    p1 = min(1.0 - epsilon, max(epsilon, float(first)))
    p2 = min(1.0 - epsilon, max(epsilon, float(second)))
    logit = 0.5 * (
        math.log(p1 / (1.0 - p1))
        + math.log(p2 / (1.0 - p2))
    )
    return 1.0 / (1.0 + math.exp(-logit))


def _isotonic_nonincreasing(values: list[float]) -> list[float]:
    """Pool-adjacent-violators fit constrained to p[0] >= p[1] >= ..."""
    if not values:
        return []

    blocks: list[list[float]] = []
    for value in values:
        blocks.append([float(value), 1.0])
        while len(blocks) >= 2:
            previous = blocks[-2][0] / blocks[-2][1]
            current = blocks[-1][0] / blocks[-1][1]
            if previous >= current:
                break
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0] + right[0], left[1] + right[1]])

    result: list[float] = []
    for total, count in blocks:
        average = total / count
        result.extend([average] * int(count))
    return result


def _resolve_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError("dtype must be bfloat16/bf16 or float32/fp32")
