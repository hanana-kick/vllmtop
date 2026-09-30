from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from statistics import fmean
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
_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


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
    input_text: str
    token_ids: tuple[int, ...] = ()
    option_values: tuple[Any, ...] = ()
    option_texts: tuple[str, ...] = ()
    normalized_threshold: float | None = None
    threshold_value: int | float | None = None
    calibration_role: str | None = None
    sequence_candidate: int | float | None = None
    sequence_source: str | None = None
    sequence_target_ids: tuple[int, ...] = ()


class QwenSingleForwardSelector:
    """Select structured JSON values using at most one model forward call."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        dtype: str = "bfloat16",
        max_candidates: int = 26,
        numeric_thresholds: int = 4,
        literal_margin: float = 0.5,
        max_numeric_literals: int = 5,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if numeric_thresholds < 2:
            raise ValueError("numeric_thresholds must be at least 2")
        if literal_margin < 0:
            raise ValueError("literal_margin must be non-negative")
        if max_numeric_literals < 1:
            raise ValueError("max_numeric_literals must be at least 1")

        self.model_id = model_id
        self.max_candidates = max_candidates
        self.numeric_thresholds = numeric_thresholds
        self.literal_margin = literal_margin
        self.max_numeric_literals = max_numeric_literals

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
                for swapped in (False, True):
                    task = self._prepare_threshold_task(
                        prompt,
                        field_index,
                        field,
                        normalized_threshold,
                        swapped=swapped,
                    )
                    rows_by_field[field_index].append(len(tasks))
                    tasks.append(task)

            literal_candidates = _numeric_literal_candidates(
                prompt,
                field,
                max_literals=self.max_numeric_literals,
            )
            if literal_candidates:
                for value, text, source in _literal_candidate_set(
                    field,
                    literal_candidates,
                ):
                    task = self._prepare_sequence_task(
                        prompt,
                        field_index,
                        field,
                        value=value,
                        candidate_text=text,
                        source=source,
                    )
                    rows_by_field[field_index].append(len(tasks))
                    tasks.append(task)

        classification_probabilities: dict[int, list[float]] = {}
        sequence_scores: dict[int, float] = {}
        forward_calls = 0
        logits_to_keep = 1

        if tasks:
            max_sequence_suffix = max(
                (len(task.sequence_target_ids) for task in tasks),
                default=0,
            )
            logits_to_keep = max(1, max_sequence_suffix + 1)

            encoded = self.tokenizer(
                [task.input_text for task in tasks],
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
                    logits_to_keep=logits_to_keep,
                )

            logits = outputs.logits.float()

            for row, task in enumerate(tasks):
                if task.sequence_target_ids:
                    log_probs = torch.log_softmax(logits[row], dim=-1)
                    suffix_len = len(task.sequence_target_ids)
                    token_logprobs: list[float] = []
                    for offset, token_id in enumerate(task.sequence_target_ids):
                        relative_prediction_pos = (
                            logits_to_keep - suffix_len + offset - 1
                        )
                        if relative_prediction_pos < 0:
                            raise RuntimeError("insufficient logits retained for sequence scoring")
                        token_logprobs.append(
                            float(
                                log_probs[
                                    relative_prediction_pos,
                                    token_id,
                                ].item()
                            )
                        )
                    sequence_scores[row] = fmean(token_logprobs)
                    continue

                ids = torch.tensor(
                    task.token_ids,
                    dtype=torch.long,
                    device=logits.device,
                )
                candidate_logits = logits[row, -1, ids]
                probabilities = torch.softmax(candidate_logits, dim=-1)
                classification_probabilities[row] = [
                    float(probabilities[i].item())
                    for i in range(len(task.token_ids))
                ]

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
                probabilities = classification_probabilities[row]
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

            threshold_rows = [
                row for row in rows if tasks[row].method == "label-swap-threshold"
            ]
            sequence_rows = [
                row for row in rows if tasks[row].method == "numeric-sequence-candidate"
            ]
            if len(threshold_rows) != self.numeric_thresholds * 2:
                raise RuntimeError(
                    f"numeric field {field.dotted_path} has an invalid calibration row count"
                )

            calibrated_probabilities: list[float] = []
            threshold_details: list[dict[str, Any]] = []

            for offset in range(0, len(threshold_rows), 2):
                normal_row = threshold_rows[offset]
                swapped_row = threshold_rows[offset + 1]
                normal_task = tasks[normal_row]
                swapped_task = tasks[swapped_row]

                if (
                    normal_task.calibration_role != "normal"
                    or swapped_task.calibration_role != "swapped"
                    or normal_task.normalized_threshold
                    != swapped_task.normalized_threshold
                ):
                    raise RuntimeError(
                        f"numeric field {field.dotted_path} has invalid calibration pairing"
                    )

                normal_probs = classification_probabilities[normal_row]
                swapped_probs = classification_probabilities[swapped_row]
                normal_true = normal_probs[normal_task.option_values.index(True)]
                swapped_true = swapped_probs[swapped_task.option_values.index(True)]
                calibrated = _mean_log_odds_probability(normal_true, swapped_true)
                calibrated_probabilities.append(calibrated)

                threshold_details.append(
                    {
                        "normalized_threshold": normal_task.normalized_threshold,
                        "threshold": normal_task.threshold_value,
                        "normal_probability_ge": normal_true,
                        "swapped_probability_ge": swapped_true,
                        "calibrated_probability_ge": calibrated,
                    }
                )

            monotonic_probabilities = _isotonic_nonincreasing(calibrated_probabilities)
            normalized_value = fmean(monotonic_probabilities)
            threshold_value = quantize_numeric(field, normalized_value)
            threshold_confidence = fmean(
                abs(probability - 0.5) * 2
                for probability in calibrated_probabilities
            )

            for item, adjusted in zip(
                threshold_details,
                monotonic_probabilities,
                strict=True,
            ):
                item["probability_ge_monotonic"] = adjusted

            chosen_value = threshold_value
            chosen_method = "label-swap-thresholds"
            chosen_confidence = threshold_confidence
            literal_details: dict[str, Any] | None = None

            if sequence_rows:
                ranked = sorted(
                    (
                        {
                            "row": row,
                            "value": tasks[row].sequence_candidate,
                            "source": tasks[row].sequence_source,
                            "score": sequence_scores[row],
                        }
                        for row in sequence_rows
                    ),
                    key=lambda item: item["score"],
                    reverse=True,
                )
                best = ranked[0]
                second = ranked[1]
                margin = best["score"] - second["score"]
                accepted = (
                    best["source"] == "literal"
                    and margin >= self.literal_margin
                )
                literal_details = {
                    "accepted": accepted,
                    "margin": margin,
                    "required_margin": self.literal_margin,
                    "ranking": ranked,
                }
                if accepted:
                    chosen_value = best["value"]
                    chosen_method = "literal-sequence"
                    chosen_confidence = margin

            selections.append((field, chosen_value))
            details.append(
                FieldSelection(
                    path=field.dotted_path,
                    value=chosen_value,
                    method=chosen_method,
                    confidence=chosen_confidence,
                    details={
                        "minimum": field.minimum,
                        "maximum": field.maximum,
                        "multiple_of": field.multiple_of,
                        "integer": field.integer,
                        "threshold_count": self.numeric_thresholds,
                        "threshold_task_count": len(threshold_rows),
                        "normalized_value": normalized_value,
                        "threshold_value": threshold_value,
                        "thresholds": threshold_details,
                        "literal_candidates": literal_details,
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
                        input_text=rendered,
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
                resolved_ids = tuple(
                    int(token_id)
                    for token_id in token_ids
                    if token_id is not None
                )
                if len(set(resolved_ids)) == len(resolved_ids):
                    return _Task(
                        field_index=field_index,
                        method="direct-choice",
                        input_text=rendered,
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
            self._single_continuation_token_id(rendered, label)
            for label in labels
        )
        if any(token_id is None for token_id in token_ids):
            raise RuntimeError(
                f"could not encode one-token surrogate labels for {field.dotted_path}"
            )
        return _Task(
            field_index=field_index,
            method="surrogate-labels",
            input_text=rendered,
            token_ids=tuple(
                int(token_id) for token_id in token_ids if token_id is not None
            ),
            option_values=field.candidates,
            option_texts=labels,
        )

    def _prepare_threshold_task(
        self,
        prompt: str,
        field_index: int,
        field: NumericFieldSpec,
        normalized_threshold: float,
        *,
        swapped: bool,
    ) -> _Task:
        threshold_value = numeric_threshold_value(field, normalized_threshold)
        labels = ("A", "B")
        token_ids = tuple(self._standalone_token_id(label) for label in labels)
        if any(token_id is None for token_id in token_ids):
            raise RuntimeError(
                f"could not encode A/B calibration labels for {field.dotted_path}"
            )

        if swapped:
            option_values: tuple[bool, bool] = (True, False)
            role = "swapped"
        else:
            option_values = (False, True)
            role = "normal"

        rendered = self._render_threshold_prompt(
            prompt,
            field,
            threshold_value,
            option_values,
        )
        return _Task(
            field_index=field_index,
            method="label-swap-threshold",
            input_text=rendered,
            token_ids=tuple(
                int(token_id) for token_id in token_ids if token_id is not None
            ),
            option_values=option_values,
            option_texts=labels,
            normalized_threshold=normalized_threshold,
            threshold_value=threshold_value,
            calibration_role=role,
        )

    def _prepare_sequence_task(
        self,
        prompt: str,
        field_index: int,
        field: NumericFieldSpec,
        *,
        value: int | float,
        candidate_text: str,
        source: str,
    ) -> _Task:
        rendered = self._render_numeric_value_prompt(prompt, field)
        base_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        full_text = rendered + candidate_text
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        prefix_len = _common_prefix_length(base_ids, full_ids)
        target_ids = tuple(full_ids[prefix_len:])
        if not target_ids:
            raise RuntimeError(
                f"empty teacher-forced suffix for numeric field {field.dotted_path}"
            )
        return _Task(
            field_index=field_index,
            method="numeric-sequence-candidate",
            input_text=full_text,
            sequence_candidate=value,
            sequence_source=source,
            sequence_target_ids=target_ids,
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

    def _render_threshold_prompt(
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
                "content": "You are a deterministic classifier. Output only A or B.",
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{prompt}\n\n"
                    f"Field: {field.dotted_path}{description}\n"
                    f"Allowed range: [{field.minimum}, {field.maximum}]\n"
                    f"Question: Is the best value for this field greater than or equal to "
                    f"{threshold_value}?\n"
                    f"{mapping}\n"
                    "Return exactly A or B."
                ),
            },
        ]
        return self._render_assistant_prefill(messages)

    def _render_numeric_value_prompt(
        self,
        prompt: str,
        field: NumericFieldSpec,
    ) -> str:
        description = f"\nField description: {field.description}" if field.description else ""
        messages = [
            {
                "role": "system",
                "content": "Return only the requested numeric value. Do not explain.",
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{prompt}\n\n"
                    f"Field: {field.dotted_path}{description}\n"
                    f"Allowed range: [{field.minimum}, {field.maximum}]\n"
                    "Return the value of this field. "
                    "If explicitly stated, use that exact value."
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
        full_ids = self.tokenizer.encode(
            rendered_prompt + text,
            add_special_tokens=False,
        )
        if (
            full_ids[: len(base_ids)] == base_ids
            and len(full_ids) == len(base_ids) + 1
        ):
            return full_ids[-1]
        return self._standalone_token_id(text)

    def _standalone_token_id(self, text: str) -> int | None:
        direct_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(direct_ids) == 1:
            return direct_ids[0]
        return None


def _numeric_literal_candidates(
    prompt: str,
    field: NumericFieldSpec,
    *,
    max_literals: int,
) -> list[tuple[int | float, str]]:
    found: list[tuple[int | float, str]] = []
    seen: set[int | float] = set()

    for match in _NUMBER_RE.finditer(prompt):
        text = match.group(0)
        try:
            numeric = float(text)
        except ValueError:
            continue
        if not math.isfinite(numeric):
            continue
        if numeric < float(field.minimum) or numeric > float(field.maximum):
            continue

        if field.integer:
            if not numeric.is_integer():
                continue
            value: int | float = int(numeric)
        else:
            value = numeric

        if field.multiple_of is not None:
            span = float(field.maximum) - float(field.minimum)
            if span == 0:
                continue
            normalized = (float(value) - float(field.minimum)) / span
            quantized = quantize_numeric(field, normalized)
            if not math.isclose(
                float(quantized),
                float(value),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                continue
            value = quantized

        if value in seen:
            continue
        seen.add(value)
        found.append((value, text))
        if len(found) >= max_literals:
            break

    return found


def _literal_candidate_set(
    field: NumericFieldSpec,
    literals: list[tuple[int | float, str]],
) -> list[tuple[int | float, str, str]]:
    candidates: list[tuple[int | float, str, str]] = []
    by_value: dict[int | float, int] = {}

    for value, text in literals:
        by_value[value] = len(candidates)
        candidates.append((value, text, "literal"))

    for normalized in (0.0, 0.5, 1.0):
        value = quantize_numeric(field, normalized)
        if value in by_value:
            continue
        by_value[value] = len(candidates)
        candidates.append(
            (
                value,
                _format_numeric_candidate(value),
                "anchor",
            )
        )

    return candidates


def _format_numeric_candidate(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), ".12g")


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    size = min(len(left), len(right))
    index = 0
    while index < size and left[index] == right[index]:
        index += 1
    return index


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
    a = min(1.0 - epsilon, max(epsilon, first))
    b = min(1.0 - epsilon, max(epsilon, second))
    mean_log_odds = 0.5 * (
        math.log(a / (1.0 - a))
        + math.log(b / (1.0 - b))
    )
    return 1.0 / (1.0 + math.exp(-mean_log_odds))


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
