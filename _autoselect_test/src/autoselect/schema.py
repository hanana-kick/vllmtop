from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Any


class SchemaError(ValueError):
    """Raised when a schema cannot be represented by the single-forward selector."""


@dataclass(frozen=True)
class ChoiceFieldSpec:
    path: tuple[str, ...]
    candidates: tuple[Any, ...]
    description: str | None = None

    @property
    def dotted_path(self) -> str:
        return ".".join(self.path)


@dataclass(frozen=True)
class NumericFieldSpec:
    path: tuple[str, ...]
    minimum: int | float
    maximum: int | float
    multiple_of: int | float | None
    integer: bool
    description: str | None = None

    @property
    def dotted_path(self) -> str:
        return ".".join(self.path)


FieldSpec = ChoiceFieldSpec | NumericFieldSpec


def compile_schema(
    schema: dict[str, Any],
    *,
    max_candidates: int = 26,
) -> list[FieldSpec]:
    """Compile a top-level JSON object schema into selectable leaf fields."""
    if not isinstance(schema, dict):
        raise SchemaError("schema must be an object")
    if schema.get("type") != "object" and "properties" not in schema:
        raise SchemaError("top-level schema must be an object with properties")

    fields: list[FieldSpec] = []
    _walk_schema(schema, (), fields, max_candidates=max_candidates)
    if not fields:
        raise SchemaError("schema contains no selectable leaf fields")
    return fields


def _walk_schema(
    schema: dict[str, Any],
    path: tuple[str, ...],
    fields: list[FieldSpec],
    *,
    max_candidates: int,
) -> None:
    if not isinstance(schema, dict):
        raise SchemaError(f"schema at {_path_name(path)} must be an object")

    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties")
        if not isinstance(properties, dict) or not properties:
            raise SchemaError(f"object at {_path_name(path)} must have non-empty properties")
        for key, child in properties.items():
            if not isinstance(key, str) or not key:
                raise SchemaError(f"invalid property name at {_path_name(path)}")
            _walk_schema(child, path + (key,), fields, max_candidates=max_candidates)
        return

    description = schema.get("description")
    if not isinstance(description, str):
        description = None

    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum:
            raise SchemaError(f"enum at {_path_name(path)} must be a non-empty array")
        if len(enum) > max_candidates:
            raise SchemaError(
                f"{_path_name(path)} has {len(enum)} enum candidates; maximum is {max_candidates}"
            )
        fields.append(
            ChoiceFieldSpec(
                path=path,
                candidates=tuple(enum),
                description=description,
            )
        )
        return

    if schema_type == "boolean":
        fields.append(
            ChoiceFieldSpec(
                path=path,
                candidates=(False, True),
                description=description,
            )
        )
        return

    if schema_type == "integer":
        minimum, maximum = _bounded_numbers(schema, path)
        if not isinstance(minimum, int) or isinstance(minimum, bool):
            raise SchemaError(f"integer minimum at {_path_name(path)} must be an integer")
        if not isinstance(maximum, int) or isinstance(maximum, bool):
            raise SchemaError(f"integer maximum at {_path_name(path)} must be an integer")
        multiple_of = schema.get("multipleOf", 1)
        if not isinstance(multiple_of, int) or isinstance(multiple_of, bool) or multiple_of <= 0:
            raise SchemaError(
                f"integer multipleOf at {_path_name(path)} must be a positive integer"
            )
        _ensure_valid_multiple(minimum, maximum, multiple_of, path)
        fields.append(
            NumericFieldSpec(
                path=path,
                minimum=minimum,
                maximum=maximum,
                multiple_of=multiple_of,
                integer=True,
                description=description,
            )
        )
        return

    if schema_type == "number":
        minimum, maximum = _bounded_numbers(schema, path)
        multiple_of = schema.get("multipleOf")
        if multiple_of is not None:
            if not _is_number(multiple_of) or multiple_of <= 0:
                raise SchemaError(
                    f"number multipleOf at {_path_name(path)} must be a positive finite number"
                )
            _ensure_valid_multiple(minimum, maximum, multiple_of, path)
        fields.append(
            NumericFieldSpec(
                path=path,
                minimum=minimum,
                maximum=maximum,
                multiple_of=multiple_of,
                integer=False,
                description=description,
            )
        )
        return

    if schema_type == "string":
        raise SchemaError(f"string at {_path_name(path)} requires enum")

    if schema_type == "array":
        raise SchemaError(f"array at {_path_name(path)} is not supported")

    raise SchemaError(
        f"unsupported leaf schema at {_path_name(path)}; "
        "use boolean, enum, bounded integer, or bounded number"
    )


def quantize_numeric(field: NumericFieldSpec, normalized_value: float) -> int | float:
    """Map a normalized [0, 1] estimate back into the schema's numeric domain."""
    normalized = min(1.0, max(0.0, float(normalized_value)))
    lo = Decimal(str(field.minimum))
    hi = Decimal(str(field.maximum))
    raw = lo + Decimal(str(normalized)) * (hi - lo)

    step_value: int | float | None = field.multiple_of
    if field.integer and step_value is None:
        step_value = 1

    if step_value is not None:
        step = Decimal(str(step_value))
        first_index = (lo / step).to_integral_value(rounding=ROUND_CEILING)
        last_index = (hi / step).to_integral_value(rounding=ROUND_FLOOR)
        index = (raw / step).to_integral_value(rounding=ROUND_HALF_UP)
        index = min(last_index, max(first_index, index))
        raw = index * step

    if field.integer:
        return int(raw)

    if raw == raw.to_integral_value():
        return int(raw)
    return float(raw)


def numeric_threshold_value(
    field: NumericFieldSpec,
    normalized_threshold: float,
) -> int | float:
    lo = Decimal(str(field.minimum))
    hi = Decimal(str(field.maximum))
    value = lo + Decimal(str(normalized_threshold)) * (hi - lo)
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def build_object(selections: list[tuple[FieldSpec, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field, value in selections:
        cursor = output
        for part in field.path[:-1]:
            child = cursor.get(part)
            if child is None:
                child = {}
                cursor[part] = child
            if not isinstance(child, dict):
                raise RuntimeError(f"path collision at {field.dotted_path}")
            cursor = child
        cursor[field.path[-1]] = value
    return output


def _bounded_numbers(
    schema: dict[str, Any],
    path: tuple[str, ...],
) -> tuple[int | float, int | float]:
    if "minimum" not in schema or "maximum" not in schema:
        raise SchemaError(f"{_path_name(path)} requires minimum and maximum")
    minimum = schema["minimum"]
    maximum = schema["maximum"]
    if not _is_number(minimum) or not _is_number(maximum):
        raise SchemaError(f"minimum/maximum at {_path_name(path)} must be finite numbers")
    if minimum > maximum:
        raise SchemaError(f"minimum exceeds maximum at {_path_name(path)}")
    return minimum, maximum


def _ensure_valid_multiple(
    minimum: int | float,
    maximum: int | float,
    multiple_of: int | float,
    path: tuple[str, ...],
) -> None:
    lo = Decimal(str(minimum))
    hi = Decimal(str(maximum))
    step = Decimal(str(multiple_of))
    first = (lo / step).to_integral_value(rounding=ROUND_CEILING) * step
    if first > hi:
        raise SchemaError(
            f"{_path_name(path)} contains no value divisible by multipleOf={multiple_of}"
        )


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _path_name(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "<root>"
