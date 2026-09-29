import pytest

from autoselect.schema import (
    ChoiceFieldSpec,
    NumericFieldSpec,
    SchemaError,
    build_object,
    compile_schema,
    quantize_numeric,
)


def test_compile_dynamic_numeric_schema_without_enumerating_range() -> None:
    fields = compile_schema(
        {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "number",
                    "minimum": -20,
                    "maximum": 80,
                },
                "score": {
                    "type": "integer",
                    "minimum": 1000,
                    "maximum": 50000,
                },
                "run": {"type": "boolean"},
            },
        }
    )

    assert isinstance(fields[0], NumericFieldSpec)
    assert fields[0].minimum == -20
    assert fields[0].maximum == 80
    assert fields[0].multiple_of is None

    assert isinstance(fields[1], NumericFieldSpec)
    assert fields[1].minimum == 1000
    assert fields[1].maximum == 50000
    assert fields[1].multiple_of == 1

    assert isinstance(fields[2], ChoiceFieldSpec)
    assert fields[2].candidates == (False, True)


def test_quantize_dynamic_ranges() -> None:
    fields = compile_schema(
        {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "number",
                    "minimum": -20,
                    "maximum": 80,
                },
                "score": {
                    "type": "integer",
                    "minimum": 1000,
                    "maximum": 50000,
                },
                "stepped": {
                    "type": "number",
                    "minimum": 0.15,
                    "maximum": 0.95,
                    "multipleOf": 0.2,
                },
            },
        }
    )

    assert quantize_numeric(fields[0], 0.75) == pytest.approx(55.0)
    assert quantize_numeric(fields[1], 0.75) == 37750
    assert quantize_numeric(fields[2], 0.51) == pytest.approx(0.6)


def test_build_object() -> None:
    fields = compile_schema(
        {
            "type": "object",
            "properties": {
                "a": {"type": "boolean"},
                "nested": {
                    "type": "object",
                    "properties": {
                        "b": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 10,
                        }
                    },
                },
            },
        }
    )
    value = build_object([(fields[0], True), (fields[1], 7.5)])
    assert value == {"a": True, "nested": {"b": 7.5}}


def test_number_no_longer_requires_multiple_of() -> None:
    fields = compile_schema(
        {
            "type": "object",
            "properties": {
                "value": {"type": "number", "minimum": -1.5, "maximum": 12.75}
            },
        }
    )
    assert isinstance(fields[0], NumericFieldSpec)
    assert fields[0].multiple_of is None


def test_reject_free_string() -> None:
    with pytest.raises(SchemaError, match="requires enum"):
        compile_schema(
            {"type": "object", "properties": {"text": {"type": "string"}}}
        )


def test_reject_non_object_root() -> None:
    with pytest.raises(SchemaError, match="top-level schema"):
        compile_schema({"type": "boolean"})


def test_reject_enum_with_too_many_candidates() -> None:
    with pytest.raises(SchemaError, match="maximum is 26"):
        compile_schema(
            {
                "type": "object",
                "properties": {"n": {"enum": list(range(30))}},
            }
        )


def test_large_integer_range_does_not_hit_candidate_limit() -> None:
    fields = compile_schema(
        {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "minimum": -1_000_000_000,
                    "maximum": 1_000_000_000,
                }
            },
        }
    )
    assert isinstance(fields[0], NumericFieldSpec)
