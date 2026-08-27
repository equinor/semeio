from collections.abc import Collection
from pathlib import Path
from typing import Any, Literal, Self

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    NonNegativeInt,
    PositiveInt,
    field_serializer,
)

from semeio.fmudesign.config_validation import SeedStrategy
from semeio.fmudesign.utils import resolve_path


def parse_value(value: object) -> object:
    if isinstance(value, str):
        return value.strip()
    # pd.isna(Collection) -> NDArray, which is ambiguous
    if not isinstance(value, Collection) and pd.isna(value):  # type: ignore[call-overload]
        return None
    return value


class GeneralInput(BaseModel):
    designtype: Literal["onebyone"]
    repeats: PositiveInt
    distribution_seed: NonNegativeInt | None
    rms_seeds: FilePath | Literal["default"] | None
    correlation_iterations: NonNegativeInt = 0
    seed_strategy: SeedStrategy = Field(
        default=SeedStrategy.JOINT, validate_default=True
    )
    background: FilePath | str | None = None

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    @classmethod
    def from_dict(cls, inputdict: dict[str, Any], input_filename: str = "") -> Self:
        general_input: dict[str, Any] = {
            str(key).strip(): parse_value(value) for key, value in inputdict.items()
        }

        # Boolean values are interpreted as valid ints, and are not caught by
        # pydantic's validation. It can be caught using strict=True, but that
        # removes the flexibility of allowing numeric strings for numeric fields.
        # No values should be boolean, so we check for all.
        for key, value in general_input.items():
            if isinstance(value, bool):
                raise ValueError(
                    f"key '{key}' cannot have boolean value, got '{value}'"
                )

        for key in ["seed_strategy", "correlation_iterations"]:
            val = general_input.get(key)
            is_none = general_input.get(key) is None
            is_none_str = isinstance(val, str) and val.lower() == "none"
            if is_none or is_none_str:
                print(
                    f"'{key}' not set in general input sheet. "
                    f"Setting to default "
                    f"{GeneralInput.model_fields[key].default}."
                )
                general_input.pop(key, None)

        for key in ["rms_seeds", "background"]:
            val = general_input.get(key)
            if isinstance(val, str):
                resolved = resolve_path(val, base_file=input_filename)
                assert isinstance(resolved, str)
                if Path(resolved).exists():
                    general_input[key] = Path(resolved)
                elif resolved.lower() == "none":
                    general_input[key] = None
                else:
                    general_input[key] = resolved

        return cls(**general_input)

    @field_serializer("rms_seeds", "background")
    def serialize_paths(self, field: Path | None) -> str | None:  # ruff: ignore[no-self-use]
        if field is None:
            return None
        return str(field)
