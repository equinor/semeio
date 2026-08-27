from pathlib import Path
from typing import Literal, Self

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
    def from_dict(
        cls, input_dict: dict[str, str | None], input_filename: str = ""
    ) -> Self:
        general_input: dict[str, str | Path | None] = dict(input_dict.items())

        for key in ["seed_strategy", "correlation_iterations"]:
            if general_input.get(key) is None:
                print(
                    f"'{key}' not set in general input sheet. "
                    f"Setting to default "
                    f"{GeneralInput.model_fields[key].default}."
                )
                general_input.pop(key, None)

        for key in ["rms_seeds", "background"]:
            if isinstance((val := general_input.get(key)), str):
                resolved = resolve_path(val, base_file=input_filename)
                assert isinstance(resolved, str)
                general_input[key] = (
                    Path(resolved) if Path(resolved).is_file() else resolved
                )

        return cls(**general_input)

    @field_serializer("rms_seeds", "background")
    def serialize_paths(self, field: Path | None) -> str | None:  # ruff: ignore[no-self-use]
        if field is None:
            return None
        return str(field)
