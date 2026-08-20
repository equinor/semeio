from collections.abc import Collection
from pathlib import Path
from typing import Any, Literal, Self

import pandas as pd
from pydantic import BaseModel, model_validator

from semeio.fmudesign.config_validation import SeedStrategy
from semeio.fmudesign.read_background import read_background
from semeio.fmudesign.utils import resolve_path, seeds_from_extern


def parse_value(value: object) -> object:
    if isinstance(value, str):
        return value.strip()
    # pd.isna(Collection) -> NDArray, which is ambiguous
    if not isinstance(value, Collection) and pd.isna(value):  # type: ignore[call-overload]
        return None
    return value


class GeneralInput(BaseModel):
    designtype: Literal["onebyone"]
    repeats: int
    distribution_seed: int | None
    rms_seeds: list[int] | Literal["default"] | None
    correlation_iterations: int = 0
    seed_strategy: SeedStrategy = SeedStrategy.JOINT
    background: Path | dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, inputdict: dict[str, Any]) -> Self:
        general_input: dict[str, Any] = {}
        for key_, value_ in inputdict.items():
            key = str(key_).strip()
            value = parse_value(value_)

            # Boolean values are interpreted as valid ints, and are not caught by
            # pydantic's validation. No values should be boolean, so we check for all
            if isinstance(value, bool):
                raise ValueError(
                    f"key '{key}' cannot have boolean value, got '{value}'"
                )

            match key:
                case "input_filename":
                    continue
                case "designtype":
                    general_input[key] = value
                case "repeats":
                    general_input[key] = value
                case "distribution_seed":
                    general_input[key] = value
                case "rms_seeds":
                    if value == "default":
                        general_input[key] = value
                    elif isinstance(value, str):
                        maybe_path = resolve_path(inputdict["input_filename"], value)
                        if isinstance(maybe_path, str):
                            general_input[key] = seeds_from_extern(maybe_path)
                    elif value is None:
                        general_input[key] = value
                    else:
                        raise ValueError(
                            "'rms_seeds' in generalinput must be 'default', 'None' or "
                            "relative path to a file from the design input file"
                        )
                case "correlation_iterations":
                    general_input[key] = value
                case "seed_strategy":
                    general_input[key] = value
                case "background":
                    background = str(value).strip()
                    if background.lower() in {"", "none"}:
                        general_input[key] = None
                    elif background.endswith(("csv", "xlsx", "txt")):
                        general_input[key] = {
                            "extern": resolve_path(
                                inputdict["input_filename"], background
                            )
                        }
                    else:
                        general_input[key] = read_background(
                            inputdict["input_filename"], background
                        )
                case _:
                    raise ValueError(
                        f"Invalid key '{key}' in general_input.\n"
                        f"Valid keys are: {', '.join(list(GeneralInput.model_fields))}"
                    )

        for key in [
            "correlation_iterations",
            "seed_strategy",
        ]:
            if key not in general_input:
                default_value = GeneralInput.model_fields[key].default
                print(
                    f"'{key}' not set in general input sheet. "
                    f"Setting to default {default_value}."
                )

        return cls(**general_input)

    @model_validator(mode="after")
    def validate_positive_ints(self) -> Self:
        if self.repeats < 1:
            raise ValueError(
                "'repeats' in generalinput must be an int greater than zero"
            )
        if self.correlation_iterations < 0:
            raise ValueError(
                "'correlation_iterations' in generalinput must be a "
                "positive int or zero"
            )
        if isinstance(self.distribution_seed, int) and self.distribution_seed < 0:
            raise ValueError(
                "'distribution_seed' in generalinput must be a positive int"
            )
        return self
