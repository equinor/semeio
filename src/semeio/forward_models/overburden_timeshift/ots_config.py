from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional

from pydantic import BaseModel, Field, conlist, field_validator, model_validator
from resdata.resfile import ResdataFile
from typing_extensions import Annotated, Self

if TYPE_CHECKING:
    ConstrainedList = List[date]
else:
    ConstrainedList = conlist(date, min_length=2, max_length=2)


class Vintages(BaseModel):
    """
    vintages:
      dpv:
      - ['1997-11-06', '1997-12-17']
      ts_simple:
      - ['1997-11-06', '1998-02-01']
      - ['1997-12-17', '1998-02-01']
    """

    ts_simple: Annotated[
        List[ConstrainedList],
        Field(
            description=(
                "Simple TimeShift geertsma algorithm. "
                "It assumes a constant velocity and is fast"
            )
        ),
    ] = []
    ts: Annotated[
        List[ConstrainedList],
        Field(
            description="TimeShift geertsma algorithm, which uses velocity, very slow"
        ),
    ] = []
    ts_rporv: Annotated[
        List[ConstrainedList],
        Field(description="Delta pressure multiplied by cell volume, relatively fast"),
    ] = []
    dpv: Annotated[
        List[ConstrainedList],
        Field(
            description=(
                "Calculates timeshift without using velocity. The velocity is only "
                "used to get the surface on the velocity grid. It uses a change in "
                "porevolume from Eclipse (RPORV in .UNRST) as input to Geertsma model."
            )
        ),
    ] = []

    @model_validator(mode="after")
    def check_not_empty(self) -> Self:
        if not self.ts and not self.ts_simple and not self.ts_rporv and not self.dpv:
            raise ValueError("Vintages must contain at least one entry")
        return self


class OTSConfig(BaseModel):
    file_format: Annotated[
        Literal[
            "irap_ascii",
            "irapascii",
            "irap_txt",
            "irapasc",
            "irap_binary",
            "irapbinary",
            "irapbin",
            "irap",
            "gri",
            "zmap",
            "storm_binary",
            "petromod",
            "ijxyz",
        ],
        Field(description="The file format of the exported surfaces"),
    ] = "irap_binary"
    seabed: Annotated[float, Field(description="The depth of the seabead in meters.")]
    rfactor: Annotated[
        float,
        Field(
            description=(
                "Scales the surface displacement "
                "between base_survey and monitor_survey, eg. 20."
            )
        ),
    ]
    above: Annotated[
        float,
        Field(
            description=(
                "Distance in meters above the reservoir where shift is calculated. "
                "The distance from the shallowest cell, eg. 100"
            )
        ),
    ]
    convention: Annotated[
        Literal[-1, 1],
        Field(
            1,
            description=(
                "Positive or negative shift can be either 1 or -1, where 1 = "
                "monitor-base and -1 = base-monitor"
            ),
        ),
    ]
    poisson: Annotated[
        float,
        Field(
            description=(
                "Poisson ratio. Describes the expansion or "
                "contraction of material in ecl_subsidence_eval"
            )
        ),
    ]
    youngs: Annotated[float, Field(0.0, description="Youngs modulus")]
    output_dir: Annotated[
        str,
        Field(
            description=(
                "Directory(ies) where the shift is written to disk. Post fixed with "
                "type of algorithm: ts, ts_simple, dpv and ts_rporv"
            )
        ),
    ]
    horizon: Annotated[
        Optional[str],
        Field(
            None,
            description=(
                "Path to result irap file, the surface mapped to the velocity grid, "
                "with the depth of horizon."
            ),
        ),
    ]
    eclbase: Annotated[str, Field(description="Path to the Eclipse case")]
    vintages_export_file: Annotated[
        Optional[str],
        Field(
            None,
            description="Path to resulting text file, which contains all computed "
            "vintage pair dates: lines of x, y, z, ts1, ts2, ts3...",
        ),
    ]
    velocity_model: Annotated[
        Optional[str],
        Field(None, description="Path to segy file containing the velocity field"),
    ]
    mapaxes: Annotated[
        bool,
        Field(
            description=(
                "Mapping axes from the global to local geometry. "
                "If False EclGrid will not apply transformation to the grid"
            )
        ),
    ]
    vintages: Annotated[
        Vintages,
        Field(
            description="Vintage date pairs: date of base and monitor survey",
            examples=Vintages.__doc__,
        ),
    ]

    @field_validator("eclbase")
    @classmethod
    def check_eclbase(cls, value: str) -> str:
        errors = []
        if not Path(f"{value}.INIT").exists():
            errors.append(f"{value}.INIT")
        if not Path(f"{value}.EGRID").exists():
            errors.append(f"{value}.EGRID")
        if not Path(f"{value}.UNRST").exists():
            errors.append(f"{value}.UNRST")
        if errors:
            raise ValueError(f"eclbase missing required file(s): {errors}")
        return value

    @model_validator(mode="after")
    def check_date_in_rst(self) -> Self:
        errors = []
        rst_dates = {d.date() for d in ResdataFile(f"{self.eclbase}.UNRST").dates}

        for field in self.vintages.model_fields_set:
            vintage = getattr(self.vintages, field)
            for dates in vintage:
                dates = set(dates)
                if not dates.issubset(rst_dates):
                    errors.append(
                        f"Dates: {dates - rst_dates} missing for property: {field}"
                    )
        if errors:
            raise ValueError(
                f"Vintages with dates not found in: {self.eclbase}.UNRST: {errors}"
            )
        return self
