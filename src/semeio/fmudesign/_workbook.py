import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)
from xlsxwriter import Workbook
from xlsxwriter.format import Format
from xlsxwriter.utility import xl_rowcol_to_cell
from xlsxwriter.worksheet import Worksheet

type Scalar = StrictStr | StrictBool | StrictInt | StrictFloat
type CellValue = Scalar | None
type PositiveInt = Annotated[StrictInt, Field(gt=0)]

_INVALID_SHEET_NAME = re.compile(r"[\[\]:*?/\\]")
_DESIGN_COLUMNS: list[CellValue] = [
    "sensname",
    "numreal",
    "type",
    "param_name",
    "senscase1",
    "value1",
    "senscase2",
    "value2",
    "dist_name",
    "dist_param1",
    "dist_param2",
    "dist_param3",
    "dist_param4",
    "decimals",
    "corr_sheet",
    "extern_file",
    "dependencies",
]
_SENSITIVITY_COLORS = (
    "#FFF2CC",
    "#DDEBF7",
    "#E2F0D9",
    "#FCE4D6",
    "#E4DFEC",
    "#E7E6E6",
    "#F4CCCC",
    "#D0E0E3",
    "#EEECE1",
)


class SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GeneralSpec(SpecModel):
    repeats: PositiveInt
    rms_seeds: StrictStr | None = None
    distribution_seed: StrictInt | None = None
    correlation_iterations: Annotated[StrictInt, Field(ge=0)] | None = None
    seed_strategy: StrictStr | None = None


class DistributionParameter(SpecModel):
    distribution: StrictStr
    values: list[Scalar] = Field(min_length=1, max_length=4)
    decimals: Annotated[StrictInt, Field(ge=0)] | None = None
    correlation: StrictStr | None = None
    dependency: StrictStr | None = None


class SensitivityBase(SpecModel):
    name: StrictStr
    realizations: PositiveInt | None = None


class SeedSensitivity(SensitivityBase):
    type: Literal["seed"]
    constants: dict[StrictStr, Scalar] = Field(default_factory=dict)


class ScenarioSensitivity(SensitivityBase):
    type: Literal["scenario"]
    cases: list[StrictStr] = Field(min_length=1, max_length=2)
    parameters: dict[StrictStr, list[Scalar]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cases(self) -> Self:
        if len(self.cases) != len(set(self.cases)):
            raise ValueError("Scenario case names must be unique")
        if any(len(values) != len(self.cases) for values in self.parameters.values()):
            raise ValueError("Scenario parameters require one value per case")
        return self


class DistributionSensitivity(SensitivityBase):
    type: Literal["distribution"]
    parameters: dict[StrictStr, DistributionParameter] = Field(min_length=1)


class ReferenceSensitivity(SensitivityBase):
    type: Literal["reference"]


class BackgroundSensitivity(SensitivityBase):
    type: Literal["background"]


class ExternalSensitivity(SensitivityBase):
    type: Literal["external"]
    file: StrictStr
    parameters: list[StrictStr] = Field(min_length=1)


type Sensitivity = Annotated[
    SeedSensitivity
    | ScenarioSensitivity
    | DistributionSensitivity
    | ReferenceSensitivity
    | BackgroundSensitivity
    | ExternalSensitivity,
    Field(discriminator="type"),
]


class BackgroundSpec(SpecModel):
    parameters: dict[StrictStr, DistributionParameter] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_dependencies(self) -> Self:
        if any(parameter.dependency for parameter in self.parameters.values()):
            raise ValueError("Background parameters cannot define dependencies")
        return self


class CorrelationSpec(SpecModel):
    parameters: list[StrictStr] = Field(min_length=1)
    matrix: list[list[StrictInt | StrictFloat]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        expected_widths = list(range(1, len(self.parameters) + 1))
        if (
            len(self.parameters) != len(set(self.parameters))
            or [len(row) for row in self.matrix] != expected_widths
        ):
            raise ValueError(
                "Correlation matrix must be lower-triangular with one row "
                "per unique parameter"
            )
        return self


class DependencySpec(SpecModel):
    source: StrictStr
    values: list[Scalar]
    targets: dict[StrictStr, list[Scalar]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        if any(len(values) != len(self.values) for values in self.targets.values()):
            raise ValueError("Dependency columns must have equal lengths")
        return self


class TableSpec(SpecModel):
    sheet: StrictStr = "Sheet1"
    columns: list[StrictStr] | None = None
    rows: list[list[CellValue]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_width(self) -> Self:
        width = len(self.columns) if self.columns is not None else len(self.rows[0])
        if width == 0 or any(len(row) != width for row in self.rows):
            raise ValueError("Auxiliary table rows must have a consistent width")
        return self


class WorkbookSpec(SpecModel):
    version: Literal[1]
    general: GeneralSpec
    sensitivities: list[Sensitivity] = Field(min_length=1)
    defaults: dict[StrictStr, CellValue]
    background: StrictStr | BackgroundSpec | None = None
    correlations: dict[StrictStr, CorrelationSpec] = Field(default_factory=dict)
    dependencies: dict[StrictStr, DependencySpec] = Field(default_factory=dict)
    auxiliary_files: dict[StrictStr, TableSpec] = Field(default_factory=dict)
    include_instructions: StrictBool = False

    @field_validator("auxiliary_files")
    @classmethod
    def validate_auxiliary_names(
        cls, auxiliary_files: dict[str, TableSpec]
    ) -> dict[str, TableSpec]:
        for name in auxiliary_files:
            if Path(name).name != name or Path(name).suffix.lower() != ".xlsx":
                raise ValueError(
                    "Auxiliary files must use simple filenames ending in .xlsx"
                )
        return auxiliary_files

    @model_validator(mode="after")
    def validate_relationships(self) -> Self:
        sensitivity_names = [item.name.casefold() for item in self.sensitivities]
        if len(sensitivity_names) != len(set(sensitivity_names)):
            raise ValueError("Sensitivity names must be unique")

        correlation_members: dict[str, list[str]] = defaultdict(list)
        for name, parameter in self._distribution_parameters():
            if parameter.correlation is not None:
                if parameter.correlation not in self.correlations:
                    raise ValueError(
                        f"Unknown correlation {parameter.correlation!r} for {name!r}"
                    )
                correlation_members[parameter.correlation].append(name)
            if parameter.dependency is not None:
                dependency = self.dependencies.get(parameter.dependency)
                if dependency is None:
                    raise ValueError(
                        f"Unknown dependency {parameter.dependency!r} for {name!r}"
                    )
                if dependency.source != name:
                    raise ValueError(
                        f"Dependency {parameter.dependency!r} must use {name!r} "
                        "as its source"
                    )

        for name, members in correlation_members.items():
            if set(members) != set(self.correlations[name].parameters):
                raise ValueError(
                    f"Correlation {name!r} parameters must match its references"
                )

        dynamic_sheets = [
            *(["background"] if isinstance(self.background, BackgroundSpec) else []),
            *self.correlations,
            *self.dependencies,
            *(["INFO"] if self.include_instructions else []),
        ]
        for name in dynamic_sheets:
            _validate_sheet_name(name)
        reserved = {"general_input", "designinput", "defaultvalues"}
        folded_names = [name.casefold() for name in [*reserved, *dynamic_sheets]]
        if len(folded_names) != len(set(folded_names)):
            raise ValueError("Generated worksheet names must be unique")
        for table in self.auxiliary_files.values():
            _validate_sheet_name(table.sheet)
        return self

    def _distribution_parameters(
        self,
    ) -> list[tuple[str, DistributionParameter]]:
        parameters = [
            (name, parameter)
            for sensitivity in self.sensitivities
            if isinstance(sensitivity, DistributionSensitivity)
            for name, parameter in sensitivity.parameters.items()
        ]
        if isinstance(self.background, BackgroundSpec):
            parameters.extend(self.background.parameters.items())
        return parameters


def load_workbook_spec(resource: Traversable) -> WorkbookSpec:
    try:
        raw_spec = yaml.safe_load(resource.read_text(encoding="utf-8"))
        return WorkbookSpec.model_validate(raw_spec)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as error:
        raise ValueError(
            f"Invalid workbook specification {resource}: {error}"
        ) from error


def render_workbook(
    spec: WorkbookSpec | Mapping[str, object], destination: str | Path
) -> list[Path]:
    try:
        workbook_spec = (
            spec
            if isinstance(spec, WorkbookSpec)
            else WorkbookSpec.model_validate(spec)
        )
    except ValidationError as error:
        raise ValueError(f"Invalid workbook specification: {error}") from error

    destination = Path(destination)
    _render_config_workbook(workbook_spec, destination)
    created = [destination]
    for filename, table in workbook_spec.auxiliary_files.items():
        auxiliary_path = destination.parent / filename
        if auxiliary_path.resolve() == destination.resolve():
            raise ValueError("An auxiliary file cannot replace the main workbook")
        _render_table_workbook(table, auxiliary_path)
        created.append(auxiliary_path)
    return created


def render_workbook_resource(
    resource: Traversable, destination: str | Path
) -> list[Path]:
    return render_workbook(load_workbook_spec(resource), destination)


def _render_config_workbook(spec: WorkbookSpec, destination: Path) -> None:
    with Workbook(destination) as workbook:
        formats = _create_formats(workbook)
        _render_general(workbook, formats, spec)
        sensitivity_rows, sensitivity_groups = _design_rows(spec.sensitivities)
        if _uses_dependencies(spec.sensitivities):
            design_columns = _DESIGN_COLUMNS
        else:
            design_columns = _DESIGN_COLUMNS[:-1]
            sensitivity_rows = [row[:-1] for row in sensitivity_rows]
        design_rows: list[list[CellValue]] = [
            design_columns,
            *sensitivity_rows,
        ]
        design_formats = [
            formats["header"],
            *[
                formats[f"sensitivity_{group % len(_SENSITIVITY_COLORS)}"]
                for group in sensitivity_groups
            ],
        ]
        _render_table(
            workbook.add_worksheet("designinput"),
            design_rows,
            formats,
            row_formats=design_formats,
        )
        default_rows: list[list[CellValue]] = [
            ["param_name", "default_value"],
            *[[name, value] for name, value in spec.defaults.items()],
        ]
        _render_table(
            workbook.add_worksheet("defaultvalues"),
            default_rows,
            formats,
        )
        if isinstance(spec.background, BackgroundSpec):
            background_rows: list[list[CellValue]] = [
                [
                    "param_name",
                    "dist_name",
                    "dist_param1",
                    "dist_param2",
                    "dist_param3",
                    "dist_param4",
                    "decimals",
                    "corr_sheet",
                ]
            ]
            for name, distribution in spec.background.parameters.items():
                background_rows.append(_background_row(name, distribution))
            _render_table(
                workbook.add_worksheet("background"),
                background_rows,
                formats,
            )
        for name, correlation in spec.correlations.items():
            correlation_rows: list[list[CellValue]] = [[None, *correlation.parameters]]
            for parameter_name, values in zip(
                correlation.parameters,
                correlation.matrix,
                strict=True,
            ):
                correlation_rows.append([parameter_name, *values])
            _render_table(workbook.add_worksheet(name), correlation_rows, formats)
        for name, dependency in spec.dependencies.items():
            dependency_rows: list[list[CellValue]] = [
                [dependency.source, *dependency.targets]
            ]
            for index, source in enumerate(dependency.values):
                dependency_rows.append(
                    [
                        source,
                        *[
                            dependency.targets[target][index]
                            for target in dependency.targets
                        ],
                    ]
                )
            _render_table(workbook.add_worksheet(name), dependency_rows, formats)
        if spec.include_instructions:
            _render_instructions(workbook.add_worksheet("INFO"), formats)


def _render_general(
    workbook: Workbook, formats: Mapping[str, Format], spec: WorkbookSpec
) -> None:
    background = (
        "background" if isinstance(spec.background, BackgroundSpec) else spec.background
    )
    rows: list[tuple[str, CellValue]] = [
        ("designtype", "onebyone"),
        ("repeats", spec.general.repeats),
        (
            "rms_seeds",
            spec.general.rms_seeds if spec.general.rms_seeds is not None else "None",
        ),
        ("background", background or "None"),
        (
            "distribution_seed",
            spec.general.distribution_seed
            if spec.general.distribution_seed is not None
            else "None",
        ),
    ]
    if spec.general.correlation_iterations is not None:
        rows.append(("correlation_iterations", spec.general.correlation_iterations))
    if spec.general.seed_strategy is not None:
        rows.append(("seed_strategy", spec.general.seed_strategy))

    worksheet = workbook.add_worksheet("general_input")
    for row, (name, value) in enumerate(rows):
        _write_cell(worksheet, row, 0, name, formats["label"])
        _write_cell(worksheet, row, 1, value, formats["input"])
    _set_widths(worksheet, rows)
    worksheet.set_zoom(95)


def _design_rows(
    sensitivities: list[Sensitivity],
) -> tuple[list[list[CellValue]], list[int]]:
    rows: list[list[CellValue]] = []
    groups: list[int] = []
    for group, sensitivity in enumerate(sensitivities):
        first_group_row = len(rows)
        if isinstance(sensitivity, SeedSensitivity):
            constants: list[tuple[str | None, Scalar | None]] = list(
                sensitivity.constants.items()
            )
            if not constants:
                constants.append((None, None))
            for index, (name, value) in enumerate(constants):
                row = _sensitivity_row(sensitivity, index)
                row[3] = name
                if name is not None:
                    row[8:10] = ["const", value]
                rows.append(row)
        elif isinstance(sensitivity, ScenarioSensitivity):
            for index, (name, values) in enumerate(sensitivity.parameters.items()):
                row = _sensitivity_row(sensitivity, index)
                row[3] = name
                if index == 0:
                    row[4] = sensitivity.cases[0]
                    if len(sensitivity.cases) == 2:
                        row[6] = sensitivity.cases[1]
                row[5] = values[0]
                if len(values) == 2:
                    row[7] = values[1]
                rows.append(row)
        elif isinstance(sensitivity, DistributionSensitivity):
            for index, (name, parameter) in enumerate(sensitivity.parameters.items()):
                row = _distribution_row(name, parameter)
                row[:3] = _sensitivity_row(sensitivity, index)[:3]
                rows.append(row)
        elif isinstance(sensitivity, ExternalSensitivity):
            for index, name in enumerate(sensitivity.parameters):
                row = _sensitivity_row(sensitivity, index)
                row[3] = name
                if index == 0:
                    row[15] = sensitivity.file
                rows.append(row)
        else:
            rows.append(_sensitivity_row(sensitivity, 0))
        groups.extend([group] * (len(rows) - first_group_row))
    return rows, groups


def _uses_dependencies(sensitivities: list[Sensitivity]) -> bool:
    return any(
        parameter.dependency is not None
        for sensitivity in sensitivities
        if isinstance(sensitivity, DistributionSensitivity)
        for parameter in sensitivity.parameters.values()
    )


def _sensitivity_row(sensitivity: Sensitivity, index: int) -> list[CellValue]:
    type_name = {
        "distribution": "dist",
        "reference": "ref",
        "external": "extern",
    }.get(sensitivity.type, sensitivity.type)
    return [
        sensitivity.name if index == 0 else None,
        sensitivity.realizations if index == 0 else None,
        type_name if index == 0 else None,
        *([None] * 14),
    ]


def _distribution_row(
    name: str,
    parameter: DistributionParameter,
) -> list[CellValue]:
    row: list[CellValue] = [None] * len(_DESIGN_COLUMNS)
    row[3] = name
    row[8] = parameter.distribution
    row[9 : 9 + len(parameter.values)] = parameter.values
    row[13] = parameter.decimals
    row[14] = parameter.correlation
    row[16] = parameter.dependency
    return row


def _background_row(name: str, parameter: DistributionParameter) -> list[CellValue]:
    return [
        name,
        parameter.distribution,
        *parameter.values,
        *([None] * (4 - len(parameter.values))),
        parameter.decimals,
        parameter.correlation,
    ]


def _render_table_workbook(spec: TableSpec, destination: Path) -> None:
    with Workbook(destination) as workbook:
        formats = _create_formats(workbook)
        rows: list[list[CellValue]] = spec.rows
        if spec.columns is not None:
            rows = [list(spec.columns), *rows]
        _render_table(
            workbook.add_worksheet(spec.sheet),
            rows,
            formats,
            has_header=spec.columns is not None,
        )


def _render_table(
    worksheet: Worksheet,
    rows: Sequence[Sequence[CellValue]],
    formats: Mapping[str, Format],
    *,
    row_formats: Sequence[Format] | None = None,
    has_header: bool = True,
) -> None:
    if row_formats is not None and len(row_formats) != len(rows):
        raise ValueError("A format must be provided for every table row")
    for row_index, values in enumerate(rows):
        if row_formats is not None:
            cell_format = row_formats[row_index]
        else:
            cell_format = (
                formats["header"] if has_header and row_index == 0 else formats["body"]
            )
        for column, value in enumerate(values):
            _write_cell(worksheet, row_index, column, value, cell_format)
    if has_header:
        worksheet.freeze_panes(1, 0)
        worksheet.set_row(0, 30)
    _set_widths(worksheet, rows)
    worksheet.set_zoom(90)


def _render_instructions(worksheet: Worksheet, formats: Mapping[str, Format]) -> None:
    lines = [
        "FMU-design example workbook",
        "Sensitivity blocks are color-banded; rows within a block share a color.",
        "Run: fmudesign run <input.xlsx> <output.xlsx>",
        "See the FMU-design documentation for configuration options.",
    ]
    for row, line in enumerate(lines):
        _write_cell(
            worksheet,
            row,
            0,
            line,
            formats["title"] if row == 0 else formats["plain"],
        )
    worksheet.set_column(0, 0, 88)


def _create_formats(workbook: Workbook) -> dict[str, Format]:
    formats = {
        "header": workbook.add_format(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#1F4E78",
                "border": 1,
                "align": "center",
                "valign": "vcenter",
                "text_wrap": True,
            }
        ),
        "body": workbook.add_format({"bg_color": "#FFF2CC", "border": 1}),
        "label": workbook.add_format(
            {"bold": True, "bg_color": "#D9EAF7", "border": 1}
        ),
        "input": workbook.add_format({"bg_color": "#FFF2CC", "border": 1}),
        "title": workbook.add_format(
            {"bold": True, "font_size": 16, "font_color": "#1F4E78"}
        ),
        "plain": workbook.add_format(),
    }
    formats.update(
        {
            f"sensitivity_{index}": workbook.add_format(
                {"bg_color": color, "border": 1}
            )
            for index, color in enumerate(_SENSITIVITY_COLORS)
        }
    )
    return formats


def _set_widths(
    worksheet: Worksheet,
    rows: Sequence[Sequence[CellValue]],
) -> None:
    width = max(len(row) for row in rows)
    for column in range(width):
        values = (row[column] for row in rows if column < len(row))
        column_width = min(
            40,
            max(10, max(len(str(value)) for value in values) + 2),
        )
        if worksheet.set_column(column, column, column_width) != 0:
            raise ValueError(f"Failed to set width for column {column + 1}")


def _write_cell(
    worksheet: Worksheet,
    row: int,
    column: int,
    value: CellValue,
    cell_format: Format,
) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite number in cell {xl_rowcol_to_cell(row, column)}")
    result = (
        worksheet.write_blank(row, column, None, cell_format)
        if value is None
        else worksheet.write(row, column, value, cell_format)
    )
    if result != 0:
        reference = xl_rowcol_to_cell(row, column)
        raise ValueError(f"Failed to write cell {reference}: XlsxWriter error {result}")


def _validate_sheet_name(name: str) -> None:
    if (
        not name
        or len(name) > 31
        or _INVALID_SHEET_NAME.search(name)
        or name.startswith("'")
        or name.endswith("'")
    ):
        raise ValueError(f"Invalid worksheet name: {name!r}")
