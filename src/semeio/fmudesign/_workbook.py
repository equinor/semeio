import math
import re
from collections.abc import Mapping
from datetime import date, datetime
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
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
from xlsxwriter.utility import xl_cell_to_rowcol, xl_col_to_name, xl_rowcol_to_cell
from xlsxwriter.worksheet import Worksheet

type CellValue = (
    StrictStr | StrictBool | StrictInt | StrictFloat | datetime | date | None
)
type FormatValue = StrictStr | StrictBool | StrictInt | StrictFloat

_CELL_REFERENCE = re.compile(r"^[A-Z]{1,3}[1-9][0-9]*$")
_COLUMN_RANGE = re.compile(r"^[A-Z]{1,3}(?::[A-Z]{1,3})?$")
_INVALID_SHEET_NAME = re.compile(r"[\[\]:*?/\\]")
_MAX_EXCEL_ROW = 1_048_576
_MAX_EXCEL_COLUMN = 16_384

_FORMAT_PROPERTIES = {
    "align",
    "bg_color",
    "bold",
    "border",
    "border_color",
    "bottom",
    "bottom_color",
    "center_across",
    "diag_border",
    "diag_color",
    "diag_type",
    "fg_color",
    "font_charset",
    "font_color",
    "font_condense",
    "font_extend",
    "font_family",
    "font_name",
    "font_only",
    "font_outline",
    "font_scheme",
    "font_script",
    "font_shadow",
    "font_size",
    "font_strikeout",
    "hidden",
    "hyperlink",
    "indent",
    "italic",
    "left",
    "left_color",
    "locked",
    "num_format",
    "pattern",
    "quote_prefix",
    "reading_order",
    "right",
    "right_color",
    "rotation",
    "shrink",
    "text_justlast",
    "text_wrap",
    "top",
    "top_color",
    "underline",
    "valign",
}


class FormatSpec(RootModel[dict[str, FormatValue]]):
    model_config = ConfigDict(strict=True)

    @field_validator("root")
    @classmethod
    def validate_properties(
        cls, properties: dict[str, FormatValue]
    ) -> dict[str, FormatValue]:
        if unknown := properties.keys() - _FORMAT_PROPERTIES:
            raise ValueError(f"Unknown format properties: {sorted(unknown)}")
        return properties


class CommentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: StrictStr
    author: StrictStr | None = None


type RowNumber = Annotated[int, Field(ge=1, le=_MAX_EXCEL_ROW)]
type Dimension = Annotated[float, Field(gt=0)]
type Zoom = Annotated[int, Field(ge=10, le=400)]
type PaperSize = Annotated[int, Field(ge=1, le=118)]


class WorksheetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: StrictStr
    rows: dict[RowNumber, list[CellValue]]
    cell_formats: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    comments: dict[StrictStr, CommentSpec] = Field(default_factory=dict)
    column_widths: dict[StrictStr, Dimension] = Field(default_factory=dict)
    row_heights: dict[RowNumber, Dimension] = Field(default_factory=dict)
    default_column_width: Dimension | None = None
    default_row_height: Dimension | None = None
    zoom: Zoom | None = None
    orientation: Literal["portrait", "landscape"] | None = None
    paper_size: PaperSize | None = None
    print_scale: Zoom | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        if not name or len(name) > 31:
            raise ValueError("Worksheet names must contain between 1 and 31 characters")
        if (
            _INVALID_SHEET_NAME.search(name)
            or name.startswith("'")
            or name.endswith("'")
        ):
            raise ValueError(f"Invalid worksheet name: {name!r}")
        return name

    @field_validator("rows")
    @classmethod
    def validate_row_width(
        cls, rows: dict[int, list[CellValue]]
    ) -> dict[int, list[CellValue]]:
        for row_number, values in rows.items():
            if len(values) > _MAX_EXCEL_COLUMN:
                raise ValueError(f"Row {row_number} exceeds Excel's column limit")
        return rows

    @field_validator("cell_formats", "comments")
    @classmethod
    def validate_cell_references(cls, values: dict[str, object]) -> dict[str, object]:
        for reference in values:
            _parse_cell_reference(reference)
        return values

    @field_validator("column_widths")
    @classmethod
    def validate_column_ranges(cls, values: dict[str, float]) -> dict[str, float]:
        for column_range in values:
            _parse_column_range(column_range)
        return values


class WorkbookSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1]
    formats: dict[StrictStr, FormatSpec] = Field(default_factory=dict)
    worksheets: list[WorksheetSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_workbook(self) -> Self:
        names = [worksheet.name.casefold() for worksheet in self.worksheets]
        if len(names) != len(set(names)):
            raise ValueError("Worksheet names must be unique")

        format_names = self.formats.keys()
        for worksheet in self.worksheets:
            if missing := set(worksheet.cell_formats.values()) - format_names:
                raise ValueError(
                    f"Unknown formats in worksheet {worksheet.name!r}: "
                    f"{sorted(missing)}"
                )
        return self


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
) -> None:
    try:
        workbook_spec = (
            spec
            if isinstance(spec, WorkbookSpec)
            else WorkbookSpec.model_validate(spec)
        )
    except ValidationError as error:
        raise ValueError(f"Invalid workbook specification: {error}") from error

    with Workbook(destination) as workbook:
        formats = {
            name: workbook.add_format(format_spec.root)
            for name, format_spec in workbook_spec.formats.items()
        }
        for worksheet_spec in workbook_spec.worksheets:
            _render_worksheet(workbook, worksheet_spec, formats)


def render_workbook_resource(resource: Traversable, destination: str | Path) -> None:
    render_workbook(load_workbook_spec(resource), destination)


def _render_worksheet(
    workbook: Workbook,
    spec: WorksheetSpec,
    formats: Mapping[str, Format],
) -> None:
    worksheet = workbook.add_worksheet(spec.name)
    _apply_worksheet_settings(worksheet, spec)
    cell_formats = {
        _parse_cell_reference(reference): formats[name]
        for reference, name in spec.cell_formats.items()
    }
    written_cells: set[tuple[int, int]] = set()

    for row_number, values in sorted(spec.rows.items()):
        row = row_number - 1
        for column, value in enumerate(values):
            cell_format = cell_formats.get((row, column))
            _write_cell(
                worksheet,
                row,
                column,
                value,
                cell_format,
                xl_rowcol_to_cell(row, column),
            )
            if value is not None or cell_format is not None:
                written_cells.add((row, column))

    for (row, column), cell_format in cell_formats.items():
        if (row, column) not in written_cells:
            _check_write_result(
                worksheet.write_blank(row, column, None, cell_format),
                xl_rowcol_to_cell(row, column),
            )

    for reference, comment in spec.comments.items():
        options = {"author": comment.author} if comment.author is not None else None
        _check_write_result(
            worksheet.write_comment(reference, comment.text, options),
            reference,
        )

    for column_range, width in spec.column_widths.items():
        first_column, last_column = _parse_column_range(column_range)
        if worksheet.set_column(first_column, last_column, width) != 0:
            raise ValueError(f"Failed to set column width for {column_range}")

    for row_number, height in spec.row_heights.items():
        if worksheet.set_row(row_number - 1, height) != 0:
            raise ValueError(f"Failed to set height for row {row_number}")


def _apply_worksheet_settings(
    worksheet: Worksheet,
    spec: WorksheetSpec,
) -> None:
    if spec.default_column_width is not None and (
        worksheet.set_column(
            0,
            _MAX_EXCEL_COLUMN - 1,
            spec.default_column_width,
        )
        != 0
    ):
        raise ValueError("Failed to set the default column width")
    if spec.default_row_height is not None:
        worksheet.set_default_row(spec.default_row_height)
    if spec.zoom is not None:
        worksheet.set_zoom(spec.zoom)
    if spec.orientation == "portrait":
        worksheet.set_portrait()
    elif spec.orientation == "landscape":
        worksheet.set_landscape()
    if spec.paper_size is not None:
        worksheet.set_paper(spec.paper_size)
    if spec.print_scale is not None:
        worksheet.set_print_scale(spec.print_scale)


def _write_cell(
    worksheet: Worksheet,
    row: int,
    column: int,
    value: CellValue,
    cell_format: Format | None,
    reference: str,
) -> None:
    if value is None:
        if cell_format is not None:
            result = worksheet.write_blank(row, column, None, cell_format)
        else:
            return
    elif isinstance(value, str):
        result = worksheet.write_string(row, column, value, cell_format)
    elif isinstance(value, bool):
        result = worksheet.write_boolean(row, column, value, cell_format)
    elif isinstance(value, int):
        result = worksheet.write_number(row, column, value, cell_format)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite number in cell {reference}")
        result = worksheet.write_number(row, column, value, cell_format)
    elif isinstance(value, datetime | date):
        result = worksheet.write_datetime(row, column, value, cell_format)
    else:
        raise TypeError(f"Unsupported value in cell {reference}: {value!r}")
    _check_write_result(result, reference)


def _check_write_result(result: int, reference: str) -> None:
    if result != 0:
        raise ValueError(f"Failed to write cell {reference}: XlsxWriter error {result}")


def _parse_cell_reference(reference: str) -> tuple[int, int]:
    if not _CELL_REFERENCE.fullmatch(reference):
        raise ValueError(f"Invalid cell reference: {reference!r}")
    row, column = xl_cell_to_rowcol(reference)
    if row >= _MAX_EXCEL_ROW or column >= _MAX_EXCEL_COLUMN:
        raise ValueError(f"Cell reference outside Excel limits: {reference!r}")
    return row, column


def _parse_column_range(column_range: str) -> tuple[int, int]:
    if not _COLUMN_RANGE.fullmatch(column_range):
        raise ValueError(f"Invalid column range: {column_range!r}")
    first_name, _, last_name = column_range.partition(":")
    first_column = _column_name_to_index(first_name)
    last_column = _column_name_to_index(last_name or first_name)
    if first_column > last_column:
        raise ValueError(f"Invalid descending column range: {column_range!r}")
    return first_column, last_column


def _column_name_to_index(name: str) -> int:
    _, column = xl_cell_to_rowcol(f"{name}1")
    if column >= _MAX_EXCEL_COLUMN or xl_col_to_name(column) != name:
        raise ValueError(f"Column outside Excel limits: {name!r}")
    return column
