import heapq
import logging
import typing
from contextlib import contextmanager
from csv import writer as csv_writer
from dataclasses import dataclass, field
from enum import Enum
from logging import basicConfig, getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from cnocr.cn_ocr import CnOcr, OcrResult
from numpy import ndarray
from pdf2image import convert_from_path

DEFAULT_OCR_MODEL = "densenet_lite_136-gru"


L = getLogger("report_ocr")


class GroupingResult(Enum):
    FIT = 0  # Right in the group
    INSUFFICIENT = 1  # Left (horizontal) to or above (vertical) the group
    SURPLUS = 2  # Right (horizontal) to or below (vertical) the group


class Orientation(Enum):
    HORIZONTAL = 1
    VERTICAL = 2

    def to_baseline_index(self) -> int:
        """baseline of a vertical group is in the X dimensison, vice versa"""
        return 0 if self == Orientation.VERTICAL else 1

    def to_sort_index(self) -> int:
        """elements in a vertical group are sorted by their Y position,
        vice versa"""
        return 1 if self == Orientation.VERTICAL else 0

    def calc_core(self, result: OcrResult) -> float:
        position = cast(ndarray, result.position)
        return position[:, self.to_baseline_index()].mean()

    def calc_sort_weight(self, result: OcrResult) -> float:
        position = cast(ndarray, result.position)
        return position[:, self.to_sort_index()].mean()

    @property
    def orthogonal(self) -> "Orientation":
        return (
            Orientation.HORIZONTAL
            if self == Orientation.VERTICAL
            else Orientation.VERTICAL
        )


@dataclass
class Group:
    orientation: Orientation
    tolerance: float = 8
    baseline: float = 0
    results: list[tuple[float, OcrResult]] = field(default_factory=list)

    def is_same_group(self, result: OcrResult) -> GroupingResult:
        diff = self.orientation.calc_core(result) - self.baseline
        if abs(diff) <= self.tolerance:
            return GroupingResult.FIT
        return (
            GroupingResult.INSUFFICIENT
            if diff < 0
            else GroupingResult.SURPLUS
        )

    def add(self, result: OcrResult):
        core = self.orientation.calc_core(result)
        weight = self.orientation.calc_sort_weight(result)
        n = len(self.results)
        self.baseline = (self.baseline * n + core) / (n + 1)
        heapq.heappush(self.results, (weight, result))

    def try_add(self, result: OcrResult) -> bool:
        if self.results and self.is_same_group(result) != GroupingResult.FIT:
            return False
        self.add(result)
        return True

    def write(self, csv: typing.Any):
        csv.writerow(ele.text for _, ele in sorted(self.results))


class LinearGrouper:
    groups: list[Group]
    orientation: Orientation
    tolerance: float

    def __init__(
        self,
        orientation: Orientation,
        tolerance: float,
    ) -> None:
        self.groups = []
        self.orientation = orientation
        self.tolerance = tolerance

    def group(self, results: list[OcrResult]):
        group_idx = 0
        for result in results:
            while (
                len(self.groups) > group_idx
                and self.groups[group_idx].is_same_group(result)
                == GroupingResult.SURPLUS
            ):
                group_idx += 1

            if len(self.groups) <= group_idx:
                self.groups.append(
                    Group(self.orientation, tolerance=self.tolerance)
                )

            if self.groups[group_idx].try_add(result):
                continue
            else:
                assert (
                    self.groups[group_idx].is_same_group(result)
                    == GroupingResult.INSUFFICIENT
                )
                new_group = Group(self.orientation, tolerance=self.tolerance)
                new_group.add(result)
                self.groups.insert(group_idx, new_group)
                group_idx += 1


@contextmanager
def pdf_first_page(pdf: str | Path) -> typing.Iterator[Path]:
    pdf = Path(pdf)
    try:
        with TemporaryDirectory(prefix="report_ocr_") as path:
            filename = pdf.stem
            L.info("dumping pdf %s to temp file %s/%s", pdf, path, filename)
            convert_from_path(
                pdf,
                fmt="png",
                single_file=True,
                output_folder=path,
                paths_only=True,
                output_file=filename,
            )
            yield Path(path) / Path(filename + ".png")
    except Exception:  # pylint: disable=broad-exception-caught
        L.exception("cannot handle pdf %s", pdf)


def ocr_first_page(ocr: CnOcr, pdf: str | Path) -> list[OcrResult]:
    try:
        with pdf_first_page(pdf) as jpg:
            L.info("OCR scanning pdf %s", pdf)
            return [
                OcrResult(**result) for result in ocr.ocr(jpg, min_box_size=3)
            ]
    except Exception:  # pylint: disable=broad-exception-caught
        L.exception("cannot OCR pdf %s", pdf)
        raise


def ocr_all_pdfs(
    data_dir: str | Path = "data",
) -> typing.Iterator[tuple[Path, list[OcrResult]]]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir:
        raise ValueError("data_dir must be a dir")

    ocr_obj = CnOcr(DEFAULT_OCR_MODEL)
    for pdf in data_dir.glob("*.pdf"):
        yield pdf, ocr_first_page(ocr_obj, pdf)


def ensure_output_dir(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir:
        raise ValueError("output must be a dir")
    return output_dir


def ocr_to_txt(
    data_dir: str | Path = "data", output_dir: str | Path = "output"
):
    L.info("OCR all pdf in %s and save text to %s", data_dir, output_dir)
    output_dir = ensure_output_dir(output_dir)

    for pdf, results in ocr_all_pdfs(data_dir):
        output_name = output_dir / Path(pdf.with_suffix(".txt").name)
        with output_name.open("w", encoding="utf-8") as output:
            output.writelines(str(result) + "\n" for result in results)


@dataclass
class TableOcrResult:
    col_idx: int
    row_idx: int
    result: OcrResult

    @property
    def sort_key(self) -> tuple[int, int]:
        return (self.row_idx, self.col_idx)


def table_layout(
    results: list[OcrResult], row_tolerance: float, column_tolerance: float
) -> list[TableOcrResult]:
    result_to_position: dict[int, TableOcrResult] = dict()
    rows = [Group(Orientation.HORIZONTAL, row_tolerance)]
    row = rows[0]
    row_idx = 0
    for result in results:
        if row.try_add(result):
            result_to_position[id(result)] = TableOcrResult(
                -1, row_idx, result
            )
            continue
        row_idx += 1
        row = Group(Orientation.HORIZONTAL, row_tolerance)
        row.add(result)
        result_to_position[id(result)] = TableOcrResult(-1, row_idx, result)
        rows.append(row)

    column_grouper = LinearGrouper(Orientation.VERTICAL, column_tolerance)
    for row in rows:
        column_grouper.group([r for _, r in sorted(row.results)])

    for col_idx, column in enumerate(column_grouper.groups):
        try:
            for _, result in column.results:
                result_to_position[id(result)].col_idx = col_idx
        except TypeError:
            L.exception("bad %s", column)
            L.error("%s", [type(r) for r in column.results])
            L.error("%s", [w for w, _ in column.results])
            raise

    return sorted(result_to_position.values(), key=lambda r: r.sort_key)


def table_text_iter(
    results: list[TableOcrResult],
    score_threshold: float = 0,
    placeholder: str = "",
) -> typing.Iterable[typing.Iterable[str]]:
    row: list[TableOcrResult] = []
    prev: typing.Optional[TableOcrResult] = None

    for r in results:
        if prev is None:
            prev = r
            row.append(r)
            continue

        if prev.row_idx != r.row_idx:
            yield [
                (
                    cell.result.text
                    if cell.result.score >= score_threshold
                    else placeholder
                )
                for cell in row
            ]
            prev = TableOcrResult(-1, r.row_idx, OcrResult("", 0))
            row = []

        while prev.col_idx < r.col_idx - 1:
            L.debug("filling between %s and %s", prev.col_idx, r.col_idx)
            fill = TableOcrResult(
                prev.col_idx + 1, prev.row_idx, OcrResult(placeholder, 1)
            )
            row.append(fill)
            prev = fill

        row.append(r)
        prev = r

    if row:
        yield [
            (
                cell.result.text
                if cell.result.score >= score_threshold
                else placeholder
            )
            for cell in row
        ]


def ocr_to_csv(
    data_dir: str | Path = "data", output_dir: str | Path = "output"
):
    L.info("OCR all pdf in %s and save csv to %s", data_dir, output_dir)
    output_dir = ensure_output_dir(output_dir)

    for pdf, results in ocr_all_pdfs(data_dir):
        output_name = output_dir / Path(pdf.with_suffix(".csv").name)
        with output_name.open("w", encoding="utf-8") as output:
            csv = csv_writer(output)
            table_results = table_layout(
                results,
                row_tolerance=8,
                column_tolerance=78,
            )
            for row in table_text_iter(table_results, 0.325):
                csv.writerow(row)


if __name__ == "__main__":
    basicConfig(
        handlers=(
            logging.StreamHandler(),
            logging.FileHandler(
                "./report_ocr.log", mode="w", encoding="utf-8"
            ),
        ),
        level=logging.INFO,
        format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s",
        force=True,
    )
    # ocr_to_txt()
    ocr_to_csv()
