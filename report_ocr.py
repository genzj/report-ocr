import heapq
import logging
import typing
from csv import writer as csv_writer
from contextlib import contextmanager
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


class Orientation(Enum):
    HORIZONTAL = 1
    VERTICAL = 2

    def to_baseline_index(self) -> int:
        """baseline of a vertical group is in the X dimensison, vice versa"""
        return 0 if self == Orientation.VERTICAL else 1

    def to_sort_index(self) -> int:
        """elements in a vertical group are sorted by their Y position, vice versa"""
        return 1 if self == Orientation.VERTICAL else 0

    def calc_core(self, result: OcrResult) -> float:
        position = cast(ndarray, result.position)
        return position[:, self.to_baseline_index()].mean()

    def calc_sort_weight(self, result: OcrResult) -> float:
        position = cast(ndarray, result.position)
        return position[:, self.to_sort_index()].mean()


@dataclass
class Group:
    orientation: Orientation
    tolerance: float = 5
    baseline: float = 0
    results: list[tuple[float, OcrResult]] = field(default_factory=list)

    def is_same_group(self, result: OcrResult) -> bool:
        return (
            abs(self.baseline - self.orientation.calc_core(result))
            <= self.tolerance
        )

    def add(self, result: OcrResult):
        core = self.orientation.calc_core(result)
        weight = self.orientation.calc_sort_weight(result)
        n = len(self.results)
        self.baseline = (self.baseline * n + core) / (n + 1)
        heapq.heappush(self.results, (weight, result))

    def try_add(self, result: OcrResult) -> bool:
        if not self.is_same_group(result):
            return False
        self.add(result)
        return True


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
            return [OcrResult(**result) for result in ocr.ocr(jpg)]
    except Exception:  # pylint: disable=broad-exception-caught
        L.exception("cannot OCR pdf %s", pdf)
        raise


def ocr_all_pdfs( data_dir: str | Path = "data") -> typing.Iterator[tuple[Path, list[OcrResult]]]:
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


def ocr_to_csv(
    data_dir: str | Path = "data", output_dir: str | Path = "output"
):
    L.info("OCR all pdf in %s and save csv to %s", data_dir, output_dir)
    output_dir = ensure_output_dir(output_dir)

    for pdf, results in ocr_all_pdfs(data_dir):
        output_name = output_dir / Path(pdf.with_suffix(".csv").name)
        with output_name.open("w", encoding="utf-8") as output:
            csv = csv_writer(output)
            group = Group(Orientation.HORIZONTAL)
            for result in results:
                if group.try_add(result):
                    continue
                csv.writerow(ele.text for _, ele in sorted(group.results))
                group = Group(Orientation.HORIZONTAL)
                group.add(result)
            # don't forget to write the last group
            csv.writerow(ele.text for _, ele in sorted(group.results))


if __name__ == "__main__":
    basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    ocr_to_txt()
    ocr_to_csv()
