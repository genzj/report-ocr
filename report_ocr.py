import logging
import typing
from contextlib import contextmanager
from logging import basicConfig, getLogger
from pathlib import Path
from tempfile import TemporaryDirectory

from cnocr.cn_ocr import CnOcr, OcrResult
from pdf2image import convert_from_path

DEFAULT_OCR_MODEL = "densenet_lite_136-gru"


L = getLogger("report_ocr")


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


def ocr_to_txt(
    data_dir: str | Path = "data", output_dir: str | Path = "output"
):
    L.info("OCR all pdf in %s and save text to %s", data_dir, output_dir)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir:
        raise ValueError("data_dir must be a dir")
    if not output_dir.is_dir:
        raise ValueError("output must be a dir")

    ocr_obj = CnOcr(DEFAULT_OCR_MODEL)
    for pdf in data_dir.glob("*.pdf"):
        output_name = output_dir / Path(pdf.with_suffix(".txt").name)
        with output_name.open("w", encoding="utf-8") as output:
            results = ocr_first_page(ocr_obj, pdf)
            output.writelines(str(result) + "\n" for result in results)


if __name__ == "__main__":
    basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    ocr_to_txt()
