import logging
from abc import ABC, abstractmethod
from csv import DictWriter, reader
from dataclasses import dataclass, field
from logging import basicConfig, getLogger
from pathlib import Path
from re import Pattern
from re import compile as compile_re
from string import ascii_uppercase

L = getLogger("extract_report")


@dataclass
class Report:
    name: str = ""
    gender: str = ""
    age: str = ""
    filename: str = ""
    registration_id: str = ""
    results: dict[str, str] = field(default_factory=dict)

    def validate(self, warn=True) -> bool:
        if any(
            not getattr(self, attr)
            for attr in self.__dataclass_fields__.keys()  # pylint: disable=no-member
        ):
            if warn:
                L.warning("some field of report %s are empty", self)
            return False
        return True

    @staticmethod
    def header() -> tuple[str, ...]:
        return (
            "文件名",
            "登记号",
            "姓名",
            "性别",
            "年龄",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "文件名": self.filename,
            "登记号": self.registration_id,
            "姓名": self.name,
            "性别": self.gender,
            "年龄": self.age,
            **self.results,
        }


class LineExtractor(ABC):
    @abstractmethod
    def can_extract(self, csv_row: list[str]) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def extract(self, csv_row: list[str]) -> dict[str, str]:
        raise NotImplementedError


class RegexExtractor(LineExtractor):
    PATTERN = compile_re(r"")
    QUICK_TAG = "xxxxxxxxxxx"

    @classmethod
    def pattern(cls) -> Pattern:
        if cls.PATTERN is RegexExtractor.PATTERN:
            raise NotImplementedError(
                "subclass should define PATTERN as a class property"
            )
        return cls.PATTERN

    @classmethod
    def quick_tag(cls) -> str:
        if cls.QUICK_TAG is RegexExtractor.QUICK_TAG:
            raise NotImplementedError(
                "subclass should define QUICK_TAG as a class property"
            )
        return cls.QUICK_TAG

    def can_extract(self, csv_row: list[str]) -> bool:
        return self.quick_tag() in " ".join(csv_row)

    def extract(self, csv_row: list[str]) -> dict[str, str]:
        L.debug("%s search %s in %r", self, self.pattern(), " ".join(csv_row))
        m = self.pattern().search(" ".join(csv_row))
        if m:
            return {k: v.strip() for k, v in m.groupdict().items()}
        L.warning(
            "%r didn't find any info from %r, its can_extract may be slack.",
            self,
            csv_row,
        )
        return {}


class NameGenderAgeExtractor(RegexExtractor):
    PATTERN = compile_re(
        r"姓名[：:](?P<name>.+?)\s+.*?性别[：:](?P<gender>.+?)\s+年龄[：:](?P<age>.+?)岁"
    )

    QUICK_TAG = "姓名"


class RegistrationIDExtractor(RegexExtractor):
    PATTERN = compile_re(r"登记号[：:]\s*(?P<registration_id>\d+)")

    QUICK_TAG = "登记号"


class ResultItemExtractor(LineExtractor):
    def can_extract(self, csv_row: list[str]) -> bool:
        return bool(
            csv_row
            and csv_row[0].strip()
            and csv_row[0].strip()[0] in ascii_uppercase
        )

    def extract(self, csv_row: list[str]) -> dict[str, str]:
        code_name_value = []
        for col in csv_row:
            col = col.strip()
            if col:
                code_name_value.append(col)
            if len(code_name_value) == 3:
                break
        # item name isn't necesarry
        item_code, _, item_value = code_name_value

        # item value may contain space because of OCR inaccuracy
        return {item_code: item_value.replace(" ", "")}


INFO_EXTRACTORS = (NameGenderAgeExtractor(), RegistrationIDExtractor())
RESULT_EXTRACTOR = ResultItemExtractor()


def extract_info_into(report: Report, csv_row: list[str]) -> dict[str, str]:
    for extractor in INFO_EXTRACTORS:
        if not extractor.can_extract(csv_row):
            continue

        L.debug(
            "row %s can be extracted by extractor %s",
            csv_row,
            extractor,
        )
        extracted = extractor.extract(csv_row)
        L.debug("extracted %s", extracted)
        break
    else:
        extracted = dict()

    for k, v in extracted.items():
        setattr(report, k, v)
    return extracted


def extract(file: Path | str) -> Report:
    file = Path(file)
    L.info("processing file %s", file)
    report = Report(filename=file.name)
    with file.open("r", encoding="utf-8") as inf:
        csv = reader(inf)
        for csv_row in csv:
            if extract_info_into(report, csv_row):
                # info row and result row are mutually exlusive so just move to
                # next row and skip the result extractor when info can be extracted
                continue
            if RESULT_EXTRACTOR.can_extract(csv_row):
                report.results.update(RESULT_EXTRACTOR.extract(csv_row))
    return report


def extract_all(csv_dir: Path | str) -> list[Report]:
    csv_dir = Path(csv_dir)
    reports = [
        extract(file)
        for file in csv_dir.glob("*.csv")
        if file.is_fifo and "merge" not in file.name
    ]
    return reports


def merge_to_csv(reports: list[Report], output: Path | str):
    # superset of all result codes
    item_codes: set[str] = set()
    for report in reports:
        item_codes.update(report.results.keys())

    header = Report.header() + tuple(item_codes)

    with Path(output).open("w", encoding="utf-8") as outf:
        csv_writer = DictWriter(outf, fieldnames=header)
        csv_writer.writeheader()
        csv_writer.writerows(
            map(
                Report.to_dict,
                reports,
            )
        )


if __name__ == "__main__":
    basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    merge_to_csv(extract_all("./output"), "./output/merge.csv")
