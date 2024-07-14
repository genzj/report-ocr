import logging
import typing
from abc import ABC, abstractmethod
from csv import DictWriter, reader
from dataclasses import dataclass, field
from enum import Enum
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
    examination_name: str = ""
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
            "医嘱名称",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "文件名": self.filename,
            "登记号": self.registration_id,
            "姓名": self.name,
            "性别": self.gender,
            "年龄": self.age,
            "医嘱名称": self.examination_name,
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
        r"姓名[：:](?P<name>.+?)\s+.*?性\s*别[：:](?P<gender>.+?)\s+年龄[：:](?P<age>.+?)岁"
    )

    QUICK_TAG = "姓名"


class RegistrationIDExtractor(RegexExtractor):
    PATTERN = compile_re(r"登记号[：:]\s*(?P<registration_id>\d+)")

    QUICK_TAG = "登记号"


class ExaminationNameExtractor(RegexExtractor):
    PATTERN = compile_re(r"医嘱名称[：:]\s*(?P<examination_name>\S+)")

    QUICK_TAG = "医嘱名称"


class ResultItemExtractor(LineExtractor):
    def can_extract(self, csv_row: list[str]) -> bool:
        return len(list(filter(lambda col: bool(col.strip()), csv_row))) >= 2

    def extract(self, csv_row: list[str]) -> dict[str, str]:
        code_name_value = []
        for col in csv_row:
            col = col.strip()
            if col:
                code_name_value.append(col)
            if len(code_name_value) == 3:
                break
        if (
            len(code_name_value) == 3
            and code_name_value[0][0].upper() in ascii_uppercase
        ):
            # 3 items and the first column begins with letter, indicating a item
            # code; item name isn't necesarry in this case
            item_code, _, item_value = code_name_value
        else:
            # use the first two columns, although the first column is
            # actually item name (usually in Chinese)
            item_code, item_value = code_name_value[:2]

        # item value may contain space because of OCR inaccuracy
        return {item_code: item_value.replace(" ", "")}


INFO_EXTRACTORS = (
    NameGenderAgeExtractor(),
    RegistrationIDExtractor(),
    ExaminationNameExtractor(),
)
RESULT_HEADER_TAGS = ("项目", "结果")
RESULT_ENDING_TAGS = ("医嘱时间", "采样时间")
RESULT_EXTRACTOR = ResultItemExtractor()


def has_all_tags(csv_row: list[str], tags: typing.Iterable[str]) -> bool:
    row = " ".join(csv_row)
    return all(
        map(
            lambda s: s in row,
            tags,
        )
    )


class ReportSection(Enum):
    INFO = 1
    REPORT = 2
    FOOTER = 3

    def calculate_section(
        self, csv_row: list[str]
    ) -> tuple[bool, "ReportSection"]:
        if has_all_tags(csv_row, RESULT_HEADER_TAGS):
            if self is not ReportSection.INFO:
                L.warning(
                    "found another header after section %s: %s",
                    self.name,
                    csv_row,
                )
            return True, ReportSection.REPORT
        if has_all_tags(csv_row, RESULT_ENDING_TAGS):
            if self is ReportSection.INFO:
                L.warning(
                    "found footer after section %s: %s", self.name, csv_row
                )
            if self is ReportSection.FOOTER:
                L.warning(
                    "found another footer after section %s: %s",
                    self.name,
                    csv_row,
                )
            return True, ReportSection.FOOTER
        return False, self


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
        section: ReportSection = ReportSection.INFO

        for csv_row in csv:
            new_section, section = section.calculate_section(csv_row)
            match [new_section, section]:
                case [_, ReportSection.INFO]:
                    extract_info_into(report, csv_row)
                case [True, ReportSection.REPORT]:
                    L.debug("found report header")
                case [False, ReportSection.REPORT]:
                    if RESULT_EXTRACTOR.can_extract(csv_row):
                        report.results.update(
                            RESULT_EXTRACTOR.extract(csv_row)
                        )
                case [_, ReportSection.FOOTER]:
                    break
    return report


def extract_all(csv_dir: Path | str) -> typing.Iterable[Report]:
    csv_dir = Path(csv_dir)
    for file in sorted(csv_dir.glob("*.csv")):
        if not file.is_file():
            L.info("skip non-file path %s", file)
            continue
        if "merge" in file.name:
            L.info("skip merged output %s", file)
            continue
        try:
            report = extract(file)
        except Exception as ex:
            L.warning("cannot extract file %s", file, exc_info=ex)
        else:
            yield report


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
        handlers=(
            logging.StreamHandler(),
            logging.FileHandler(
                "./extract_report.log", mode="w", encoding="utf-8"
            ),
        ),
        level=logging.INFO,
        format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s",
        force=True,
    )
    reports = list(extract_all("./output"))
    merge_to_csv(reports, "./output/merge.csv")
