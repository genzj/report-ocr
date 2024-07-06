import logging
from abc import ABC, abstractmethod
from csv import reader, DictWriter
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


class InfoLineExtractor(ABC):
    @abstractmethod
    def can_extract(self, csv_row: list[str]) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def extract(self, csv_row: list[str]) -> dict[str, str]:
        raise NotImplementedError


class RegexExtractor(InfoLineExtractor):
    PATTERN = compile_re(r"")
    QUICK_TAG = "xxxxxxxxxxx"

    @classmethod
    def pattern(cls) -> Pattern:
        return cls.PATTERN

    @classmethod
    def quick_tag(cls) -> str:
        return cls.QUICK_TAG

    def can_extract(self, csv_row: list[str]) -> bool:
        return self.quick_tag() in " ".join(csv_row)

    def extract(self, csv_row: list[str]) -> dict[str, str]:
        L.debug("%s search %s in %r", self, self.pattern(), " ".join(csv_row))
        m = self.pattern().search(" ".join(csv_row))
        if m:
            return {k: v.strip() for k, v in m.groupdict().items()}
        # TODO should be an internal error here
        return {}


class NameGenderAgeExtractor(RegexExtractor):
    PATTERN = compile_re(
        r"姓名[：:](?P<name>.+?)\s+.*?性别[：:](?P<gender>.+?)\s+年龄[：:](?P<age>.+?)岁"
    )

    QUICK_TAG = "姓名"


class RegistrationIDExtractor(RegexExtractor):
    PATTERN = compile_re(r"登记号[：:]\s*(?P<registration_id>\d+)")

    QUICK_TAG = "登记号"


INFO_EXTRACTORS = (NameGenderAgeExtractor(), RegistrationIDExtractor())


def extract(file: Path | str) -> Report:
    file = Path(file)
    report = Report(filename=file.name)
    with file.open("r", encoding="utf-8") as inf:
        csv = reader(inf)
        for csv_row in csv:
            for extractor in INFO_EXTRACTORS:
                if extractor.can_extract(csv_row):
                    L.info(
                        "row %s can be extracted by extractor %s",
                        csv_row,
                        extractor,
                    )
                    info = extractor.extract(csv_row)
                    L.info("extracted %s", info)
                    for k, v in info.items():
                        setattr(report, k, v)
                    continue
            if (
                csv_row
                and csv_row[0].strip()
                and csv_row[0].strip()[0] in ascii_uppercase
            ):
                item_code = csv_row[0].strip()
                item_name = None
                for col in csv_row[1:]:
                    col = col.strip()
                    if col and item_name:
                        item_value = col.replace(" ", "")
                        break
                    elif col:
                        item_name = col
                else:
                    item_value = ""

                report.results[item_code] = item_value

    return report


def extract_all(csv_dir: Path | str) -> list[Report]:
    csv_dir = Path(csv_dir)
    reports = [extract(file) for file in csv_dir.glob("*.csv")]
    return reports


def merge_to_csv(reports: list[Report], output: Path | str):
    item_codes = set()
    for report in reports:
        item_codes.update(report.results.keys())
    header = [
        "文件名",
        "登记号",
        "姓名",
        "性别",
        "年龄",
    ] + list(item_codes)
    with Path(output).open("w", encoding="utf-8") as outf:
        csv_writer = DictWriter(outf, fieldnames=header)
        csv_writer.writeheader()
        csv_writer.writerows(
            map(
                lambda report: {
                    "文件名": report.filename,
                    "登记号": report.registration_id,
                    "姓名": report.name,
                    "性别": report.gender,
                    "年龄": report.age,
                    **report.results,
                },
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
