import json
from datetime import datetime

import pandas as pd


def add_dates(data: pd.DataFrame):
    election_dates = [
        datetime(year=2016, month=3, day=5),
        datetime(year=2020, month=2, day=29),
        datetime(year=2023, month=9, day=30),
        datetime(year=2012, month=3, day=10),
        datetime(year=2010, month=6, day=12),
    ]
    slovak_months = {
        "januar": 1,
        "februar": 2,
        "marec": 3,
        "april": 4,
        "maj": 5,
        "jun": 6,
        "jul": 7,
        "august": 8,
        "september": 9,
        "oktober": 10,
        "november": 11,
        "december": 12,
    }
    datetimes = [
        datetime(year=int(year), month=slovak_months[month], day=1)
        for year, month in zip(data["year"], data["month"])
    ]

    data["date"] = datetimes
    del data["year"]
    del data["month"]

    def diff_month(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    dates_to_elections = []
    for i in data["date"]:
        diffs = [j for j in map(lambda x: diff_month(x, i), election_dates) if j >= 0]

        if diffs:
            dates_to_elections.append(min(diffs))
        else:
            dates_to_elections.append(None)

    data["to_election"] = dates_to_elections


def map_names(data: pd.DataFrame, mapper_filename: str):
    mapper = json.load(open(mapper_filename))
    data["political_party"] = data["political_party"].map(mapper)


def main(CSV_NAME: str, JSON_MAPPER_NAME: str):
    data = pd.read_csv(CSV_NAME)
    add_dates(data)
    map_names(data, JSON_MAPPER_NAME)
    data.drop("index", inplace=True, axis="columns")
    data.sort_values(by="to_election", inplace=True)
    data.to_csv("src/focus_scraping/pdf_data_extraction/focus_to_election.csv", index=False)


if __name__ == "__main__":
    main("src/focus_scraping/pdf_data_extraction/focus.csv", "src/focus_scraping/pdf_data_extraction/mapper.json")
