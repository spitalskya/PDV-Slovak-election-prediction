from datetime import datetime
from dateutil.relativedelta import relativedelta
import csv
import pandas as pd
import numpy as np


"""
Script with functions used to clean up, achieve consistency and make sense of the data we have.
Were run in no particular order.
"""


def subtract_month(date_str: str) -> str:
    return (datetime.strptime(date_str, "%Y-%m-%d") - relativedelta(months=1)).strftime("%Y-%m-%d")


def closest_date_before(date_str: str, dates: list[str]) -> str:
    # gets closest date before `date_str` in `dates`
    compare_date_obj: datetime = datetime.strptime(date_str, "%Y-%m-%d")
    
    closest_date: str = None
    distance_days: float = float("inf")
    for date in dates:
        date_obj: datetime = datetime.strptime(date, "%Y-%m-%d")
        if (date_obj < compare_date_obj) and (compare_date_obj - date_obj).days < distance_days:
            closest_date = date_obj.strftime("%Y-%m-%d")
            distance_days = (compare_date_obj - date_obj).days 
    return closest_date


def join_all_polls_data() -> None:
    # joins polls data stored in various formats (scrapping, by hand...) into one clean dataframe
    def try_convert_float(value):
        try:
            return float(value)
        except ValueError:
            return value
        
    polls1: pd.DataFrame = pd.read_csv("data/raw/polls/polls_by_hand_focus_failed.csv", na_values="NA")
    polls1 = polls1.loc[:, polls1.iloc[0] != "interpolacia"]
    
    polls2: pd.DataFrame = pd.read_csv("data/raw/polls/polls_by_hand_older.csv", na_values="NA")
    polls3: pd.DataFrame = pd.read_csv("data/raw/polls/polls_by_hand_newer.csv", na_values="NA")
        
    polls_by_hand = pd.merge(polls1, polls2, on="political_party", how="outer")
    polls_by_hand = pd.merge(polls_by_hand, polls3, on="political_party", how="outer")
    polls_by_hand.iloc[:, 1:] = polls_by_hand.iloc[:, 1:].fillna(0)
    
    polls_by_hand = polls_by_hand.map(try_convert_float)
    
    polls_by_hand.to_csv("data/raw/polls/polls_by_hand.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC, na_rep="NA")
    polls_by_hand = polls_by_hand.iloc[1:, :]       # drop agencies
    
    focus_polls: pd.DataFrame = pd.read_csv("data/raw/polls/focus_polls.csv", na_values="NA")
    
    all_polls = pd.merge(focus_polls, polls_by_hand, on="political_party", how="outer")
    all_polls.fillna(0)
    
    missing_polls: list[str] = get_all_missing_polls(all_polls)
    # ['2024-01-01', '2023-10-01', '2020-09-01', '2020-08-01', '2020-03-01', '2019-07-01', '2018-07-01', '2017-12-01', '2012-03-01', '2010-11-01']
    for missing in missing_polls:
        all_polls[missing] = np.nan
    
    sorted_columns: list[str] = sorted(all_polls.columns[1:], key = lambda c: datetime.strptime(c, "%Y-%m-%d"))
    all_polls = all_polls.reindex(columns=["political_party"] + sorted_columns)   

    for col in all_polls.columns[1:]:
        all_polls[col] = all_polls[col].astype(float)
    all_polls.iloc[:, 1:] = all_polls.iloc[:, 1:].interpolate(method="linear", axis=1).round(1)
    
    convert_time = lambda t: datetime.strptime(t, "%Y-%m-%d").strftime("%Y-%m") if t != "political_party" else t
    all_polls = all_polls.rename(columns=convert_time)
    
    all_polls.to_csv("data/polls_data.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC, na_rep="NA")


def election_result(party: str, election_date: str, elections_df: pd.DataFrame) -> float:
    # gets election result of passed party in given election
    election_year: str = election_date[:election_date.find("-")]
    
    try:
        return elections_df[elections_df["political_party"] == party][election_year].values[0]
    except IndexError:
        return 0


def get_all_missing_polls(df) -> None:
    # gets strings of all months from which we did not found any election polls
    last_poll = "2024-11-01"
    first_poll = "2010-01-01"
    
    poll = last_poll
    missing = []
    while True:
        if poll not in df.columns: 
            missing.append(poll)
        
        poll = subtract_month(poll)
        if datetime.strptime(poll, "%Y-%m-%d") < datetime.strptime(first_poll, "%Y-%m-%d"):
            break
    return missing


def remove_failed_scraped_data() -> None:
    # removes data that were badly scraped
    failed_focus_scrapes: list[str] = [
        "2010-08-01", "2010-11-01", "2011-10-01", "2012-03-01", "2013-12-01", 
        "2014-04-01", "2015-03-01", "2016-01-01", "2016-05-01", "2017-08-01", 
        "2017-12-01", "2018-08-01", "2019-05-01", "2020-08-01", "2020-03-01", 
        "2021-12-01", "2022-05-01", "2023-12-01", "2024-07-01", "2024-11-01"
        ]
    
    focus_polls: pd.DataFrame = pd.read_csv("data/raw/polls/focus_polls.csv", na_values="NA")
    focus_polls = focus_polls.drop(columns=failed_focus_scrapes)
    focus_polls.to_csv("data/raw/polls/focus_polls.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC, na_rep="NA")


def add_missing_polls() -> None:
    # add polls collected by hand (replacement for badly scraped polls)
    polls_updated: pd.DataFrame = pd.read_csv("data/raw/polls/focus_polls_updated.csv", na_values="NA")
    def rename_cols(col):
        if col == "political_party": return col
        month, day, year = col.split("/")
        return f"{year}-{int(month):02d}-{int(day):02d}"
    
    polls_updated = polls_updated.rename(columns=rename_cols)
    
    """# two polls in september 2023, keep poll closer to election 2023
    polls_updated = polls_updated.drop(columns="2023.1-09-01")"""

    # two polls in september 2023, keep poll one month after august 2023 poll
    polls_updated = polls_updated.drop(columns="2023-09-01").rename(columns={"2023.1-09-01": "2023-09-01"})   
    
    polls_incomplete: pd.DataFrame = pd.read_csv("data/raw/polls/polls_incomplete.csv", na_values="NA")
    
    missing_columns = sorted(set(polls_updated.columns) - set(polls_incomplete.columns))
    for col in missing_columns:
        polls_incomplete[col] = polls_updated[col]
        
    sorted_columns: list[str] = sorted(polls_incomplete.columns[1:], key = lambda c: datetime.strptime(c, "%Y-%m-%d"))
    polls_incomplete = polls_incomplete.reindex(columns=["political_party"] + sorted_columns)
    polls_incomplete.to_csv("data/raw/polls/focus_polls.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC, na_rep="NA")


def create_polls_agencies_table() -> None:
    # create table that stores agencies for each used poll
    focus_polls: pd.DataFrame = pd.read_csv("data/raw/polls/focus_polls.csv")
    data_focus = {
        "poll_month": list(focus_polls.columns[1:]),
        "agency": ["FOCUS Centrum pre sociálnu a marketingovú analýzu, s r.o. v skratke FOCUS, s r.o."]  * len(list(focus_polls.columns[1:]))
        }
    
    conversion = {
        "MEDIAN_SK": "MEDIAN SK, s.r.o.",
        "median_sk": "MEDIAN SK, s.r.o.",
        "POLIS": "Polis Slovakia, s.r.o.",
        "MVK": "AGENTÚRA MVK s.r.o.",
        "AKO": "AKO, s.r.o.",
        "ako": "AKO, s.r.o.",
        "FOCUS": "FOCUS Centrum pre sociálnu a marketingovú analýzu, s r.o. v skratke FOCUS, s r.o.",
        "focus": "FOCUS Centrum pre sociálnu a marketingovú analýzu, s r.o. v skratke FOCUS, s r.o.",
        "nms": "NMS Market Research Slovakia s.r.o.",
    }
    
    polls_by_hand: pd.DataFrame = pd.read_csv("data/raw/polls/polls_by_hand.csv")
    data_by_hand = {
        "poll_month": list(polls_by_hand.columns[1:]),
        "agency": [conversion[agency] for agency in polls_by_hand.iloc[0, 1:]]
    }
    
    interpolated_months = ['2024-01-01', '2023-10-01', '2020-09-01', '2020-08-01', '2020-03-01', '2019-07-01', '2018-07-01', '2017-12-01', '2012-03-01', '2010-11-01']
    data_interpolated = {
        "poll_month": interpolated_months,
        "agency": ["linearly interpolated from neighboring months"] * len(interpolated_months)
    }

    data = {
        "poll_month": data_focus["poll_month"] + data_by_hand["poll_month"] + data_interpolated["poll_month"],
        "agency": data_focus["agency"] + data_by_hand["agency"] + data_interpolated["agency"]
    }
    data["poll_month"] = list(map(lambda t: datetime.strptime(t, "%Y-%m-%d").strftime("%Y-%m"), data["poll_month"]))
    
    pd.DataFrame(data).sort_values(by="poll_month").to_csv("data/polls_agencies.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC, na_rep="NA")
    

def split_polls_by_election() -> None:
    # create dataset in which every political party has their result in a particular election 
    # and their polls 12 months before election, 
    # along with whether they were in coalition/opposition before the election
    election_dates: list[str] = ("2023-09-30", "2020-02-29", "2016-03-05", "2012-03-10")
    polls: pd.DataFrame = pd.read_csv("data/raw/polls/focus_polls.csv", na_values="NA")
    polls_dates: list[str] = list(polls.columns)[1:]
    
    elections_data: pd.DataFrame = pd.read_csv("data/election_results.csv", na_values="NA").fillna(0)
    coalitions_data: pd.DataFrame = pd.read_csv("data/elected_parties.csv", na_values="NA")
    
    def was_in_coalition(party, until) -> int:
        try:
            return coalitions_data[(coalitions_data["until"] == until) & (coalitions_data["political_party"] == party)]["coalition"].values[0]
        except IndexError:
            return 0
    def was_in_opposition(party, until) -> int:
        try:
            return int(
                coalitions_data[(coalitions_data["until"] == until) & (coalitions_data["political_party"] == party)]["coalition"].values[0] == 0
                )
        except IndexError:
            return 0
        
    data = []
    for election_date in election_dates:
        data_election = {"political_party": polls["political_party"],
                         "election_date": [election_date] * len(polls),
                         "election_result": [election_result(party, election_date, elections_data)
                                             for party in polls["political_party"]]
                         }
        data_election["elected_to_parliament"] = [
            int(res >= 7) 
            if election_date == "2020-02-29" and data_election["political_party"][i] == "progresivne_slovensko"
            else int(res >= 5) 
            for i, res in enumerate(data_election["election_result"])
            ]
        data_election["in_coalition_before"] = [was_in_coalition(party, election_date) for party in polls["political_party"]]
        data_election["in_opposition_before"] = [was_in_opposition(party, election_date) for party in polls["political_party"]]
                
        
        poll_before = closest_date_before(election_date, polls_dates)
        for i in range(12):
            data_election[i+1] = polls[poll_before]
            poll_before = closest_date_before(poll_before, polls_dates)
        
        data.append(pd.DataFrame(data_election))
    
    split_polls = pd.concat(data, ignore_index=True)
    split_polls.to_csv("data/polls_by_election.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC, na_rep="NA")


def make_train_test_datasets() -> None:
    # make train and test datasets to be used later
    polls = pd.read_csv("data/polls_by_election.csv", na_values="NA")
    
    threshold = 1.5
    first_poll_column_index = list(polls.columns).index("1")
    polls = polls[(polls.iloc[:, first_poll_column_index:] >= threshold).any(axis=1)]
    
    train_frac = 0.8
    n_train = round(len(polls) * train_frac)
    polls_train = polls.sample(n=n_train)
    polls_test = polls.drop(polls_train.index)
    
    polls_train.to_csv("data/polls_by_election_train.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC, na_rep="NA")
    polls_test.to_csv("data/polls_by_election_test.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC, na_rep="NA")
    
    
if __name__ == "__main__":
    pass
