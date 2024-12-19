from datetime import datetime
from dateutil.relativedelta import relativedelta
import csv
import pandas as pd


def subtract_month(date_str: str) -> str:
    return (datetime.strptime(date_str, "%Y-%m-%d") - relativedelta(months=1)).strftime("%Y-%m-%d")


def closest_date_before(date_str: str, dates: list[str]) -> str:
    compare_date_obj: datetime = datetime.strptime(date_str, "%Y-%m-%d")
    
    closest_date: str = None
    distance_days: float = float("inf")
    for date in dates:
        date_obj: datetime = datetime.strptime(date, "%Y-%m-%d")
        if (date_obj < compare_date_obj) and (compare_date_obj - date_obj).days < distance_days:
            closest_date = date_obj.strftime("%Y-%m-%d")
            distance_days = (compare_date_obj - date_obj).days 
    return closest_date


def election_result(party: str, election_date: str, elections_df: pd.DataFrame) -> float:
    election_year: str = election_date[:election_date.find("-")]
    
    try:
        return elections_df[elections_df["political_party"] == party][election_year].values[0]
    except IndexError:
        return 0

def main() -> None:
    election_dates: list[str] = ("2023-09-30", "2020-02-29", "2016-03-05", "2012-03-10")
    polls: pd.DataFrame = pd.read_csv("data/focus_polls.csv", na_values="NA")
    polls_dates: list[str] = list(polls.columns)[1:]
    
    for election_date in election_dates:
        print("election", election_date)
        poll_before = closest_date_before(election_date, polls_dates)
        print("first poll before", poll_before)
        for _ in range(12):
            if poll_before not in polls_dates:
                print(poll_before)
            poll_before = subtract_month(poll_before)
        print()

def main2() -> None:
    election_dates: list[str] = ("2023-09-30", "2020-02-29", "2016-03-05", "2012-03-10")
    polls: pd.DataFrame = pd.read_csv("data/focus_polls.csv", na_values="NA")
    polls_dates: list[str] = list(polls.columns)[1:]
    
    elections_data: pd.DataFrame = pd.read_csv("data/election_results.csv", na_values="NA").fillna(0)
        
    data = []
    for election_date in election_dates:
        data_election = {"political_party": polls["political_party"],
                         "election_date": [election_date] * len(polls),
                         "election_result": [election_result(party, election_date, elections_data)
                                             for party in polls["political_party"]]
                         }
        
        
        poll_before = closest_date_before(election_date, polls_dates)
        for i in range(12):
            print(poll_before)
            data_election[i+1] = polls[poll_before]
            poll_before = closest_date_before(poll_before, polls_dates)
        
        data.append(pd.DataFrame(data_election))
        print()
    
    split_polls = pd.concat(data, ignore_index=True)
    split_polls.to_csv("data/polls_by_election.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC, na_rep="NA")
    
    
if __name__ == "__main__":
    main()
