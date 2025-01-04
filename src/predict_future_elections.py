import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import Holt
from src.utils import join_dfs_with_diff


def prepare_data(relevant_parties: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepares data for election prediction. Polls of relevant parties and dataset
    used to predict difference between election result and value in poll before election
    (described in report).

    Args:
        relevant_parties (list[str]): parties that are relevant for future elections

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 
            first dataframe: polls data for each party
            second dataframe: "information" about the party and its interactions with general data
    """
    # get the raw polls data, filter only rows for relevant parties
    relevant_parties = sorted(relevant_parties)
    
    polls_data: pd.DataFrame = pd.read_csv("data/polls_data.csv", na_values="NA", index_col="political_party")
    X_polls: pd.DataFrame = polls_data.loc[polls_data.index.isin(relevant_parties)]
    
     # since there is no general data for 2024, we assume its the same as 2023 and use this "workaround"
    X_party_data: pd.DataFrame = pd.DataFrame(
        data={"election_date": pd.to_datetime("2023-09-30")},       
        index=relevant_parties
    )
    X_party_data["political_party"] = relevant_parties
    
    # add data whether the party is currently in coalition/opposition
    coalition_data: pd.DataFrame = pd.read_csv("data/elected_parties.csv", na_values="NA")
    X_party_data = pd.merge(
        X_party_data,
        coalition_data[coalition_data["until"] == "now"].drop(columns="until"),
        on="political_party",
        how="left"
    ).rename(columns={"coalition": "in_coalition_before"}).set_index("political_party")
    X_party_data["in_opposition_before"] = 1 - X_party_data["in_coalition_before"]
    X_party_data = X_party_data.fillna(0)
    
    # add data from political compass
    X_party_data = pd.merge(
        X_party_data,
        pd.read_csv("data/political_compass_data.csv", na_values="NA"),
        on="political_party"
    )
    
    # add the general data
    X_party_data = join_dfs_with_diff(X_party_data, pd.read_csv("data/general_data.csv", na_values="NA"))
    
    # remove the redundant variables for regression
    redundant_variables: list[str] = ["election_date", "year"]
    X_party_data = X_party_data.drop(columns=redundant_variables)
    X_party_data = X_party_data.set_index("political_party")
    
    # add variable interactions
    interactions: PolynomialFeatures = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X_party_data = pd.DataFrame(
        data=interactions.fit_transform(X_party_data),
        columns=interactions.get_feature_names_out(X_party_data.columns)
        )
    
    # remove column of zeros
    X_party_data = X_party_data.drop(columns="in_coalition_before in_opposition_before")
    
    # return the prepared data
    X_party_data.index = relevant_parties
    return X_polls.sort_index(axis=0), X_party_data.sort_index(axis=0)


def predict_election_poll_difference(X_party_data: pd.DataFrame) -> np.ndarray:
    """Predicts difference of election result and last poll before election.

    Args:
        X_party_data (pd.DataFrame): data about each party

    Returns:
        np.ndarray: prediction of the difference
    """
    # coefficients for linear regression trained in src/prediction_models.py
    election_poll_difference_model: LinearRegression = LinearRegression()
    election_poll_difference_model.intercept_ = -1.422803951315247
    election_poll_difference_model.coef_ = np.array([1.16370568e+00, 2.81747988e+00, 6.94022258e-06])
    
    # column subset used for linear regression, found in src/prediction_models.py
    column_subset_for_linear_regression: list[str] = [
        "in_coalition_before Liberalism-Conservatism", 
        "in_opposition_before Pension expenditures (per capita)", 
        "Unemployment rate (%) GDP (per capita)"
        ]
    election_poll_difference_model.feature_names_in_ = np.array(column_subset_for_linear_regression)
        
    # predict the difference
    X_party_data = X_party_data.loc[:, column_subset_for_linear_regression]
    return election_poll_difference_model.predict(X_party_data)
    

def predict_polls_month_before_election(X_polls: pd.DataFrame, elections_in_n_months: int = 1) -> np.ndarray:
    """Forecast the development of polls for n future months

    Args:
        X_polls (pd.DataFrame): polls data
        elections_in_n_months (int, optional): how many months to predict. 
            Defaults to 1.

    Returns:
        np.ndarray: prediction of polls for n months
    """
    def get_time_series_from_polls(polls: pd.Series) -> pd.Series:
        """Converts provided series to time series pandas structure

        Args:
            polls (pd.Series): series to be converted

        Returns:
            pd.Series: series with index as a proper monthly time series
        """
        last_poll = pd.to_datetime(polls.index[-1])
        time_series: pd.Series = pd.Series(
            index = pd.date_range(end=last_poll, periods=len(polls), freq="M")
            )
        time_series[:] = pd.to_numeric(polls)
        return time_series
    
    # if election is in one month, return last polls
    if elections_in_n_months == 1:
        return X_polls.iloc[:, -1].to_numpy()
    
    # predict polls for each row separately by Holt's double exponential smoothing
    predictions: list[float] = []
    for party_name, party_polls in X_polls.iterrows():
        polls_time_series: pd.Series = get_time_series_from_polls(party_polls)
        holt: Holt = Holt(polls_time_series)
        holt = holt.fit()
        # print(f"{party_name}: alpha={holt.params["smoothing_level"]}, beta={holt.params["smoothing_trend"]}")
        predictions.append(holt.forecast(elections_in_n_months).iloc[-1])
    
    # return the predictions
    return np.array(predictions)
    
    
def allocate_seats_from_percentages(election_results_df, percent_column, total_votes, total_seats=150, threshold=5):
    # ! whole function generated by ChatGPT after a conversation with initial prompt: 
    #       "i have data in format political_party: election result. 
    #       Generate a function that would convert the result into seats 
    #       in parliament for slovak parlamentary system"
    """
    Allocate parliamentary seats based on percentage results using the d'Hondt method with a threshold.

    Parameters:
        election_results_df (pd.DataFrame): DataFrame with party names and vote percentages.
        percent_column (str): Column name in the DataFrame containing the vote percentages.
        total_votes (int): Total number of votes cast (used to convert percentages to absolute votes).
        total_seats (int): Total number of seats in parliament (default: 150 for Slovakia).
        threshold (float): Minimum percentage of votes required for a party to qualify for seats (default: 5%).

    Returns:
        pd.DataFrame: A DataFrame with party names, vote percentages, and allocated seats.
    """
    # Filter parties based on the threshold
    qualified_parties = election_results_df[election_results_df[percent_column] >= threshold].copy()
    
    # Calculate absolute votes from percentages
    qualified_parties["Votes"] = (qualified_parties[percent_column] / 100) * total_votes
    
    # Prepare a list to store the quotients
    quotients = []
    for party, votes in zip(qualified_parties.index, qualified_parties["Votes"]):
        for divisor in range(1, total_seats + 1):
            quotients.append((votes / divisor, party))
    
    # Sort quotients in descending order (highest quotient gets the seat)
    quotients.sort(reverse=True, key=lambda x: x[0])
    
    # Allocate seats
    seat_allocation = {party: 0 for party in qualified_parties.index}
    for _, party in quotients[:total_seats]:
        seat_allocation[party] += 1
    
    # Add the allocated seats to the qualified_parties DataFrame
    qualified_parties["Seats"] = qualified_parties.index.map(seat_allocation)
    
    # Prepare the final result
    final_results = election_results_df.copy()
    final_results["Seats"] = final_results.index.map(seat_allocation).fillna(0).astype(int)
    
    return final_results["Seats"]

    
def main() -> None:
    """Predict elections if they were held in december 2024 and in may 2025
    """
    
    # relevant parties for elections
    relevant_parties: list[str] = [
        "progresivne_slovensko", "hlas_sd", "smer_sd", "olano", "sas", "kdh", 
        "republika", "sns", "sme_rodina", "madarska_aliancia", "demokrati"
        ]
    
    # get the data
    X_polls, X_party_data = prepare_data(relevant_parties)
    
    # predict the polls after relevant for december 2024 and may 2025 elections
    predictions_in_1_months: np.ndarray = predict_polls_month_before_election(X_polls, 1)
    predictions_in_6_months: np.ndarray = predict_polls_month_before_election(X_polls, 6)
    
    # predict the difference between election result and poll value
    election_poll_diff: np.ndarray = predict_election_poll_difference(X_party_data)    
    
    # display the results
    results: pd.DataFrame = pd.DataFrame(
        data={
            "election 12-2024": predictions_in_1_months + election_poll_diff,
            "election 05-2025": predictions_in_6_months + election_poll_diff
            },
        index=X_polls.index
    )
    
    # sum of results needs to be 100
    results = (100 * results / results.sum(axis=0)).round(2)
    
    # add the allocated seats in parliament (approximate the voters count by 2023 voters count)
    results["seats 12-2024"] = allocate_seats_from_percentages(results, "election 12-2024", total_votes=3007123)
    results["seats 05-2025"] = allocate_seats_from_percentages(results, "election 05-2025", total_votes=3007123)
    results = results.iloc[:, [0,2,1,3]].sort_values(by="election 12-2024", ascending=False)
    
    print(results)
    

if __name__ == "__main__":
    main()
