import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import Holt
from src.utils import join_dfs_with_diff


def prepare_data(relevant_parties: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    polls_data: pd.DataFrame = pd.read_csv("data/polls_data.csv", na_values="NA", index_col="political_party")
    X_polls: pd.DataFrame = polls_data.loc[polls_data.index.isin(relevant_parties)]
    
    joint_data: pd.DataFrame = pd.DataFrame(
        data={"election_date": pd.to_datetime("2023-09-30")},        # since there is no general data for 2024, we assume its the same as 2023 and use this "workaround"
        index=relevant_parties
    )
    joint_data["political_party"] = relevant_parties
    
    coalition_data: pd.DataFrame = pd.read_csv("data/elected_parties.csv", na_values="NA")
    joint_data = pd.merge(
        joint_data,
        coalition_data[coalition_data["until"] == "now"].drop(columns="until"),
        on="political_party",
        how="left"
    ).rename(columns={"coalition": "in_coalition_before"}).set_index("political_party")
    joint_data["in_opposition_before"] = 1 - joint_data["in_coalition_before"]
    joint_data = joint_data.fillna(0)
        
    joint_data = pd.merge(
        joint_data,
        pd.read_csv("data/political_compass_data.csv", na_values="NA"),
        on="political_party"
    )
    
    joint_data = join_dfs_with_diff(joint_data, pd.read_csv("data/general_data.csv", na_values="NA"))
        
    redundant_variables: list[str] = ["election_date", "year"]
    joint_data = joint_data.drop(columns=redundant_variables)
    joint_data = joint_data.set_index("political_party")
    
    interactions: PolynomialFeatures = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X_party_data = pd.DataFrame(
        data=interactions.fit_transform(joint_data),
        columns=interactions.get_feature_names_out(joint_data.columns)
        )
    X_party_data.index = relevant_parties
    
    return X_polls, X_party_data


def predict_election_poll_difference(X_party_data: pd.DataFrame) -> np.ndarray:
    election_poll_difference_model: LinearRegression = LinearRegression()
    election_poll_difference_model.intercept_ = -1.422803951315247
    election_poll_difference_model.coef_ = np.array([0.00000000e+00, 1.16370568e+00, 2.81747988e+00, 6.94022258e-06])
    
    column_subset_for_linear_regression: list[str] = [
        'in_coalition_before in_opposition_before', 'in_coalition_before Liberalism-Conservatism', 
        'in_opposition_before Pension expenditures (per capita)', 'Unemployment rate (%) GDP (per capita)'
        ]
        
    
    X_party_data = X_party_data.loc[:, column_subset_for_linear_regression]
    print(X_party_data)
    return election_poll_difference_model.predict(X_party_data)
    

def predict_polls_month_before_election(X_polls: pd.DataFrame, elections_in_n_months: int = 1) -> np.ndarray:
    def get_time_series_from_polls(polls: pd.Series) -> pd.Series:
        last_poll = pd.to_datetime(polls.index[-1])
        time_series: pd.Series = pd.Series(
            index = pd.date_range(end=last_poll, periods=len(polls), freq="M")
            )
        time_series[:] = pd.to_numeric(polls)
        return time_series
    
    if elections_in_n_months == 1:
        return X_polls.iloc[:, -1].to_numpy()
    
    predictions: list[float] = []
    for _, party_polls in X_polls.iterrows():
        polls_time_series: pd.Series = get_time_series_from_polls(party_polls)
        holt: Holt = Holt(polls_time_series)
        predictions.append(holt.fit().forecast(elections_in_n_months).iloc[-1])
    
    return np.array(predictions)
    
    
def allocate_seats_from_percentages(election_results_df, percent_column, total_votes, total_seats=150, threshold=5):
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
    qualified_parties['Votes'] = (qualified_parties[percent_column] / 100) * total_votes
    
    # Prepare a list to store the quotients
    quotients = []
    for party, votes in zip(qualified_parties.index, qualified_parties['Votes']):
        for divisor in range(1, total_seats + 1):
            quotients.append((votes / divisor, party))
    
    # Sort quotients in descending order (highest quotient gets the seat)
    quotients.sort(reverse=True, key=lambda x: x[0])
    
    # Allocate seats
    seat_allocation = {party: 0 for party in qualified_parties.index}
    for _, party in quotients[:total_seats]:
        seat_allocation[party] += 1
    
    # Add the allocated seats to the qualified_parties DataFrame
    qualified_parties['Seats'] = qualified_parties.index.map(seat_allocation)
    
    # Prepare the final result
    final_results = election_results_df.copy()
    final_results['Seats'] = final_results.index.map(seat_allocation).fillna(0).astype(int)
    
    return final_results['Seats']

    
    


def main() -> None:
    relevant_parties: list[str] = [
        "progresivne_slovensko", "hlas_sd", "smer_sd", "olano", "sas", "kdh", 
        "republika", "sns", "sme_rodina", "madarska_aliancia", "demokrati"
        ]
    
    X_polls, X_party_data = prepare_data(relevant_parties)
    X_polls = X_polls.sort_index(axis=0)
    X_party_data = X_party_data.sort_index(axis=0)

    
    election_poll_diff: np.ndarray = predict_election_poll_difference(X_party_data)    
    print(election_poll_diff)

    predictions_in_1_months: np.ndarray = predict_polls_month_before_election(X_polls, 1)
    predictions_in_6_months: np.ndarray = predict_polls_month_before_election(X_polls, 6)
    
    results: pd.DataFrame = pd.DataFrame(
        data={
            "1 month": predictions_in_1_months + election_poll_diff,
            "6 months": predictions_in_6_months + election_poll_diff
            },
        index=X_polls.index
    )
    
    results = (100 * results / results.sum(axis=0)).round(2)
    
    results["seats 1 month"] = allocate_seats_from_percentages(results, "1 month", 3007123)
    results["seats 6 months"] = allocate_seats_from_percentages(results, "6 months", 3007123)
    
    print(X_polls["2024-11"])
    print(results)
    

if __name__ == "__main__":
    main()
