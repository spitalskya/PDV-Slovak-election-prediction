import pandas as pd
from sklearn.preprocessing import StandardScaler

def join_dfs(main_data: pd.DataFrame, data_general: pd.DataFrame) -> pd.DataFrame:
    """function to join 'general data' and 'polls_by_election' """

    main_data["election_date"] = pd.to_datetime(main_data["election_date"]) # make a date format from string
    main_data["year"] = main_data["election_date"].dt.year # new column year

    variables_to_extract: list[str] = [
        "Unemployment rate (%)", "GDP (per capita)", "Risk of poverty rate", "Total household income", 
        "Pension expenditures (per capita)", "Total inflation rate (%)", "Gasoline price 95 octane", 
        "Expenditures on research and development"
    ]
    # These variables will be added from general data to main df

    variable: str
    for variable in variables_to_extract:
        data_general_long = data_general.melt(id_vars=["indicator"], var_name="year", value_name=variable)
        data_general_long = data_general_long[data_general_long["indicator"] == variable]
        data_general_long['year'] = data_general_long['year'].astype(int) # cast string to int
        main_data = pd.merge(main_data, data_general_long[['year', variable]], on='year', how='left') # merge two dfs

    return main_data


def join_dfs_with_diff(main_data: pd.DataFrame, data_general: pd.DataFrame) -> pd.DataFrame:
    """function to join 'general data' and 'polls_by_election' 
    same as join_dfs(), but also adds variables corresponding to difference of values of indicators at
    the end and at the beginning of electoral term
    """

    main_data["election_date"] = pd.to_datetime(main_data["election_date"]) # make a date format from string
    main_data["year"] = main_data["election_date"].dt.year # new column year

    variables_to_extract: list[str] = [
        "Unemployment rate (%)", "GDP (per capita)", "Risk of poverty rate", "Total household income", 
        "Pension expenditures (per capita)", "Gasoline price 95 octane",            # removed "Total inflation rate (%)", 
        "Expenditures on research and development"
    ]
    # These variables will be added from general data to main df

    previous_election: dict[int: int] = {
        2023: 2020,
        2020: 2016,
        2016: 2012,
        2012: 2010
    }

    variable: str
    for variable in variables_to_extract:
        data_general_long = data_general.melt(id_vars=["indicator"], var_name="year", value_name=variable)
        data_general_long = data_general_long[data_general_long["indicator"] == variable]
        data_general_long['year'] = data_general_long['year'].astype(int) # cast string to int  
        
        get_data_from_previous_election = lambda x: data_general[data_general["indicator"] == variable][str(previous_election[x])].iloc[0]
        
        for election_year in previous_election: # add difference
            data_general_long[variable + " diff"] = data_general_long[variable]
            data_general_long.loc[data_general_long["year"] == election_year, variable + " diff"] -= get_data_from_previous_election(election_year)
        
        main_data = pd.merge(main_data, data_general_long[['year', variable]], on='year', how='left') # merge two dfs

    return main_data


def make_interactions(data: pd.DataFrame, variables: list[str], to_interact: list[str]) -> pd.DataFrame:
    """function for creating interaction variables"""

    variable: str
    for variable in variables:
        interaction: str
        for interaction in to_interact:
            try:
                data[f"{variable}_{interaction}"] = data[variable] * data[interaction]
            except Exception as e:
                print(f"Variables ({variable, interaction}) could not be interacted because of an error: {e}")
    return data

def extract_and_scale(data: pd.DataFrame, variables_to_extract: list[str]):
    """function for extracting variables and z-scaling"""

    extracted_data: pd.DataFrame = data.loc[:, variables_to_extract]

    scaler: StandardScaler = StandardScaler()

    return scaler.fit_transform(extracted_data)


if __name__ == "__main__":
    path1 = "data/polls_by_election.csv"
    path2 = "data/general_data.csv"
    
    
    main_data = pd.read_csv(path1)
    data_general = pd.read_csv(path2)

    new_data: pd.DataFrame = join_dfs_with_diff(main_data, data_general)
    print(new_data)
    