from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import SequentialFeatureSelector, RFECV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from statsmodels.tsa.api import Holt    
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from src.utils import join_dfs, make_interactions, join_dfs_with_diff


def choose_polls_prediction_model() -> None:
    def get_time_series_from_polls(polls: pd.Series) -> pd.Series:
        last_poll = pd.to_datetime(polls.index[-1])
        time_series: pd.Series = pd.Series(
            index = pd.date_range(end=last_poll, periods=len(polls), freq="M")
            )
        time_series[:] = pd.to_numeric(polls)
        return time_series

    def holt_mle_predict_n(polls_time_series: pd.Series, n: int = 12) -> pd.Series:   
        return Holt(polls_time_series).fit().forecast(n)
    
    def holt_grid_search_predict_n(polls_time_series: pd.Series, n: int = 12) -> pd.Series:   
        best_mse = float("inf")
        best_params = None
        train, test = polls_time_series[:-6], polls_time_series[-6:]

        for alpha in np.linspace(0.01, 1, 10):
            for beta in np.linspace(0.01, 1, 10):
                model = Holt(train)
                fit_model = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)

                forecast = fit_model.forecast(len(test))

                mse = np.sqrt((abs(forecast - test)**2).mean())
                if mse < best_mse:
                    best_mse = mse
                    best_params = (alpha, beta)
        
        final_model = Holt(
            polls_time_series
        ).fit(smoothing_level=best_params[0], smoothing_trend=best_params[1], optimized=False)

        # Forecast future values
        forecast = final_model.forecast(steps=n)       
        return forecast
    
    def arima_predict_n(polls_time_series: pd.Series, n: int = 1) -> pd.Series:
        stepwise_model = auto_arima(
            polls_time_series,
            information_criterion="aic",
            start_p=0, max_p=5,
            start_q=0, max_q=5,
            d=None,  # Automatically determine 'd'
            seasonal=False,  # Set True if your data has a seasonal component
            trace=False,  # Print the process
            error_action='ignore',  # Ignore errors during fitting
            suppress_warnings=True,  # Suppress warnings
            stepwise=True,  # Perform stepwise search,
            scoring="mse",
            out_of_sample_size=6
        )

        # Print the best model's order
        # print(f"Optimal ARIMA order: {stepwise_model.order}")
        
        model = ARIMA(polls_time_series, order=stepwise_model.order)#, trend="t")
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=n)

        future_dates = pd.date_range(start=polls_time_series.index[-1] + pd.Timedelta(days=1), periods=n, freq="M")

        forecast_series = pd.Series(forecast, index=future_dates)
        return forecast_series


    polls_data: pd.DataFrame = pd.read_csv("data/polls_data.csv", na_values="NA", index_col="political_party")

    # choose by how well do models predict polls year before 2023 election
    
    # "demokrati" do not have data year before 2023 election, therefore they are not included
    relevant_parties: list[str] = ["progresivne_slovensko", "hlas_sd", "smer_sd", "olano", "sas", "kdh", "republika", "sns", "sme_rodina", "madarska_aliancia"]
    results: dict[str, list[float]] = {
        "arima": [],
        "holt_mle": [],
        "holt_grid_search": []
    }
    
    dont_include_polls_after: str = "2023-08"
    polls_data = polls_data.iloc[:, :polls_data.columns.get_loc(dont_include_polls_after) + 1]
    
    MSE = lambda y, y_pred: ((y - y_pred)**2).mean()
    
    print("Calculating for:", end=" ")
    for party_name in relevant_parties:    
        print(party_name, end=", ", flush=True)
        party_polls: pd.Series = polls_data.loc[party_name]
                
        party_polls_ts: pd.Series = get_time_series_from_polls(party_polls)
        
        months_to_predict: int = 12
        train, test = party_polls_ts[:-months_to_predict], party_polls_ts[-months_to_predict:]
        results["holt_mle"].append(
            MSE(test, holt_mle_predict_n(train, months_to_predict))
        )
        results["holt_grid_search"].append(
            MSE(test, holt_grid_search_predict_n(train, months_to_predict))
        )
        results["arima"].append(
            MSE(test, arima_predict_n(train, months_to_predict))
        )
    print()
    
    for model in results:
        print(f"{model}: {sum(results[model]) / len(results[model])}")
    
    """    
    arima: 9.55449280499384
    holt_mle: 8.915272536955003
    holt_grid_search: 11.987891172946794
    """


def prepare_data(path_to_polls_by_election: str, with_political_compass: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    polls_by_election: pd.DataFrame
    if not with_political_compass:
        polls_by_election = pd.read_csv(path_to_polls_by_election, na_values="NA") #
    else:
        polls_by_election = pd.read_csv(path_to_polls_by_election, na_values="NA", index_col="political_party") 
        polls_by_election = pd.merge(
            polls_by_election,
            pd.read_csv("data/political_compass_data.csv"),
            on="political_party"
        )
    
    general_data: pd.DataFrame = pd.read_csv("data/general_data.csv", na_values="NA")
    
    polls_with_general_data: pd.DataFrame = join_dfs_with_diff(polls_by_election, general_data)
    
    redundant_variables: list[str] = ["political_party", "election_date", "election_result", "elected_to_parliament", "year"]
    polls_columns: list[str] = [str(i) for i in range(1, 13)]
    
    X: pd.DataFrame = polls_with_general_data.drop(
        columns=redundant_variables + polls_columns
    )
    y: pd.Series = polls_by_election["election_result"] - polls_by_election["1"]
    
    interactions: PolynomialFeatures = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X = pd.DataFrame(
        data=interactions.fit_transform(X),
        columns=interactions.get_feature_names_out(X.columns)
        )
        
    return X, y


def choose_linear_model(path_to_data: str, with_political_compass: bool = False,
                        scorer: str = "neg_mean_squared_error", cv: int = 10, max_columns: int = 20
                        ) -> None:   
    def choose_features_for_linear_regression(X: pd.DataFrame, y: pd.Series, 
                                              scorer: str, cv: int, max_columns: int
                                              ) -> np.ndarray:
        cv_scores: list[float] = []
        supports: list[np.ndarray] = []
        current_support: np.ndarray = np.array([False] * X.shape[1])
        
        print("getting column subset for linear regression")
        
        # intercept prediction
        cv_scores.append(cross_val_score(
            LinearRegression(), pd.DataFrame(data=[0] * X.shape[0]), y, scoring=scorer, cv=cv
        ).mean())
        supports.append(current_support.copy())
        
        # predictions with >= 1 variable
        for n_features in range(1, max_columns + 1):
            print(f"{100*(n_features - 1)/max_columns:.0f}%", end=" ", flush=True)
            # greedily add column that maximizes cross-val score (no matter the max score in previous iteration)
            cv_scores_iteration: list[float] = []
            
            # go through columns
            for new_column in range(len(current_support)):
                if current_support[new_column]:
                    cv_scores_iteration.append(-float("inf"))
                    continue
                new_support: np.ndarray = current_support.copy()
                new_support[new_column] = True
                cv_scores_iteration.append(cross_val_score(
                    LinearRegression(), X.loc[:, new_support], y, scoring=scorer, cv=cv
                ).mean())
            
            # store the best cross-val score for `n_features` columns with its support
            argmax_col: int = cv_scores_iteration.index(max(cv_scores_iteration))
            current_support[argmax_col] = True
            cv_scores.append(cv_scores_iteration[argmax_col])
            supports.append(current_support.copy())
        
        # return the best greedy support
        print("100%")
        result: np.ndarray = supports[cv_scores.index(max(cv_scores))]
        print("greedy optimal column subset for linear regression:", list(X.columns[result]))
        return result
    
    
    def choose_lambda_for_regularized_regression(X: pd.DataFrame, y: pd.Series, 
                                                 estimator: Lasso | Ridge, scorer: str, cv: int
                                                 ) -> tuple[BaseEstimator, np.ndarray, StandardScaler]:
        cv_scores: list[float] = []
        lambdas: list[float] = []
        
        print("getting lambda for", estimator().__class__.__name__)    
        for lam in np.linspace(0.1, 10, 100):
            cv_scores.append(cross_val_score(
                estimator(alpha=lam, max_iter=5000), X, y, scoring=scorer, cv=cv
            ).mean())
            lambdas.append(lam)
            
        result: float = lambdas[cv_scores.index(max(cv_scores))]
        print("optimal lambda in [0.1;10]:", result)
        return result
    
    X_train, y_train = prepare_data(path_to_data, with_political_compass)
    
    X_scaler: StandardScaler = StandardScaler().fit(X_train)
    X_train_scaled: pd.DataFrame = pd.DataFrame(
        data = X_scaler.transform(X_train),
        columns=X_train.columns
        )
    
    cols_to_unscale: str = ["in_coalition_before", "in_opposition_before"]
    for col in cols_to_unscale:
        X_train_scaled[col] = X_train[col]
    
    
    class LastPollPredictor(BaseEstimator):
        def fit(self, X, y=None) -> None:
            return self

        def predict(self, X):
            return [0] * X.shape[0]
       
    X_lin_reg_support: np.ndarray = choose_features_for_linear_regression(
        X_train_scaled, y_train, 
        scorer=scorer, cv=cv, max_columns=max_columns
        )
    
    lambda_lasso: float = choose_lambda_for_regularized_regression(
        X_train_scaled, y_train, Lasso, scorer=scorer, cv=cv
        )
    lambda_ridge: float = choose_lambda_for_regularized_regression(
        X_train_scaled, y_train, Ridge, scorer=scorer, cv=cv
        )
    
    estimators: list[BaseEstimator] = [LinearRegression(), Lasso(alpha=lambda_lasso), Ridge(alpha=lambda_ridge), LastPollPredictor()]
    columns_subsets: list[np.ndarray] = [X_lin_reg_support] + [np.array([True] * X_train.shape[1])] * 3
    
    cv_scores: list[float] = []
    for estimator, column_subset in zip(estimators, columns_subsets):
        cv_scores.append(
            cross_val_score(estimator, X_train_scaled.loc[:, column_subset], y_train, scoring=scorer, cv=5).mean()
            )
    
    argmax_cv_scores: int = cv_scores.index(max(cv_scores))
    best_column_subset: np.ndarray = columns_subsets[argmax_cv_scores]
    best_estimator: BaseEstimator = estimators[argmax_cv_scores].fit(X_train_scaled.loc[:, best_column_subset], y_train)
    
    print("chosen model:", best_estimator.__class__.__name__)
    return best_estimator, best_column_subset, X_scaler


def evaluate_linear_model(with_political_compass: bool) -> None:
    estimator, column_subset, scaler = choose_linear_model("data/polls_by_election_train.csv", with_political_compass)
    
    X_test, y_test = prepare_data("data/polls_by_election_test.csv", with_political_compass)
    X_test_scaled: pd.DataFrame = pd.DataFrame(
        data=scaler.transform(X_test),
        columns=X_test.columns
        )
    
    cols_to_unscale: str = ["in_coalition_before", "in_opposition_before"]
    for col in cols_to_unscale:
        X_test_scaled[col] = X_test[col]

    results = pd.DataFrame(data={
        "pred": estimator.predict(X_test_scaled.loc[:, column_subset]),
        "true": y_test,
        "naive": 0
        })
        
    MSE = lambda y, y_pred: ((y - y_pred) ** 2).mean()
    MAE = lambda y, y_pred: abs(y - y_pred).mean()
    R2 = lambda y, y_pred: 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    print("Evaluating chosen model on test data: ")
    print("MSE model:", MSE(results["true"], results["pred"]))
    print("MSE naive:", MSE(results["true"], results["naive"]))
    print("MAE model:", MAE(results["true"], results["pred"]))
    print("MAE naive:", MAE(results["true"], results["naive"]))    
    print("R2 model:", R2(results["true"], results["pred"]))
    print("R2 naive:", R2(results["true"], results["naive"]))


def evaluate_linear_models_to_predict_election_poll_difference() -> None:
    print("without compass")
    evaluate_linear_model(with_political_compass=False)

    print("#" * 20)
    
    print("with compass")
    evaluate_linear_model(with_political_compass=True)
        
    """
    without compass
    getting column subset for linear regression
    0% 5% 10% 15% 20% 25% 30% 35% 40% 45% 50% 55% 60% 65% 70% 75% 80% 85% 90% 95% 100%
    greedy optimal column subset for linear regression: ['in_opposition_before']
    getting lambda for Lasso
    optimal lambda in [0.1;10]: 0.30000000000000004
    getting lambda for Ridge
    optimal lambda in [0.1;10]: 10.0
    chosen model: LinearRegression
    Evaluating chosen model on test data: 
    MSE model: 7.242646896974216
    MSE naive: 7.991146153846153
    MAE model: 1.7428137651821862
    MAE naive: 1.7176923076923079
    R2 model: 0.06577248582971373
    R2 naive: -0.030776277357666304
    ####################
    with compass
    getting column subset for linear regression
    0% 5% 10% 15% 20% 25% 30% 35% 40% 45% 50% 55% 60% 65% 70% 75% 80% 85% 90% 95% 100%
    greedy optimal column subset for linear regression: ['in_coalition_before in_opposition_before', 'in_coalition_before Liberalism-Conservatism', 'in_opposition_before Pension expenditures (per capita)', 'Unemployment rate (%) GDP (per capita)']
    getting lambda for Lasso
    optimal lambda in [0.1;10]: 1.2000000000000002
    getting lambda for Ridge
    optimal lambda in [0.1;10]: 10.0
    chosen model: LinearRegression
    Evaluating chosen model on test data: 
    MSE model: 6.533403453027842
    MSE naive: 11.287239999999997
    MAE model: 1.724073555787856
    MAE naive: 2.2239999999999998
    R2 model: 0.3682928452363886
    R2 naive: -0.09135006230628373
    """
    
    
def train_optimal_linear_model() -> None:
    model: LinearRegression = LinearRegression()
    X_train, y_train = prepare_data("data/polls_by_election_train.csv", with_political_compass=True)
    X_test, y_test = prepare_data("data/polls_by_election_test.csv", with_political_compass=True)
    
    X: pd.DataFrame = pd.concat([X_train, X_test]).reset_index(drop=True)
    y: pd.Series = pd.concat([y_train, y_test]).reset_index(drop=True)
    
    optimal_column_subset: list[str] = [
        'in_coalition_before in_opposition_before', 'in_coalition_before Liberalism-Conservatism', 
        'in_opposition_before Pension expenditures (per capita)', 'Unemployment rate (%) GDP (per capita)'
        ]
    
    model.fit(X.loc[:, optimal_column_subset], y)
    print(model.intercept_)
    print(model.coef_)
    """
    -1.422803951315247
    [0.00000000e+00 1.16370568e+00 2.81747988e+00 6.94022258e-06]
    """
    


def main() -> None:
    train_optimal_linear_model()
       
if __name__ == "__main__":
    main()
