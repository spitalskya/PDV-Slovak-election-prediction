import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import numpy
from sklearn.base import BaseEstimator
from typing import Callable, Optional
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


numpy.random.seed(42)

def add_variables(main_data: pd.DataFrame, data_general: pd.DataFrame) -> pd.DataFrame:
    """Add columns 'unemployment_in_coallition' and 'Total inflation rate (%)' into our main dataframe"""

    main_data["election_date"] = pd.to_datetime(main_data["election_date"]) # make a date format from string
    main_data["year"] = main_data["election_date"].dt.year # new column year

    #-----------------------------------------------------------------------------------------------------------------------------
    # Add column in_coalition * unemployment_rate
    data_general_long = data_general.melt(id_vars=['indicator'], var_name='year', value_name='unemployment_rate') # make the table in long format
    data_general_long = data_general_long[data_general_long["indicator"] == "Unemployment rate (%)"] # take only unemployment rate
    data_general_long['year'] = data_general_long['year'].astype(int) # cast string to int
    data_general_long['unemployment_rate'] = data_general_long['unemployment_rate'].astype(float) # cast string to float
    df_main = pd.merge(main_data, data_general_long[['year', 'unemployment_rate']], on='year', how='left') # merge two dfs
    df_main["unemployment_in_coallition"] = df_main["in_coalition_before"] * df_main["unemployment_rate"] # 1 / 0 * unemployment_rate
    #-----------------------------------------------------------------------------------------------------------------------------
    # Add column inflation
    data_general_long = data_general.melt(id_vars=['indicator'], var_name='year', value_name='Total inflation rate (%)')
    data_general_long = data_general_long[data_general_long["indicator"] == "Total inflation rate (%)"]
    data_general_long['year'] = data_general_long['year'].astype(int)
    data_general_long['unemployment_rate'] = data_general_long['Total inflation rate (%)'].astype(float)
    df_main = pd.merge(df_main, data_general_long[['year', 'Total inflation rate (%)']], on='year', how='left') # merge two dfs

    return df_main

def extract_variables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract features and targets from dataframe"""

    variables: list[str] = list(map(str, range(1, 13))) + ["unemployment_in_coallition", "Total inflation rate (%)"]
    target: str = "elected_to_parliament"

    return data.loc[:, variables], data.loc[:, target]


def segment_by_year(df: pd.DataFrame) -> \
                        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Segments data into four categories based on the year of election"""

    df['election_date'] = pd.to_datetime(df['election_date'])
    data_2012 = df[df["election_date"].dt.year == 2012]
    data_2016 = df[df["election_date"].dt.year == 2016]
    data_2020 = df[df["election_date"].dt.year == 2020]
    data_2023 = df[df["election_date"].dt.year == 2023]

    return data_2012, data_2016, data_2020, data_2023

def run_basic_models(models: list[BaseEstimator], X_train: pd.DataFrame,
                     X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame,
                     metrics: list[Callable]) -> dict[str, dict[str, float]]:
    """Function that fits multiple models and returns various metric evaluation for these models
    Args:
        models: list[model] -> array of models to be used
        X_train: pd.Dataframe -> training data
        y_train: pd.Dataframe -> training targets
        X_test: pd.Dataframe -> testing data
        y_test: pd.Dataframe -> testing targets
        metrics: list[metric] -> array of metric to be computed
    Returns:
        dictionary which stores calculated value (predictions ? ) for every model for every metric"""
    
    assert all([hasattr(model, "fit") for model in models]), "All models must have method called 'fit'"
    assert all([hasattr(model, "predict") for model in models]), "All models must have method called 'predict'"

    result: dict[str, dict[str, float]] = dict()

    model: BaseEstimator
    for model in models:

        # result[model.__name__] = dict()
        model_instance = model()
        model_instance.fit(X=X_train, y=y_train)

        predicted = model_instance.predict(X=X_test)
        result[model.__name__] = predicted


        # for metric in metrics:
        #     score = metric(y_true=y_test, y_pred=predicted)

        #     result[model.__name__][metric.__name__] = score
    
    return result

class Ensemble:
    models: list[BaseEstimator]
    metric: Callable
    weights: Optional[list[float]]
    fitted_classifiers: list[BaseEstimator]

    def __init__(self, models: list[BaseEstimator], metric: Callable, weights: Optional[list[float]] = None):
        """
        Args:
        models: list[model] -> array of models to be used
        metric: Callable -> metric that is used for determining the 'weight' of model in final prediction
        weights: list[float] -> optional parameter, if given, no additional scores for final prediction are computed"""

        assert all([hasattr(model, "fit") for model in models]), "All models must have method called 'fit'"
        assert all([hasattr(model, "predict") for model in models]), "All models must have method called 'predict'"
        
        if weights is not None:
            assert len(models) == len(weights), "If given weights, must have same number of elements as 'models' "

        self.models = models
        self.metric = metric
        self.weights = weights
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:

        self.fitted_classifiers = []

        for model in self.models:
            """fit each model"""
            classifier = model()

            classifier.fit(X=X, y=y)
            self.fitted_classifiers.append(classifier)
        
        if self.weights is None:
            """calculate i-th weight for i-th model"""
            scores = []
            for classifier in self.fitted_classifiers:
                predicted = classifier.predict(X=X)

                scores.append(self.metric(y_true=y, y_pred=predicted))
                
            """normalize the weights"""
            total = sum(scores)
            scores = [value / total for value in scores]
            self.weights = scores.copy()
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5):
        predictions: list[np.ndarray] = []

        for classifier in self.fitted_classifiers:
            predictions.append(np.array(classifier.predict(X=X)))
        
        weighted_sum: np.ndarray = np.dot(np.array(predictions).T, np.array(self.weights))

        return (weighted_sum >= threshold).astype(int)


def plot_all_confusion_matrices(y_true: pd.Series, model_predictions: dict, ensemble_pred: np.ndarray) -> None:
    """Plots all confusion matrices side by side for comparison"""
    models = list(model_predictions.keys()) + ['Ensemble']
    predictions = list(model_predictions.values()) + [ensemble_pred]

    num_models = len(models)
    plt.figure(figsize=(5 * num_models, 4))

    for i, (model_name, y_pred) in enumerate(zip(models, predictions), 1):
        cm = confusion_matrix(y_true, y_pred)
        plt.subplot(1, num_models, i)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Elected', 'Elected'], 
                    yticklabels=['Not Elected', 'Elected'])
        plt.ylabel('Actual') if i == 1 else plt.ylabel('')
        plt.xlabel('Predicted')
        plt.title(f"Confusion Matrix: {model_name}")

    plt.tight_layout()
    plt.show()


def main() -> None:
    
    path_train: str = r"C:\Users\antal\Desktop\matfyz\Princípy DAV\PDV-Slovak-election-prediction\data\polls_by_election_train.csv"
    path_test: str = r"C:\Users\antal\Desktop\matfyz\Princípy DAV\PDV-Slovak-election-prediction\data\polls_by_election_test.csv"
    path_general_data: str = r"C:\Users\antal\Desktop\matfyz\Princípy DAV\PDV-Slovak-election-prediction\data\general_data.csv"

    data_train: pd.DataFrame = pd.read_csv(path_train)
    data_test: pd.DataFrame = pd.read_csv(path_test)
    data_general: pd.DataFrame = pd.read_csv(path_general_data)


    data_train = add_variables(data_train, data_general)
    data_test = add_variables(data_test, data_general)

    X_train, y_train = extract_variables(data=data_train)
    X_test, y_test = extract_variables(data=data_test)

    models: list[BaseEstimator] = [LogisticRegression, SVC, DecisionTreeClassifier]

    models_predictions = run_basic_models(models=models, X_train=X_train, y_train=y_train,
                                          X_test=X_test, y_test=y_test, metrics=[accuracy_score])
    
    ensemble = Ensemble(models=models, metric=accuracy_score)
    ensemble.fit(X=X_train, y=y_train)
    ensemble_predictions = ensemble.predict(X=X_test, threshold=0.5)

    plot_all_confusion_matrices(y_true=y_test, model_predictions=models_predictions, ensemble_pred=ensemble_predictions)




if __name__ == "__main__":
    main()
