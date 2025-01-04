from typing import Callable, Optional, Type
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from src.utils import join_dfs, make_interactions, extract_and_scale

np.random.seed(42)


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

class Ensemble(ClassifierMixin, BaseEstimator):
    models: list[BaseEstimator]
    metric: Callable
    weights: Optional[list[float]]
    fitted_classifiers: list[BaseEstimator]
    classes_: np.ndarray
    threshold: float

    def __init__(self, models: list[Type[BaseEstimator]],
                 metric: Callable, threshold: float = 0.5,
                 weights: Optional[list[float]] = None) -> None:
        """
        Args:
        models: list[model] -> array of models to be used
        metric: Callable -> metric that is used for determining the 'weight' of model in final prediction
        weights: list[float] -> optional parameter, if given, no additional scores for final prediction are computed
        threshold: float -> hyperparameter for predict method"""

        assert all([hasattr(model, "fit") for model in models]), "All models must have method called 'fit'"
        assert all([hasattr(model, "predict") for model in models]), "All models must have method called 'predict'"

        assert 0 < threshold < 1, "Threshold must be between 0 and 1"
        
        if weights is not None:
            assert len(models) == len(weights), "If given weights, must have same number of elements as 'models' "

        self.models = models
        self.metric = metric
        self.weights = weights
        self.threshold = threshold
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:

        self.fitted_classifiers = []
        self.classes_ = np.unique(y)

        for model in self.models:
            # fit each model
            classifier = model()

            classifier.fit(X=X, y=y)
            self.fitted_classifiers.append(classifier)
        
        if self.weights is None:
            # calculate i-th weight for i-th model
            scores = []
            for classifier in self.fitted_classifiers:
                predicted = classifier.predict(X=X)

                scores.append(self.metric(y_true=y, y_pred=predicted))
                
            # normalize the weights
            total = sum(scores)
            scores = [value / total for value in scores]
            self.weights = scores.copy()
    
    def predict(self, X: pd.DataFrame):
        predictions: list[np.ndarray] = []

        for classifier in self.fitted_classifiers:
            predictions.append(np.array(classifier.predict(X=X)))
        
        weighted_sum: np.ndarray = np.dot(np.array(predictions).T, np.array(self.weights))

        return (weighted_sum >= self.threshold).astype(int)


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
        plt.title(f"{model_name}")

    plt.tight_layout()
    plt.show()

def optimal_features(estimator: BaseEstimator,
                     X_train: pd.DataFrame,
                     y_train: pd.DataFrame,
                     X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """estimates the best subset of features for given estimator based on accuracy
    Args:
        estimator: model for which the optimal subset of features to be determined
        X_train: training features
        X_test: testing features
        y_train: training targets
    Returns:
        selected X_train features, selected X_test features"""
    
    sfs = SequentialFeatureSelector(estimator=estimator, tol=0,
                                    direction="forward",
                                    scoring="accuracy", cv=5)
    sfs.fit(X_train, y_train)

    X_train_selected = sfs.transform(X_train)
    X_test_selected = sfs.transform(X_test)

    return X_train_selected, X_test_selected


def main() -> None:
    
    path_train: str = "data/polls_by_election_train.csv"
    path_test: str = "data/polls_by_election_test.csv"
    path_general_data: str = "data/general_data.csv"

    data_train: pd.DataFrame = pd.read_csv(path_train)
    data_test: pd.DataFrame = pd.read_csv(path_test)
    data_general: pd.DataFrame = pd.read_csv(path_general_data)
    # loading original data

    y_train: pd.DataFrame = data_train.loc[:, "elected_to_parliament"]
    y_test: pd.DataFrame = data_test.loc[:, "elected_to_parliament"]

    combined_data_train: pd.DataFrame = join_dfs(main_data=data_train, data_general=data_general)
    combined_data_test: pd.DataFrame = join_dfs(main_data=data_test, data_general=data_general)
    # combining general data and polls

    interacted_data_train: pd.DataFrame = make_interactions(data=combined_data_train,
                                                            variables=["in_coalition_before"],
                                                            to_interact=["Unemployment rate (%)", "Pension expenditures (per capita)", "Expenditures on research and development"])
    interacted_data_test: pd.DataFrame = make_interactions(data=combined_data_test,
                                                            variables=["in_coalition_before"],
                                                            to_interact=["Unemployment rate (%)", "Pension expenditures (per capita)", "Expenditures on research and development"])
    """making interactions"""
    
    final_variables: list[str] = [
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        "in_coalition_before_Unemployment rate (%)",
        "in_coalition_before_Pension expenditures (per capita)",
        "in_coalition_before_Expenditures on research and development",
        "Risk of poverty rate", "Total household income", "Total inflation rate (%)",
        "Gasoline price 95 octane",
    ]
    # extracting and scaling data to final format
    final_train = extract_and_scale(data=interacted_data_train, variables_to_extract=final_variables)
    final_test = extract_and_scale(data=interacted_data_test, variables_to_extract=final_variables)

    #--------------------------------------------------------------
    # Determine the optimal features using sequential feature selection
    models: list[BaseEstimator] = [LogisticRegression, SVC, DecisionTreeClassifier]
    ensemble = Ensemble(models=models, metric=accuracy_score)
    ensemble_train, ensemble_test = optimal_features(estimator=ensemble, X_train=final_train,
                                                     y_train=y_train, X_test=final_test)
    #--------------------------------------------------------------
    # Determine the optimal threshold using cross-validation
    thresholds = np.linspace(0, 1, 100)
    param_grid = {
        'threshold': thresholds
    }

    grid_search = GridSearchCV(
        estimator=ensemble, 
        param_grid=param_grid,
        scoring="accuracy",
        cv=5
    )
    grid_search.fit(ensemble_train, y_train)
    best_threshold = grid_search.best_params_["threshold"]
    #--------------------------------------------------------------
    # final ensemble
    final_ensemble = Ensemble(models=models, metric=accuracy_score, threshold=best_threshold)
    final_ensemble.fit(ensemble_train, y=y_train)
    #--------------------------------------------------------------
    # Logistic regression
    log_reg: LogisticRegression = LogisticRegression()
    log_reg_train, log_reg_test = optimal_features(estimator=log_reg, X_train=final_train, 
                                                   y_train=y_train, X_test=final_test)
    
    final_log_reg = LogisticRegression()
    final_log_reg.fit(X=log_reg_train, y=y_train)
    #--------------------------------------------------------------
    # Decision treee
    tree: DecisionTreeClassifier = DecisionTreeClassifier()
    tree_train, tree_test = optimal_features(estimator=tree, X_train=final_train, 
                                             y_train=y_train, X_test=final_test)
    
    final_tree = DecisionTreeClassifier()
    final_tree.fit(X=tree_train, y=y_train)

    #--------------------------------------------------------------
    # Support Vector Classification
    svc: SVC = SVC()
    svc_train, svc_test = optimal_features(estimator=svc, X_train=final_train, 
                                           y_train=y_train, X_test=final_test)
    
    final_svc = SVC()
    final_svc.fit(X=svc_train, y=y_train)
    #--------------------------------------------------------------

    print("Ensemble accuracy: ", accuracy_score(y_true=y_test, y_pred=final_ensemble.predict(X=ensemble_test)))
    print("Logistic regression accuracy: ", accuracy_score(y_true=y_test, y_pred=final_log_reg.predict(X=log_reg_test)))
    print("Decision tree accuracy: ", accuracy_score(y_true=y_test, y_pred=final_tree.predict(X=tree_test)))
    print("SVC accuracy: ", accuracy_score(y_true=y_test, y_pred=final_svc.predict(X=svc_test)))
    #--------------------------------------------------------------

    plot_all_confusion_matrices(y_true=y_test, model_predictions={
        "Logistic regression": final_log_reg.predict(X=log_reg_test),
        "Decision Tree": final_tree.predict(X=tree_test),
        "SVC": final_svc.predict(X=svc_test)
    }, ensemble_pred=final_ensemble.predict(X=ensemble_test))


if __name__ == "__main__":
    main()
