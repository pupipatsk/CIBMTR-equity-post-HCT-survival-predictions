import warnings
warnings.filterwarnings("ignore")
# import optuna
from autogluon.tabular import TabularPredictor
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
# from datetime import datetime
# import joblib
# import os


class AutoGluonModel:
    def __init__(self):
        """
        Initialize the class with the number of Optuna trials.
        :param target_feature: Target column name
        :param df: Dataframe
        """
        self.model = None
        self.mean_val_auc = None        

    def fit(self, df, target_feature):
        """
        Train the model.
        :param df: DataFrame.
        :param target_feature: Target feature name (string).
        """        
        self.model = TabularPredictor(label=target_feature, eval_metric="roc_auc")
        self.model.fit(df)
        # self.save_model()

    def predict_proba(self, X_test):
        """
        Make probability predictions with the trained model.
        :param X_test: Test features.
        :return: Predicted probabilities for the positive class.
        """
        if self.model is None:
            raise ValueError("You need to train the model first!")
        return self.model.evaluate(X_test, silent=True)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data using AUC.
        :param X_test: Test features.
        :param y_test: True labels.
        :return: AUC score.
        """
        if self.model is None:
            raise ValueError("You need to train the model first!")
        y_pred_proba = self.predict_proba(X_test)
        return roc_auc_score(y_test, y_pred_proba)
    
    def get_leaderboard(self, test_data):
        """
        Evaluate the model on test data using AUC.
        :param test_data: DataFrame contains both X and y.
        :return: Rankings.
        """
        return self.model.leaderboard(test_data)

# Example Usage
if __name__ == "__main__":
    pass