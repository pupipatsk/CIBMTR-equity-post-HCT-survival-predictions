import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import joblib
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignore user warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Ignore runtime warnings

class XGBoostModel:
    def __init__(self, n_trials=50, use_gpu=False):
        """
        Initialize the class with the number of Optuna trials.
        :param n_trials: Number of trials for Optuna hyperparameter tuning.
        """
        self.n_trials = n_trials
        self.best_params = None
        self.model = None
        self.mean_val_auc = None
        self.use_gpu = use_gpu

    def objective(self, trial, X_train, y_train):
        """
        Objective function for Optuna to minimize.
        :param trial: Optuna trial object.
        :param X_train: Training features.
        :param y_train: Training labels.
        :return: AUC score.
        """

        tree_method = "gpu_hist" if self.use_gpu else "hist"
        predictor = "gpu_predictor" if self.use_gpu else "cpu_predictor"

        # TODO: CHANGE TUNING RANGE
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "scale_pos_weight": trial.suggest_uniform("scale_pos_weight", 5.0, 50.0),
            "tree_method": tree_method,  # Use GPU/CPU dynamically
            "predictor": predictor,  # Use GPU/CPU predictor dynamically
        }

        # Use stratified K-fold for better AUC evaluation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
        auc_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            if isinstance(X_train, pd.DataFrame):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            else:
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = XGBClassifier(**params, use_label_encoder=False, eval_metric="auc")
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            y_pred_proba = model.predict_proba(X_val)[:, 1]  # Get probability of positive class
            auc_scores.append(roc_auc_score(y_val, y_pred_proba))

        return sum(auc_scores) / len(auc_scores)

    def tune(self, X_train, y_train, verbose=True):
        """
        Perform hyperparameter tuning using Optuna with a progress bar.
        :param X_train: Training features.
        :param y_train: Training labels.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")

        # Progress bar setup
        n_trials = self.n_trials
        pbar = tqdm(total=n_trials, desc="Tuning XGBoost", position=0, leave=True)

        def progress_callback(study, trial):
            pbar.update(1)  # Update progress bar by one step

        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train),
            n_trials=n_trials,
            callbacks=[progress_callback]
        )

        # Store the best parameters
        self.best_params = study.best_params
        self.mean_val_auc = self.cross_validate(X_train, y_train)  # Save mean CV AUC

        if verbose:
            print("\nBest parameters:", self.best_params)
            print(f"Mean Validation AUC Score: {self.mean_val_auc:.4f}")

        # Close the progress bar
        pbar.close()

    # def tune(self, X_train, y_train, verbose=True):
    #     """
    #     Perform hyperparameter tuning using Optuna and store the mean validation AUC score.
    #     :param X_train: Training features.
    #     :param y_train: Training labels.
    #     """
    #     optuna.logging.set_verbosity(optuna.logging.WARNING)
    #     study = optuna.create_study(direction="maximize")
    #     study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=self.n_trials)
        
    #     # Store the best parameters and calculate the mean validation AUC
    #     self.best_params = study.best_params
    #     self.mean_val_auc = self.cross_validate(X_train, y_train)

    #     if verbose:
    #         print("Best parameters:", self.best_params)
    #         print(f"Mean Validation AUC Score with Best Params: {self.mean_val_auc:.4f}")

    def cross_validate(self, X_train, y_train):
        """
        Perform Stratified K-Fold cross-validation and return the mean validation AUC score.
        :param X_train: Training features.
        :param y_train: Training labels.
        :return: Mean validation AUC score from cross-validation.
        """
        if self.best_params is None:
            raise ValueError("You need to tune the model first to obtain the best parameters!")
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
        auc_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            if isinstance(X_train, pd.DataFrame):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            else:
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # Initialize model with the best parameters
            model = XGBClassifier(**self.best_params, use_label_encoder=False, eval_metric="auc")
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            # Predict probabilities for the positive class
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc_scores.append(roc_auc_score(y_val, y_pred_proba))

        mean_auc = sum(auc_scores) / len(auc_scores)
        return mean_auc


    def fit(self, X_train, y_train):
        """
        Train the model using the best parameters from Optuna.
        :param X_train: Training features.
        :param y_train: Training labels.
        """
        if self.best_params is None:
            raise ValueError("You need to tune the model first!")
        self.model = XGBClassifier(**self.best_params, use_label_encoder=False, eval_metric="auc")
        self.model.fit(X_train, y_train)
        self.save_model()

    def predict_proba(self, X_test):
        """
        Make probability predictions with the trained model.
        :param X_test: Test features.
        :return: Predicted probabilities for the positive class.
        """
        if self.model is None:
            raise ValueError("You need to train the model first!")
        return self.model.predict_proba(X_test)[:, 1]

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

    def save_model(self):
        """
        Save the trained model and best parameters to a file with a dynamic filename.
        """
        if self.model is None:
            raise ValueError("You need to train the model first!")

        timestamp = datetime.now().strftime("%m%d-%H%M")
        
        dir_path = os.path.join("models", "saved")
        os.makedirs(dir_path, exist_ok=True)
        file_name = f"xgboost-AUC{self.mean_val_auc:.4f}-{timestamp}.joblib"
        path = os.path.join(dir_path, file_name)
        # path = f"/models/saved/lgbm-AUC{self.mean_val_auc:.4f}-{timestamp}.joblib"

        to_save = {
            "model": self.model,
            "best_params": self.best_params,
            "mean_val_auc": self.mean_val_auc,
        }
        joblib.dump(to_save, path)
        print(f"Model and best parameters saved as {path}")

    def get_mean_val_auc(self):
        if self.mean_val_auc is None:
            raise ValueError("You need to tune the model first!")
        return self.mean_val_auc


# Example Usage
if __name__ == "__main__":
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

    # Create and use the class
    xgb_optuna = XGBoostModel(n_trials=50)
    xgb_optuna.tune(X_train, y_train)  # Hyperparameter tuning
    xgb_optuna.fit(X_train, y_train)  # Train with best parameters
    auc_score = xgb_optuna.cross_validate(X_test, y_test)  # Evaluate on test data
    print("Test AUC Score:", auc_score)
