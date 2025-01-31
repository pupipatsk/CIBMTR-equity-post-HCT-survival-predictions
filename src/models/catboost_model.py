import warnings
warnings.filterwarnings("ignore")
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import os


class CatBoostModel:
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
        self.base_params = {
            # "task_type": "GPU" if self.use_gpu else "CPU",
            "eval_metric": "AUC",
            "verbose": False,
            "thread_count": -1,
        }

    def objective(self, trial, X_train, y_train):
        """
        Objective function for Optuna to maximize AUC.
        :param trial: Optuna trial object.
        :param X_train: Training features.
        :param y_train: Training labels.
        :return: Mean validation AUC score from cross-validation.
        """
        # params = {
        #     # Number of trees/iterations: affects training duration and overfitting
        #     "iterations": trial.suggest_int("iterations", 500, 5000, step=100),  # Larger range for fine-grained control
        #     # Tree depth: balances model complexity and overfitting
        #     "depth": trial.suggest_int("depth", 4, 10),
        #     # Learning rate: step size for gradient updates, lower values reduce overfitting risk
        #     "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        #     # L2 regularization: penalizes large leaf values to prevent overfitting
        #     "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
        #     # Sampling parameters: for regularization and reducing overfitting
        #     "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        #     # Feature sampling: controls feature usage per tree
        #     "rsm": trial.suggest_float("rsm", 0.5, 1.0),  # Random subspace method
        #     # Custom loss function metric
        #     "custom_metric": ["AUC"],  # Set AUC explicitly as the evaluation metric
        #     # Allow GPU for faster computation if available
        #     # "task_type": "GPU" if self.use_gpu else "CPU",
        #     # Random seed for reproducibility
        #     # "random_seed": 42,
        #     # Border count for quantizing features
        #     "border_count": trial.suggest_int("border_count", 32, 256),
        # }
        params = {
            "iterations": trial.suggest_int("iterations", 100, 5000),
            "depth": trial.suggest_int("depth", 3, 17),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.5),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        }

        # Use Stratified K-Fold for better AUC evaluation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            if isinstance(X_train, pd.DataFrame):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            else:
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
            classes = np.unique(y_train)
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y_train
            )
            class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
            model = CatBoostClassifier(**self.base_params, **params, class_weights=[class_weights_dict.get(cls, 1.0) for cls in classes])
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
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
        pbar = tqdm(total=n_trials, desc="Tuning Catboost", position=0, leave=True)

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

    def fit(self, X_train, y_train):
        """
        Train the model using the best parameters from Optuna.
        :param X_train: Training features.
        :param y_train: Training labels.
        """
        if self.best_params is None:
            raise ValueError("You need to tune the model first!")
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )
        class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
        self.model = CatBoostClassifier(**self.best_params, class_weights=[class_weights_dict.get(cls, 1.0) for cls in classes])
        self.model.fit(X_train, y_train)
        self.save_model()

    def cross_validate(self, X_train, y_train):
        if self.best_params is None:
            raise ValueError("You need to tune the model first to obtain the best parameters!")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
        auc_scores = []

        # Compute class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )
        class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

        for train_idx, val_idx in skf.split(X_train, y_train):
            if isinstance(X_train, pd.DataFrame):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            else:
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
            model = CatBoostClassifier(
                **self.best_params,
                class_weights=[class_weights_dict.get(cls, 1.0) for cls in classes]
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=10)

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc_scores.append(roc_auc_score(y_val, y_pred_proba))

        return np.mean(auc_scores)

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
        file_name = f"catboost-AUC{self.mean_val_auc:.4f}-{timestamp}.joblib"
        # path = os.path.join("src", "models", "saved", file_name)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and use the class
    catboost_optuna = CatBoostModel(n_trials=50)
    catboost_optuna.tune(X_train, y_train)  # Hyperparameter tuning
    catboost_optuna.fit(X_train, y_train)  # Train with best parameters
    auc_score = catboost_optuna.cross_validate(X_test, y_test)  # Evaluate on test data
    print("Test AUC Score:", auc_score)

    # Save the trained model
    catboost_optuna.save_model()