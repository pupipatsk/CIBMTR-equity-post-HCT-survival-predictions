import warnings
warnings.filterwarnings("ignore")
from autogluon.tabular import TabularPredictor
from ..data.dataloader import load_data
from .catboost_model import CatBoostModel
from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel
import os
import pandas as pd

class Trainer:
    def __init__(self, use_gpu=False):
        # TODO: Setup Data Path and Target Name
        self.target = "default_12month"

        self.repo_root = self._get_repository_root()
        self.data_dir = os.path.join(self.repo_root, "data", "featured")

        train_filename = "train_featured_data-allbin.parquet"
        test_filename = "test_featured_data-1-allbin.parquet"
        self.train_data_path = os.path.join(self.data_dir, train_filename)
        self.test_data_path = os.path.join(self.data_dir, test_filename)

        self.df_train = load_data(self.train_data_path)
        self.df_test = load_data(self.test_data_path)

        self.X_train = self.df_train.drop(self.target, axis=1)
        self.y_train = self.df_train[self.target]
        self.X_test = self.df_test
        self.use_gpu = use_gpu

    @staticmethod
    def _get_repository_root() -> str:
        """
        Determine the root directory of the repository.

        Returns:
            str: The absolute path of the repository root.
        """
        current_file_path = os.path.abspath(__file__)
        repository_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        return repository_root

    def train(self):

        # Catbost Model
        catboost_optuna = CatBoostModel(n_trials=200)
        catboost_optuna.tune(self.X_train, self.y_train)  # Hyperparameter tuning
        catboost_optuna.fit(self.X_train, self.y_train)  # Train with best parameters
        # catboost_auc_score = catboost_optuna.cross_validate(self.X_test, self.y_test)  # Evaluate on test data
        # catboost_optuna.save_model()

        # xgb_optuna = XGBoostModel(n_trials=100, use_gpu=self.use_gpu)
        # xgb_optuna.tune(self.X_train, self.y_train)  # Hyperparameter tuning
        # xgb_optuna.fit(self.X_train, self.y_train)  # Train with best parameters
        # df_sub = pd.read_csv(r"C:\Users\Tonza\Desktop\Code\Aihack-Thailand-2024\data\raw\csv\submission_template_for_public.csv")
        # df_sub['default_12month'] = xgb_optuna.predict_proba(self.X_test)  
        # df_sub.to_csv("tmp_sub.csv", index=False)

        return
        # LightGBM Model
        lightgbm_optuna = LightGBMModel(n_trials=50)
        lightgbm_optuna.tune(self.X_train, self.y_train)  # Hyperparameter tuning
        lightgbm_optuna.fit(self.X_train, self.y_train)  # Train with best parameters
        # lightgbm_auc_score = lightgbm_optuna.cross_validate(self.X_test, self.y_test)  # Evaluate on test data
        # lightgbm_optuna.save_model()

        # XGBoost Model
        xgb_optuna = XGBoostModel(n_trials=50)
        xgb_optuna.tune(self.X_train, self.y_train)  # Hyperparameter tuning
        xgb_optuna.fit(self.X_train, self.y_train)  # Train with best parameters
        # xgb_auc_score = xgb_optuna.cross_validate(self.X_test, self.y_test)  # Evaluate on test data
        # xgb_optuna.save_model()

        # AutoGluon
        # auto_gluon = TabularPredictor(label=self.target, eval_metric="roc_auc")
        # auto_gluon.fit(self.df_train, time_limit=60*60)
        # auto_gluon_leaderboard = auto_gluon.leaderboard()


if __name__ == "__main__":
    # Load dataset
    trainer = Trainer()
    trainer.train()
