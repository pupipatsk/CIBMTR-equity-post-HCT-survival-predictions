import numpy as np
import pandas as pd
import json
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
)


class NumericalTransformer:
    def __init__(self, log_dir=os.path.join("logs", "transform_logs")):
        """
        Initializes the Feature_transformer with training and test datasets and a logistic regression model.
        """
        self.model = LogisticRegression(random_state=69)
        self.log_dict = dict()
        self.log_dir = log_dir

    def save_log_dict(self):
        """
        Saves the log dictionary as a JSON file with a unique name.
        """
        # Create a unique filename using the current timestamp
        timestamp = time.strftime("%Y%m%d_%H%M")
        filename = f"{timestamp}.json"
        log_file = os.path.join(self.log_dir, filename)

        try:
            with open(log_file, "w") as f:
                json.dump(self.log_dict, f, indent=4)
            print(f"Log saved to {log_file}")
        except Exception as e:
            print(f"Failed to save log_dict to {log_file}: {e}")

    def fit(self, df_train, df_test):
        """
        Finds the best transformation for each feature by evaluating its impact on the AUC score
        in logistic regression and logs the results.
        """
        # Separate features and target
        X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
        X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]

        # Define transformations
        transformations = {
            "none": lambda: None,  # No transformation
            "standard_scaler": StandardScaler,
            "minmax_scaler": MinMaxScaler,
            "log": lambda: None,  # Special case for custom log transform
            "sqrt": lambda: None,  # Special case for custom sqrt transform
            "boxcox": lambda: PowerTransformer(method="box-cox"),
            "yeo-johnson": lambda: PowerTransformer(method="yeo-johnson"),
        }

        for col in X_train.columns:
            best_auc = 0
            best_transform = None

            for name, transformer_func in transformations.items():
                # Apply transformation to the column
                try:
                    X_train_transformed = X_train.copy()
                    X_test_transformed = X_test.copy()

                    if name in ["log", "sqrt"]:  # Custom transformations
                        if name == "log":
                            X_train_transformed[col] = np.log1p(X_train[col])
                            X_test_transformed[col] = np.log1p(X_test[col])
                        elif name == "sqrt":
                            X_train_transformed[col] = np.sqrt(X_train[col].clip(lower=0))
                            X_test_transformed[col] = np.sqrt(X_test[col].clip(lower=0))
                    else:  # Scaler/transformer methods
                        transformer = transformer_func()
                        X_train_transformed[[col]] = transformer.fit_transform(X_train[[col]])
                        X_test_transformed[[col]] = transformer.transform(X_test[[col]])

                    # Train and evaluate logistic regression model
                    self.model.fit(X_train_transformed, y_train)
                    y_pred = self.model.predict_proba(X_test_transformed)[:, 1]
                    auc_score = roc_auc_score(y_test, y_pred)

                    # Update best transformation if necessary
                    if auc_score > best_auc:
                        best_auc = auc_score
                        best_transform = name
                except Exception as e:
                    print(f"Transformation {name} failed for column {col}: {e}")
                    continue

            # Log the results for the column
            self.log_dict[col] = {"best_transform": best_transform, "score": best_auc}
            # print(f"Best transformation for {col}: {best_transform} with AUC {best_auc}")
        self.save_log_dict()


    def transform(self, df_train, df_test):
        """
        Applies the best transformation for each feature based on the logged results
        and returns the transformed datasets with renamed columns.
        """
        X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
        X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]

        X_train_transformed = pd.DataFrame()
        X_test_transformed = pd.DataFrame()

        for col, info in self.log_dict.items():
            best_transform = info["best_transform"]
            new_col_name = col  # Default column name

            if best_transform in [None, "none"]:
                # Keep original column without any transformation
                X_train_transformed[col] = X_train[col]
                X_test_transformed[col] = X_test[col]
                continue

            if best_transform == "log":
                new_col_name = f"{col}_log"
                X_train_transformed[new_col_name] = np.log1p(X_train[col])
                X_test_transformed[new_col_name] = np.log1p(X_test[col])
            elif best_transform == "sqrt":
                new_col_name = f"{col}_sqrt"
                X_train_transformed[new_col_name] = np.sqrt(X_train[col].clip(lower=0))
                X_test_transformed[new_col_name] = np.sqrt(X_test[col].clip(lower=0))
            else:
                # Initialize the appropriate transformer
                transformer = {
                    "standard_scaler": StandardScaler(),
                    "minmax_scaler": MinMaxScaler(),
                    "boxcox": PowerTransformer(method="box-cox"),
                    "yeo-johnson": PowerTransformer(method="yeo-johnson"),
                }.get(best_transform)

                if transformer is None:
                    raise ValueError(f"Unknown transformation: {best_transform}")

                new_col_name = f"{col}_{best_transform}"
                transformer.fit(X_train[[col]])
                X_train_transformed[new_col_name] = transformer.transform(X_train[[col]]).flatten()
                X_test_transformed[new_col_name] = transformer.transform(X_test[[col]]).flatten()

        # Reattach the target columns to the transformed datasets
        df_train_transformed = pd.concat([X_train_transformed, y_train.reset_index(drop=True)], axis=1)
        df_test_transformed = pd.concat([X_test_transformed, y_test.reset_index(drop=True)], axis=1)

        return df_train_transformed, df_test_transformed

    def fit_transform(self, df_train, df_test):
        """
        Fits the transformer to find the best transformations and then applies them,
        returning transformed datasets with renamed columns.
        """
        self.fit(df_train, df_test)
        return self.transform(df_train, df_test)
