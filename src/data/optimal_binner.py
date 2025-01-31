from optbinning import OptimalBinning
import pandas as pd

class OptimalBinner:
    def __init__(self, target):
        if not target:
            raise ValueError("Target column name must be specified.")
        self.target = target
        self.binning_metadata = {"numerical": {}, "categorical": {}}

    def _validate_target_column(self, df):
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataset.")

    def _fit_numerical(self, df, numerical_features):
        self._validate_target_column(df)
        metadata = {}
        for feature in numerical_features:
            x = df[feature]
            y = df[self.target]
            optb = OptimalBinning(name=feature, dtype="numerical", solver="cp")
            optb.fit(x, y)
            metadata[feature] = optb
        self.binning_metadata["numerical"] = metadata

    def _fit_categorical(self, df, categorical_features):
        self._validate_target_column(df)
        metadata = {}
        for feature in categorical_features:
            x = df[feature]
            y = df[self.target]
            optb = OptimalBinning(name=feature, dtype="categorical", solver="mip", cat_cutoff=0.1)
            optb.fit(x, y)
            metadata[feature] = optb
        self.binning_metadata["categorical"] = metadata

    def _transform_numerical(self, df, verbose=True):
        df_transformed = pd.DataFrame(index=df.index)
        for feature, optb in self.binning_metadata["numerical"].items():
            if feature in df.columns:
                if feature == self.target:
                    df_transformed[feature] = df[feature]
                else:
                    df_transformed[f"{feature}_WOE"] = optb.transform(df[feature], metric="woe")

                if verbose:
                    print(f"Binning '{feature}' to '{feature}_WOE'.")
        return df_transformed

    def _transform_categorical(self, df):
        df_transformed = pd.DataFrame(index=df.index)
        for feature, optb in self.binning_metadata["categorical"].items():
            if feature in df.columns:
                if feature == self.target:
                    df_transformed[feature] = df[feature]
                else:
                    df_transformed[f"{feature}_WOE"] = optb.transform(df[feature], metric="woe")

                if True:
                    print(f"Binning '{feature}' to '{feature}_WOE'.")
        return df_transformed

    def fit(self, df_train, numerical_features, categorical_features):
        """
        Fit binning models using labeled training data.
        """
        self._fit_numerical(df_train, numerical_features)
        self._fit_categorical(df_train, categorical_features)

    def transform(self, df, numerical_features, categorical_features):
        """
        Transform data using fitted binning models.
        """
        df_num_binned = self._transform_numerical(df)
        df_cat_binned = self._transform_categorical(df)
        return pd.concat([df_num_binned, df_cat_binned], axis=1)

    def fit_transform(self, df_train, df_test, numerical_features, categorical_features):
        """
        Fit and transform training data, and only transform test data.
        """
        self.fit(df_train, numerical_features, categorical_features)
        df_train_transformed = self.transform(df_train, numerical_features, categorical_features)
        df_test_transformed = self.transform(df_test, numerical_features, categorical_features)
        return df_train_transformed, df_test_transformed
