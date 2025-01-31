import os
import pandas as pd
from .dataloader import load_data, save_data
from .numerical_transformer import NumericalTransformer
from sklearn.preprocessing import OrdinalEncoder
import time
from .optimal_binner import OptimalBinner


class FeatureEngineer:
    def __init__(self):
        """
        Initialize the FeatureEngineer class.

        Attributes:
            repo_root (str): The root directory of the repository.
            data_dir (str): The directory containing the processed data files.
            train_data_path (str): The path to the processed training data file.
            test_data_path (str): The path to the processed test data file.
            df_train (pandas.DataFrame): The original training data.
            df_test (pandas.DataFrame): The original test data.
            df_train_featured (pandas.DataFrame): The transformed training data.
            df_test_featured (pandas.DataFrame): The transformed test data.
            numerical_features (list): List of numerical feature column names.
            categorical_features (list): List of categorical feature column names.
            ordinal_features (list): List of ordinal categorical feature column names.
            nominal_features (list): List of nominal categorical feature column names.
        """
        self.repo_root = self._get_repository_root()
        self.data_dir = os.path.join(self.repo_root, "data", "processed")

        train_filename = "train_processed_data.parquet"
        # test_filename = "test_processed_data-1.parquet"
        test_filename = "20241209-1444-test_processed_data-1.parquet"

        self.train_data_path = os.path.join(self.data_dir, train_filename)
        self.test_data_path = os.path.join(self.data_dir, test_filename)

        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.df_train_featured = pd.DataFrame()
        self.df_test_featured = pd.DataFrame()

        self.target = 'default_12month'
        self.numerical_features = []
        self.categorical_features = []
        self.ordinal_features = []
        self.nominal_features = []
        self.datetime_features = []

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

    def load_datasets(self, verbose=True):
        """
        Load the training and test datasets.

        Args:
            verbose (bool): If True, prints a message after loading the datasets.
        """
        self.df_train = load_data(self.train_data_path)
        self.df_test = load_data(self.test_data_path)
        if verbose:
            print("FeatureEngineer: Datasets loaded successfully.")

    def save_datasets(self, file_format="parquet", verbose=True):
        """
        Save the processed datasets.

        Args:
            file_format (str): The file format for saving datasets (default is "parquet").
            verbose (bool): If True, prints a message after saving the datasets.
        """
        destination_dir = os.path.join(self.repo_root, "data", "featured")
        save_data(self.df_train_featured, file_name="train_featured_data", file_directory=destination_dir, file_format=file_format)
        save_data(self.df_test_featured, file_name="test_featured_data-1", file_directory=destination_dir, file_format=file_format)
        # save_data(self.df_test_featured, file_name="20241209-1355-test_processed_data-2", file_directory=destination_dir, file_format=file_format)

        if verbose:
            print("FeatureEngineer: Featured datasets saved successfully.")

    def identify_features(self, verbose=True):
        """
        Identify numerical, ordinal, and nominal features in the datasets.

        Args:
            verbose (bool): If True, prints the identified features.
        """
        # Identify datetime features
        self.datetime_features = [
            'pms_i_ymd',
            'date_of_birth',
        ]

        # Identify numerical features
        # self.numerical_features = self.df_train.select_dtypes(include=["number"]).columns.tolist()
        self.numerical_features = [
            "number_of_children",
            "number_of_resident",
            "living_period_year",
            "living_period_month",
            "c_number_of_employee",
            "c_monthly_salary",
            "c_number_of_working_year",
            "c_number_of_working_month",
            "r_expected_credit_limit",
            "r_allloan_case",
            "r_allloan_amount",
            "r_additional_income",
            "r_spouse_income",
            "Overdraft_count",
            "Personal Loan_count",
            "Mortgage_count",
            "Credit Card_count",
            "Automobile installment purchase_count",
            "Other installment purchase_count",
            "Loan for agriculture_count",
            "Other Loans_count",
            "Overdraft_balance",
            "Personal Loan_balance",
            "Mortgage_balance",
            "Credit Card_balance",
            "Automobile installment purchase_balance",
            "Other installment purchase_balance",
            "Loan for agriculture_balance",
            "Other Loans_balance",
            "Bank inquiry_count",
            "Consumer finance inquiry_count",
            "Leasing enquiry_count"
        ]

        # Identify categorical features
        # self.categorical_features = self.df_train.select_dtypes(include=["object", "category"]).columns.tolist()
        self.categorical_features = list(set(self.df_train.columns) - set(self.numerical_features) - set(self.datetime_features))

        # Separate ordinal and nominal categorical features
        self.ordinal_features = [
            'date_of_birth_week',
            'c_position',
        ]
        self.nominal_features = list(set(self.categorical_features) - set(self.ordinal_features))

        if verbose:
            print(f"Numerical features ({len(self.numerical_features)}): {self.numerical_features}")
            print(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
            print(f"Ordinal features ({len(self.ordinal_features)}): {self.ordinal_features}")
            print(f"Nominal features ({len(self.nominal_features)}): {self.nominal_features}")
            print(f"Datetime features ({len(self.datetime_features)}): {self.datetime_features}")
            print(f"All==Num+Cat+Datetime: {len(self.df_train.columns) == len(self.numerical_features) + len(self.categorical_features) + len(self.datetime_features)}")
            print(f"Cat==Ord+Nom: {len(self.categorical_features) == len(self.ordinal_features) + len(self.nominal_features)}")

    def drop_features(self, verbose=True):
        """
        Drop features that are not needed for modeling.

        Updates:
            df_train (pandas.DataFrame): Updated with dropped features.
            df_test (pandas.DataFrame): Updated with dropped features.
        """
        # Drop features that are not needed for modeling
        features_to_drop = [
            "Area",
            "Shop Name",
            "date_of_birth_week",
            "tel_category", # very low variety
            "living_period_year"
            "c_monthly_salary",
            "c_number_of_working_year",
            "c_salary_payment_methods",
            "c_date_of_salary_payment",
            "r_additional_income",
            "r_spouse_income",
            "r_generalcode1",
            "r_generalcode2",
            "r_generalcode4",
            "postal_code",
            "c_postal_code"
        ]
        features_to_drop.extend(self.datetime_features)

        self.df_train = self.df_train.drop(columns=features_to_drop, errors="ignore")
        self.df_test = self.df_test.drop(columns=features_to_drop, errors="ignore")

        # update features
        self.numerical_features = list(set(self.numerical_features) - set(features_to_drop))
        self.categorical_features = list(set(self.categorical_features) - set(features_to_drop))
        self.ordinal_features = list(set(self.ordinal_features) - set(features_to_drop))
        self.nominal_features = list(set(self.nominal_features) - set(features_to_drop))

        if verbose:
            print(f"Dropped features: {features_to_drop}.")

    # def transform_numerical_features(self):
    #     """
    #     Transform numerical features using the NumericalTransformer.

    #     Updates:
    #         df_train_featured (pandas.DataFrame): Updated with transformed numerical features.
    #         df_test_featured (pandas.DataFrame): Updated with transformed numerical features.
    #     """
    #     if not self.numerical_features:
    #         raise ValueError("Numerical features have not been identified. Run identify_features() first.")

    #     df_train_num = self.df_train[self.numerical_features]
    #     df_test_num = self.df_test[self.numerical_features]

    #     numerical_transformer = NumericalTransformer()
    #     df_train_num_transformed, df_test_num_transformed = numerical_transformer.fit_transform(df_train_num, df_test_num)

    #     # Update featured datasets with transformed numerical features
    #     self.df_train_featured = pd.concat([self.df_train_featured, df_train_num_transformed], axis=1)
    #     self.df_test_featured = pd.concat([self.df_test_featured, df_test_num_transformed], axis=1)

    # def transform_ordinal_features(self):
    #     """
    #     Encode categorical features as an integer array.
    #     The outputs are discrete numbers between 0 to n_categories - 1.
    #     """
    #     # TODO: Map enc to pipeline.
    #     enc = OrdinalEncoder()
    #     df_train_ordinals = enc.fit_transform(self.df_train[self.ordinal_features])
    #     df_test_ordinals = enc.fit_transform(self.df_test[self.ordinal_features])

    #     df_train_ordinals = pd.DataFrame(df_train_ordinals, columns=self.ordinal_features)
    #     df_test_ordinals = pd.DataFrame(df_test_ordinals, columns=self.ordinal_features)

    #     self.df_train_featured = pd.concat([self.df_train_featured, df_train_ordinals], axis=1)
    #     self.df_test_featured = pd.concat([self.df_test_featured, df_test_ordinals], axis=1)

    # def transform_nominal_features(self):
    #     pass

    # def transform_categorical_features(self):
    #     """
    #     One-hot encode all categorical features and ensure alignment between train and test datasets.
    #     """
    #     # Generate one-hot encodings for train and test sets
    #     df_train_dummies = pd.get_dummies(self.df_train[self.categorical_features])
    #     df_test_dummies = pd.get_dummies(self.df_test[self.categorical_features])

    #     # Align columns between train and test sets (to handle mismatched categories)
    #     df_train_dummies, df_test_dummies = df_train_dummies.align(df_test_dummies, axis=1, fill_value=0)

    #     # Concatenate the encoded features with the respective featured DataFrames
    #     self.df_train_featured = pd.concat([self.df_train_featured, df_train_dummies], axis=1)
    #     self.df_test_featured = pd.concat([self.df_test_featured, df_test_dummies], axis=1)

    def transform_categorical_features(self):
        # self.transform_ordinal_features()
        # self.transform_nominal_features()
        pass

    def transform(self, verbose=True):
        """
        Transform the datasets by applying optimal binning for numerical and categorical features.
        """
        # TODO: Implement ordinal and nominal feature transformations
        # self.transform_ordinal_features()
        # self.transform_nominal_features()

        # Binning every features
        if verbose:
                print("Initializing OptimalBinner...")
        optimal_binner = OptimalBinner(target=self.target)

        if verbose:
            print("Fitting and Transforming Training Data...")
        self.df_train_featured, self.df_test_featured = optimal_binner.fit_transform(
            df_train=self.df_train,
            df_test=self.df_test,
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features
        )

        if verbose:
            print("Optimal Binning completed.")

    def new_features(self, verbose=True):
        """
        Create new features for both training and test datasets.

        Args:
            verbose (bool): If True, prints a message upon completion.
        """
        def _create_features(df):
            """
            Helper function to create features in a given DataFrame.
            """
            # mathematical
            df['employment_duration'] = df['c_number_of_working_year'] + df['c_number_of_working_month'] / 12
            df['residence_duration'] = df['living_period_year'] + df['living_period_month'] / 12
            df['total_income'] = df['c_monthly_salary'] + df['r_additional_income'] + df['r_spouse_income']
            df['credit_utilization'] = df['r_expected_credit_limit'] / df['total_income']
            df['dependency_ratio'] = df['c_monthly_salary'] / df['number_of_children']
            df['living_period_month'] = df['living_period_year'] * 12 + df['living_period_month']
            df['c_number_of_working_month'] = df['c_number_of_working_year'] * 12 + df['c_number_of_working_month']
            # datetime
            if 'pms_i_ymd' in df.columns and 'date_of_birth' in df.columns:
                df['pms_i_ymd'] = pd.to_datetime(df['pms_i_ymd'], errors='coerce')
                df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
                df['c_age_app'] = (df['pms_i_ymd'] - df['date_of_birth']).dt.days / 365.25
            return df

        # Apply feature creation to both train and test datasets
        self.df_train = _create_features(self.df_train)
        self.df_test = _create_features(self.df_test)

        if verbose:
            print("New features created successfully.")

    def preprocess(self, verbose=True):
        """
        Preprocess the datasets by modifying specific features based on requirements.

        Args:
            verbose (bool): If True, prints messages indicating preprocessing steps.
        """
        def _preprocess_features(df):
            """
            Helper function to preprocess features in a given DataFrame.
            """
            # Group 'F1' and 'F2' in the gender feature as 'F'
            if 'gender' in df.columns:
                df['gender'] = df['gender'].replace({'F1': 'F', 'F2': 'F'})

            # Set values > 8000 in 'r_allloan_case' to NaN
            if 'r_allloan_case' in df.columns:
                df.loc[df['r_allloan_case'] > 8000, 'r_allloan_case'] = pd.NA

            return df

        # Apply preprocessing to both train and test datasets
        self.df_train = _preprocess_features(self.df_train)
        self.df_test = _preprocess_features(self.df_test)

        if verbose:
            print("Preprocessing completed: Gender feature grouped.")
            print("Preprocessing completed: r_allloan_case values > 8000 set to NaN.")

    # def map_provinces(self, column_name, verbose=True):
    #     """
    #     Map province codes to province names using a predefined dictionary.

    #     Args:
    #         column_name (str): The name of the column containing province codes.
    #         verbose (bool): If True, prints a message after mapping is completed.

    #     Updates:
    #         Maps province codes in the specified column to their corresponding names.
    #     """
    #     # !Fix
    #     provinces = {
    #         "10": "Bangkok",
    #         "11": "Samut Prakan",
    #         "12": "Nonthaburi",
    #         "13": "Pathum Thani",
    #         "14": "Phra Nakhon Si Ayutthaya",
    #         "15": "Ang Thong",
    #         "16": "Loburi",
    #         "17": "Sing Buri",
    #         "18": "Chai Nat",
    #         "19": "Saraburi",
    #         "20": "Chon Buri",
    #         "21": "Rayong",
    #         "22": "Chanthaburi",
    #         "23": "Trat",
    #         "24": "Chachoengsao",
    #         "25": "Prachin Buri",
    #         "26": "Nakhon Nayok",
    #         "27": "Sa Kaeo",
    #         "30": "Nakhon Ratchasima",
    #         "31": "Buri Ram",
    #         "32": "Surin",
    #         "33": "Si Sa Ket",
    #         "34": "Ubon Ratchathani",
    #         "35": "Yasothon",
    #         "36": "Chaiyaphum",
    #         "37": "Amnat Charoen",
    #         "39": "Nong Bua Lam Phu",
    #         "40": "Khon Kaen",
    #         "41": "Udon Thani",
    #         "42": "Loei",
    #         "43": "Nong Khai",
    #         "44": "Maha Sarakham",
    #         "45": "Roi Et",
    #         "46": "Kalasin",
    #         "47": "Sakon Nakhon",
    #         "48": "Nakhon Phanom",
    #         "49": "Mukdahan",
    #         "50": "Chiang Mai",
    #         "51": "Lamphun",
    #         "52": "Lampang",
    #         "53": "Uttaradit",
    #         "54": "Phrae",
    #         "55": "Nan",
    #         "56": "Phayao",
    #         "57": "Chiang Rai",
    #         "58": "Mae Hong Son",
    #         "60": "Nakhon Sawan",
    #         "61": "Uthai Thani",
    #         "62": "Kamphaeng Phet",
    #         "63": "Tak",
    #         "64": "Sukhothai",
    #         "65": "Phitsanulok",
    #         "66": "Phichit",
    #         "67": "Phetchabun",
    #         "70": "Ratchaburi",
    #         "71": "Kanchanaburi",
    #         "72": "Suphan Buri",
    #         "73": "Nakhon Pathom",
    #         "74": "Samut Sakhon",
    #         "75": "Samut Songkhram",
    #         "76": "Phetchaburi",
    #         "77": "Prachuap Khiri Khan",
    #         "80": "Nakhon Si Thammarat",
    #         "81": "Krabi",
    #         "82": "Phangnga",
    #         "83": "Phuket",
    #         "84": "Surat Thani",
    #         "85": "Ranong",
    #         "86": "Chumphon",
    #         "90": "Songkhla",
    #         "91": "Satun",
    #         "92": "Trang",
    #         "93": "Phatthalung",
    #         "94": "Pattani",
    #         "95": "Yala",
    #         "96": "Narathiwat",
    #         "97": "Buogkan"
    #     }

    #     # Map the provinces for both train and test datasets
    #     self.df_train[column_name] = self.df_train[column_name].map(provinces)
    #     self.df_test[column_name] = self.df_test[column_name].map(provinces)

    #     if verbose:
    #         print(f"Province mapping completed for column: {column_name}")

    def run(self):
        """
        Execute the feature engineering pipeline.
        """
        time_start = time.time()
        print("Feature engineering pipeline started...")

        self.load_datasets()
        self.identify_features()
        self.new_features()
        self.preprocess()
        self.drop_features()
        self.transform()
        self.save_datasets()

        time_end = time.time()
        print(f"Feature engineering pipeline completed in {time_end - time_start:.2f} seconds.")

if __name__ == "__main__":
    feature_engineer = FeatureEngineer()
    feature_engineer.run()
