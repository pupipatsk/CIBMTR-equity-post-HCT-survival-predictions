import os
import pandas as pd
from .dataloader import load_data, save_data

class Preprocessor:
    def __init__(self):
        """
        Initialize the Preprocessor class.

        Attributes:
            repo_root (str): The root directory of the repository.
            raw_data_dir (str): The directory containing the raw data files.
            train_data_path (str): The path to the training data file.
            test_data_path (str): The path to the test data file.
            df_train (pandas.DataFrame): The training data.
            df_test (pandas.DataFrame): The test data but only input features, no label. (X_test)
        """
        self.repo_root = self._get_repository_root()
        self.raw_data_dir = self.get_raw_data_directory()

        self.train_data_path = os.path.join(self.raw_data_dir, "train_data.parquet")
        self.test_data_path = os.path.join(self.raw_data_dir, "private_dataset_without_gt_1.csv")

        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()

        self.target = "default_12month"

    @staticmethod
    def _get_repository_root() -> str:
        current_file_path = os.path.abspath(__file__)
        repository_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        return repository_root

    def get_raw_data_directory(self) -> str:
        raw_data_dir = os.path.join(self.repo_root, "data", "raw", "parquet")
        return raw_data_dir

    def load_datasets(self, verbose=True):
        self.df_train = load_data(self.train_data_path)
        self.df_test = load_data(self.test_data_path)
        if verbose:
            print("Preprocessor: Datasets loaded successfully.")

    def apply_dtypes(self):
        dtypes_dict = {
            # ID columns
            "ID": "string",

            # Date columns
            "pms_i_ymd": "datetime64[ns]",
            "date_of_birth": "datetime64[ns]",

            # Nominal columns
            "Area": "object",
            "Province": "object",
            "Shop Name": "object",
            "gender": "object",
            "marital_status": "object",
            "postal_code": "object",
            "tel_category": "object",
            "type_of_residence": "object",
            "c_postal_code": "object",
            "c_business_type": "object",
            "c_occupation": "object",
            "c_employment_status": "object",
            "c_salary_payment_methods": "object",
            "c_date_of_salary_payment": "object",
            "media": "object",
            "place_for_sending_information": "object",
            "r_propose": "object",
            "r_generalcode1": "object",
            "r_generalcode2": "object",
            "r_generalcode3": "object",
            "r_generalcode4": "object",
            "r_generalcode5": "object",
            "apply": "object",

            # Ordinal columns
            "date_of_birth_week": "int64",  # Assuming integers for ordinal values
            "c_position": "int64",  # Assuming integers for ordinal values

            # Numerical columns
            "number_of_children": "float64",
            "number_of_resident": "float64",
            "living_period_year": "float64",
            "living_period_month": "float64",
            "c_number_of_employee": "float64",
            "c_monthly_salary": "float64",
            "c_number_of_working_year": "float64",
            "c_number_of_working_month": "float64",
            "r_expected_credit_limit": "float64",
            "r_allloan_case": "float64",
            "r_allloan_amount": "float64",
            "r_additional_income": "float64",
            "r_spouse_income": "float64",
            "Overdraft_count": "float64",
            "Personal Loan_count": "float64",
            "Mortgage_count": "float64",
            "Credit Card_count": "float64",
            "Automobile installment purchase_count": "float64",
            "Other installment purchase_count": "float64",
            "Loan for agriculture_count": "float64",
            "Other Loans_count": "float64",
            "Overdraft_balance": "float64",
            "Personal Loan_balance": "float64",
            "Mortgage_balance": "float64",
            "Credit Card_balance": "float64",
            "Automobile installment purchase_balance": "float64",
            "Other installment purchase_balance": "float64",
            "Loan for agriculture_balance": "float64",
            "Other Loans_balance": "float64",
            "Bank inquiry_count": "float64",
            "Consumer finance inquiry_count": "float64",
            "Leasing enquiry_count": "float64",

            # Target column
            "default_12month": "bool",  # Assuming binary target
        }
        self.df_train = self.df_train.astype(dtypes_dict)
        self.df_test = self.df_test.astype(dtypes_dict)

    @staticmethod
    def drop_unique_cols(df) -> pd.DataFrame:
        """
        TODO: check col to drop
        """
        unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
        if unique_cols:
            df = df.drop(columns=unique_cols)
            print(f"Dropped unique columns: {unique_cols}.")
        else:
            print("No unique columns to drop.")
        return df

    def drop_unique(self):
        self.df_train = self.drop_unique_cols(self.df_train)
        self.df_test = self.drop_unique_cols(self.df_test)

    def clean(self):
        """
        Clean the dataset by replacing invalid or missing values.
        """
        # Replace '*' with NaN in 'Leasing enquiry_count'
        self.df_train['Leasing enquiry_count'] = self.df_train['Leasing enquiry_count'].replace('*', pd.NA)
        self.df_test['Leasing enquiry_count'] = self.df_test['Leasing enquiry_count'].replace('*', pd.NA)
        # Convert the column to numeric if required (optional)
        self.df_train['Leasing enquiry_count'] = pd.to_numeric(self.df_train['Leasing enquiry_count'], errors='coerce')
        self.df_test['Leasing enquiry_count'] = pd.to_numeric(self.df_test['Leasing enquiry_count'], errors='coerce')
        print("Cleaned 'Leasing enquiry_count' column by replacing '*' with NaN.")

    def preprocessing(self):
        self.load_datasets()
        self.clean()
        # self.apply_dtypes()
        self.drop_unique()
        self.save_datasets()

    def save_datasets(self, file_format="parquet"):
        destination_dir = os.path.join(self.repo_root, "data", "processed")
        save_data(self.df_train, file_name="train_processed_data", file_directory=destination_dir, file_format=file_format)
        save_data(self.df_test, file_name="test_processed_data-1", file_directory=destination_dir, file_format=file_format)
        print("Datasets saved successfully.")

def main():
    preprocessor = Preprocessor()
    preprocessor.preprocessing()
    preprocessor.save_datasets()

if __name__ == "__main__":
    main()
