import os
import numpy as np
import pandas as pd
import time


def optimize_memory_usage(df, verbose=True) -> pd.DataFrame:
    """
    Optimize the memory usage of a pandas DataFrame by using the smallest possible data types.

    Args:
        df (pandas.DataFrame): The input DataFrame to be optimized.

    Returns:
        pandas.DataFrame: The optimized DataFrame with reduced memory usage.
    """
    initial_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        # Numerical
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            # Integer
            if str(col_type)[:3] == "int":
                # Unsigned Integer
                if c_min >= 0:
                    if c_max < np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    elif c_max < np.iinfo(np.uint64).max:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
            # Float
            else:
                if (c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        # Categorical
        else:
            df[col] = df[col].astype("object")

    optimized_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print(f"Memory usage: Before={initial_mem:.2f}MB -> After={optimized_mem:.2f}MB, Decreased by {100 * (initial_mem - optimized_mem) / initial_mem:.1f}%")

    return df

def load_data(file_path) -> pd.DataFrame:
    """
    Load a pandas DataFrame from a CSV or Parquet file and optimize its memory usage.

    Args:
        file_path (str): The path to the data file (CSV or Parquet).

    Returns:
        pandas.DataFrame: The loaded and optimized DataFrame.

    Raises:
        ValueError: If the file extension is not supported.
    """

    # Load data
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Parquet file.")

    # Optimize memory usage
    df_optimized = optimize_memory_usage(df)

    print("Data loaded successfully.")
    return df_optimized

def save_data(df, file_name, file_directory, file_format="parquet"):
    """
    Save a pandas DataFrame to a CSV or Parquet file.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved.
        file_name (str): The name of the output file.
        file_directory (str): The directory where the output file will be saved.
        file_format (str): The format of the output file (CSV or Parquet).

    Raises:
        ValueError: If the file format is not supported.
    """
    time_now = time.strftime("%Y%m%d-%H%M")
    filename = f"{time_now}-{file_name}.{file_format}"
    filepath = os.path.join(file_directory, filename)

    # save
    if file_format == "parquet":
        df.to_parquet(filepath)
    elif file_format == "csv":
        df.to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
