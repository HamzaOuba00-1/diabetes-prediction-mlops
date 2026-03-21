import pandas as pd

from src.config import ORIGINAL_TARGET_COLUMN, TARGET_COLUMN


def load_raw_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the raw diabetes dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the raw CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with normalized target column name.
    """
    df = pd.read_csv(file_path)

    if ORIGINAL_TARGET_COLUMN in df.columns:
        df = df.rename(columns={ORIGINAL_TARGET_COLUMN: TARGET_COLUMN})

    return df