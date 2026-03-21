import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

from src.config import (
    TARGET_COLUMN,
    BINARY_VARS,
    ORDINAL_VARS,
    CONTINUOUS_VARS,
)

from src.visualization.eda import plot_missing_values


def print_initial_overview(df: pd.DataFrame) -> None:
    """
    Display an initial overview of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    """
    print("\nDataset overview")
    print("-" * 60)
    print(f"Shape: {df.shape}")
    print("\nColumn types:")
    print(df.dtypes)
    print("\nDescriptive statistics:")
    print(df.describe().round(2))


def print_original_target_distribution(df: pd.DataFrame) -> None:
    """
    Display the distribution of the original 3-class target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the original target.
    """
    labels = {
        0.0: "No diabetes",
        1.0: "Prediabetes",
        2.0: "Diabetes",
    }

    counts = df[TARGET_COLUMN].value_counts().sort_index()

    print("\nOriginal target distribution")
    print("-" * 60)
    for value, count in counts.items():
        print(
            f"Class {int(value)} - {labels[value]}: "
            f"{count:,} ({count / len(df) * 100:.1f}%)"
        )


def recode_target_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recode the original target variable into a binary classification target.

    Original mapping
    ----------------
    0 -> No diabetes
    1 -> Prediabetes
    2 -> Diabetes

    New mapping
    -----------
    0 -> No diabetes or Prediabetes
    1 -> Diabetes

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with recoded target.
    """
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(
        lambda value: 0 if value in [0.0, 1.0] else 1
    )
    print("Distribution après recodage :")
    print_binary_target_distribution(df)


    return df


def print_binary_target_distribution(df: pd.DataFrame) -> None:
    """
    Display the distribution of the binary target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the binary target.
    """
    counts = df[TARGET_COLUMN].value_counts().sort_index()

    print("\nBinary target distribution")
    print("-" * 60)
    for value, count in counts.items():
        label = (
            "No diabetes or Prediabetes"
            if value == 0
            else "Diabetes"
        )
        print(
            f"Class {value} - {label}: "
            f"{count:,} ({count / len(df) * 100:.1f}%)"
        )


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataset into features and target.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Features and target.
    """
    features = [column for column in df.columns if column != TARGET_COLUMN]
    x = df[features].copy()
    y = df[TARGET_COLUMN].copy()

    return x, y


def cast_feature_types(x: pd.DataFrame) -> pd.DataFrame:
    """
    Cast feature columns to memory-efficient and semantically coherent dtypes.

    Parameters
    ----------
    x : pd.DataFrame
        Feature matrix.

    Returns
    -------
    pd.DataFrame
        Typed feature matrix.
    """
    x = x.copy()

    for column in BINARY_VARS:
        x[column] = x[column].astype("int8")

    for column in ORDINAL_VARS + CONTINUOUS_VARS:
        x[column] = x[column].astype("float32")

    return x


def check_missing_values(x: pd.DataFrame) -> None:
    """
    Print missing values report.

    Parameters
    ----------
    x : pd.DataFrame
        Feature matrix.
    """
    # Visualisation des valeurs manquantes

    plot_missing_values(x)

    missing_values = x.isnull().sum()
    total_missing = missing_values.sum()

    print("\nMissing values report")
    print("-" * 60)

    if total_missing == 0:
        print("No missing values detected.")
    else:
        print(f"Total missing values: {total_missing:,}")
        print(missing_values[missing_values > 0])


def clean_dataset(
    x: pd.DataFrame, 
    y: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Remove duplicated rows and rows containing missing values
    from the reconstructed dataset composed of X and y.

    Parameters
    ----------
    x : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Cleaned feature matrix and target vector.
    """

    df_work = x.copy()
    df_work[TARGET_COLUMN] = y.values

    print("\nDataset cleaning report")
    print("-" * 60)
    print(f"Initial shape: {df_work.shape}")

    # -------------------------------------------------
    # Missing values
    # -------------------------------------------------

    plot_missing_values(df_work)

    missing_per_column = df_work.isnull().sum()
    total_missing = missing_per_column.sum()

    if total_missing == 0:
        print("No missing values detected.")
    else:
        rows_with_missing = df_work.isnull().any(axis=1).sum()

        print(f"Total missing values: {total_missing:,}")
        print(f"Rows containing missing values: {rows_with_missing:,}")

        df_work = df_work.dropna().reset_index(drop=True)

        print(f"Removed rows with missing values: {rows_with_missing:,}")

    # -------------------------------------------------
    # Duplicates
    # -------------------------------------------------

    duplicates_count = df_work.duplicated().sum()

    print("\nDuplicate rows report")
    print("-" * 60)
    print(f"Detected duplicates: {duplicates_count:,}")

    if duplicates_count > 0:
        df_work = df_work.drop_duplicates().reset_index(drop=True)
        print(f"Removed duplicates: {duplicates_count:,}")
    else:
        print("No duplicates found.")

    print(f"Final dataset shape: {df_work.shape}")

    # -------------------------------------------------
    # Split back to features and target
    # -------------------------------------------------

    x_clean = df_work.drop(columns=[TARGET_COLUMN])
    y_clean = df_work[TARGET_COLUMN]

    return x_clean, y_clean
    