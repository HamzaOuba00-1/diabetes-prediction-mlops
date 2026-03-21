import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_datasets(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    columns_to_scale: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Apply MinMax scaling on selected columns.

    The scaler is fitted only on the training set to avoid data leakage.

    Parameters
    ----------
    x_train : pd.DataFrame
        Training features.
    x_val : pd.DataFrame
        Validation features.
    x_test : pd.DataFrame
        Test features.
    columns_to_scale : list[str]
        Columns to normalize.

    Returns
    -------
    tuple
        Scaled train, validation, test sets and fitted scaler.
    """
    scaler = MinMaxScaler()

    x_train_scaled = x_train.copy()
    x_val_scaled = x_val.copy()
    x_test_scaled = x_test.copy()

    x_train_scaled[columns_to_scale] = scaler.fit_transform(
        x_train[columns_to_scale].astype(float)
    )
    x_val_scaled[columns_to_scale] = scaler.transform(
        x_val[columns_to_scale].astype(float)
    )
    x_test_scaled[columns_to_scale] = scaler.transform(
        x_test[columns_to_scale].astype(float)
    )

    return x_train_scaled, x_val_scaled, x_test_scaled, scaler


def print_scaling_summary(x_train_scaled: pd.DataFrame, columns_to_scale: list[str]) -> None:
    """
    Print min and max values for scaled columns on the training set.
    """
    print("\nScaling summary on training set")
    print("-" * 60)
    print(x_train_scaled[columns_to_scale].describe().loc[["min", "max"]].round(3))