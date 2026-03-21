import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE


def split_train_validation_test(
    x: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
    valid_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split the dataset into train, validation, and test sets using stratification.

    Parameters
    ----------
    x : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    train_size : float, default=0.70
        Proportion of the dataset for training.
    valid_size : float, default=0.15
        Proportion of the dataset for validation.
    test_size : float, default=0.15
        Proportion of the dataset for test.

    Returns
    -------
    tuple
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if round(train_size + valid_size + test_size, 2) != 1.00:
        raise ValueError("train_size + valid_size + test_size must equal 1.0")

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=(1 - train_size),
        stratify=y,
        random_state=RANDOM_STATE,
    )

    relative_test_size = test_size / (valid_size + test_size)

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    n_total = len(x)
    print(f"{'Split':<12} {'Lignes':>8}  {'%':>6}  {'Diabétiques':>12}  {'Taux':>6}")
    print("-" * 50)
    for name, X_s, y_s in [("Train", x_train, y_train), ("Validation", x_val, y_val), ("Test", x_test, y_test)]:
        print(f"{name:<12} {len(X_s):>8,}  {len(X_s)/n_total*100:>5.1f}%  {y_s.sum():>12,}  {y_s.mean()*100:>5.1f}%")

    return x_train, x_val, x_test, y_train, y_val, y_test


def print_split_summary(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Print summary statistics for train, validation, and test splits.
    """
    total_rows = len(x_train) + len(x_val) + len(x_test)

    print("\nDataset split summary")
    print("-" * 72)
    print(f"{'Split':<12} {'Rows':>10} {'Share':>10} {'Positive':>12} {'Rate':>10}")
    print("-" * 72)

    for name, x_part, y_part in [
        ("Train", x_train, y_train),
        ("Validation", x_val, y_val),
        ("Test", x_test, y_test),
    ]:
        print(
            f"{name:<12} "
            f"{len(x_part):>10,} "
            f"{len(x_part) / total_rows * 100:>9.1f}% "
            f"{int(y_part.sum()):>12,} "
            f"{y_part.mean() * 100:>9.1f}%"
        )