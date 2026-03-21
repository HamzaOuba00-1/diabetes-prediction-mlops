from pathlib import Path
import pandas as pd

from src.config import TARGET_COLUMN


def save_processed_datasets(
    output_dir: Path,
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    full_clean_dataset: pd.DataFrame,
) -> None:
    """
    Save processed train, validation, test, and full cleaned datasets.

    Parameters
    ----------
    output_dir : Path
        Destination directory.
    x_train, x_val, x_test : pd.DataFrame
        Processed feature matrices.
    y_train, y_val, y_test : pd.Series
        Corresponding targets.
    full_clean_dataset : pd.DataFrame
        Full cleaned dataset before split.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = x_train.copy()
    train_df[TARGET_COLUMN] = y_train.values

    val_df = x_val.copy()
    val_df[TARGET_COLUMN] = y_val.values

    test_df = x_test.copy()
    test_df[TARGET_COLUMN] = y_test.values

    train_df.to_csv(output_dir / "train_cleaned.csv", index=False)
    val_df.to_csv(output_dir / "val_cleaned.csv", index=False)
    test_df.to_csv(output_dir / "test_cleaned.csv", index=False)
    full_clean_dataset.to_csv(output_dir / "dataset_cleaned_full.csv", index=False)

    print("\nSaved files")
    print("-" * 60)
    print(f"- {output_dir / 'train_cleaned.csv'}")
    print(f"- {output_dir / 'val_cleaned.csv'}")
    print(f"- {output_dir / 'test_cleaned.csv'}")
    print(f"- {output_dir / 'dataset_cleaned_full.csv'}")