import pandas as pd

from src.config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    COLUMNS_TO_SCALE,
    TRAIN_SIZE,
    VALID_SIZE,
    TEST_SIZE,
)
from src.data.load_data import load_raw_dataset
from src.data.preprocessing import (
    print_initial_overview,
    print_original_target_distribution,
    recode_target_to_binary,
    print_binary_target_distribution,
    split_features_target,
    cast_feature_types,
    check_missing_values,
    remove_duplicates,
)
from src.data.splitting import split_train_validation_test, print_split_summary
from src.data.scaling import scale_datasets, print_scaling_summary
from src.data.save_data import save_processed_datasets


def main() -> None:
    """
    Execute the complete preprocessing pipeline for Sprint 1.
    """
    print("Starting preprocessing pipeline...")

    # 1. Load raw dataset
    df_raw = load_raw_dataset(str(RAW_DATA_FILE))

    # 2. Initial exploration logs
    print_initial_overview(df_raw)
    print_original_target_distribution(df_raw)

    # 3. Recode target to binary classification
    df_binary = recode_target_to_binary(df_raw)
    print_binary_target_distribution(df_binary)

    # 4. Split features and target
    x, y = split_features_target(df_binary)

    print("\nFeature / target split")
    print("-" * 60)
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Number of features: {x.shape[1]}")
    print(f"Features: {', '.join(x.columns)}")

    # 5. Cast dtypes
    x = cast_feature_types(x)

    print("\nFeature types after casting")
    print("-" * 60)
    print(x.dtypes)

    # 6. Missing values check
    check_missing_values(x)

    # 7. Duplicate removal
    x_clean, y_clean = remove_duplicates(x, y)

    # Full cleaned dataset before split
    full_clean_df = x_clean.copy()
    full_clean_df[TARGET_COLUMN] = y_clean.values

    # 8. Split datasets
    x_train, x_val, x_test, y_train, y_val, y_test = split_train_validation_test(
        x_clean,
        y_clean,
        train_size=TRAIN_SIZE,
        valid_size=VALID_SIZE,
        test_size=TEST_SIZE,
    )
    print_split_summary(x_train, x_val, x_test, y_train, y_val, y_test)

    # 9. Scale selected features
    x_train_scaled, x_val_scaled, x_test_scaled, scaler = scale_datasets(
        x_train,
        x_val,
        x_test,
        columns_to_scale=COLUMNS_TO_SCALE,
    )
    print_scaling_summary(x_train_scaled, COLUMNS_TO_SCALE)

    # 10. Save processed outputs
    save_processed_datasets(
        output_dir=PROCESSED_DATA_DIR,
        x_train=x_train_scaled,
        x_val=x_val_scaled,
        x_test=x_test_scaled,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        full_clean_dataset=full_clean_df,
    )

    print("\nPreprocessing pipeline completed successfully.")


if __name__ == "__main__":
    main()