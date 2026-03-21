from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "sprint1"

# Dataset file names
RAW_DATA_FILE = RAW_DATA_DIR / "diabetes_012_health_indicators_BRFSS2015.csv"

# Target column
TARGET_COLUMN = "Diabetes_binary"

# Original column name in Kaggle file
ORIGINAL_TARGET_COLUMN = "Diabetes_012"

# Feature groups
BINARY_VARS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
]

ORDINAL_VARS = [
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "Age",
    "Education",
    "Income",
]

CONTINUOUS_VARS = [
    "BMI",
]

COLUMNS_TO_SCALE = [
    "BMI",
    "MentHlth",
    "PhysHlth",
    "Age",
    "GenHlth",
    "Education",
    "Income",
]

# Split configuration
TRAIN_SIZE = 0.70
VALID_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42