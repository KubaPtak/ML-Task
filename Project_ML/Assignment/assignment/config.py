import os
import pandas as pd

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")
# Dataset folders
AREA_DIR = os.path.join(DATASETS_DIR, "area")
SMOKING_DIR = os.path.join(DATASETS_DIR, "smoking")
HOSPITAL_BEDS_DIR = os.path.join(DATASETS_DIR, "hospital_beds")
HEALTH_EXPENDITURE_DIR = os.path.join(DATASETS_DIR, "health_expenditure")
POPULATION_DIR = os.path.join(DATASETS_DIR, "population")

# CSV paths
AREA_DATASET_PATH = os.path.join(AREA_DIR, "API_AG.LND.TOTL.K2_DS2_en_csv_v2_21556.csv")
SMOKING_DATASET_PATH = os.path.join(
    SMOKING_DIR, "API_SH.PRV.SMOK_DS2_en_csv_v2_31160.csv"
)
HOSPITAL_BEDS_DATASET_PATH = os.path.join(
    HOSPITAL_BEDS_DIR, "API_SH.MED.BEDS.ZS_DS2_en_csv_v2_23583.csv"
)
HEALTH_EXPENDITURE_DATASET_PATH = os.path.join(
    HEALTH_EXPENDITURE_DIR, "API_SH.XPD.CHEX.PP.CD_DS2_en_csv_v2_34418.csv"
)
POPULATION_DATASET_PATH = os.path.join(
    POPULATION_DIR, "WPP2019_PopulationByAgeSex_Medium.csv"
)

COVID19_TRAIN_DATASET_PATH = os.path.join(
    DATASETS_DIR, "covid19-global-forecasting-week-1", "train.csv"
)
COVID19_TEST_DATASET_PATH = os.path.join(
    DATASETS_DIR, "covid19-global-forecasting-week-1", "test.csv"
)

# Constants

cat_features = ["Province/State", "Country/Region"]
targets = ["LogNewConfirmedCases", "LogNewFatalities"]
location_columns = ["Country/Region", "Province/State"]

# Split dates
LAST_TRAIN_DATE = pd.Timestamp(2020, 3, 11)
LAST_EVAL_DATE = pd.Timestamp(2020, 3, 24)
LAST_TEST_DATE = pd.Timestamp(2020, 4, 23)
