import pandas as pd
import os, glob
import catboost as cb
from assignment.config import MODELS_DIR, targets, PREDICTIONS_DIR


def get_hubei_coords(df):
    for index, row in df.iterrows():
        if row["Province/State"] == "Hubei":
            return (row["Lat"], row["Long"])

    raise Exception("Hubei not found in data")


def merge_with_column_drop(left_df, right_df, right_df_column="Country"):
    df = pd.merge(
        left=left_df,
        right=right_df,
        how="left",
        left_on="Country/Region",
        right_on=right_df_column,
    )
    df.drop(columns=right_df_column, inplace=True)

    return df


def remap_country_name_from_world_bank_to_df_name(country):
    return {
        "Bahamas, The": "The Bahamas",
        "Brunei Darussalam": "Brunei",
        "Congo, Rep.": "Congo (Brazzaville)",
        "Congo, Dem. Rep.": "Congo (Kinshasa)",
        "Czech Republic": "Czechia",
        "Egypt, Arab Rep.": "Egypt",
        "Iran, Islamic Rep.": "Iran",
        "Korea, Rep.": "Korea, South",
        "Kyrgyz Republic": "Kyrgyzstan",
        "Russian Federation": "Russia",
        "Slovak Republic": "Slovakia",
        "St. Lucia": "Saint Lucia",
        "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
        "United States": "US",
        "Venezuela, RB": "Venezuela",
    }.get(country, country)


def remap_country_name_from_un_wpp_to_df_name(country):
    return {
        "Bahamas": "The Bahamas",
        "Bolivia (Plurinational State of)": "Bolivia",
        "Brunei Darussalam": "Brunei",
        "China, Taiwan Province of China": "Taiwan*",
        "Congo": "Congo (Brazzaville)",
        "Côte d'Ivoire": "Cote d'Ivoire",
        "Democratic Republic of the Congo": "Congo (Kinshasa)",
        "Gambia": "The Gambia",
        "Iran (Islamic Republic of)": "Iran",
        "Republic of Korea": "Korea, South",
        "Republic of Moldova": "Moldova",
        "Réunion": "Reunion",
        "Russian Federation": "Russia",
        "United Republic of Tanzania": "Tanzania",
        "United States of America": "US",
        "Venezuela (Bolivarian Republic of)": "Venezuela",
        "Viet Nam": "Vietnam",
    }.get(country, country)


def find_latest_model(target):
    pattern = os.path.join(MODELS_DIR, f"covid_19_model_{target}_*.cbm")
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def load_latest_models():
    models = {}
    for t in targets:
        path = find_latest_model(t)
        if not path:
            return None
        m = cb.CatBoostRegressor()
        m.load_model(path)
        models[t] = m
    return models


def load_latest_predictions():
    pattern = os.path.join(PREDICTIONS_DIR, "predictions_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No predictions found in {PREDICTIONS_DIR}. "
            "Please run `poetry run predict` first."
        )
    files.sort(key=os.path.getmtime, reverse=True)
    latest_path = files[0]
    print(f"Loading predictions from: {latest_path}")
    return pd.read_csv(latest_path)
