import os
import pandas as pd
import urllib.request
import zipfile
import requests
import io

from assignment.config import (
    AREA_DIR,
    COVID19_TEST_DATASET_PATH,
    COVID19_TRAIN_DATASET_PATH,
    SMOKING_DIR,
    HOSPITAL_BEDS_DIR,
    HEALTH_EXPENDITURE_DIR,
    POPULATION_DIR,
    AREA_DATASET_PATH,
    SMOKING_DATASET_PATH,
    HOSPITAL_BEDS_DATASET_PATH,
    HEALTH_EXPENDITURE_DATASET_PATH,
    POPULATION_DATASET_PATH,
)


def load_data() -> pd.DataFrame:
    _download_additional_datasets()

    original_train_df = pd.read_csv(COVID19_TRAIN_DATASET_PATH, parse_dates=["Date"])
    original_test_df = pd.read_csv(COVID19_TEST_DATASET_PATH, parse_dates=["Date"])

    last_original_train_date = original_train_df["Date"].max()

    original_test_wo_train_df = original_test_df.drop(
        index=original_test_df[
            original_test_df["Date"] <= last_original_train_date
        ].index
    )

    main_df = pd.concat(
        [original_train_df, original_test_wo_train_df], ignore_index=True
    )

    from_cruise_ships = main_df["Province/State"].isin(
        ["From Diamond Princess", "Grand Princess"]
    )
    main_df.loc[from_cruise_ships, ["Province/State", "Country/Region"]] = main_df.loc[
        from_cruise_ships, ["Country/Region", "Province/State"]
    ].values

    return main_df


def load_area_df(converters):
    return pd.read_csv(AREA_DATASET_PATH, skiprows=4, converters=converters)


def load_population_df(converters):
    return pd.read_csv(
        POPULATION_DATASET_PATH,
        usecols=["Location", "Time", "AgeGrp", "PopMale", "PopFemale", "PopTotal"],
        parse_dates=["Time"],
        converters=converters,
    )


def load_smoking_df():
    return pd.read_csv(SMOKING_DATASET_PATH, skiprows=4)


def load_hospital_beds_df():
    return pd.read_csv(HOSPITAL_BEDS_DATASET_PATH, skiprows=4)


def load_health_expenditure_df():
    return pd.read_csv(HEALTH_EXPENDITURE_DATASET_PATH, skiprows=4)


def _download_additional_datasets():
    _download_data_set(
        AREA_DIR,
        "area.zip",
        "http://api.worldbank.org/v2/en/indicator/AG.LND.TOTL.K2?downloadformat=csv",
    )
    _download_data_set(
        SMOKING_DIR,
        "smoking.zip",
        "http://api.worldbank.org/v2/en/indicator/SH.PRV.SMOK?downloadformat=csv",
    )
    _download_data_set(
        HOSPITAL_BEDS_DIR,
        "hospital_beds.zip",
        "http://api.worldbank.org/v2/en/indicator/SH.MED.BEDS.ZS?downloadformat=csv",
    )
    _download_data_set(
        HEALTH_EXPENDITURE_DIR,
        "health_expenditure.zip",
        "http://api.worldbank.org/v2/en/indicator/SH.XPD.CHEX.PP.CD?downloadformat=csv",
    )

    url = "https://github.com/ordinaryevidence/leep-cea/raw/refs/heads/master/WPP2019_PopulationByAgeSex_Medium.zip"
    os.makedirs(POPULATION_DIR, exist_ok=True)

    response = requests.get(url)
    response.raise_for_status()
    zipfile_bytes = io.BytesIO(response.content)

    with zipfile.ZipFile(zipfile_bytes) as z:
        csv_filename = [f for f in z.namelist() if f.endswith(".csv")][0]
        with z.open(csv_filename) as csvfile, open(
            POPULATION_DATASET_PATH, "wb"
        ) as out_file:
            out_file.write(csvfile.read())


def _download_data_set(dir_name: str, zip_filename: str, url: str):
    os.makedirs(dir_name, exist_ok=True)
    zip_path = os.path.join(dir_name, zip_filename)

    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded {zip_filename}")
    else:
        print(f"{zip_filename} already exists, skipping download.")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            dest_path = os.path.join(dir_name, file)
            if not os.path.exists(dest_path):
                zip_ref.extract(file, dir_name)
                print(f"Extracted: {file}")
            else:
                print(f"Skipped (already exists): {file}")
