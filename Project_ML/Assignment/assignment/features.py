import numpy as np
import pandas as pd
from typing import List
import geopy.distance
import re

from assignment.data_load import (
    load_area_df,
    load_health_expenditure_df,
    load_population_df,
    load_smoking_df,
)
from assignment.utils import (
    get_hubei_coords,
    merge_with_column_drop,
    remap_country_name_from_world_bank_to_df_name,
    remap_country_name_from_un_wpp_to_df_name,
)


def process_data(main_df: pd.DataFrame) -> pd.DataFrame:
    main_df = process_location(main_df)

    main_df, world_bank_converters, un_wpp_converters = (
        process_confirmed_case_and_fatality(main_df)
    )

    # Area
    area_df = load_area_df(world_bank_converters)
    area_df = process_area_df(area_df)
    main_df = merge_with_column_drop(main_df, area_df, right_df_column="Country Name")

    # Population
    population_df = load_population_df(un_wpp_converters)
    aggregated_population_df = process_population_df(population_df)
    main_df = merge_with_column_drop(
        main_df, aggregated_population_df, right_df_column="Location"
    )

    # Density
    main_df = _add_country_population_density(main_df)

    # Smoking
    smoking_df = load_smoking_df()
    smoking_df = process_smoking_df(smoking_df)
    main_df = merge_with_column_drop(
        main_df, smoking_df, right_df_column="Country Name"
    )

    # Health expenditure
    health_expenditure_df = load_health_expenditure_df()
    health_expenditure_df = process_health_expenditure_df(health_expenditure_df)
    main_df = merge_with_column_drop(
        main_df, health_expenditure_df, right_df_column="Country Name"
    )

    return main_df


# ---------------------- Location / target transformations ---------------------- #


def _preprocess_location(
    df: pd.DataFrame, location_columns: List[str] = ["Country/Region", "Province/State"]
):
    df.sort_values(by="Date", inplace=True)
    for column in location_columns:
        df[column].fillna("", inplace=True)
    return df


def _is_cumulative(increment_series):
    for v in increment_series:
        if (not np.isnan(v)) and (v < 0):
            return False
    return True


def _add_prev_day_columns(df: pd.DataFrame, days_history_size: int = 30):
    for field in ["LogNewConfirmedCases", "LogNewFatalities"]:
        df[field] = np.nan
        for prev_day in range(1, days_history_size + 1):
            df[f"{field}_prev_day_{prev_day}"] = np.nan
    return df


def _process_location_group(
    df: pd.DataFrame,
    days_history_size: int = 30,
    location_columns: List[str] = ["Country/Region", "Province/State"],
):
    for location_name, location_df in df.groupby(location_columns):
        for field in ["ConfirmedCases", "Fatalities"]:
            new_values = location_df[field].values.copy()
            new_values[1:] -= new_values[:-1]
            if not _is_cumulative(new_values):
                print(
                    f"{field} for {location_name} is not valid cumulative series, drop it"
                )
                df.drop(index=location_df.index, inplace=True)
                break
            log_new_values = np.log1p(new_values)
            df.loc[location_df.index, f"LogNew{field}"] = log_new_values

            for prev_day in range(1, days_history_size + 1):
                df.loc[
                    location_df.index[prev_day:], f"LogNew{field}_prev_day_{prev_day}"
                ] = log_new_values[:-prev_day]
    return df


def process_location(df: pd.DataFrame) -> pd.DataFrame:
    df = _preprocess_location(df)
    df = _add_prev_day_columns(df)
    df = _process_location_group(df)
    return df


# ---------------------- Feature engineering ---------------------- #


def process_area_df(area_df: pd.DataFrame) -> pd.DataFrame:
    year_columns = [str(year) for year in range(1960, 2020)]
    area_df["CountryArea"] = area_df[year_columns].apply(
        lambda row: row[row.last_valid_index()] if row.last_valid_index() else np.nan,
        axis="columns",
    )
    return area_df[["Country Name", "CountryArea"]]


def process_confirmed_case_and_fatality(
    main_df: pd.DataFrame, thresholds: List[int] = [1, 10, 100]
):
    first_date = min(main_df["Date"])
    main_df["Day"] = (main_df["Date"] - first_date).dt.days.astype("int32")
    main_df["WeekDay"] = main_df["Date"].transform(lambda d: d.weekday())

    for threshold in thresholds:
        main_df[f"Days_since_ConfirmedCases={threshold}"] = np.nan
        main_df[f"Days_since_Fatalities={threshold}"] = np.nan

    for location_name, location_df in main_df.groupby(
        ["Country/Region", "Province/State"]
    ):
        for field in ["ConfirmedCases", "Fatalities"]:
            for threshold in thresholds:
                first_day = (
                    location_df["Day"].loc[location_df[field] >= threshold].min()
                )
                if not np.isnan(first_day):
                    main_df.loc[
                        location_df.index, f"Days_since_{field}={threshold}"
                    ] = location_df["Day"].transform(
                        lambda day: -1 if (day < first_day) else (day - first_day)
                    )

    origin_coords = get_hubei_coords(main_df)
    main_df["Distance_to_origin"] = main_df.apply(
        lambda row: geopy.distance.distance(
            (row["Lat"], row["Long"]), origin_coords
        ).km,
        axis="columns",
    )

    world_bank_converters = {
        "Country Name": remap_country_name_from_world_bank_to_df_name
    }
    un_wpp_converters = {"Location": remap_country_name_from_un_wpp_to_df_name}

    return main_df, world_bank_converters, un_wpp_converters


def process_population_df(population_df: pd.DataFrame) -> pd.DataFrame:
    population_df = population_df.loc[
        (population_df["Time"] >= pd.Timestamp(2014, 1, 1))
        & (population_df["Time"] <= pd.Timestamp(2019, 1, 1))
    ]

    aggregated_population_df = pd.DataFrame()
    for (location, time), group_df in population_df.groupby(["Location", "Time"]):
        pop_by_age_groups = [0] * 5
        pop_male = 0
        pop_female = 0

        for _, row in group_df.iterrows():
            age_grp_start = int(re.split(r"[\-\+]", row["AgeGrp"])[0])
            pop_by_age_groups[min(age_grp_start // 20, 4)] += (
                row["PopMale"] + row["PopFemale"]
            )
            pop_male += row["PopMale"]
            pop_female += row["PopFemale"]

        aggregated_population_df = pd.concat(
            [
                aggregated_population_df,
                pd.DataFrame(
                    [
                        {
                            "Location": location,
                            "Time": time,
                            "CountryPop_0-20": pop_by_age_groups[0],
                            "CountryPop_20-40": pop_by_age_groups[1],
                            "CountryPop_40-60": pop_by_age_groups[2],
                            "CountryPop_60-80": pop_by_age_groups[3],
                            "CountryPop_80+": pop_by_age_groups[4],
                            "CountryPopMale": pop_male,
                            "CountryPopFemale": pop_female,
                            "CountryPopTotal": pop_male + pop_female,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    aggregated_population_df = aggregated_population_df.sort_values(
        "Time"
    ).drop_duplicates(["Location"], keep="last")
    aggregated_population_df.drop(columns="Time", inplace=True)

    return aggregated_population_df


def _add_country_population_density(df: pd.DataFrame) -> pd.DataFrame:
    df["CountryPopDensity"] = df["CountryPopTotal"] / df["CountryArea"]
    return df


def process_smoking_df(smoking_df: pd.DataFrame) -> pd.DataFrame:
    recent_year_columns = [str(year) for year in range(2010, 2020)]
    smoking_df["CountrySmokingRate"] = smoking_df[recent_year_columns].apply(
        lambda row: row[row.last_valid_index()] if row.last_valid_index() else np.nan,
        axis="columns",
    )
    return smoking_df[["Country Name", "CountrySmokingRate"]]


def process_health_expenditure_df(health_expenditure_df: pd.DataFrame) -> pd.DataFrame:
    recent_year_columns = [str(year) for year in range(2010, 2020)]
    health_expenditure_df[
        "CountryHealthExpenditurePerCapitaPPP"
    ] = health_expenditure_df[recent_year_columns].apply(
        lambda row: row[row.last_valid_index()] if row.last_valid_index() else np.nan,
        axis="columns",
    )
    return health_expenditure_df[
        ["Country Name", "CountryHealthExpenditurePerCapitaPPP"]
    ]
