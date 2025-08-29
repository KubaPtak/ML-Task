import pandas as pd
import numpy as np
import catboost as cb
import os
from datetime import datetime

from assignment.config import (
    cat_features,
    location_columns,
    LAST_TRAIN_DATE,
    LAST_EVAL_DATE,
    LAST_TEST_DATE,
    PREDICTIONS_DIR,
)
from assignment.utils import load_latest_models
from assignment.data_load import load_data
from assignment.train import preprocess_df, split_dfs
from assignment.features import process_data


def _save_predictions(train_df, eval_df, test_df):
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)  # Create folder if it doesn't exist
    stamp = datetime.now().strftime("%Y%m%d")
    final_df = pd.concat([train_df, eval_df, test_df])
    path = os.path.join(PREDICTIONS_DIR, f"predictions_{stamp}.csv")
    final_df.to_csv(path, index=False)

    print(f"Predictions saved to {path}")


def _predict_for_dataset(
    df, features_df, prev_day_df, first_date, last_date, update_features_data, models
):
    df["PredictedLogNewConfirmedCases"] = np.nan
    df["PredictedLogNewFatalities"] = np.nan
    df["PredictedConfirmedCases"] = np.nan
    df["PredictedFatalities"] = np.nan

    for day in pd.date_range(first_date, last_date):
        day_df = df[df["Date"] == day]
        day_features_pool = cb.Pool(
            features_df.loc[day_df.index], cat_features=cat_features
        )

        for prediction_type in ["LogNewConfirmedCases", "LogNewFatalities"]:
            df.loc[day_df.index, "Predicted" + prediction_type] = np.maximum(
                models[prediction_type].predict(day_features_pool), 0.0
            )

        day_predictions_df = df.loc[day_df.index][
            location_columns
            + ["PredictedLogNewConfirmedCases", "PredictedLogNewFatalities"]
        ]

        for field in ["ConfirmedCases", "Fatalities"]:
            prev_day_field = field if day == first_date else ("Predicted" + field)
            merged_df = day_predictions_df.merge(
                right=prev_day_df[location_columns + [prev_day_field]],
                how="inner",
                on=location_columns,
            )

            df.loc[day_df.index, "Predicted" + field] = merged_df.apply(
                lambda row: row[prev_day_field]
                + np.rint(np.expm1(row["PredictedLogNew" + field])),
                axis="columns",
            ).values

        if update_features_data:
            for next_day in pd.date_range(day + pd.Timedelta(days=1), last_date):
                next_day_features_df = features_df[df["Date"] == next_day]
                merged_df = next_day_features_df[location_columns].merge(
                    right=day_predictions_df, how="inner", on=location_columns
                )

                prev_day_idx = (next_day - day).days
                for prediction_type in ["LogNewConfirmedCases", "LogNewFatalities"]:
                    features_df.loc[
                        next_day_features_df.index,
                        prediction_type + "_prev_day_%s" % prev_day_idx,
                    ] = merged_df["Predicted" + prediction_type].values

        prev_day_df = df.loc[day_df.index]


def cli_entrypoint():
    models = load_latest_models()
    if not models:
        raise RuntimeError(
            "No trained models found. Please run `poetry run train` first."
        )

    main_df = load_data()

    train_df, eval_df, test_df = split_dfs(main_df)
    processed_df = process_data(main_df)
    train_df, eval_df, test_df = split_dfs(processed_df)
    eval_features_df, _ = preprocess_df(eval_df)
    test_features_df, _ = preprocess_df(test_df)

    first_eval_date = LAST_TRAIN_DATE + pd.Timedelta(days=1)
    first_test_date = LAST_EVAL_DATE + pd.Timedelta(days=1)

    prev_day_df = train_df.loc[train_df["Date"] == LAST_TRAIN_DATE]
    _predict_for_dataset(
        eval_df,
        eval_features_df,
        prev_day_df,
        first_eval_date,
        LAST_EVAL_DATE,
        update_features_data=False,
        models=models,
    )

    prev_day_df = eval_df.loc[eval_df["Date"] == LAST_EVAL_DATE]
    _predict_for_dataset(
        test_df,
        test_features_df,
        prev_day_df,
        first_test_date,
        LAST_TEST_DATE,
        update_features_data=True,
        models=models,
    )
    _save_predictions(train_df, eval_df, test_df)
