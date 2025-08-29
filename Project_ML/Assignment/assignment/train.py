import glob
import os
from re import split
import pandas as pd
import catboost as cb
from datetime import datetime

from assignment.config import (
    MODELS_DIR,
    cat_features,
    targets,
    LAST_TRAIN_DATE,
    LAST_EVAL_DATE,
)
from assignment.utils import load_latest_models
from assignment.data_load import load_data
from assignment.features import process_data


def cli_entrypoint():
    models = load_latest_models()
    if models:
        print("Using existing models, skipping training.")
        return models

    main_df = load_data()
    processed_df = process_data(main_df)
    models = _train(processed_df, iterations=1000)
    _save_models(models)
    return models


def preprocess_df(df: pd.DataFrame):
    labels = df[["LogNewConfirmedCases", "LogNewFatalities"]].copy()
    features_df = df.drop(
        columns=[
            "Id",
            "ForecastId",
            "ConfirmedCases",
            "LogNewConfirmedCases",
            "Fatalities",
            "LogNewFatalities",
            "Date",
        ]
    ).copy()
    return features_df, labels


def split_dfs(main_df: pd.DataFrame):
    train_df = main_df[main_df["Date"] <= LAST_TRAIN_DATE].copy()
    eval_df = main_df[
        (main_df["Date"] > LAST_TRAIN_DATE) & (main_df["Date"] <= LAST_EVAL_DATE)
    ].copy()
    test_df = main_df[main_df["Date"] > LAST_EVAL_DATE].copy()
    return train_df, eval_df, test_df


def _train(main_df: pd.DataFrame, iterations: int = 1000):

    train_df, eval_df, _ = split_dfs(main_df)

    train_features_df, train_labels = preprocess_df(train_df)
    eval_features_df, eval_labels = preprocess_df(eval_df)

    catboost_models = {}
    for prediction_name in targets:
        model = cb.CatBoostRegressor(has_time=True, iterations=iterations)
        model.fit(
            train_features_df,
            train_labels[prediction_name],
            eval_set=(eval_features_df, eval_labels[prediction_name]),
            cat_features=cat_features,
            verbose=100,
        )
        print(
            "CatBoost: prediction of %s: RMSLE on validation = %s"
            % (prediction_name, model.evals_result_["validation"]["RMSE"][-1])
        )
        catboost_models[prediction_name] = model

    return catboost_models


def _save_models(models: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    for name, model in models.items():
        path = os.path.join(MODELS_DIR, f"covid_19_model_{name}_{stamp}.cbm")
        model.save_model(path)
        print(f"Saved: {path}")
