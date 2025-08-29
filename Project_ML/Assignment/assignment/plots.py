import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from assignment.config import (
    LAST_TRAIN_DATE,
    LAST_EVAL_DATE,
)

from assignment.utils import load_latest_predictions


def plot_graph(main_df, country_region, province_state, field, log_scale=True):
    location_df = main_df.loc[
        (main_df["Country/Region"] == country_region)
        & (main_df["Province/State"] == province_state)
    ]

    title = (
        f"{field} for {country_region}"
        if not province_state
        else f"{field} for {country_region}, {province_state}"
    )

    plt.figure(figsize=(16, 10))
    plt.suptitle(title, fontsize=14)
    if log_scale:
        plt.yscale("log")

    for sub_field in [field, "Predicted" + field]:
        if sub_field in location_df.columns:
            plt.plot(location_df["Date"], location_df[sub_field], label=sub_field)

    ax = plt.gca()
    transform_for_text = matplotlib.transforms.blended_transform_factory(
        ax.transData, ax.transAxes
    )

    first_eval_date = LAST_TRAIN_DATE + pd.Timedelta(days=1)
    first_test_date = LAST_EVAL_DATE + pd.Timedelta(days=1)

    plt.axvline(x=LAST_TRAIN_DATE, color="#000000")
    plt.text(first_eval_date, 0.95, "eval", transform=transform_for_text)
    plt.axvline(x=LAST_EVAL_DATE, color="#000000")
    plt.text(first_test_date, 0.95, "test", transform=transform_for_text)

    plt.legend()
    plt.show()


def cli_entrypoint():

    main_df = load_latest_predictions()
    main_df["Date"] = pd.to_datetime(main_df["Date"])
    plot_graph(main_df, "US", "Kansas", "ConfirmedCases")
    plot_graph(main_df, "US", "Kansas", "Fatalities")
    plot_graph(main_df, "China", "Hubei", "ConfirmedCases")
    plot_graph(main_df, "China", "Hubei", "Fatalities")
    # plot_graph(main_df, "Germany", '', "ConfirmedCases", log_scale=False)
    # plot_graph(main_df, "Germany", '', "Fatalities", log_scale=False)
