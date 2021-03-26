from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from sklearn import metrics

ALT_TITLE_CONFIG = {"fontSize": 14, "offset": 10, "orient": "top", "anchor": "middle"}
ALT_SCHEME = "tableau10"


def read_md(file):
    return Path(f"resources/markdown/{file}").read_text()


@st.cache
def read_train_test():
    train = pd.read_csv(
        f"https://raw.githubusercontent.com/sbunzel/iml-playground/main/resources/car-insurance-cold-calls/train.csv"
    )
    test = pd.read_csv(
        f"https://raw.githubusercontent.com/sbunzel/iml-playground/main/resources/car-insurance-cold-calls/test.csv"
    )
    return train, test


def prediction_histogram(df, p_min, p_max):
    if df["target"].nunique() == 1:
        df = df.assign(focus=lambda df: df["prediction"].between(p_min, p_max))
        color = alt.Color(
            "focus:N",
            legend=None,
            scale=alt.Scale(scheme=ALT_SCHEME),
        )
    else:
        color = alt.Color("target:N", scale=alt.Scale(scheme=ALT_SCHEME))
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X(
                "prediction:Q",
                bin=alt.Bin(step=0.01),
                scale=alt.Scale(domain=(0, 1)),
                title="Predicted Probability of class 1",
            ),
            y=alt.Y("count()", title="Number of Predictions"),
            color=color,
            tooltip=["target"],
        )
        .properties(
            width="container", height=300, title="Distribution of Model Predictions"
        )
        .configure_title(**ALT_TITLE_CONFIG)
    )


def class_performance(y_test, test_preds, threshold: float) -> pd.DataFrame:
    report = metrics.classification_report(
        y_true=y_test,
        y_pred=test_preds > threshold,
        output_dict=True,
    )
    df = (
        pd.DataFrame(report)[[str(c) for c in y_test.unique()]]
        .T.astype({"support": int})[["f1-score", "precision", "recall", "support"]]
        .reset_index()
        .rename(columns={"index": "target"})
    )
    return df


def plot_class_performance(perf_df: pd.DataFrame) -> alt.Chart:
    plot_df = perf_df.drop(columns="support").melt(
        id_vars=["target"], var_name="metric", value_name="score"
    )

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("score:Q", title="Score", scale=alt.Scale(domain=(0, 1))),
            y=alt.Y("target:N", title=""),
            row=alt.Row("metric:N", title=""),
            color=alt.Color(
                "metric:N", legend=None, scale=alt.Scale(scheme=ALT_SCHEME)
            ),
        )
        .properties(width="container", title="Classification Report")
        .configure_title(**ALT_TITLE_CONFIG)
    )
