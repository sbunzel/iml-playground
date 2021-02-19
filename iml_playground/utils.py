from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


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
            scale=alt.Scale(scheme="tableau10"),
        )
    else:
        color = alt.Color("target:N", scale=alt.Scale(scheme="tableau10"))
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X(
                "prediction:Q",
                bin=alt.Bin(step=0.01),
                scale=alt.Scale(domain=(0, 1)),
                title="Predicted Probability",
            ),
            y=alt.Y("count()", title="Number of Predictions"),
            color=color,
            tooltip=["target"],
        )
        .properties(
            width="container", height=300, title="Distribution of Model Predictions"
        )
        .configure_title(fontSize=14, offset=10, orient="top", anchor="middle")
    )
