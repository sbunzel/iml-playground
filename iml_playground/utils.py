from pathlib import Path

import pandas as pd
import streamlit as st


def read_md(dataset: str, file: str):
    return Path(f"resources/markdown/{dataset}/{file}").read_text()


@st.cache
def read_train_test(dataset: str):
    train = pd.read_csv(
        f"https://raw.githubusercontent.com/sbunzel/iml-playground/main/resources/data/{dataset}/train.csv"  # noqa
    )
    test = pd.read_csv(
        f"https://raw.githubusercontent.com/sbunzel/iml-playground/main/resources/data/{dataset}/test.csv"  # noqa
    )
    return train, test
