from pathlib import Path

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
