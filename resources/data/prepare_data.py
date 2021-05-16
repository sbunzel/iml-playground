from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

data_path = Path(__file__).parent


def prepare_test_data():
    X, y = make_regression(
        n_samples=150,
        n_features=2,
        n_informative=1,
        n_targets=1,
        shuffle=False,
        effective_rank=None,
        coef=False,
        random_state=42,
    )
    rg = np.random.default_rng(seed=42)
    simple_regression = (
        pd.DataFrame(X, columns=["PredictiveFeature", "RandomFeature"])
        .assign(
            Target=y,
            PredictiveFeatureCorrelated=lambda df: df["PredictiveFeature"] * 1.1
            + rg.normal(loc=0.0, scale=3e-5, size=df.shape[0]),
            PredictiveFeatureWithNoise=lambda df: df["PredictiveFeature"]
            + rg.normal(loc=0.0, scale=5.0, size=df.shape[0]),
        )
        .round(4)[
            [
                "PredictiveFeature",
                "PredictiveFeatureCorrelated",
                "PredictiveFeatureWithNoise",
                "RandomFeature",
                "Target",
            ]
        ]
    )
    simple_regression[:100].to_csv(
        data_path / "simple-regression" / "train.csv", index=False
    )
    simple_regression[100:].to_csv(
        data_path / "simple-regression" / "test.csv", index=False
    )


def prepare_car_insurance():
    COLMAPPER = {
        "Marital": "MaritalStatus",
        "Education": "EducationLevel",
        "Default": "HasCreditDefault",
        "HHInsurance": "HasHHInsurance",
        "CarLoan": "HasCarLoan",
        "Communication": "CommunicationType",
        "LastContactDay": "LastContactDayOfMonth",
        "NoOfContacts": "NumContactsCurrentCampaign",
        "PrevAttempts": "NumAttemptsPrevCampaign",
        "Outcome": "PrevCampaignOutcome",
    }
    TARGET = "CarInsurance"
    base_path = data_path / "car-insurance-cold-calls"
    df = (
        pd.read_csv(
            base_path / "raw.csv",
            parse_dates=["CallStart", "CallEnd"],
            date_parser=partial(pd.to_datetime, format="%H:%M:%S"),
        )
        .drop(columns="Id")
        .assign(
            LastCallDurationSecs=lambda df: (
                df["CallEnd"] - df["CallStart"]
            ).dt.seconds,
            LastCallHourOfDay=lambda df: df["CallStart"].dt.hour,
        )
        .rename(COLMAPPER, axis=1)
        .drop(["CallStart", "CallEnd"], axis=1)
    )
    for c in ["HasCarLoan", "HasCreditDefault", "HasHHInsurance"]:
        df[c] = df[c].map({0: "no", 1: "yes"})

    features = list(set(df.columns) - set([TARGET]))
    train, test = train_test_split(
        df[features + [TARGET]], test_size=0.3, random_state=42, stratify=df[TARGET]
    )
    train.to_csv(base_path / "train.csv", index=False)
    test.to_csv(base_path / "test.csv", index=False)


def main():
    print("Starting data preparation.")

    prepare_test_data()
    prepare_car_insurance()

    print("Data preparation done.")


if __name__ == "__main__":
    main()
