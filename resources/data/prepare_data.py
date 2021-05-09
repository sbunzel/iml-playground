from functools import partial
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

data_path = Path(__file__).parent


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


# def prepare_stroke():
#     base_path = data_path / "stroke-prediction"
#     df = pd.read_csv(base_path / "raw.csv").drop(columns=["id"])
#     train, test = train_test_split(
#         df, test_size=0.3, random_state=42, stratify=df["stroke"]
#     )
#     train.to_csv(base_path / "train.csv", index=False)
#     test.to_csv(base_path / "test.csv", index=False)


def main():
    print("Starting data preparation.")

    prepare_car_insurance()
    # prepare_stroke()

    print("Data preparation done.")


if __name__ == "__main__":
    main()
