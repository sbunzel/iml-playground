from dataclasses import dataclass
from typing import Any, Dict

import altair as alt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def plot_prediction_histogram(
    y_pred: pd.DataFrame,
    p_min: float,
    p_max: float,
    title_config: Dict[str, Any],
    scheme: str = "tableau10",
) -> alt.Chart:
    df = pd.DataFrame(data={"target": 1, "prediction": y_pred}).assign(
        focus=lambda df: df["prediction"].between(p_min, p_max)
    )
    color = alt.Color(
        "focus:N",
        legend=None,
        scale=alt.Scale(scheme=scheme),
    )
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
        .configure_title(**title_config)
    )


@dataclass
class Model:
    """A container for the trained model and the relevant training and prediction data."""

    train: pd.DataFrame
    test: pd.DataFrame
    target: str

    def __post_init__(self):
        self.X_train, self.y_train = (
            self.train.drop(columns=[self.target]),
            self.train[self.target],
        )
        self.X_test, self.y_test = (
            self.test.drop(columns=[self.target]),
            self.test[self.target],
        )
        clf = RandomForestClassifier(
            n_estimators=20, min_samples_leaf=3, max_depth=12, random_state=42
        ).fit(self.X_train, self.y_train)
        self.y_pred = clf.predict_proba(self.X_test)[:, 1]
