from typing import Any, Dict

import altair as alt
import pandas as pd


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
