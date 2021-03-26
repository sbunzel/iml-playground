from typing import Any, Dict

import altair as alt
import numpy as np
import pandas as pd
from sklearn import metrics


def plot_class_performance(
    y_test,
    test_preds,
    threshold: float,
    title_config: Dict[str, Any],
    scheme: str = "tableau10",
) -> alt.Chart:
    perf_df = _class_performance(
        y_test=y_test, test_preds=test_preds, threshold=threshold
    )
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
            color=alt.Color("metric:N", legend=None, scale=alt.Scale(scheme=scheme)),
        )
        .properties(width="container", title="Classification Report")
        .configure_title(**title_config)
    )


def _class_performance(
    y_test: np.ndarray, test_preds: np.ndarray, threshold: float
) -> pd.DataFrame:
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
