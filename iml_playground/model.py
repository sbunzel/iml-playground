from dataclasses import dataclass
from typing import Any, Dict

import altair as alt
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .dataset import Dataset


@dataclass
class Model:
    """A container for the trained model and predictions."""

    ds: Dataset
    estimator_name: str

    def __post_init__(self):
        self._init_estimator()
        self._fit_estimator()
        self._register_predictions()

    def _init_estimator(self):
        estimators = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=20, min_samples_leaf=3, max_depth=12, random_state=42
            ),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=20, min_samples_leaf=3, max_depth=12, random_state=42
            ),
        }
        self.estimator = estimators[self.estimator_name]
        if self.estimator._estimator_type == "classifier":
            self.task = "classification"
        elif self.estimator._estimator_type == "regressor":
            self.task = "regression"
        else:
            raise ValueError(
                f"Estimator type '{self.estimator._estimator_type}' is not supported. Use 'classifier' or 'regressor'."  # noqa
            )

    def _fit_estimator(self):
        self.estimator = self.estimator.fit(self.ds.X_train, self.ds.y_train)

    def _register_predictions(self):
        if self.task == "classification":
            self.y_pred = self.estimator.predict_proba(self.ds.X_test)[:, 1]
        elif self.task == "regression":
            self.y_pred = self.estimator.predict(self.ds.X_test)

    def plot_prediction_histogram(
        self, p_min: float, p_max: float, altair_config: Dict[str, Any]
    ) -> alt.Chart:
        df = pd.DataFrame(data={"target": 1, "prediction": self.y_pred}).assign(
            focus=lambda df: df["prediction"].between(p_min, p_max)
        )
        color = alt.Color(
            "focus:N",
            legend=None,
            scale=alt.Scale(scheme=altair_config["scheme"]),
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
            .configure_title(**altair_config["title_config"])
        )

    def plot_class_performance(
        self,
        threshold: float,
        altair_config: Dict[str, Any],
    ) -> alt.Chart:
        perf_df = self._class_performance(threshold=threshold)
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
                    "metric:N",
                    legend=None,
                    scale=alt.Scale(scheme=altair_config["scheme"]),
                ),
            )
            .properties(width="container", title="Classification Report")
            .configure_title(**altair_config["title_config"])
        )

    def _class_performance(self, threshold: float) -> pd.DataFrame:
        report = metrics.classification_report(
            y_true=self.ds.y_test,
            y_pred=self.y_pred > threshold,
            output_dict=True,
        )
        df = (
            pd.DataFrame(report)[[str(c) for c in self.ds.y_test.unique()]]
            .T.astype({"support": int})[["f1-score", "precision", "recall", "support"]]
            .reset_index()
            .rename(columns={"index": "target"})
        )
        return df
