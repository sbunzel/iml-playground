from typing import Any, Dict

import altair as alt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .dataset import Dataset


class BaseModel:
    def __init__(self, ds: Dataset, estimator_name: str):
        self.ds = ds
        self.estimator_name = estimator_name
        self._init_estimator()
        self._fit_estimator()

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


class Model(BaseModel):
    def __init__(self, ds: Dataset, estimator_name: str):
        super().__init__(ds, estimator_name)
        if self.task == "classification":
            self.model = ClassificationModel(ds, estimator_name)
        else:
            self.model = RegressionModel(ds, estimator_name)
        self.model._register_predictions()

    def plot_predictions(self, altair_config: Dict[str, Any], **kwargs) -> alt.Chart:
        return self.model.plot_predictions(altair_config, **kwargs)

    def plot_performance(self, altair_config: Dict[str, Any], **kwargs) -> alt.Chart:
        return self.model.plot_performance(altair_config, **kwargs)

    @property
    def description(self) -> str:
        return self.model.description


class ClassificationModel(BaseModel):
    def __init__(self, ds: Dataset, estimator_name: str):
        super().__init__(ds, estimator_name)

    def _register_predictions(self):
        self.y_pred = self.estimator.predict_proba(self.ds.X_test)[:, 1]

    def plot_predictions(self, altair_config: Dict[str, Any], **kwargs) -> alt.Chart:
        p_min = kwargs["p_min"]
        p_max = kwargs["p_max"]
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

    def plot_performance(
        self, altair_config: Dict[str, Any], threshold: float
    ) -> alt.Chart:
        perf_df = self._class_performance(
            y_true=self.ds.y_test, y_pred=self.y_pred, threshold=threshold
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
                color=alt.Color(
                    "metric:N",
                    legend=None,
                    scale=alt.Scale(scheme=altair_config["scheme"]),
                ),
            )
            .properties(width="container", title="Classification Report")
            .configure_title(**altair_config["title_config"])
        )

    @staticmethod
    def _class_performance(
        y_true: np.ndarray, y_pred: np.ndarray, threshold: float
    ) -> pd.DataFrame:
        report = metrics.classification_report(
            y_true=y_true,
            y_pred=y_pred > threshold,
            output_dict=True,
        )
        df = (
            pd.DataFrame(report)[[str(v) for v in y_true.unique()]]
            .T.astype({"support": int})[["f1-score", "precision", "recall", "support"]]
            .reset_index()
            .rename(columns={"index": "target"})
        )
        return df

    @property
    def description(self) -> str:
        return (
            f"The model is a {self.estimator_name} with the following parameters: "
            + f"`{str(self.estimator)}`"
        )


class RegressionModel(BaseModel):
    def __init__(self, ds: Dataset, estimator_name: str):
        super().__init__(ds, estimator_name)

    def _register_predictions(self):
        self.y_pred = self.estimator.predict(self.ds.X_test)

    def plot_predictions(self, altair_config: Dict[str, Any]) -> alt.Chart:
        chart = self.plot_actuals_predicted(y_true=self.ds.y_test, y_pred=self.y_pred)
        return chart.configure_title(**altair_config["title_config"])

    def plot_performance(self, altair_config: Dict[str, Any]) -> alt.Chart:
        plot_df = pd.DataFrame(
            {"Actual": self.ds.y_test, "Predicted": self.y_pred}
        ).assign(**{"Residual": lambda df: (df["Actual"] - df["Predicted"])})
        return (
            alt.Chart(plot_df)
            .mark_point()
            .encode(x="Actual", y="Residual")
            .properties(width="container", height=300, title="Actuals vs. Residuals")
            .configure_title(**altair_config["title_config"])
        )

    @staticmethod
    def plot_actuals_predicted(y_true: np.ndarray, y_pred: np.ndarray) -> alt.Chart:
        plot_df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
        return (
            alt.Chart(plot_df)
            .mark_point()
            .encode(x="Actual", y="Predicted")
            .properties(
                width="container",
                height=300,
                title="Actuals vs. Predicted",
            )
        )

    @property
    def description(self) -> str:
        if isinstance(self.estimator, LinearRegression):
            param_description = str(
                {
                    name: round(value, 2)
                    for name, value in zip(
                        ["Intercept", *self.ds.X_train.columns],
                        [self.estimator.intercept_, *self.estimator.coef_],
                    )
                }
            )
        elif isinstance(self.estimator, RandomForestRegressor):
            param_description = str(self.estimator)
        return (
            f"The model is a {self.estimator_name} with the following parameters: "
            + f"`{param_description}`"
        )
