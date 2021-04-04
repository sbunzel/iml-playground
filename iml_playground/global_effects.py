from typing import Any, Dict

import altair as alt
import pandas as pd
from sklearn import inspection

from .ale import _first_order_ale_quant, _get_centres
from .model import Model


class GlobalEffects:
    def __init__(self, model: Model, method: str, feature: str) -> None:
        self.model = model
        self.feature = feature
        X = model.X_test.copy()
        self.X = X.loc[
            X[feature].between(X[feature].quantile(0.05), X[feature].quantile(0.95))
        ]
        methods = {
            "partial_dependence_plot": self._calculate_partial_dependence,
            "accumulated_local_effects": self._calculate_ale,
        }
        try:
            self.avg_effect, self.values = methods[method](feature=feature)
        except KeyError:
            raise ValueError(
                f"'{method}' is not a valid method. Possible options are {list(methods.keys())}"
            )

    def _calculate_partial_dependence(self, feature: str):
        res = inspection.partial_dependence(
            estimator=self.model.estimator,
            X=self.X,
            features=[feature],
            kind="average",
            percentiles=(0, 1),
        )
        return res["average"], res["values"][0]

    def _calculate_ale(self, feature: str):
        effects, quantiles = _first_order_ale_quant(
            self.model.estimator.predict,
            self.X,
            feature,
            50,
        )
        values = _get_centres(quantiles)
        return effects, values

    def plot(self, title_config: Dict[str, Any]) -> alt.Chart:
        effects_df = pd.DataFrame(
            {
                "Average Effect": self.avg_effect.reshape(-1),
                self.feature: self.values,
            }
        )
        effect_line = (
            alt.Chart(effects_df)
            .mark_line()
            .encode(
                x=alt.X(f"{self.feature}:Q"),
                y=alt.Y("Average Effect:Q"),
            )
        )

        min_value, max_value = (self.values.min(), self.values.max())
        values_df = self.model.X_test[[self.feature]]
        values_df = values_df.loc[values_df[self.feature].between(min_value, max_value)]
        step_size = (max_value - min_value) / 10
        values_hist = (
            alt.Chart(values_df)
            .mark_bar(opacity=0.3)
            .encode(
                alt.X(f"{self.feature}:Q", bin=alt.Bin(step=step_size)),
                y=alt.Y("count()", title="Number of Samples"),
            )
        )
        chart = values_hist + effect_line
        return (
            chart.properties(
                title=f"Global Effect of Feature '{self.feature}' on the Predictions",
                height=350,
            )
            .configure_title(**title_config)
            .resolve_axis(y="independent")
            .resolve_scale(y="independent")
        )
