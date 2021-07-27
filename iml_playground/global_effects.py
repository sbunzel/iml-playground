from typing import Any, Dict, Tuple

import altair as alt
import numpy as np
import pandas as pd
from sklearn import inspection

from .model import Model


class GlobalEffects:
    def __init__(self, model: Model, method: str, feature: str) -> None:
        self.model = model
        self.ds = model.ds
        self.feature = feature
        self.X = self.ds.X_test.loc[
            self.ds.X_test[feature].between(
                self.ds.X_test[feature].quantile(0.05),
                self.ds.X_test[feature].quantile(0.95),
            )
        ]
        methods = {
            "partial_dependence_plot": self._calculate_partial_dependence,
        }
        try:
            self.avg_effect, self.values = methods[method](feature=feature)
        except KeyError:
            raise ValueError(
                f"'{method}' is not a valid method. Possible options are {list(methods.keys())}"
            )

    def _calculate_partial_dependence(
        self, feature: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        # apparently, this works better with inspection.partial_dependence, not really sure why...
        feature_id = self.X.columns.get_loc(feature)
        pd = inspection.partial_dependence(
            estimator=self.model.estimator,
            X=self.X,
            features=[feature_id],
            kind="average",
            percentiles=(0, 1),
        )
        effects, values = pd["average"].reshape(-1), pd["values"][0]
        return effects, values

    def plot(self, altair_config: Dict[str, Any]) -> alt.Chart:
        # Line plot of average effects
        effects_df = pd.DataFrame(
            {
                "Average Effect": self.avg_effect,
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
        # Histogram of feature values
        min_value, max_value = (self.values.min(), self.values.max())
        values_df = self.ds.X_test[[self.feature]].loc[
            self.ds.X_test[self.feature].between(min_value, max_value)
        ]
        step_size = (max_value - min_value) / 10
        values_hist = (
            alt.Chart(values_df)
            .mark_bar(opacity=0.3)
            .encode(
                alt.X(
                    f"{self.feature}:Q",
                    bin=alt.Bin(step=step_size),
                    axis=alt.Axis(title=self.feature),
                ),
                y=alt.Y("count()", title="Number of Samples"),
            )
        )
        # Layered combination of line and histogram
        chart = values_hist + effect_line
        return (
            chart.properties(
                title=f"Global Effect of Feature '{self.feature}' on the Predictions",
                height=350,
            )
            .configure_title(**altair_config["title_config"])
            .resolve_axis(y="independent")
            .resolve_scale(y="independent")
        )
