import altair as alt
import pandas as pd
from sklearn import inspection

from .model import Model


class GlobalEffects:
    def __init__(self, model: Model) -> None:
        self.model = model

    def _calculate_effects(self, features):
        res = inspection.partial_dependence(
            estimator=self.model.estimator,
            X=self.model.X_test,
            features=features,
            kind="average",
        )
        return res["average"], res["values"][0]

    def plot(self, feature):
        avg_effect, values = self._calculate_effects(features=[feature])
        df = pd.DataFrame(
            {"Partial Dependence": avg_effect.reshape(-1), feature: values}
        )
        return (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=f"{feature}:Q",
                y=alt.Y("Partial Dependence:Q", scale=alt.Scale(domain=(0, 1))),
            )
        )
