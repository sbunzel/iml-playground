from typing import Any, Dict

import altair as alt
import numpy as np
import pandas as pd
from sklearn import inspection


class FeatureImportance:
    def __init__(self, model, top_n: int = 15) -> None:
        self.model = model
        self.ds = model.ds
        self.top_n = top_n
        self.top_n_imp, self.sorted_names = self._calculate()

    def _calculate(self):
        imp = inspection.permutation_importance(
            estimator=self.model.estimator,
            X=self.ds.X_test,
            y=self.ds.y_test,
            scoring="roc_auc",
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        sorted_idx = np.median(imp.importances, axis=1).argsort()[-self.top_n :]
        sorted_names = list(self.model.ds.feature_names[sorted_idx])
        top_n_imp = imp.importances[sorted_idx]
        return top_n_imp, sorted_names

    def plot(self, altair_config: Dict[str, Any]) -> alt.Chart:
        df = pd.DataFrame(
            {
                "Feature": np.repeat(self.sorted_names, self.top_n_imp.shape[0]),
                "Importance": self.top_n_imp.reshape(-1),
            }
        ).round(6)
        chart = (
            alt.Chart(df)
            .mark_boxplot(outliers=True)
            .encode(
                x="Importance:Q",
                y=alt.Y("Feature:N", sort=self.sorted_names[::-1], title=""),
            )
            .properties(title="Permutation Feature Importances", height=300)
            .configure_title(**altair_config["title_config"])
        )
        return chart
