from typing import Any, Dict

import altair as alt
import numpy as np
import pandas as pd
from sklearn import inspection


class FeatureImportance:
    def __init__(self, model) -> None:
        self.model = model
        self.ds = model.ds
        self.sorted_imp, self.sorted_names = self._calculate()

    def _calculate(self):
        scoring = (
            "roc_auc" if self.model.estimator._estimator_type == "classifier" else None
        )
        imp = inspection.permutation_importance(
            estimator=self.model.estimator,
            X=self.ds.X_test,
            y=self.ds.y_test,
            scoring=scoring,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        sorted_idx = np.median(imp.importances, axis=1).argsort()
        sorted_names = list(self.model.ds.feature_names[sorted_idx])
        sorted_imp = imp.importances[sorted_idx]
        return sorted_imp, sorted_names

    def _build_plot_df(self, top_n: int) -> pd.DataFrame:
        top_n_imp = self.sorted_imp[-top_n:]
        top_n_names = self.sorted_names[-top_n:]
        return pd.DataFrame(
            {
                "Feature": np.repeat(top_n_names, top_n_imp.shape[1]),
                "Importance": top_n_imp.reshape(-1),
            }
        ).round(6)

    def plot(self, altair_config: Dict[str, Any], top_n: int = 15) -> alt.Chart:
        plot_df = self._build_plot_df(top_n=top_n)
        chart = (
            alt.Chart(plot_df)
            .mark_boxplot(outliers=True)
            .encode(
                x="Importance:Q",
                y=alt.Y("Feature:N", sort=self.sorted_names[::-1], title=""),
            )
            .properties(title="Permutation Feature Importances", height=300)
            .configure_title(**altair_config["title_config"])
        )
        return chart
