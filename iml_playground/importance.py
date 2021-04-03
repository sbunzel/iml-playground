from typing import Any, Dict

import altair as alt
import numpy as np
import pandas as pd
from sklearn import inspection


def plot_permutation_importance(
    model, title_config: Dict[str, Any], top_n: int = 15
) -> alt.Chart:
    imp = inspection.permutation_importance(
        estimator=model.estimator,
        X=model.X_test,
        y=model.y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    sorted_idx = np.median(imp.importances, axis=1).argsort()[-top_n:]
    sorted_names = list(model.feature_names[sorted_idx])
    top_n_imp = imp.importances[sorted_idx]
    df = pd.DataFrame(
        {
            "Feature": np.repeat(sorted_names, top_n_imp.shape[0]),
            "Importance": top_n_imp.reshape(-1),
        }
    ).round(6)
    chart = (
        alt.Chart(df)
        .mark_boxplot(outliers=True)
        .encode(
            x="Importance:Q", y=alt.Y("Feature:N", sort=sorted_names[::-1], title="")
        )
        .properties(title="Permutation Feature Importances", height=300)
        .configure_title(**title_config)
    )
    return chart
