from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


@dataclass
class Model:
    """A container for the trained model and the relevant training and prediction data."""

    train: pd.DataFrame
    test: pd.DataFrame
    target: str

    def __post_init__(self):
        self.X_train, self.y_train = (
            self.train.drop(columns=[self.target]),
            self.train[self.target],
        )
        self.X_test, self.y_test = (
            self.test.drop(columns=[self.target]),
            self.test[self.target],
        )
        self.feature_names = self.X_train.columns
        self.estimator = RandomForestClassifier(
            n_estimators=20, min_samples_leaf=3, max_depth=12, random_state=42
        ).fit(self.X_train, self.y_train)
        self.y_pred = self.estimator.predict_proba(self.X_test)[:, 1]
