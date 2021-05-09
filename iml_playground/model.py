from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class Model:
    """A container for the trained model and the relevant training and prediction data."""

    train: pd.DataFrame
    test: pd.DataFrame
    target: str

    def __post_init__(self):
        self._split_x_y()
        self._register_feature_names()
        self._impute_and_encode()
        self._fit_estimator()
        self._register_predictions()

    def _split_x_y(self):
        self.X_train, self.y_train = (
            self.train.drop(columns=[self.target]),
            self.train[self.target],
        )
        self.X_test, self.y_test = (
            self.test.drop(columns=[self.target]),
            self.test[self.target],
        )

    def _register_feature_names(self):
        self.cat_features = [
            c
            for c in self.X_train.columns
            if pd.api.types.is_object_dtype(self.X_train[c])
        ]
        self.num_features = [
            c for c in self.X_train.columns if c not in self.cat_features
        ]

    def _impute_and_encode(self):
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(sparse=False, handle_unknown="ignore")),
            ]
        )
        transfomer = ColumnTransformer(
            [
                ("cat", cat_pipe, self.cat_features),
                (
                    "num",
                    SimpleImputer(strategy="mean"),
                    self.num_features,
                ),
            ],
            sparse_threshold=0,
        )
        X_train_trans = transfomer.fit_transform(self.X_train)
        X_test_trans = transfomer.transform(self.X_test)
        if len(self.cat_features) > 0:
            cat_features_transformed = (
                transfomer.named_transformers_["cat"]
                .named_steps["onehot"]
                .get_feature_names(input_features=self.cat_features)
            )
            self.feature_names = np.r_[cat_features_transformed, self.num_features]
        else:
            self.feature_names = np.array(self.num_features)
        self.X_train = pd.DataFrame(X_train_trans, columns=self.feature_names)
        self.X_test = pd.DataFrame(X_test_trans, columns=self.feature_names)

    def _fit_estimator(self):
        self.estimator = RandomForestClassifier(
            n_estimators=20, min_samples_leaf=3, max_depth=12, random_state=42
        ).fit(self.X_train, self.y_train)

    def _register_predictions(self):
        self.y_pred = self.estimator.predict_proba(self.X_test)[:, 1]
