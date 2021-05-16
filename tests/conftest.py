from pathlib import Path

import pandas as pd
import pytest
from iml_playground import Dataset, FeatureImportance, Model


@pytest.fixture
def regression_ds() -> Dataset:
    base_path = (
        Path(__file__).parent.parent / "resources" / "data" / "simple-regression"
    )
    train = pd.read_csv(base_path / "train.csv")
    test = pd.read_csv(base_path / "test.csv")
    return Dataset(train, test, target="Target")


@pytest.fixture
def regression_model(regression_ds: Dataset) -> Model:
    return Model(ds=regression_ds, estimator_name="Linear Regression")


@pytest.fixture
def regression_importance(regression_model: Model) -> FeatureImportance:
    return FeatureImportance(model=regression_model)
