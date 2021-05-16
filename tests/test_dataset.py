from iml_playground import Dataset


def test_dataset_has_required_attributes(regression_ds: Dataset):
    expected_attributes = [
        "cat_features",
        "feature_names",
        "num_features",
        "target",
        "test",
        "train",
        "X_test",
        "X_train",
        "y_test",
        "y_train",
    ]
    assert all(a in dir(regression_ds) for a in expected_attributes)


def test_dataset_contains_no_nas(regression_ds: Dataset):
    assert regression_ds.train.isna().sum().sum() == 0
    assert regression_ds.test.isna().sum().sum() == 0


def test_dataset_sample_contains_all_columns(regression_ds: Dataset):
    sample = regression_ds.sample
    assert all(sample.columns == regression_ds.test.columns)
