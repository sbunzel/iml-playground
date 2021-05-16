from iml_playground import FeatureImportance


def test_importance_has_required_attributes(regression_importance: FeatureImportance):
    expected_attributes = ["ds", "model", "sorted_names", "sorted_imp"]
    assert all(a in dir(regression_importance) for a in expected_attributes)


def test_importance_is_sorted_correctly(regression_importance: FeatureImportance):
    assert regression_importance.sorted_names[-2:] == [
        "PredictiveFeatureCorrelated",
        "PredictiveFeature",
    ]
