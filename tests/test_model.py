from iml_playground import Model


def test_model_has_required_attributes(regression_model: Model):
    expected_attributes = ["ds", "estimator", "estimator_name", "y_pred"]
    assert all(a in dir(regression_model) for a in expected_attributes)


def test_model_predictions_are_1d(regression_model: Model):
    assert len(regression_model.y_pred.shape) == 1
