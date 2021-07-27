from iml_playground import FeatureImportance


def test_importance_has_required_attributes(regression_importance: FeatureImportance):
    expected_attributes = ["ds", "model", "sorted_names", "sorted_imp"]
    assert all(a in dir(regression_importance) for a in expected_attributes)


def test_importance_is_sorted_correctly(regression_importance: FeatureImportance):
    assert regression_importance.sorted_names[-2:] == [
        "PredictiveFeatureCorrelated",
        "PredictiveFeature",
    ]


def test_plot_data_can_be_derived_from_importances(
    regression_importance: FeatureImportance,
):
    plot_df = regression_importance._build_plot_df(top_n=2)
    assert plot_df.shape == (20, 2)
    assert list(plot_df.columns) == ["Feature", "Importance"]
    assert sorted(list(plot_df["Feature"].unique())) == [
        "PredictiveFeature",
        "PredictiveFeatureCorrelated",
    ]
