import streamlit as st

import iml_playground as iml
from iml_playground import performance, predictions, utils

ALT_TITLE_CONFIG = {"fontSize": 14, "offset": 10, "orient": "top", "anchor": "middle"}
ALT_SCHEME = "tableau10"
TARGET = "CarInsurance"


def main():

    st.markdown("# IML Playground")
    st.markdown("**_An exploration of the world of interpretable machine learning_**")

    st.markdown("## The Dataset")

    left, right = st.beta_columns(2)
    train, test = utils.read_train_test()
    with left:
        st.markdown(utils.read_md("dataset.md"))
    with right:
        st.dataframe(test.head(100), height=300)

    st.markdown("## Model Predictions and Performance")

    model = iml.Model(train, test, target=TARGET)

    left, right = st.beta_columns(2)
    with left:
        distribution_plot = st.empty()
        threshold = st.slider(
            "Set the threshold for classifying an observation as class 1",
            0.0,
            1.0,
            0.5,
        )
        chart = predictions.plot_prediction_histogram(
            model.y_pred,
            p_min=threshold,
            p_max=1,
            title_config=ALT_TITLE_CONFIG,
            scheme=ALT_SCHEME,
        )
        distribution_plot.altair_chart(chart, use_container_width=True)
    with right:
        st.markdown(utils.read_md("model_predictions.md"))

    left, right = st.beta_columns(2)
    with left:
        chart = performance.plot_class_performance(
            y_test=model.y_test,
            test_preds=model.y_pred,
            threshold=threshold,
            title_config=ALT_TITLE_CONFIG,
            scheme=ALT_SCHEME,
        )
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.markdown(utils.read_md("model_performance.md"))

    st.markdown("## Feature Importance")

    left, right = st.beta_columns(2)
    with left:
        st.markdown(utils.read_md("feature_importance.md"))
    with right:
        imp = iml.FeatureImportance(model=model, top_n=10)
        chart = imp.plot(title_config=ALT_TITLE_CONFIG)
        st.altair_chart(chart, use_container_width=True)

    st.markdown("## Global Effects")

    GLOBAL_EFFECTS_METHODS = ["partial_dependence_plot", "accumulated_local_effects"]

    left, right = st.beta_columns(2)
    with right:
        global_effects_method = st.selectbox(
            label="Select a method to explore",
            options=GLOBAL_EFFECTS_METHODS,
            format_func=lambda s: s.replace("_", " ").title(),
        )
        st.markdown(utils.read_md(f"{global_effects_method}.md"))
    with left:
        global_effects_feature = st.selectbox(
            label="Select a feature to calculate global effects for",
            options=imp.sorted_names[::-1],
        )
        chart = iml.GlobalEffects(
            model=model,
            method=global_effects_method,
            feature=global_effects_feature,
        ).plot(title_config=ALT_TITLE_CONFIG)
        st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()
