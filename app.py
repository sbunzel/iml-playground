import streamlit as st

import iml_playground as iml
from iml_playground import importance, performance, predictions, utils

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
        chart = importance.plot_permutation_importance(
            model, title_config=ALT_TITLE_CONFIG, top_n=10
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("## Global Effects")
    left, right = st.beta_columns(2)
    with left:
        pd_feature = st.selectbox(
            label="Select a feature to calculate Partial Dependence for",
            options=model.feature_names,
        )
        chart = iml.GlobalEffects(model=model).plot(feature=pd_feature)
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.write("Placeholder")


if __name__ == "__main__":
    main()
