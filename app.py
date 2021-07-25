import streamlit as st

import iml_playground as iml
from iml_playground import utils

ALTAIR_CONFIG = {
    "title_config": {"fontSize": 14, "offset": 10, "orient": "top", "anchor": "middle"},
    "scheme": "tableau10",
}
DATASET_CONFIG = {
    "car-insurance-cold-calls": {
        "target": "CarInsurance",
        "models": ["Random Forest Classifier"],
    },
    "simple-regression": {
        "target": "Target",
        "models": ["Linear Regression", "Random Forest Regressor"],
    },
}


def main():

    st.markdown("# IML Playground")
    st.markdown("**_An exploration of the world of interpretable machine learning_**")

    st.markdown("## The Dataset")

    dataset_name = st.sidebar.selectbox(
        label="Select a dataset",
        options=list(DATASET_CONFIG.keys()),
        format_func=lambda s: s.replace("-", " ").title(),
    )

    left, right = st.beta_columns(2)
    with left:
        st.markdown(utils.read_md(dataset=dataset_name, file="dataset.md"))
    with right:
        train, test = utils.read_train_test(dataset=dataset_name)
        dataset = iml.Dataset(
            train, test, target=DATASET_CONFIG[dataset_name]["target"]
        )
        st.dataframe(dataset.sample, height=300)

    st.markdown("## Model Predictions and Performance")
    estimator_name = st.sidebar.selectbox(
        label="Select the model type to use",
        options=DATASET_CONFIG[dataset_name]["models"],
    )
    model = iml.Model(ds=dataset, estimator_name=estimator_name)

    left, right = st.beta_columns(2)
    with left:
        if model.task == "classification":
            distribution_plot = st.empty()
            threshold = st.slider(
                "Set the threshold for classifying an observation as class 1",
                0.0,
                1.0,
                0.5,
            )
            chart = model.plot_predictions(
                altair_config=ALTAIR_CONFIG, p_min=threshold, p_max=1
            )
            distribution_plot.altair_chart(chart, use_container_width=True)
        else:
            chart = model.plot_predictions(altair_config=ALTAIR_CONFIG)
            st.altair_chart(chart, use_container_width=True)
    with right:
        st.markdown(utils.read_md(dataset=dataset_name, file="model_predictions.md"))

    left, right = st.beta_columns(2)
    with left:
        if model.task == "classification":
            chart = model.plot_performance(
                altair_config=ALTAIR_CONFIG, threshold=threshold
            )
        else:
            chart = model.plot_performance(
                altair_config=ALTAIR_CONFIG,
            )
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.markdown(utils.read_md(dataset=dataset_name, file="model_performance.md"))

    st.markdown("## Feature Importance")

    left, right = st.beta_columns(2)
    with left:
        st.markdown(utils.read_md(dataset=dataset_name, file="feature_importance.md"))
    with right:
        imp = iml.FeatureImportance(model=model)
        chart = imp.plot(altair_config=ALTAIR_CONFIG, top_n=10)
        st.altair_chart(chart, use_container_width=True)

    st.markdown("## Global Effects")

    GLOBAL_EFFECTS_METHODS = ["partial_dependence_plot"]

    left, right = st.beta_columns(2)
    with right:
        global_effects_method = st.selectbox(
            label="Select a method to explore",
            options=GLOBAL_EFFECTS_METHODS,
            format_func=lambda s: s.replace("_", " ").title(),
        )
        st.markdown(
            utils.read_md(dataset=dataset_name, file=f"{global_effects_method}.md")
        )
    with left:
        global_effects_feature = st.selectbox(
            label="Select a feature to calculate global effects for",
            options=imp.sorted_names[::-1],
        )
        chart = iml.GlobalEffects(
            model=model,
            method=global_effects_method,
            feature=global_effects_feature,
        ).plot(altair_config=ALTAIR_CONFIG)
        st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()
