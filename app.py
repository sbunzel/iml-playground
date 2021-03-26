import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from iml_playground import utils


def main():

    st.markdown("# IML Playground")
    st.markdown("**_An exploration of interpretable machine learning_**")

    st.markdown("## The Dataset")

    left_col, right_col = st.beta_columns(2)
    train, test = utils.read_train_test()
    with left_col:
        st.dataframe(test.head(100), height=300)
    with right_col:
        st.markdown(utils.read_md("dataset.md"))

    st.markdown("## Model Predictions and Performance")

    # TODO: Factor out model training and prediction
    TARGET = "CarInsurance"
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
    model = RandomForestClassifier(
        n_estimators=20, min_samples_leaf=3, max_depth=12, random_state=42
    ).fit(X_train, y_train)
    test_preds = model.predict_proba(X_test)[:, 1]
    pred_df = pd.DataFrame(data={"target": 1, "prediction": test_preds})

    left_col, right_col = st.beta_columns(2)
    with left_col:
        distribution_plot = st.empty()
        threshold = st.slider(
            "Set the threshold for classifying an observation as class 1",
            0.0,
            1.0,
            0.5,
        )
        chart = utils.prediction_histogram(pred_df, p_min=threshold, p_max=1)
        distribution_plot.altair_chart(chart, use_container_width=True)
    with right_col:
        st.markdown(utils.read_md("model_predictions.md"))

    left_col, right_col = st.beta_columns(2)
    with left_col:
        perf_df = utils.class_performance(y_test, test_preds, threshold)
        chart = utils.plot_class_performance(perf_df)
        st.altair_chart(chart, use_container_width=True)
    with right_col:
        st.markdown(utils.read_md("model_performance.md"))


if __name__ == "__main__":
    main()
