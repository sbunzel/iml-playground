import altair as alt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st


def plot_predictions(df, p_min, p_max):
    if df["target"].nunique() == 1:
        df = df.assign(focus=lambda df: df["prediction"].between(p_min, p_max))
        color = alt.Color(
            "focus:N",
            legend=None,
            scale=alt.Scale(scheme="tableau10"),
        )
    else:
        color = alt.Color("target:N", scale=alt.Scale(scheme="tableau10"))
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X(
                "prediction:Q",
                bin=alt.Bin(step=0.01),
                scale=alt.Scale(domain=(0, 1)),
                title="Predicted Probability",
            ),
            y=alt.Y("count()", title="Number of Predictions"),
            color=color,
            tooltip=["target"],
        )
        .properties(
            width="container", height=300, title="Distribution of Model Predictions"
        )
        .configure_title(fontSize=14, offset=10, orient="top", anchor="middle")
    )


def main():

    st.markdown("# IML Playground")
    st.markdown("**_An exploration into the world of interpretable machine learning_**")

    st.markdown("## The Dataset")
    train = pd.read_csv(
        f"https://raw.githubusercontent.com/sbunzel/iml-playground/main/data/car-insurance-cold-calls/train.csv"
    )
    test = pd.read_csv(
        f"https://raw.githubusercontent.com/sbunzel/iml-playground/main/resources/car-insurance-cold-calls/test.csv"
    )
    st.dataframe(test.head(100), height=300)

    # Model training
    TARGET = "CarInsurance"
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
    model = RandomForestClassifier(
        n_estimators=20, min_samples_leaf=3, max_depth=12, random_state=42
    ).fit(X_train, y_train)
    test_preds = model.predict_proba(X_test)[:, 1]

    st.markdown("## Predictions")
    distribution_plot = st.empty()
    threshold = st.slider(
        "Set the threshold for classifying an observation as class 1",
        0.0,
        1.0,
        0.5,
    )
    pred_df = pd.DataFrame(test_preds, columns=y_train.unique()).melt(
        var_name="target", value_name="prediction"
    )
    chart = plot_predictions(pred_df, p_min=threshold, p_max=1)
    distribution_plot.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()