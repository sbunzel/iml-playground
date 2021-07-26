This dataset is constructed for illustratory purposes. It was put together using scikit-learn's [`make_regression`](https://github.com/sbunzel/iml-playground/blob/8e8544a3986fa41c00c8abcff24fc81c127fd862/resources/data/prepare_data.py#L12) utility and contains four feature columns and one target column. The features are:

* one predictive feature
* one random feature
* one feature that is correlated with the predictive feature
* and one feature that is correlated with the predictive feature, but has noise added to it

The target is real valued making this a regression task.
