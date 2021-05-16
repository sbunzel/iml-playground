# IML Playground

*Exploring interpretable machine learning*

IML Playground is a tool to explore and play with interpretable machine learning methods. The inspiration for the name comes from the [TensorFlow Playground](https://playground.tensorflow.org/).

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/sbunzel/iml-playground/main/app.py)

Or read more in the [design doc](https://steffenbunzel.com/blog/design-docs/).

## Getting Started

To create a local development environment run

    conda env create --name iml_playground --file=environment-dev.yml

To activate the environment run

    conda activate iml_playground

To update this environment with the production dependencies run

    conda env update --file=environment.yml

## Testing

To run the unit tests, first install the source code locally by running

    pip install -e .

from the root of this directory. Then, you can execute the tests through the pytest CLI

    pytest tests/

## Contributing

This is a side project I do for fun and I can't promise I'll still be working on it in 3 months. However, this also means there'll always be tons of things to improve. So, if you'd like to contribute, I'd be happy to talk to you. Just open an issue to get started!

Before contributing code, please set up the pre-commit hooks to reduce errors and ensure consistent formatting

    pre-commit install

## Contact

[Steffen Bunzel](https://steffenbunzel.com/contact/)
