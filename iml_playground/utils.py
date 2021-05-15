from pathlib import Path


def read_md(file):
    return Path(f"resources/markdown/{file}").read_text()
