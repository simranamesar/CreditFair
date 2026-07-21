"""Shared pytest fixtures for the CreditFair suite.

This is the SINGLE source of fixtures. Do not redefine df/split/etc. in the test
module or they will shadow these and may miss the dataset.

Data lookup order:
  1. $GERMAN_DATA (explicit path)
  2. known project locations (incl. Week-1/dataset)
  3. walk up from this file and the cwd, plus sibling '*/dataset' folders
Run from the project root:  pytest --cov=creditfair --cov-report=term-missing
"""
import glob
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

# Make `import creditfair` work in a flat layout OR a src/ layout, whether this
# conftest sits in the project root or in tests/.
for _p in (HERE, os.path.join(HERE, "src"),
           os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "src")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import creditfair as cf  # noqa: E402


def _find_data():
    env = os.environ.get("GERMAN_DATA")
    if env and os.path.exists(env):
        return env

    roots = []
    for start in (HERE, os.getcwd()):
        d = start
        for _ in range(6):               # walk up to 6 levels
            roots.append(d)
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent

    # prefer the project's data/ folder, then anywhere in the tree
    patterns = [os.path.join("data", "german.data"),
                "german.data", "german.data.csv",
                os.path.join("*", "german.data"),
                os.path.join("*", "dataset", "german.data"),
                os.path.join("*", "*", "german.data")]
    for r in dict.fromkeys(roots):       # de-dup, keep order
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(r, pat)))
            if hits:
                return hits[0]

    pytest.skip("german.data not found. Place it in the project's data/ folder, "
                "or set GERMAN_DATA=/path/to/german.data.")


@pytest.fixture(scope="session")
def data_path():
    return _find_data()


@pytest.fixture(scope="session")
def df(data_path):
    return cf.load_decode(data_path)


@pytest.fixture(scope="session")
def split(df):
    X, y = df[cf.FEATURES], df["bad"]
    s = cf.make_split(X, y, df.index)
    s["Xtr"], s["Xval"], s["Xte"], _ = cf.cap_outliers(s["Xtr"], s["Xval"], s["Xte"])
    return s


@pytest.fixture(scope="session")
def rf_and_scores(df, split):
    rf = cf.fit_risk_model(split["Xtr"], split["ytr"])
    p = rf.predict_proba(split["Xte"])[:, 1]
    return rf, p


@pytest.fixture(scope="session")
def cfmodel(df, split):
    """The deployed CreditFairModel (RF risk + cost-tuned EG decision)."""
    return cf.CreditFairModel.fit(split["Xtr"], split["ytr"], split["itr"], df,
                                  Xval=split["Xval"], yval=split["yval"])
