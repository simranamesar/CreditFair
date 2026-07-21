# CreditFair — Responsible Credit-Scoring

A fair, cost-aware, explainable, human-in-the-loop credit-scoring system on the UCI
German Credit dataset (Responsible-AI & Data-Ethics, SRH Heidelberg, SS 2026).

Pipeline: **Random-Forest risk score → intersectional cost-aware ExponentiatedGradient
fair decision → every decision human-reviewed (P1–P4)**. No protected attribute (or its
proxy) is ever used to decide; the threshold is cost-tuned to the official 5:1 matrix.

## Project structure
```
CreditFair/
├── README.md
├── requirements.txt
├── pytest.ini
├── conftest.py                     # shared test fixtures (locates the dataset)
├── data/german.data                # UCI German Credit dataset
├── src/creditfair.py               # the tested model module (all core logic)
├── tests/test_creditfair.py        # 31 tests: fairness-regression + weakness-detectors
├── notebooks/CreditFair-Weeks1-3.ipynb   # merged Weeks 1–3
├── dashboard/
│   ├── app.py                      # Streamlit dashboard (imports src/creditfair.py)
│   ├── creditfair_dashboard.py     # SAME dashboard as ONE self-contained file
│   └── .streamlit/config.toml      # dark theme
└── docs/
    └── CreditFair-Documentation.pdf
```

## Setup
    pip install -r requirements.txt

## Tests (from the project root)
    pytest                                              # 31 passed
    pytest --cov=creditfair --cov-report=term-missing   # ~99% coverage

## Dashboard
    cd dashboard
    streamlit run app.py            # or: streamlit run creditfair_dashboard.py

Seven tabs: Overview (KPIs + SHAP), Fairness monitor, Drift monitor (interactive PSI),
Review queue (human confirm/override + P4 bulk-approve), Credit assessment (verdict card +
plain-language summary), Audit log (session record + CSV export), Compliance.

## Notebook
Open notebooks/CreditFair-Weeks1-3.ipynb — Week 1 (EDA) → Week 2 (model bake-off, RF chosen)
→ Week 3 (SHAP, robustness, deployed EG, tests). The final cell runs the test suite live.

## Reproducibility
Tree-model fairness numbers (e.g. baseline sex DI) shift ±0.05–0.08 across machines (not
bit-reproducible). The pattern — all three protected attributes lifted toward/over the 0.80
four-fifths line after mitigation — is stable. Run the notebook and dashboard on one machine
for a consistent submission.
