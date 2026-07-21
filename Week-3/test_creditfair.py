"""CreditFair test suite (Week 3, Goal 3).

Fixtures (df, split, rf_and_scores, cfmodel) live in conftest.py — the single
source — so they are NOT redefined here (redefining would shadow conftest and
could miss the dataset).

Run from the project root:
    pytest --cov=creditfair --cov-report=term-missing
The headline test is `test_fairness_regression_eg_improves_sex` — it FAILS if the
mitigation stops improving fairness, i.e. it guards the ethical core of the project.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Backstop import (conftest also sets this up): work in flat OR src/ layout.
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_here, "..", "src"), os.path.join(_here, ".."),
           _here, os.path.join(_here, "src")):
    _p = os.path.abspath(_p)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import creditfair as cf  # noqa: E402


# ----------------------------------------------------------- data
def test_load_decode_shape_and_rate(df):
    assert df.shape[0] == 1000
    assert abs(df["bad"].mean() - 0.30) < 0.02
    assert set(df["sex"].unique()) == {"male", "female"}
    assert df["risk"].isin(["good", "bad"]).all()


def test_decode_no_raw_codes_leak(df):
    # every categorical should be decoded (no leftover 'A##' codes)
    for c in ["checking_status", "savings", "housing", "purpose"]:
        assert not df[c].astype(str).str.match(r"^A\d").any()


def test_feature_policy_excludes_protected():
    assert "sex" not in cf.FEATURES
    assert "foreign_worker" not in cf.FEATURES
    assert "age_years" in cf.FEATURES          # age kept (lawful)
    assert "marital_status" not in cf.FEATURES  # dropped: perfect sex proxy in this data


# ----------------------------------------------------------- split
def test_split_sizes(split):
    assert len(split["Xtr"]) == 600
    assert len(split["Xval"]) == 200
    assert len(split["Xte"]) == 200


def test_split_no_leakage(split):
    itr, ival, ite = set(split["itr"]), set(split["ival"]), set(split["ite"])
    assert itr.isdisjoint(ival)
    assert itr.isdisjoint(ite)
    assert ival.isdisjoint(ite)


def test_split_stratified(split):
    for k in ("ytr", "yval", "yte"):
        assert abs(split[k].mean() - 0.30) < 0.03


def test_caps_no_leakage_from_test(df):
    X, y = df[cf.FEATURES], df["bad"]
    s = cf.make_split(X, y, df.index)
    _, _, _, caps = cf.cap_outliers(s["Xtr"], s["Xval"], s["Xte"])
    # cap is the TRAIN quantile, independent of val/test
    assert abs(caps["duration_months"] - s["Xtr"]["duration_months"].quantile(0.99)) < 1e-9


# ----------------------------------------------------------- metrics
def test_total_cost_matrix():
    # 1 bad approved (fn) = 5 ; 1 good refused (fp) = 1
    yt = np.array([1, 0, 0, 1])
    yp = np.array([0, 1, 0, 1])   # bad approved (fn), good refused (fp), ok, ok
    assert cf.total_cost(yt, yp) == 5 + 1


def test_total_cost_perfect_is_zero():
    yt = np.array([1, 0, 1, 0])
    assert cf.total_cost(yt, yt) == 0


def test_di_of_equal_groups():
    yp = np.array([0, 1, 0, 1]); s = np.array(["a", "a", "b", "b"])
    assert cf.di_of(yp, s) == 1.0            # 50% approval both groups


def test_di_of_unequal():
    # group a approved 100%, group b 0% -> DI 0
    yp = np.array([0, 0, 1, 1]); s = np.array(["a", "a", "b", "b"])
    assert cf.di_of(yp, s) == 0.0


def test_tune_threshold_picks_min_cost():
    # crafted probs: threshold should land where cost is lowest
    yval = np.array([1, 1, 0, 0, 1])
    proba = np.array([0.9, 0.8, 0.1, 0.2, 0.7])
    t = cf.tune_threshold(proba, yval)
    # at a low threshold all bads are caught (cost 0 fn) -> should be <=0.7
    assert 0.05 <= t <= 0.7


# ----------------------------------------------------------- oversight
def test_review_all_rejections_are_P1():
    assert cf.review_category(0.9, 0.5, 1, "male", "36-50", "no") == "P1"
    assert cf.review_category(0.1, 0.5, 1, "female", "18-25", "yes") == "P1"


def test_review_borderline_approval_is_P2():
    assert cf.review_category(0.52, 0.5, 0, "male", "36-50", "no") == "P2"


def test_review_disadvantaged_approval_is_P3():
    assert cf.review_category(0.1, 0.5, 0, "female", "36-50", "yes") == "P3"
    assert cf.review_category(0.1, 0.5, 0, "male", "18-25", "no") == "P3"


def test_review_clear_approval_is_P4():
    assert cf.review_category(0.05, 0.5, 0, "male", "36-50", "no") == "P4"


def test_every_application_gets_one_category():
    cats = {cf.review_category(p, 0.5, d, sx, ag, fw)
            for p in (0.05, 0.52, 0.9) for d in (0, 1)
            for sx in ("male", "female") for ag in ("18-25", "36-50") for fw in ("yes", "no")}
    assert cats <= {"P1", "P2", "P3", "P4"}


# ----------------------------------------------------------- reweigh
def test_reweigh_shape_and_range(df):
    g = df[cf.PROTECTED].iloc[:100]
    y = df["bad"].iloc[:100]
    w = cf.reweigh(y, g)
    assert len(w) == 100
    assert w.min() >= 0.3 - 1e-9 and w.max() <= 4.0 + 1e-9


# ----------------------------------------------------------- reproducibility
def test_split_reproducible(df):
    X, y = df[cf.FEATURES], df["bad"]
    a = cf.make_split(X, y, df.index, seed=42)
    b = cf.make_split(X, y, df.index, seed=42)
    assert list(a["ite"]) == list(b["ite"])


# ----------------------------------------------------------- summarise + explain (real model)
def test_summarise_keys(df, split, rf_and_scores):
    rf, p = rf_and_scores
    yp = (p >= 0.3).astype(int)
    groups = df.loc[split["ite"], cf.PROTECTED]
    out = cf.summarise(split["yte"], yp, groups)
    assert set(out) == set(cf.PROTECTED)
    for a in cf.PROTECTED:
        assert 0 <= out[a]["DI"] <= 1


def test_explain_returns_reasons(df, split):
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    lr = Pipeline([("pre", cf.make_pre()),
                   ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                              random_state=cf.RS))]).fit(split["Xtr"], split["ytr"])
    reasons = cf.explain(lr, split["Xte"].iloc[[0]], top=3)
    assert len(reasons) == 3
    assert all(d in ("risk", "safe") for _, d in reasons)


# ----------------------------------------------------------- THE fairness regression test
@pytest.mark.slow
def test_fairness_regression_eg_improves_sex(df, split, rf_and_scores):
    """Guards the ethical core: intersectional EG must IMPROVE sex DI over baseline.

    If a future change breaks mitigation, this test fails — fairness can't silently
    regress.
    """
    rf, p = rf_and_scores
    groups = df.loc[split["ite"], cf.PROTECTED]
    thr = cf.tune_threshold(rf.predict_proba(split["Xval"])[:, 1], split["yval"])
    di_base = cf.di_of((p >= thr).astype(int), groups["sex"].values)

    pre = cf.make_pre()
    Xtr_t = pre.fit_transform(split["Xtr"])
    Xte_t = pre.transform(split["Xte"])
    Xtr_t = np.asarray(Xtr_t.todense()) if hasattr(Xtr_t, "todense") else np.asarray(Xtr_t)
    Xte_t = np.asarray(Xte_t.todense()) if hasattr(Xte_t, "todense") else np.asarray(Xte_t)
    key = df.loc[split["itr"], cf.PROTECTED].astype(str).agg("|".join, axis=1).values
    eg = cf.mitigate_eg(Xtr_t, split["ytr"], key)
    yp_mit = (eg._pmf_predict(Xte_t)[:, 1] >= 0.5).astype(int)
    di_eg = cf.di_of(yp_mit, groups["sex"].values)

    assert di_eg >= di_base, f"EG did not improve sex DI ({di_eg:.3f} < {di_base:.3f})"
    assert di_eg >= 0.80, f"mitigated sex DI below four-fifths ({di_eg:.3f})"


# ----------------------------------------------------------- deployable model (Goal 4)
@pytest.mark.slow
def test_creditfairmodel_decide_structure(cfmodel, df, split):
    row = split["Xte"].iloc[[0]]
    g = df.loc[split["ite"][0], cf.PROTECTED].to_dict()
    out = cfmodel.decide(row, g)
    assert set(out) == {"risk_score", "eg_score", "decision", "review", "reasons"}
    assert 0.0 <= out["risk_score"] <= 1.0
    assert out["decision"] in ("APPROVE", "REJECT")
    assert out["review"] in ("P1", "P2", "P3", "P4")
    assert len(out["reasons"]) >= 1


@pytest.mark.slow
def test_creditfairmodel_reject_is_p1(cfmodel, df, split):
    # a rejection must always be routed P1 (mandatory review)
    for j in range(len(split["Xte"])):
        row = split["Xte"].iloc[[j]]
        g = df.loc[split["ite"][j], cf.PROTECTED].to_dict()
        out = cfmodel.decide(row, g)
        if out["decision"] == "REJECT":
            assert out["review"] == "P1"
            break


@pytest.mark.slow
def test_eg_threshold_is_cost_tuned(cfmodel):
    # the deployed EG cut is tuned on validation (5:1), not hardcoded to 0.5
    assert 0.05 <= cfmodel.threshold <= 0.95


@pytest.mark.slow
def test_decision_explanation_is_faithful_and_nonempty(cfmodel, df, split):
    # reasons explain the EG decision (via mixture-weighted coef), and an adverse-action
    # notice for any rejection must never be empty (GDPR Art.22)
    for j in range(len(split["Xte"])):
        row = split["Xte"].iloc[[j]]
        g = df.loc[split["ite"][j], cf.PROTECTED].to_dict()
        out = cfmodel.decide(row, g)
        if out["decision"] == "REJECT":
            notice = cfmodel.adverse_action(row)
            assert "  - " in notice                       # at least one listed reason
            assert len(cfmodel.explain_decision(row)) >= 1
            break


# ============================================================================
# WEAKNESS-DETECTOR tests — these feed real/constructed inputs and would FAIL
# on an actual defect (leakage, broken model, overfitting, inverted logic).
# (Distinct from the unit tests above, which check pure-function arithmetic.)
# ============================================================================

def test_no_protected_attribute_leaks_into_features(split):
    # sex, nationality, marital (its proxy) and the audit binning must NEVER be modelled
    for bad in ["sex", "foreign_worker", "marital_status", "personal_status_sex", "age_group"]:
        assert bad not in cf.FEATURES, f"{bad} leaked into FEATURES"
    assert set(split["Xtr"].columns) == set(cf.FEATURES)   # the real matrix matches the policy


@pytest.mark.slow
def test_model_beats_baseline(split, rf_and_scores):
    # a working risk model must rank far better than random (0.5); catches a broken pipeline
    from sklearn.metrics import roc_auc_score
    _, p = rf_and_scores
    assert roc_auc_score(split["yte"], p) > 0.70


@pytest.mark.slow
def test_generalizes_val_vs_test(split, rf_and_scores):
    # validation and test AUC (both held-out) must be close -> not overfit to one split
    from sklearn.metrics import roc_auc_score
    rf, p_te = rf_and_scores
    a_val = roc_auc_score(split["yval"], rf.predict_proba(split["Xval"])[:, 1])
    a_te = roc_auc_score(split["yte"], p_te)
    assert abs(a_val - a_te) < 0.12, f"large val/test gap ({a_val:.2f} vs {a_te:.2f})"


@pytest.mark.slow
def test_risky_profile_scores_higher_than_safe(cfmodel, split):
    # metamorphic: a clearly-worse applicant must score HIGHER risk than a clearly-safe one.
    # This feeds constructed inputs; a fixed-array unit test cannot catch an inverted model.
    base = {c: split["Xtr"][c].mode()[0] for c in cf.CAT}
    base.update({c: float(split["Xtr"][c].median()) for c in cf.NUM})
    safe = {**base, "checking_status": ">=200 DM", "savings": ">=1000 DM",
            "credit_history": "paid till now", "duration_months": 6,
            "credit_amount_DM": 800, "employment_since": ">=7 yrs"}
    risky = {**base, "checking_status": "<0 DM", "savings": "<100 DM",
             "credit_history": "critical/other credits", "duration_months": 60,
             "credit_amount_DM": 12000, "employment_since": "unemployed"}
    Xs = pd.DataFrame([safe])[cf.FEATURES]
    Xr = pd.DataFrame([risky])[cf.FEATURES]
    assert cfmodel.risk_score(Xr) > cfmodel.risk_score(Xs) + 0.20


@pytest.mark.slow
def test_worsening_savings_does_not_lower_risk(cfmodel, split):
    # metamorphic: making one input strictly worse (savings high -> none) must not REDUCE risk
    base = {c: split["Xtr"][c].mode()[0] for c in cf.CAT}
    base.update({c: float(split["Xtr"][c].median()) for c in cf.NUM})
    good = pd.DataFrame([{**base, "savings": ">=1000 DM"}])[cf.FEATURES]
    worse = pd.DataFrame([{**base, "savings": "<100 DM"}])[cf.FEATURES]
    assert cfmodel.risk_score(worse) >= cfmodel.risk_score(good) - 0.02