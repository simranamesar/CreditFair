"""CreditFair — Week 3 core module.

Pure, importable functions refactored out of the Week-2 notebook so they can be
unit-tested. No plotting, no notebook globals — everything is a function with
explicit inputs. This is the backbone the test suite (tests/test_creditfair.py)
and the deployable predict() pipeline hang off.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

RS = 42
COST = {"fp_good_as_bad": 1, "fn_bad_as_good": 5}   # official Statlog 5:1 matrix

COLS = ["checking_status", "duration_months", "credit_history", "purpose",
        "credit_amount_DM", "savings", "employment_since", "installment_rate_pct",
        "personal_status_sex", "other_debtors", "residence_since_yrs", "property",
        "age_years", "other_installment_plans", "housing", "existing_credits",
        "job", "people_liable", "telephone", "foreign_worker", "risk"]

SEX = {"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"}
MARITAL = {"A91": "divorced/separated", "A92": "married/div/sep", "A93": "single",
           "A94": "married/widowed", "A95": "single"}
DECODE = {
    "checking_status": {"A11": "<0 DM", "A12": "0-200 DM", "A13": ">=200 DM", "A14": "no account"},
    "credit_history": {"A30": "no credits/all paid", "A31": "all paid this bank",
                       "A32": "paid till now", "A33": "past delay", "A34": "critical/other credits"},
    "purpose": {"A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment",
                "A43": "radio/TV", "A44": "domestic appliances", "A45": "repairs",
                "A46": "education", "A47": "vacation", "A48": "retraining", "A49": "business", "A410": "others"},
    "savings": {"A61": "<100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">=1000 DM", "A65": "unknown/none"},
    "employment_since": {"A71": "unemployed", "A72": "<1 yr", "A73": "1-4 yrs", "A74": "4-7 yrs", "A75": ">=7 yrs"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "life insurance", "A123": "car/other", "A124": "unknown/none"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {"A171": "unemployed/unskilled non-res", "A172": "unskilled resident",
            "A173": "skilled", "A174": "management/self-emp"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

NUM = ["age_years", "duration_months", "credit_amount_DM", "installment_rate_pct",
       "residence_since_yrs", "existing_credits", "people_liable"]
CAT = ["checking_status", "credit_history", "purpose", "savings", "employment_since",
       "other_debtors", "property", "other_installment_plans", "housing", "job",
       "telephone"]
# NOTE: marital_status is DROPPED from the decision. In this dataset it is a *perfect*
# proxy for sex (every female is coded married/div/sep; there are no female-single rows),
# so keeping it would let the model reconstruct sex exactly — undermining the legal
# safeguard. It's still decoded for auditing, just never used to decide.
FEATURES = NUM + CAT                 # EXCLUDES sex, foreign_worker AND marital_status; KEEPS age
PROTECTED = ["sex", "age_group", "foreign_worker"]


# ---------------------------------------------------------------- data
def load_decode(path: str) -> pd.DataFrame:
    """Load german.data, decode A** codes, split sex/marital, derive bad + age_group."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    df["sex"] = df["personal_status_sex"].map(SEX)                 # audit-only
    df["marital_status"] = df["personal_status_sex"].map(MARITAL)  # kept feature
    for c, m in DECODE.items():
        df[c] = df[c].map(m).fillna(df[c])
    df["risk"] = df["risk"].map({1: "good", 2: "bad"})
    df["bad"] = (df["risk"] == "bad").astype(int)
    df["age_group"] = pd.cut(df["age_years"], [18, 25, 35, 50, 120],
                             labels=["18-25", "26-35", "36-50", "50+"]).astype(str)
    return df


def make_split(X, y, index, seed: int = RS):
    """Stratified 600/200/200 train/val/test. Returns dict of frames + index arrays."""
    Xtr, Xtmp, ytr, ytmp, itr, itmp = train_test_split(
        X, y, index, test_size=0.4, stratify=y, random_state=seed)
    Xval, Xte, yval, yte, ival, ite = train_test_split(
        Xtmp, ytmp, itmp, test_size=0.5, stratify=ytmp, random_state=seed)
    return dict(Xtr=Xtr.copy(), Xval=Xval.copy(), Xte=Xte.copy(),
                ytr=ytr, yval=yval, yte=yte, itr=itr, ival=ival, ite=ite)


def cap_outliers(Xtr, Xval, Xte, cols=("duration_months", "credit_amount_DM"), q=0.99):
    """Cap numeric outliers at the TRAIN q-quantile; apply to val/test (no leakage)."""
    caps = {}
    Xtr, Xval, Xte = Xtr.copy(), Xval.copy(), Xte.copy()
    for col in cols:
        hi = Xtr[col].quantile(q)
        caps[col] = hi
        for D in (Xtr, Xval, Xte):
            D[col] = D[col].clip(upper=hi)
    return Xtr, Xval, Xte, caps


def make_pre(num=NUM, cat=CAT) -> ColumnTransformer:
    """One-hot categoricals + scale numerics."""
    return ColumnTransformer([("num", StandardScaler(), list(num)),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), list(cat))])


# ---------------------------------------------------------------- metrics
def total_cost(yt, yp) -> int:
    """5:1 cost: false-approve (bad->good) costs 5, false-refuse (good->bad) costs 1."""
    yt = np.asarray(yt); yp = np.asarray(yp)
    fp = int(((yp == 1) & (yt == 0)).sum())        # good refused
    fn = int(((yp == 0) & (yt == 1)).sum())        # bad approved
    return fp * COST["fp_good_as_bad"] + fn * COST["fn_bad_as_good"]


def di_of(yp, s) -> float:
    """Disparate impact = min/max approval rate across groups (four-fifths metric)."""
    s = np.asarray(s); yp = np.asarray(yp)
    rates = [(yp[s == g] == 0).mean() for g in pd.unique(s) if (s == g).sum() > 0]
    mx = max(rates)
    return float(min(rates) / mx) if mx > 0 else float("nan")


def tune_threshold(proba_val, yval, thrs=None) -> float:
    """Pick the cost-minimising threshold on VALIDATION (never test)."""
    if thrs is None:
        thrs = np.linspace(0.05, 0.95, 37)
    yval = np.asarray(yval)
    costs = [total_cost(yval, (proba_val >= t).astype(int)) for t in thrs]
    return float(thrs[int(np.argmin(costs))])


def summarise(yt, yp, groups: pd.DataFrame) -> dict:
    """Per-attribute DI + equal-opportunity gap + equalized-odds gap."""
    from sklearn.metrics import recall_score
    yt = np.asarray(yt); yp = np.asarray(yp)
    out = {}
    for c in groups.columns:
        s = groups[c].values
        rb, fp = [], []
        for g in pd.unique(s):
            m = s == g
            if m.sum() == 0:
                continue
            rb.append(recall_score(yt[m], yp[m], zero_division=0))
            goods = yt[m] == 0
            fp.append(((yp[m] == 1) & goods).sum() / max(goods.sum(), 1))
        out[c] = dict(DI=round(di_of(yp, s), 3),
                      eo_gap=round(max(rb) - min(rb), 3),
                      eodds_gap=round(max(max(rb) - min(rb), max(fp) - min(fp)), 3))
    return out


# ---------------------------------------------------------------- mitigation
def reweigh(y, gdf: pd.DataFrame, clip=(0.3, 4.0)) -> np.ndarray:
    """Intersectional reweighing weights so group and outcome decorrelate."""
    y = np.asarray(y)
    key = gdf.astype(str).agg("|".join, axis=1).values
    w = np.ones(len(y))
    py = {lv: (y == lv).mean() for lv in np.unique(y)}
    for gv in np.unique(key):
        pg = (key == gv).mean()
        for lv in np.unique(y):
            m = (key == gv) & (y == lv)
            po = m.mean()
            if po > 0:
                w[m] = np.clip(pg * py[lv] / po, *clip)
    return w


def fit_risk_model(Xtr, ytr, seed=RS) -> Pipeline:
    """Random-Forest risk scorer (class-weighted)."""
    return Pipeline([("pre", make_pre()),
                     ("clf", RandomForestClassifier(n_estimators=400,
                                                    class_weight="balanced",
                                                    random_state=seed))]).fit(Xtr, ytr)


def mitigate_eg(Xtr_t, ytr, sensitive_key, seed=RS):
    """Intersectional, cost-aware ExponentiatedGradient (DemographicParity, 1:5).

    Returns the fitted mitigator; use ._pmf_predict(X)[:,1] >= 0.5 for decisions.
    Never uses a protected attribute at decision time (constraint is training-only).
    """
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, ErrorRate
    obj = ErrorRate(costs={"fp": COST["fp_good_as_bad"], "fn": COST["fn_bad_as_good"]})
    eg = ExponentiatedGradient(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        constraints=DemographicParity(), objective=obj)
    eg.fit(Xtr_t, ytr, sensitive_features=sensitive_key)
    return eg


# ---------------------------------------------------------------- oversight
def review_category(p_bad, threshold, decision, sex, age_group, foreign, band=0.05) -> str:
    """Every application is reviewed; this returns the PRIORITY tier (P1..P4).

    P1 every rejection (mandatory, GDPR Art.22) · P2 borderline approval ·
    P3 approval in a disadvantaged subgroup · P4 clear approval (sampled).
    """
    borderline = abs(p_bad - threshold) <= band
    disadvantaged = (sex == "female" and foreign == "yes") or (age_group == "18-25")
    if decision == 1:
        return "P1"
    if borderline:
        return "P2"
    if disadvantaged:
        return "P3"
    return "P4"


def explain(pipe, Xrow, top=4):
    """Reason codes from an interpretable LR companion: signed feature contributions."""
    prep, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    xt = prep.transform(Xrow)
    xt = xt.toarray()[0] if hasattr(xt, "toarray") else np.asarray(xt)[0]
    names = prep.get_feature_names_out()
    contrib = xt * clf.coef_[0]
    idx = np.argsort(np.abs(contrib))[::-1][:top]
    return [(names[i], "risk" if contrib[i] > 0 else "safe") for i in idx]


# ---------------------------------------------------------------- deployable bundle (Week 3, Goal 4)
def _dense(a):
    return np.asarray(a.todense()) if hasattr(a, "todense") else np.asarray(a)


_HUMAN = {
    "checking_status_<0 DM": "an overdrawn checking account",
    "checking_status_no account": "no checking account on file",
    "savings_<100 DM": "low savings",
    "savings_unknown/none": "no recorded savings",
    "credit_history_critical/other credits": "a critical prior-credit history",
    "credit_history_past delay": "past payment delays",
    "housing_rent": "renting rather than owning a home",
    "housing_for free": "living in free housing (no rent record)",
    "duration_months": "a long requested loan term",
    "credit_amount_DM": "a large requested amount",
    "employment_since_unemployed": "being currently unemployed",
    "employment_since_<1 yr": "short time in current employment",
    "purpose_business": "the loan purpose (business)",
    "other_installment_plans_bank": "other outstanding installment plans",
    "installment_rate_pct": "a high installment-to-income rate",
}


def _humanize(name: str) -> str:
    key = name.replace("cat__", "").replace("num__", "")
    return _HUMAN.get(key, key.replace("_", " "))


class CreditFairModel:
    """One predict path: RF risk score -> EG fairness decision -> reason codes -> review tier.

    The EG constraint is applied at *training* time only, so no protected attribute is
    ever used to decide. Reason codes and the adverse-action notice explain the **deployed
    EG decision** (EG is a mixture of linear learners, so a mixture-weighted coefficient
    gives a faithful explanation) — not a different model's risk score. `decide()` returns
    everything a caseworker screen needs.
    """
    def __init__(self, risk_model, eg, pre_eg, lr_companion, threshold=0.5, band=0.05):
        self.risk_model = risk_model
        self.eg = eg
        self.pre_eg = pre_eg
        self.lr = lr_companion            # standalone LR, kept for reconciliation only
        self.threshold = threshold        # EG decision cut, cost-tuned on validation (5:1)
        self.band = band

    @classmethod
    def fit(cls, Xtr, ytr, itr, df, Xval=None, yval=None, seed: int = RS):
        rf = fit_risk_model(Xtr, ytr, seed)
        lr = Pipeline([("pre", make_pre()),
                       ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=seed))]).fit(Xtr, ytr)
        pre_eg = make_pre()
        Xtr_t = _dense(pre_eg.fit_transform(Xtr))
        key = df.loc[itr, PROTECTED].astype(str).agg("|".join, axis=1).values
        eg = mitigate_eg(Xtr_t, ytr, key, seed)
        thr = 0.5
        if Xval is not None and yval is not None:          # cost-tune the EG cut on validation
            eg_pv = eg._pmf_predict(_dense(pre_eg.transform(Xval)))[:, 1]
            thr = tune_threshold(eg_pv, yval)
        return cls(rf, eg, pre_eg, lr, threshold=thr)

    def risk_score(self, Xrow) -> float:
        return float(self.risk_model.predict_proba(Xrow)[:, 1][0])

    def _eg_coef(self):
        """Mixture-weighted coefficient vector of the EG ensemble (its LR base learners)."""
        w = np.asarray(self.eg.weights_, dtype=float)
        coef = None
        for wk, est in zip(w, self.eg.predictors_):
            if wk == 0 or not hasattr(est, "coef_"):
                continue
            c = np.asarray(est.coef_[0], dtype=float) * wk
            coef = c if coef is None else coef + c
        if coef is None:                                    # fallback: standalone LR companion
            coef = np.asarray(self.lr.named_steps["clf"].coef_[0], dtype=float)
        return coef

    def _contrib(self, Xrow):
        xt = _dense(self.pre_eg.transform(Xrow))[0]
        return xt * self._eg_coef(), self.pre_eg.get_feature_names_out()

    def explain_decision(self, Xrow, top=5):
        """Reason codes for the DEPLOYED EG decision (faithful — EG is linear)."""
        contrib, names = self._contrib(Xrow)
        idx = np.argsort(np.abs(contrib))[::-1][:top]
        return [(names[i], "risk" if contrib[i] > 0 else "safe") for i in idx]

    def adverse_action(self, Xrow, top=4) -> str:
        """GDPR Art.22 adverse-action notice from the DEPLOYED decision — guaranteed non-empty."""
        contrib, names = self._contrib(Xrow)
        pos = [i for i in np.argsort(contrib)[::-1] if contrib[i] > 0]   # risk-increasing first
        chosen = pos[:top] if pos else list(np.argsort(np.abs(contrib))[::-1][:top])  # fallback
        body = "\n".join("  - " + _humanize(names[i]) for i in chosen)
        return ("NOTICE OF ADVERSE ACTION\n"
                "Your credit application was declined. The principal reasons were:\n" + body +
                "\n\nThis decision was reviewed by a person. You have the right to an "
                "explanation and to contest it (GDPR Art. 22).")

    def decide(self, Xrow, group: dict) -> dict:
        """group = {'sex':..,'age_group':..,'foreign_worker':..}. Returns full decision record."""
        risk = self.risk_score(Xrow)
        eg_p = float(self.eg._pmf_predict(_dense(self.pre_eg.transform(Xrow)))[:, 1][0])
        decision = int(eg_p >= self.threshold)                  # 1 = reject; cost-tuned cut
        tier = review_category(eg_p, self.threshold, decision,
                               group["sex"], group["age_group"], group["foreign_worker"], self.band)
        return {"risk_score": round(risk, 3),
                "eg_score": round(eg_p, 3),
                "decision": "REJECT" if decision else "APPROVE",
                "review": tier,                                 # P1..P4 — always human-reviewed
                "reasons": self.explain_decision(Xrow)}         # faithful to the deployed decision
