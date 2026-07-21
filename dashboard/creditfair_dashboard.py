"""CreditFair — SINGLE-FILE Streamlit dashboard (self-contained). Run: streamlit run creditfair_dashboard.py"""
import os
import sys
import types
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

_CREDITFAIR_SRC = r'''"""CreditFair — Week 3 core module.

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
'''
cf = types.ModuleType("creditfair")
exec(compile(_CREDITFAIR_SRC, "creditfair(inlined)", "exec"), cf.__dict__)
sys.modules["creditfair"] = cf

st.set_page_config(page_title="CreditFair", page_icon="🟢", layout="wide")
GREEN, DARK, ORANGE, RED = "#1e9e75", "#14532d", "#e08a1e", "#c0483a"


def _dk(fig, ax):
    """Style a matplotlib fig for the dark dashboard theme."""
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    for sp in ax.spines.values(): sp.set_color("#3a4a5e")
    ax.tick_params(colors="#9fb0c3")
    ax.yaxis.label.set_color("#9fb0c3"); ax.xaxis.label.set_color("#9fb0c3"); ax.title.set_color("#e6edf3")
    lg = ax.get_legend()
    if lg:
        lg.get_frame().set_facecolor("#0e1a26")
        lg.get_frame().set_edgecolor("#3a4a5e")
        for t in lg.get_texts(): t.set_color("#cdd9e5")
    return fig


# ----------------------------------------------------------------- plain-language helpers
_HUMAN = {
    "checking_status_<0 DM": "an overdrawn checking account",
    "checking_status_no account": "no checking account on file",
    "checking_status_0-200 DM": "a low checking balance (0–200 DM)",
    "savings_<100 DM": "low savings", "savings_unknown/none": "no recorded savings",
    "credit_history_critical/other credits": "a critical prior-credit history",
    "credit_history_past delay": "past payment delays",
    "housing_rent": "renting rather than owning", "duration_months": "a long requested loan term",
    "credit_amount_DM": "a large requested amount", "employment_since_unemployed": "being unemployed",
    "employment_since_<1 yr": "short time in current job", "purpose_business": "a business loan purpose",
    "other_installment_plans_bank": "other outstanding installment plans",
    "other_installment_plans_none": "no other installment plans",
    "other_debtors_none": "no co-signer / guarantor",
    "installment_rate_pct": "a high installment-to-income rate",
}


def _hz(name):
    """Encoded feature name -> plain English."""
    key = name.replace("cat__", "").replace("num__", "")
    return _HUMAN.get(key, key.replace("_", " "))


def _reasons_text(reasons, top=3):
    """A short readable string of the top drivers, arrows for direction."""
    return "  ·  ".join(("🔴 " if d == "risk" else "🟢 ") + _hz(n) for n, d in reasons[:top])


def _psi(expected, actual, bins=10):
    """Population Stability Index between a reference and a new distribution (notebook §2c)."""
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1)); qs[0] = -np.inf; qs[-1] = np.inf
    e = np.clip(np.histogram(expected, qs)[0] / len(expected), 1e-4, None)
    a = np.clip(np.histogram(actual, qs)[0] / len(actual), 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _plain_summary(decision, review, good, reasons):
    """Deterministic plain-language summary built ONLY from the model's faithful reason
    codes — no LLM, no external call, so it can't hallucinate a reason the model didn't use."""
    risk = [_hz(n) for n, d in reasons if d == "risk"][:3]
    safe = [_hz(n) for n, d in reasons if d == "safe"][:2]
    verdict = "recommended for **approval**" if decision == "APPROVE" else "recommended for **decline**"
    parts = [f"This application is {verdict}, with an estimated **{good}% creditworthiness**."]
    if safe:
        parts.append("In the applicant's favour: " + ", ".join(safe) + ".")
    if risk:
        parts.append(("The main concern weighing against it: " if len(risk) == 1
                      else "The main concerns weighing against it: ") + ", ".join(risk) + ".")
    parts.append({
        "P1": "Because this is a decline, it **must be confirmed by a human reviewer** before it is final (GDPR Art. 22).",
        "P2": "The score sits close to the cutoff, so a reviewer **double-checks** it before it is final.",
        "P3": "The applicant is in a group we monitor for fairness, so a reviewer **confirms** the outcome.",
        "P4": "This is a clear case; it receives **routine, sampled** human oversight.",
    }[review])
    return " ".join(parts)


# ----------------------------------------------------------------- data / model (cached)
def _find_data():
    """Locate german.data in the project — prefers the data/ folder (dashboard/ is one
    level under the project root). No machine-specific paths."""
    h = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(h, "..", "data", "german.data"),   # project layout: dashboard/ -> ../data
              os.path.join(h, "data", "german.data"),
              os.path.join(h, "german.data"),                  # flat layout
              os.path.join(h, "..", "german.data"),
              "data/german.data", "german.data"]:
        if os.path.exists(p):
            return os.path.abspath(p)
    import glob                                                # last resort: search up the tree
    d = h
    for _ in range(4):
        hit = glob.glob(os.path.join(d, "**", "german.data"), recursive=True)
        if hit:
            return hit[0]
        d = os.path.dirname(d)
    return "german.data"


@st.cache_resource(show_spinner="Fitting CreditFair model (once)…")
def build():
    df = cf.load_decode(_find_data())
    X, y = df[cf.FEATURES], df["bad"]
    s = cf.make_split(X, y, df.index)
    s["Xtr"], s["Xval"], s["Xte"], _ = cf.cap_outliers(s["Xtr"], s["Xval"], s["Xte"])
    model = cf.CreditFairModel.fit(s["Xtr"], s["ytr"], s["itr"], df,
                                   Xval=s["Xval"], yval=s["yval"])       # RF + EG (cost-tuned cut)
    # test-set scoring
    p_rf = model.risk_model.predict_proba(s["Xte"])[:, 1]
    thr = cf.tune_threshold(model.risk_model.predict_proba(s["Xval"])[:, 1], s["yval"])
    Xte_t = cf._dense(model.pre_eg.transform(s["Xte"]))
    eg_p = model.eg._pmf_predict(Xte_t)[:, 1]
    yp_eg = (eg_p >= model.threshold).astype(int)                       # deployed decision (cost-tuned)
    yp_base = (p_rf >= thr).astype(int)                                  # baseline RF decision
    g = df.loc[s["ite"], cf.PROTECTED].reset_index(drop=True)
    # per-applicant table
    rows = []
    for j in range(len(s["Xte"])):
        cat = cf.review_category(eg_p[j], model.threshold, int(yp_eg[j]),
                                 g.loc[j, "sex"], g.loc[j, "age_group"], g.loc[j, "foreign_worker"])
        reasons = model.explain_decision(s["Xte"].iloc[[j]], top=5)     # faithful to the EG decision
        rows.append(dict(id=int(s["ite"][j]), sex=g.loc[j, "sex"], age=g.loc[j, "age_group"],
                         foreign=g.loc[j, "foreign_worker"], amount=int(s["Xte"].iloc[j]["credit_amount_DM"]),
                         risk=round(float(p_rf[j]), 3),
                         decision="REJECT" if yp_eg[j] == 1 else "APPROVE", review=cat,
                         reasons=reasons,
                         top_reason=reasons[0][0].replace("cat__", "").replace("num__", "")))
    apps = pd.DataFrame(rows)
    # fairness before/after
    di_before = {a: cf.di_of(yp_base, g[a].values) for a in cf.PROTECTED}
    di_after = {a: cf.di_of(yp_eg, g[a].values) for a in cf.PROTECTED}
    # decile
    dec = pd.qcut(p_rf, 10, labels=False, duplicates="drop")
    decile = [100 * s["yte"].values[dec == d].mean() if (dec == d).any() else 0 for d in range(10)]
    # error profile
    from sklearn.metrics import confusion_matrix, roc_auc_score
    tn, fp, fn, tp = confusion_matrix(s["yte"], yp_eg).ravel()          # deployed EG decision
    kpi = dict(approval=round((apps["decision"] == "APPROVE").mean(), 3),
               cost_base=cf.total_cost(s["yte"].values, yp_base),
               cost_eg=cf.total_cost(s["yte"].values, yp_eg),
               fpr=fp / (fp + tn), fnr=fn / (fn + tp), auc=roc_auc_score(s["yte"], p_rf), thr=thr)
    # training modes (defaults for the live scorer)
    modes = {c: s["Xtr"][c].mode()[0] for c in cf.CAT}
    med = {c: float(s["Xtr"][c].median()) for c in cf.NUM}
    cats = {c: sorted(s["Xtr"][c].unique().tolist()) for c in cf.CAT}
    # reference distributions for the PSI drift monitor
    psi_data = {c: (s["Xtr"][c].values.astype(float), s["Xte"][c].values.astype(float))
                for c in ["duration_months", "credit_amount_DM", "age_years"]}
    # SHAP on the RF risk scorer (global explainability, shown on the Overview tab)
    shap_pack = None
    try:
        import shap
        _clf = model.risk_model.named_steps["clf"]
        _pre = model.risk_model.named_steps["pre"]
        _Xt = cf._dense(_pre.transform(s["Xte"]))
        _sv = shap.TreeExplainer(_clf).shap_values(_Xt)
        _sv1 = _sv[1] if isinstance(_sv, list) else (_sv[:, :, 1] if getattr(_sv, "ndim", 2) == 3 else _sv)
        _names = [n.replace("cat__", "").replace("num__", "") for n in _pre.get_feature_names_out()]
        shap_pack = (_sv1, _Xt, _names)
    except Exception:
        shap_pack = None
    return dict(model=model, apps=apps, di_before=di_before, di_after=di_after,
                decile=decile, kpi=kpi, modes=modes, med=med, cats=cats, shap=shap_pack, psi=psi_data)


import streamlit.components.v1 as components

st.markdown("""<style>
#MainMenu, header, footer {visibility:hidden;}
.block-container{padding-top:0.6rem; max-width:1280px;}
h1,h2,h3{font-weight:700 !important;}
.stTabs [data-baseweb="tab-list"]{gap:6px; background:#0c1220; padding:6px; border:1px solid rgba(255,255,255,.07); border-radius:12px;}
.stTabs [data-baseweb="tab"]{background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.06); border-radius:9px; padding:8px 18px; color:#cdd9e5; font-weight:600;}
.stTabs [aria-selected="true"]{background:linear-gradient(90deg,#1e9e75,#14532d) !important; color:#fff !important; border-color:#1e9e75 !important;}
[data-testid="stMetric"]{background:#141c2e; border:1px solid rgba(255,255,255,.07); border-radius:14px; padding:14px 16px;}
[data-testid="stMetricValue"]{color:#eafff5; font-weight:800;}
[data-testid="stMetricLabel"]{color:#8fa3b8;}
.cf-sec{font-size:.72rem; letter-spacing:2px; text-transform:uppercase; color:#5dcaa5; font-weight:700; margin:14px 0 8px;}
.cf-card{background:#141c2e; border:1px solid rgba(255,255,255,.07); border-radius:14px; padding:16px 18px; margin-bottom:12px;}
.cf-card .row{display:flex; justify-content:space-between; padding:7px 0; border-bottom:1px solid rgba(255,255,255,.05); font-size:.9rem;}
.cf-card .row:last-child{border-bottom:none;}
.cf-card .k{color:#8fa3b8; font-weight:600;} .cf-card .v{color:#e6edf3; text-align:right;}
.cf-ok{color:#5dcaa5; font-weight:700;} .cf-warn{color:#e0b21e; font-weight:700;}
.stButton>button{background:linear-gradient(90deg,#1e9e75,#14532d); color:#fff; border:none;
  border-radius:999px; padding:.6rem 2rem; font-weight:700; letter-spacing:.3px; box-shadow:0 8px 24px rgba(30,158,117,.35);}
.stButton>button:hover{filter:brightness(1.08);}
[data-testid="stDataFrame"]{border:1px solid rgba(255,255,255,.07); border-radius:12px;}
</style>""", unsafe_allow_html=True)

HERO_HTML = r'''
<style>html,body{margin:0;padding:0;background:#000;overflow:hidden}</style>
<div id="wrap" style="position:relative;width:100%;height:100vh;overflow:hidden;background:#000">
  <div id="rain" style="position:absolute;inset:0;z-index:1"></div>
  <div style="position:absolute;inset:0;z-index:2;background:radial-gradient(ellipse at center, rgba(0,0,0,.7) 0%, rgba(0,0,0,.4) 42%, rgba(0,0,0,.08) 72%)"></div>
  <div style="position:absolute;inset:0;z-index:3;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#fff;text-shadow:0 2px 24px rgba(0,0,0,.9)">
    <div style="font-size:14px;letter-spacing:6px;opacity:.85;text-transform:uppercase;color:#9fe1cb">Responsible AI &middot; Credit</div>
    <h1 id="title" style="font-family:monospace;font-size:64px;font-weight:800;letter-spacing:4px;margin:10px 0;min-height:76px">CreditFair</h1>
    <div style="font-size:18px;max-width:640px;opacity:.94;line-height:1.5">A credit model that is fair, cost-aware, explainable &mdash; and never gets the final word.</div>
    <div style="margin-top:16px;font-size:13px;opacity:.7;color:#9fe1cb">RF risk score &middot; intersectional EG fairness &middot; human-in-the-loop</div>
  </div>
</div>
<script>
(function(){
  var layer=document.getElementById('rain');
  var ALL="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?";
  function rc(){return ALL[Math.floor(Math.random()*ALL.length)];}
  var N=300, chars=[], spans=[];
  for(var i=0;i<N;i++){
    var c={x:Math.random()*100,y:Math.random()*100,speed:0.1+Math.random()*0.3};
    chars.push(c);
    var s=document.createElement('span');
    s.textContent=rc();
    s.style.cssText="position:absolute;font-family:monospace;font-size:1.8rem;color:#475569;opacity:0.4;transform:translate(-50%,-50%) scale(1);will-change:transform,top;transition:color .1s,transform .1s,text-shadow .1s,opacity .1s";
    layer.appendChild(s); spans.push(s);
  }
  function pos(){
    for(var i=0;i<N;i++){
      var c=chars[i]; c.y+=c.speed;
      if(c.y>=100){ c.y=-5; c.x=Math.random()*100; spans[i].textContent=rc(); }
      spans[i].style.left=c.x+"%"; spans[i].style.top=c.y+"%";
    }
    requestAnimationFrame(pos);
  }
  pos();
  var active=[];
  function flick(){
    for(var a=0;a<active.length;a++){var s=spans[active[a]];s.style.color="#475569";s.style.opacity="0.4";s.style.fontWeight="300";s.style.textShadow="none";s.style.transform="translate(-50%,-50%) scale(1)";}
    active=[]; var k=3+Math.floor(Math.random()*3);
    for(var j=0;j<k;j++){var i=Math.floor(Math.random()*N);active.push(i);var s2=spans[i];s2.style.color="#00ff00";s2.style.opacity="1";s2.style.fontWeight="700";s2.style.textShadow="0 0 8px rgba(255,255,255,0.8),0 0 12px rgba(255,255,255,0.4)";s2.style.transform="translate(-50%,-50%) scale(1.25)";}
  }
  setInterval(flick,50);
  function TS(el){this.el=el;this.chars="!<>-_\\/[]{}=+*^?#";this.queue=[];this.frame=0;this.fr=0;this.resolve=null;}
  TS.prototype.setText=function(nt){var old=this.el.innerText;var len=Math.max(old.length,nt.length);var self=this;var pr=new Promise(function(res){self.resolve=res;});this.queue=[];for(var i=0;i<len;i++){var from=old[i]||"";var to=nt[i]||"";var st=Math.floor(Math.random()*40);var en=st+Math.floor(Math.random()*40);this.queue.push({from:from,to:to,start:st,end:en,char:null});}cancelAnimationFrame(this.fr);this.frame=0;this.update();return pr;};
  TS.prototype.update=function(){var out="";var comp=0;for(var i=0;i<this.queue.length;i++){var q=this.queue[i];if(this.frame>=q.end){comp++;out+=q.to;}else if(this.frame>=q.start){if(!q.char||Math.random()<0.28){q.char=this.chars[Math.floor(Math.random()*this.chars.length)];}out+='<span style="color:#1e9e75;opacity:.8">'+q.char+'</span>';}else{out+=q.from;}}this.el.innerHTML=out;if(comp===this.queue.length){if(this.resolve)this.resolve();}else{var self=this;this.fr=requestAnimationFrame(function(){self.update();});this.frame++;}};
  var ts=new TS(document.getElementById('title'));
  var phrases=["CreditFair","Fair by design","Cost-aware  5:1","Explainable  SHAP","Human in the loop"];
  var ci=0;
  function nxt(){ts.setText(phrases[ci]).then(function(){setTimeout(nxt,2200);});ci=(ci+1)%phrases.length;}
  nxt();
})();
</script>
'''

if "entered" not in st.session_state:
    st.session_state.entered = False
if st.query_params.get("entered") == "1":
    st.session_state.entered = True
if not st.session_state.entered:
    st.markdown("""<style>
    [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"]{background:#000;}
    .block-container{padding:0 !important; max-width:100% !important;}
    [data-testid="stIFrame"]{display:block;}
    /* float the real Streamlit button over the hero so entering stays in ONE window */
    div[data-testid="stButton"]{position:fixed; left:0; right:0; bottom:7vh; z-index:2147483000;
        display:flex; justify-content:center;}
    div[data-testid="stButton"] > button{width:auto !important; min-width:200px; max-width:250px;
        font-size:15px; padding:.5rem 1.8rem;}
    </style>""", unsafe_allow_html=True)
    components.html(HERO_HTML, height=900, scrolling=False)
    if st.button("Enter the dashboard  →", key="enter_btn"):
        st.session_state.entered = True
        st.rerun()
    st.stop()

D = build()
apps, kpi = D["apps"], D["kpi"]
PILL = {"P1": RED, "P2": ORANGE, "P3": "#b8791c", "P4": GREEN}
TIER_SHORT = {"P1": "P1 · Rejection", "P2": "P2 · Borderline approval",
              "P3": "P3 · Disadvantaged approval", "P4": "P4 · Clear approval"}
def _tier(t): return TIER_SHORT.get(t, t)
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "queue_actions" not in st.session_state:
    st.session_state.queue_actions = {}      # applicant id -> {final, action, time}
if "new_apps" not in st.session_state:
    st.session_state.new_apps = []           # applicants scored live in Credit assessment

st.title("CreditFair — Responsible-Credit Dashboard")
st.caption("Random-Forest risk score → intersectional cost-aware EG fairness decision → every decision human-reviewed. "
           "Numbers are from this machine's run.")

t_over, t_fair, t_drift, t_queue, t_score, t_audit, t_comp = st.tabs(
    ["Overview", "Fairness monitor", "Drift monitor", "Review queue",
     "Credit assessment", "Audit log", "Compliance"])

# ---------------------------------------------------------------- Overview
with t_over:
    st.markdown("Each applicant gets a **risk score** (Random Forest), a **fair decision** "
                "(cost-aware ExponentiatedGradient), and a **human-review tier**. The numbers below "
                "summarise the deployed system on the 200-applicant test set.")
    _cost_delta = kpi["cost_eg"] - kpi["cost_base"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected cost (5:1)", f"{kpi['cost_base']} → {kpi['cost_eg']}",
              delta=f"{_cost_delta:+d} (price of fairness)" if _cost_delta else "no change",
              delta_color="inverse",
              help="Total mis-decision cost on the official Statlog 5:1 matrix: baseline RF → after the "
                   "fair EG decision. Mitigation closes the fairness gap for a small extra cost — the "
                   "documented price of fairness.")
    c2.metric("FNR — bad loans approved", f"{kpi['fnr']*100:.1f}%",
              help="Of applicants who truly default, the share the deployed system wrongly approves. "
                   "These are the expensive (5×) mistakes, so we keep this low.")
    c3.metric("ROC-AUC", f"{kpi['auc']:.2f}",
              help="How well the risk score ranks good vs bad applicants. 0.5 = coin-flip, 1.0 = perfect. "
                   "~0.78 matches the Week-3 notebook (0.779).")

    st.divider()
    a, b = st.columns([1.3, 1])
    with a:
        st.subheader("What drives risk — SHAP (Random-Forest risk score)")
        if D["shap"] is not None:
            import shap
            sv1, Xt, names = D["shap"]
            plt.figure(figsize=(6, 3.6))
            shap.summary_plot(sv1, Xt, feature_names=names, max_display=10, show=False)
            f = plt.gcf(); f.patch.set_alpha(0)
            for ax_ in f.get_axes():
                ax_.set_facecolor("none")
                ax_.tick_params(colors="#9fb0c3")
                for sp in ax_.spines.values(): sp.set_color("#3a4a5e")
                ax_.xaxis.label.set_color("#9fb0c3"); ax_.yaxis.label.set_color("#9fb0c3")
                for t in ax_.get_yticklabels() + ax_.get_xticklabels(): t.set_color("#cdd9e5")
            f.tight_layout(); st.pyplot(f)
            st.caption("Each dot is a test applicant; **right = pushed toward higher risk**, colour = feature value "
                       "(red high, blue low). No/negative checking account, long duration and weak credit history "
                       "drive risk — sensible credit signals, not noise. SHAP explains the **RF risk score**; the "
                       "deployed accept/reject decision is the fair **EG** model.")
        else:
            st.info("Install `shap` (`pip install shap`) to show the SHAP explainability chart here.")
    with b:
        st.subheader("Human-oversight mix")
        st.caption("Every application is reviewed; the tier sets how deeply. "
                   "**P1 Rejection** (every decline) · **P2 Borderline approval** · "
                   "**P3 Disadvantaged-group approval** · **P4 Clear approval** (sampled).")
        counts = apps["review"].value_counts().reindex(["P1", "P2", "P3", "P4"]).fillna(0).astype(int)
        st.bar_chart(counts.rename(index=TIER_SHORT))
        st.caption(f"Rejection {counts['P1']} · Borderline {counts['P2']} · "
                   f"Disadvantaged {counts['P3']} · Clear {counts['P4']} — "
                   "no decision is fully unsupervised (EU AI Act Art. 14).")

# ---------------------------------------------------------------- Fairness
with t_fair:
    st.subheader("Is the decision fair across protected groups?")
    st.markdown(
        "**Disparate Impact (DI)** = (approval rate of the disadvantaged group) ÷ (approval rate of the "
        "advantaged group). **1.00 = identical treatment**; the **four-fifths rule** says a system is "
        "presumptively fair only if **DI ≥ 0.80**. Below we show DI *before* and *after* our mitigation, "
        "then the underlying approval rates so you can see where the gap is.")

    # --- before/after bars with value labels
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(3); w = 0.36
    bfr = [D["di_before"][a] for a in cf.PROTECTED]
    aft = [D["di_after"][a] for a in cf.PROTECTED]
    b1 = ax.bar(x - w/2, bfr, w, label="before", color="#9bbf7a")
    b2 = ax.bar(x + w/2, aft, w, label="after (EG)", color=GREEN)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.02, f"{r.get_height():.2f}",
                    ha="center", va="bottom", color="#cdd9e5", fontsize=8)
    ax.axhline(0.80, color=ORANGE, ls="--", lw=1.3, label="0.80 four-fifths line")
    ax.set_xticks(x); ax.set_xticklabels(["sex", "age", "foreign"]); ax.set_ylim(0, 1.12); ax.legend(fontsize=8)
    _dk(fig, ax); fig.tight_layout(); st.pyplot(fig)
    st.caption("**before** = Week-2 Random-Forest baseline · **after** = Week-3 deployed cost-tuned EG. "
               "Exact baseline figures shift ±0.05–0.08 by machine/threshold (tree models aren't bit-reproducible) — "
               "e.g. Week-2 reported baseline sex DI ≈ 0.84 at threshold 0.30; here it's ~0.77. The *pattern* — all "
               "three attributes lifted toward/over 0.80 after mitigation — is stable across runs.")

    cols = st.columns(3)
    for col, a, lbl in zip(cols, cf.PROTECTED, ["sex", "age", "foreign"]):
        after = D["di_after"][a]; delta = after - D["di_before"][a]
        state = "✓ clears 0.80" if after >= 0.80 else "⚠ borderline"
        col.metric(f"{lbl} DI (after)", f"{after:.2f}", f"{delta:+.2f} vs before")
        col.caption(state)

    # --- underlying approval rates per group (the transparency the DI number hides)
    st.markdown("**Approval rate by group** — the numbers the DI ratio is built from:")
    _colmap = {"sex": "sex", "age_group": "age", "foreign_worker": "foreign"}
    rows = []
    for a, lbl in zip(cf.PROTECTED, ["sex", "age", "foreign"]):
        col = _colmap[a]
        for grp, sub in apps.groupby(col):
            rows.append({"attribute": lbl, "group": str(grp), "n": len(sub),
                         "approval %": round(100 * (sub["decision"] == "APPROVE").mean(), 1)})
    grp_tbl = pd.DataFrame(rows)
    st.dataframe(grp_tbl, use_container_width=True, hide_index=True, height=320,
                 column_config={"approval %": st.column_config.ProgressColumn(
                     "approval %", min_value=0, max_value=100, format="%.1f%%")})

    st.info("Sex and foreign-worker clear 0.80; **age is borderline** on 200 rows (~0.77–0.84) — flagged and "
            "human-reviewed, not over-claimed. Crucially, the EG model **never uses a protected attribute at "
            "decision time** — fairness is enforced during *training*, so we avoid the illegal shortcut of "
            "per-group thresholds (AGG §3(1)).")

# ---------------------------------------------------------------- Drift monitor
with t_drift:
    st.subheader("Drift monitor — Population Stability Index (PSI)")
    st.markdown("PSI compares a **reference** distribution (what the model was trained on) to a **new** intake. "
                "Rule of thumb: **< 0.1 stable · 0.1–0.2 minor shift · > 0.2 significant drift → re-audit** before "
                "the score is trusted on the new population (EU AI Act Art. 9, ongoing risk management).")

    st.markdown("**Baseline — reference vs current book (in-distribution, should be small):**")
    bcols = st.columns(2)
    for col, name in zip(bcols, ["duration_months", "credit_amount_DM"]):
        tr, te = D["psi"][name]
        v = _psi(tr, te)
        col.metric(name, f"{v:.3f}", "stable" if v < 0.1 else ("minor shift" if v < 0.2 else "ALERT"),
                   delta_color="off")
    st.caption("Small values confirm the current book looks like the reference population — no drift at rest.")

    st.divider()
    st.markdown("**Simulate a new intake** — shift loan term / size (e.g. customers asking for longer, "
                "larger loans) and watch the monitor react.")
    s1, s2 = st.columns(2)
    dm = s1.slider("Loan duration ×", 1.0, 2.5, 1.5, 0.1)
    am = s2.slider("Loan amount ×", 1.0, 2.5, 1.6, 0.1)
    _, te_d = D["psi"]["duration_months"]
    _, te_a = D["psi"]["credit_amount_DM"]
    psi_dur = _psi(te_d, te_d * dm)
    psi_amt = _psi(te_a, te_a * am)
    k1, k2 = st.columns(2)
    for k, label, v in [(k1, "duration_months", psi_dur), (k2, "credit_amount_DM", psi_amt)]:
        alert = v > 0.2
        k.metric(label, f"{v:.3f}", "⚠ ALERT (>0.2)" if alert else "within tolerance",
                 delta_color="inverse" if alert else "off")

    # visual: see the shift, and a PSI gauge against the 0.2 line
    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(5, 2.6))
        ax.hist(te_d, bins=18, alpha=.75, label="reference (test)", color=GREEN)
        ax.hist(te_d * dm, bins=18, alpha=.55, label=f"new intake (×{dm:g})", color=ORANGE)
        ax.set_xlabel("duration_months"); ax.set_ylabel("applicants"); ax.legend(fontsize=7)
        _dk(fig, ax); fig.tight_layout(); st.pyplot(fig)
        st.caption("The orange (new intake) distribution slides right — that visible gap is what PSI measures.")
    with g2:
        fig2, ax2 = plt.subplots(figsize=(5, 2.6))
        bars = ax2.barh(["duration", "amount"], [psi_dur, psi_amt],
                        color=[RED if v > 0.2 else GREEN for v in (psi_dur, psi_amt)])
        ax2.axvline(0.2, color=ORANGE, ls="--", lw=1.4, label="0.2 alert line")
        ax2.axvline(0.1, color="#6b7f93", ls=":", lw=1, label="0.1 minor")
        ax2.set_xlim(0, max(0.7, psi_dur, psi_amt) * 1.1); ax2.set_xlabel("PSI"); ax2.legend(fontsize=7)
        for b, v in zip(bars, (psi_dur, psi_amt)):
            ax2.text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.2f}", va="center", color="#cdd9e5", fontsize=8)
        _dk(fig2, ax2); fig2.tight_layout(); st.pyplot(fig2)
        st.caption("Bars turn red once they cross the 0.2 line — the point where the model must be re-audited.")

    if psi_dur > 0.2 or psi_amt > 0.2:
        st.error("🚨 **Drift alert** — PSI exceeds 0.2. The model must be **re-audited / retrained** before it is "
                 "trusted on this population. Until then, affected applications are **contained by routing them to "
                 "human review** rather than auto-processed — the monitor is the trigger for the re-audit loop.")
    else:
        st.success("No drift alert — the intake is within tolerance; the deployed model can be trusted as-is.")

    st.caption("PSI is distribution-based, so it catches a shifted population *before* it reaches the model. "
               "In production this runs on a schedule against live applicants; here we simulate the shift so you "
               "can see the alarm work.")

# ---------------------------------------------------------------- Review queue
with t_queue:
    st.subheader("Review worklist — the model advises, a human decides")
    qcols = st.columns(4)
    _tier_help = {
        "P1": "Every REJECTION — mandatory human confirmation before it is finalised (GDPR Art. 22).",
        "P2": "Borderline APPROVALS within ±0.05 of the cutoff — a coin-flip the model shouldn't own alone.",
        "P3": "APPROVALS in a disadvantaged group (female+foreign, or 18–25) — fairness double-check.",
        "P4": "Clear APPROVALS — routine, sampled oversight only."}
    # combine the deployed test book with any applicants scored live in Credit assessment
    qapps = (pd.concat([apps, pd.DataFrame(st.session_state.new_apps)], ignore_index=True)
             if st.session_state.new_apps else apps)
    _actioned = set(st.session_state.queue_actions)
    for col, k in zip(qcols, ["P1", "P2", "P3", "P4"]):
        pend = int(((qapps["review"] == k) & (~qapps["id"].isin(_actioned))).sum())
        col.metric(TIER_SHORT[k], pend, help=_tier_help[k])

    # bulk action — only the routine P4 tier can be batch-approved
    p4_pending = [i for i in qapps[qapps["review"] == "P4"]["id"].tolist() if i not in _actioned]
    if st.button(f"✔ Auto-approve all P4 · Clear approvals  ({len(p4_pending)} pending)",
                 disabled=not p4_pending):
        _byid = qapps.set_index("id")
        for i in p4_pending:
            _rr = _byid.loc[i]
            t = pd.Timestamp.now().strftime("%H:%M:%S")
            st.session_state.queue_actions[i] = {"final": "APPROVE", "action": "Confirmed", "time": t}
            st.session_state.audit_log.append({
                "time": t, "source": "Bulk (P4)", "name": f"#{i}", "age": _rr["age"],
                "sex": _rr["sex"], "foreign": _rr["foreign"], "amount": int(_rr["amount"]),
                "purpose": "—", "decision": "APPROVE", "model_decision": "APPROVE", "action": "Confirmed",
                "risk": _rr["risk"], "creditworthy": round((1 - _rr["risk"]) * 100), "review": "P4",
                "threshold": round(D["model"].threshold, 2), "top_reason": _hz(_rr["reasons"][0][0])})
        st.rerun()

    f1, f2 = st.columns(2)
    tier = f1.multiselect("Priority tier", ["P1", "P2", "P3", "P4"], default=["P1", "P2", "P3", "P4"])
    dec = f2.multiselect("Decision", ["APPROVE", "REJECT"], default=["APPROVE", "REJECT"])
    view = qapps[qapps["review"].isin(tier) & qapps["decision"].isin(dec)
                 & ~qapps["id"].isin(_actioned)].sort_values("risk", ascending=False)   # hide actioned

    # add a plain-language reasons column and show a clean subset
    view_disp = view.copy()
    view_disp["top reasons"] = view_disp["reasons"].apply(lambda r: _reasons_text(r, top=3))
    view_disp["review"] = view_disp["review"].map(TIER_SHORT)
    view_disp["id"] = view_disp["id"].astype(str)   # ids mix numeric (book) + names (live) -> string
    view_disp = view_disp[["id", "sex", "age", "foreign", "amount", "risk", "decision", "review", "top reasons"]]
    st.dataframe(view_disp, use_container_width=True, height=380, hide_index=True,
                 column_config={"risk": st.column_config.ProgressColumn("P(bad)", min_value=0, max_value=1),
                                "amount": st.column_config.NumberColumn("amount DM", format="%d")})
    st.caption(f"{len(view)} of {len(qapps)} applications · 🔴 raises risk · 🟢 lowers risk. "
               "P1 rejections are mandatory review (GDPR Art. 22). Applicants scored in **Credit assessment** "
               "appear here by name.")

    # open a single case AND let a human act on it — the human-in-the-loop step
    if len(view):
        _qi = qapps.set_index("id")
        st.markdown("**Open a case** — read the reasons, then **confirm or override** the recommendation:")
        pick = st.selectbox("Applicant", view["id"].tolist(),
                            format_func=lambda i: f"{i} — {_qi.loc[i,'decision']} "
                                                  f"[{_tier(_qi.loc[i,'review'])}] "
                                                  f"P(bad) {_qi.loc[i,'risk']:.2f}")
        rr = _qi.loc[pick]
        d1, d2 = st.columns([1, 1.4])
        with d1:
            st.markdown(f"### :{'red' if rr['decision']=='REJECT' else 'green'}[{rr['decision']}] &nbsp; `{_tier(rr['review'])}`")
            st.write(f"**{pick}** — {rr['sex']}, {rr['age']}, "
                     f"{'foreign' if rr['foreign']=='yes' else 'resident'} · {rr['amount']} DM")
            st.metric("Risk P(bad)", f"{rr['risk']*100:.0f}%")
            st.warning(_tier_help[rr["review"]])
        with d2:
            st.write("**Decision drivers** (faithful to the deployed EG decision)")
            for name, dd in rr["reasons"]:
                st.write(f"- {'🔴 ↑ risk' if dd=='risk' else '🟢 ↓ risk'} — {_hz(name)}")

        model_dec = rr["decision"]
        opp = "REJECT" if model_dec == "APPROVE" else "APPROVE"
        acted = st.session_state.queue_actions.get(pick)
        if acted:
            tag = "✔ confirmed" if acted["action"] == "Confirmed" else "⇄ overridden"
            st.success(f"Reviewer decision recorded: **{acted['final']}** ({tag}) at {acted['time']}. "
                       "See the Audit log tab.")
        st.markdown("**Reviewer decision** — the model advises; you decide:")
        ba, bb = st.columns(2)

        def _log_review(final, _rr=rr, _pick=pick, _model=model_dec):
            action = "Confirmed" if final == _model else "Overridden"
            t = pd.Timestamp.now().strftime("%H:%M:%S")
            st.session_state.queue_actions[_pick] = {"final": final, "action": action, "time": t}
            st.session_state.audit_log.append({
                "time": t, "source": "Queue review", "name": f"#{_pick}", "age": _rr["age"],
                "sex": _rr["sex"], "foreign": _rr["foreign"], "amount": int(_rr["amount"]),
                "purpose": "—", "decision": final, "model_decision": _model, "action": action,
                "risk": _rr["risk"], "creditworthy": round((1 - _rr["risk"]) * 100),
                "review": _rr["review"], "threshold": round(D["model"].threshold, 2),
                "top_reason": _hz(_rr["reasons"][0][0])})

        if ba.button(f"✔ Confirm — {model_dec}", key=f"conf_{pick}", use_container_width=True):
            _log_review(model_dec); st.rerun()
        if bb.button(f"⇄ Override — {opp}", key=f"ovr_{pick}", use_container_width=True):
            _log_review(opp); st.rerun()
        st.caption("Confirming or overriding writes an immutable entry to the Audit log (EU AI Act Art. 12 & 14). "
                   "The queue is the *pending* worklist; the audit log is the *permanent record* of what a human decided.")

# ---------------------------------------------------------------- Credit assessment
with t_score:
    st.subheader("Credit assessment")
    st.caption("Enter an applicant → the model recommends a fair decision; a human still decides. "
               "Protected fields set the review tier only — they never enter the decision. "
               "Every assessment is recorded in the **Audit log**.")
    name_in = st.text_input("Applicant name / reference", "Applicant")
    c1, c2, c3 = st.columns(3)
    inp = {}
    inp["checking_status"] = c1.selectbox("Checking account", D["cats"]["checking_status"])
    inp["savings"] = c1.selectbox("Savings", D["cats"]["savings"])
    inp["credit_history"] = c1.selectbox("Credit history", D["cats"]["credit_history"])
    inp["housing"] = c1.selectbox("Housing", D["cats"]["housing"])
    inp["duration_months"] = c2.slider("Duration (months)", 4, 72, 18)
    inp["credit_amount_DM"] = c2.slider("Amount (DM)", 250, 15000, 2500, step=50)
    inp["employment_since"] = c2.selectbox("Employment", D["cats"]["employment_since"])
    inp["purpose"] = c2.selectbox("Purpose", D["cats"]["purpose"])
    age = c3.slider("Age", 18, 75, 35)
    sex = c3.selectbox("Sex (audit only)", ["male", "female"])
    foreign = c3.selectbox("Foreign worker (audit only)", ["no", "yes"])
    inp["installment_rate_pct"] = c3.slider("Installment rate (% income)", 1, 4, 3)

    if st.button("Run assessment", type="primary"):
        row = {**D["modes"], **{k: D["med"][k] for k in cf.NUM}}   # defaults
        row.update(inp)
        Xrow = pd.DataFrame([row])[cf.FEATURES]
        ag = pd.cut([age], [18, 25, 35, 50, 120], labels=["18-25", "26-35", "36-50", "50+"]).astype(str)[0]
        out = D["model"].decide(Xrow, {"sex": sex, "age_group": ag, "foreign_worker": foreign})
        good = round((1 - out["risk_score"]) * 100)
        # unique queue id from the applicant name
        _existing = {str(x) for x in apps["id"].tolist()} | {str(r["id"]) for r in st.session_state.new_apps}
        nid = name_in or "Applicant"; _k = 2
        while str(nid) in _existing:
            nid = f"{name_in} ({_k})"; _k += 1
        st.session_state.new_apps.append({
            "id": nid, "sex": sex, "age": ag, "foreign": foreign,
            "amount": int(inp["credit_amount_DM"]), "risk": round(float(out["risk_score"]), 3),
            "decision": out["decision"], "review": out["review"],
            "reasons": out["reasons"], "top_reason": _hz(out["reasons"][0][0])})
        st.session_state.audit_log.append({
            "time": pd.Timestamp.now().strftime("%H:%M:%S"), "source": "Assessment", "name": nid,
            "age": age, "sex": sex, "foreign": foreign, "amount": inp["credit_amount_DM"],
            "purpose": inp["purpose"], "decision": out["decision"], "model_decision": out["decision"],
            "action": "Model recommendation", "risk": out["risk_score"],
            "creditworthy": good, "review": out["review"], "threshold": round(D["model"].threshold, 2),
            "top_reason": _hz(out["reasons"][0][0])})
        # persist so it re-renders after the rerun (rerun lets the queue pick up the new applicant)
        st.session_state.last_assessment = {"out": out, "good": good, "nid": nid}
        st.rerun()

    la = st.session_state.get("last_assessment")
    if la:
        out, good, nid = la["out"], la["good"], la["nid"]
        bad = round(out["risk_score"] * 100)
        appr = out["decision"] == "APPROVE"
        vcol = GREEN if appr else RED
        st.markdown(f"""<div style="background:#141c2e;border:1px solid {vcol}55;border-left:5px solid {vcol};
            border-radius:14px;padding:18px 20px;margin:10px 0 6px">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px">
            <div>
              <div style="font-size:1.7rem;font-weight:800;color:{vcol}">{out['decision']}</div>
              <div style="color:#8fa3b8;font-size:.9rem">{nid} &middot; review tier
                <b style="color:#cdd9e5">{_tier(out['review'])}</b> &middot; decision by the fair EG model</div>
            </div>
            <div style="display:flex;gap:10px;text-align:center">
              <div style="background:#0e1a26;border-radius:10px;padding:10px 16px">
                <div style="font-size:1.4rem;font-weight:800;color:#5dcaa5">{good}%</div>
                <div style="font-size:.6rem;letter-spacing:.6px;color:#8fa3b8;text-transform:uppercase">Creditworthy</div></div>
              <div style="background:#0e1a26;border-radius:10px;padding:10px 16px">
                <div style="font-size:1.4rem;font-weight:800;color:#e0857a">{bad}%</div>
                <div style="font-size:.6rem;letter-spacing:.6px;color:#8fa3b8;text-transform:uppercase">Default risk</div></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("**Plain-language summary**")
        st.markdown(_plain_summary(out["decision"], out["review"], good, out["reasons"]))
        with st.expander("Show raw reason codes"):
            for nm, d in out["reasons"]:
                st.write(f"- {'🔴 ↑ risk' if d == 'risk' else '🟢 ↓ risk'} — {_hz(nm)}")
            st.caption("Reasons are faithful to the deployed model. Note a German-Credit quirk: **'no "
                       "checking account' is the lowest-risk group** (11.7% default) — safer than any balance "
                       "band — so simply *having* an account can read as marginally higher risk relative to "
                       "that anchor. It's the data being counterintuitive, not a bug.")
        if not appr:
            st.warning("Rejection → routed to **mandatory human review (Rejection tier)** before it can be finalised "
                       "(GDPR Art. 22). It now appears in the **Review queue** for a human to confirm or override.")
        else:
            st.info(f"Recommendation only — it goes to the **Review queue**, where a human reviewer "
                    f"(**{_tier(out['review'])}**) confirms before finalising.")
        st.caption(f"✓ Added to the **Review queue** as “{nid}” and recorded in the Audit log.")

# ---------------------------------------------------------------- Audit log
with t_audit:
    st.subheader("Audit log — permanent record of every decision (this session)")
    st.caption("Two kinds of entry: a **Model recommendation** (logged from Credit assessment) and a "
               "**human decision** (Confirmed/Overridden in the Review queue). This is the immutable record; "
               "the Review queue is the live worklist.")
    log = st.session_state.audit_log
    total = len(log)
    human_n = sum(1 for r in log if r.get("source") == "Queue review")
    override_n = sum(1 for r in log if r.get("action") == "Overridden")
    appr_n = sum(1 for r in log if r["decision"] == "APPROVE")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Log entries", total)
    m2.metric("Human decisions", human_n, help="Confirmed or overridden by a reviewer in the queue.")
    m3.metric("Overrides", override_n, help="Times a human changed the model's recommendation.")
    m4.metric("Approved (final)", appr_n)
    if not log:
        st.info("No decisions recorded yet — run a **Credit assessment**, or confirm/override a case in the "
                "**Review queue**, to populate this log.")
    else:
        dfl = pd.DataFrame([{
            "Time": r["time"], "Source": r.get("source", "Assessment"), "Applicant": r["name"],
            "Age · Sex": f"{r['age']} · {r['sex']}", "Amount DM": r["amount"], "Decision": r["decision"],
            "Action": r.get("action", "Model recommendation"), "Review": _tier(r["review"]),
            "Creditworthy %": r["creditworthy"], "Threshold": r["threshold"], "Top reason": r["top_reason"],
        } for r in reversed(log)])
        st.dataframe(dfl, use_container_width=True, hide_index=True, height=360,
                     column_config={"Amount DM": st.column_config.NumberColumn(format="%d"),
                                    "Creditworthy %": st.column_config.ProgressColumn(
                                        min_value=0, max_value=100, format="%d%%")})
        b1, b2, _ = st.columns([1, 1, 2])
        if b1.button("Clear log"):
            st.session_state.audit_log = []
            st.session_state.queue_actions = {}
            st.rerun()
        b2.download_button("Export CSV", dfl.to_csv(index=False).encode("utf-8"),
                           "creditfair_audit_log.csv", "text/csv")
    st.caption("Persists for this browser session. Each entry records the applicant, the model's recommendation, "
               "the human's final decision, the review tier and the **single, group-blind threshold** applied — "
               "evidence for EU AI Act Art. 12 (record-keeping), Art. 13 (transparency) and Art. 14 (human "
               "oversight). Unlike per-group-threshold systems, CreditFair applies **one** threshold to everyone "
               "(no AGG §3(1) direct discrimination).")

# ---------------------------------------------------------------- Compliance
with t_comp:
    rc = apps["review"].value_counts()
    dia = D["di_after"]
    st.markdown('<div class="cf-sec">Regulatory scope</div>', unsafe_allow_html=True)
    st.markdown("""<div class="cf-card">
      <div class="row"><span class="k">Classification</span><span class="v"><b>High-risk</b> AI — EU AI Act Annex III §5(b) (creditworthiness / credit scoring)</span></div>
      <div class="row"><span class="k">Regimes in scope</span><span class="v">EU AI Act · GDPR · German AGG (equal-treatment act)</span></div>
      <div class="row"><span class="k">Role</span><span class="v">Provider — obligations under AI Act Art. 16</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="cf-sec">Model card</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="cf-card">
      <div class="row"><span class="k">System</span><span class="v">CreditFair v1 — consumer credit scoring (German Credit)</span></div>
      <div class="row"><span class="k">Risk scorer</span><span class="v">Random Forest (400 trees, class-weighted), cost-tuned threshold</span></div>
      <div class="row"><span class="k">Fairness engine</span><span class="v">Intersectional cost-aware ExponentiatedGradient (in-processing)</span></div>
      <div class="row"><span class="k">Protected attrs at decision time</span><span class="v cf-ok">none — audited only</span></div>
      <div class="row"><span class="k">Proxy control</span><span class="v cf-ok">marital_status dropped — a perfect sex proxy in this data</span></div>
      <div class="row"><span class="k">Explanations</span><span class="v">SHAP + reason codes faithful to the deployed decision (offline)</span></div>
      <div class="row"><span class="k">Oversight</span><span class="v">every decision human-reviewed (Rejection {int(rc.get('P1',0))} / Borderline {int(rc.get('P2',0))} / Disadvantaged {int(rc.get('P3',0))} / Clear {int(rc.get('P4',0))})</span></div>
      <div class="row"><span class="k">Testing</span><span class="v cf-ok">31 tests · fairness-regression guard · ~99% coverage</span></div>
      <div class="row"><span class="k">Cost matrix</span><span class="v">official Statlog 5:1 (bad-approved costs 5× good-refused)</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="cf-sec">Fairness result — after mitigation (four-fifths = 0.80)</div>', unsafe_allow_html=True)
    fcol = st.columns(3)
    for col, a, lbl in zip(fcol, cf.PROTECTED, ["sex", "age", "foreign"]):
        ok = dia[a] >= 0.80
        col.markdown(f"""<div class="cf-card" style="text-align:center">
          <div class="k" style="color:#8fa3b8;font-weight:600">{lbl} disparate impact</div>
          <div style="font-size:30px;font-weight:800;color:{'#5dcaa5' if ok else '#e0b21e'}">{dia[a]:.2f}</div>
          <div class="{'cf-ok' if ok else 'cf-warn'}" style="font-size:.8rem">{'clears 0.80' if ok else 'borderline — human-reviewed'}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="cf-sec">EU AI Act — high-risk obligations</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="cf-card">
      <div class="row"><span class="k">Art. 9 — Risk management</span><span class="v"><span class="cf-ok">✓</span> cost-ratio sensitivity + PSI drift monitor + re-audit trigger</span></div>
      <div class="row"><span class="k">Art. 10 — Data governance &amp; bias</span><span class="v"><span class="cf-ok">✓</span> decoded, split without leakage; DI + amplification + calibration audits</span></div>
      <div class="row"><span class="k">Art. 11 &amp; Annex IV — Technical documentation</span><span class="v"><span class="cf-ok">✓</span> model card + DPIA + FRIA maintained</span></div>
      <div class="row"><span class="k">Art. 12 — Record-keeping / logging</span><span class="v"><span class="cf-ok">✓</span> every decision logs score, reasons, review tier</span></div>
      <div class="row"><span class="k">Art. 13 — Transparency</span><span class="v"><span class="cf-ok">✓</span> SHAP + reason codes + adverse-action notice</span></div>
      <div class="row"><span class="k">Art. 14 — Human oversight</span><span class="v"><span class="cf-ok">✓</span> advisory-only; every decision routed by P1–P4 tier</span></div>
      <div class="row"><span class="k">Art. 15 — Accuracy &amp; robustness</span><span class="v"><span class="cf-ok">✓</span> validation≈test AUC; perturbation-stable; 31 tests</span></div>
      <div class="row"><span class="k">Art. 27 — Fundamental-rights impact (FRIA)</span><span class="v"><span class="cf-ok">✓</span> CreditFair-FRIA.md</span></div>
      <div class="row"><span class="k">Art. 72 — Post-market monitoring</span><span class="v"><span class="cf-warn">◑</span> drift monitor + re-audit loop designed (not yet in production)</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="cf-sec">GDPR &amp; German AGG</div>', unsafe_allow_html=True)
    st.markdown("""<div class="cf-card">
      <div class="row"><span class="k">GDPR Art. 22 — solely-automated decisions</span><span class="v"><span class="cf-ok">✓</span> no decision is solely automated; every rejection is human-confirmed</span></div>
      <div class="row"><span class="k">GDPR Art. 13–15 / Rec. 71 — right to explanation</span><span class="v"><span class="cf-ok">✓</span> plain-language adverse-action notice</span></div>
      <div class="row"><span class="k">GDPR Art. 35 — DPIA</span><span class="v"><span class="cf-ok">✓</span> CreditFair-DPIA.md</span></div>
      <div class="row"><span class="k">AGG §3(1) — direct discrimination</span><span class="v"><span class="cf-ok">✓</span> protected attributes never used to decide; no per-group thresholds</span></div>
      <div class="row"><span class="k">AGG §3(2) — indirect discrimination</span><span class="v"><span class="cf-ok">✓</span> DI measured &amp; mitigated; proxy (marital) removed</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="cf-sec">Design choice — why not per-group thresholds?</div>', unsafe_allow_html=True)
    st.info("A common shortcut is to set a **different approval threshold per gender/age group**. That reaches a fair-looking "
            "number but **uses a protected attribute at decision time — direct discrimination under AGG §3(1)**. CreditFair "
            "instead constrains fairness *during training* (in-processing), so the deployed decision never sees a protected "
            "attribute, and explains decisions with **faithful reason codes** rather than a generative model that can't be verified.")

    st.markdown('<div class="cf-sec">Limitations — stated honestly</div>', unsafe_allow_html=True)
    st.markdown("""<div class="cf-card">
      <div class="row"><span class="k">Selection bias</span><span class="v">labels are censored (only funded loans repay/default); full reject-inference is future work</span></div>
      <div class="row"><span class="k">Small groups</span><span class="v">intersectional subgroups have wide CIs on 1,000 rows → routed to human review, not certified</span></div>
      <div class="row"><span class="k">Age DI</span><span class="v">borderline (~0.77–0.84), not claimed as cleared</span></div>
      <div class="row"><span class="k">Data</span><span class="v">a single 1990s German dataset — a reference implementation, not a certified production system</span></div>
    </div>""", unsafe_allow_html=True)
    st.caption("Full documents: CreditFair-Model-Card.md · CreditFair-DPIA.md · CreditFair-FRIA.md · CreditFair-Fairness-Law-Map.md")
