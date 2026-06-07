"""
Random Forest version of the Table 2 baseline experiments:
    Base Model
    Base + KL (ordinal)
    Base + esKOA (original)
    Base + esKOA (alternative)

This script uses grouped cross-validated predicted probabilities, with
participant ID as the group, so knees from the same participant are kept in the
same fold. Random Forests do not produce odds ratios, so the odds-ratio column
is reported as "n/a (RF)".

The workbook contains disease_activity_orig and disease_activity_new as
continuous scores rather than obvious 0/1 flags. By default, this script
classifies esKOA as score >= 0. Change the thresholds below if your project
documentation defines a different cutoff.

Output:
    1. Terminal summary
    2. table2_random_forest_summary.csv
    3. table2_random_forest_predictions.csv
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style, apply openpyxl's default",
    category=UserWarning,
)

try:
    from scipy import stats
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This script needs scipy, scikit-learn, openpyxl, pandas, and numpy.\n"
        "Install them in your py311 environment, for example:\n"
        "    pip install openpyxl pandas numpy scipy scikit-learn\n"
        "Then rerun this script."
    ) from exc


DATA_PATH = Path("/Users/jiangxiaohan/Desktop/materials of summer project/combined data.xlsx")
SHEET_NAME = "COMPARABLE"

OUT_SUMMARY = Path("table2_random_forest_summary.csv")
OUT_PREDICTIONS = Path("table2_random_forest_predictions.csv")

ESKOA_ORIGINAL_THRESHOLD = 0
ESKOA_ALTERNATIVE_THRESHOLD = 0

N_SPLITS = 5
RANDOM_STATE = 20260606
N_ESTIMATORS = 500

# 200 cluster-bootstrap reps is fast for interactive work. Increase to 1000+
# only when you need more stable CI estimates.
BOOTSTRAP_REPS = 200
BOOTSTRAP_SEED = 20260606


def hosmer_lemeshow_test(y_true: pd.Series, y_prob: pd.Series, groups: int = 10) -> tuple[float, float]:
    """Hosmer-Lemeshow calibration test using deciles of predicted risk."""
    tmp = pd.DataFrame({"y": y_true.to_numpy(), "p": y_prob.to_numpy()}).dropna()
    tmp["bin"] = pd.qcut(tmp["p"], q=groups, duplicates="drop")
    grouped = tmp.groupby("bin", observed=False)
    obs = grouped["y"].sum()
    exp = grouped["p"].sum()
    n = grouped.size()
    denom = exp * (1 - exp / n)
    hl_stat = (((obs - exp) ** 2) / denom.replace(0, np.nan)).sum()
    dof = max(len(obs) - 2, 1)
    p_value = stats.chi2.sf(hl_stat, dof)
    return float(hl_stat), float(p_value)


def format_ci(value: float, ci_low: float, ci_high: float) -> str:
    return f"{value:.2f} ({ci_low:.2f}-{ci_high:.2f})"


def prepare_baseline_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    df = df[df["visit"].astype("string").str.lower().eq("v00")].copy()

    df["kr_5yr"] = (
        df["v99KRstatus"].eq(3)
        & df["v99KRmonths"].notna()
        & df["v99KRmonths"].le(60)
    ).astype(int)

    df["age"] = pd.to_numeric(df["V00AGE"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["v00bmi"], errors="coerce")
    df["sex"] = df["P02SEX"].astype("category")
    df["race"] = df["P02RACE"].astype("category")
    df["kl_ordinal"] = pd.to_numeric(df["V00XRKL"], errors="coerce")

    orig_score = pd.to_numeric(df["disease_activity_orig"], errors="coerce")
    alt_score = pd.to_numeric(df["disease_activity_new"], errors="coerce")
    df["eskoa_original"] = (orig_score >= ESKOA_ORIGINAL_THRESHOLD).astype("float")
    df.loc[orig_score.isna(), "eskoa_original"] = np.nan
    df["eskoa_alternative"] = (alt_score >= ESKOA_ALTERNATIVE_THRESHOLD).astype("float")
    df.loc[alt_score.isna(), "eskoa_alternative"] = np.nan

    # Use one complete-case analytic sample so all rows compare the same knees.
    cols = [
        "ID",
        "side",
        "kr_5yr",
        "age",
        "sex",
        "race",
        "bmi",
        "kl_ordinal",
        "eskoa_original",
        "eskoa_alternative",
    ]
    analytic = df[cols].dropna().copy()
    analytic["ID"] = analytic["ID"].astype("string")
    return analytic


def make_rf_pipeline(feature_cols: list[str]) -> Pipeline:
    categorical_cols = [c for c in feature_cols if c in {"sex", "race"}]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    try:
        encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
    )
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        min_samples_leaf=5,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("rf", rf)])


def grouped_cv_predict_proba(data: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Out-of-fold predicted probabilities with participant-level grouping."""
    groups = data["ID"]
    y = data["kr_5yr"].astype(int)
    X = data[feature_cols]

    n_groups = groups.nunique()
    n_splits = min(N_SPLITS, n_groups)
    if n_splits < 2:
        raise ValueError("Need at least two participant groups for grouped cross-validation.")

    preds = pd.Series(np.nan, index=data.index, dtype="float")
    splitter = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        print(f"  Fold {fold}/{n_splits}...")
        model = make_rf_pipeline(feature_cols)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds.iloc[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

    return preds


def continuous_reclassification(y: pd.Series, p_base: pd.Series, p_new: pd.Series) -> dict[str, float]:
    """Continuous NRI components comparing new predictions to base predictions."""
    diff = p_new - p_base
    event = y.eq(1)
    nonevent = y.eq(0)

    event_up = diff[event].gt(0).mean()
    event_down = diff[event].lt(0).mean()
    nonevent_down = diff[nonevent].lt(0).mean()
    nonevent_up = diff[nonevent].gt(0).mean()

    kr_correct = event_up - event_down
    no_kr_correct = nonevent_down - nonevent_up
    nri = kr_correct + no_kr_correct

    base_slope = p_base[event].mean() - p_base[nonevent].mean()
    new_slope = p_new[event].mean() - p_new[nonevent].mean()
    idi = new_slope - base_slope

    return {
        "kr_correct": float(kr_correct),
        "no_kr_correct": float(no_kr_correct),
        "nri": float(nri),
        "idi": float(idi),
    }


def bootstrap_reclassification_ci(
    data: pd.DataFrame,
    p_base: pd.Series,
    p_new: pd.Series,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, tuple[float, float]]:
    """Participant-cluster bootstrap CI for NRI and IDI from fixed predictions."""
    rng = np.random.default_rng(seed)
    rows = data[["ID", "kr_5yr"]].copy()
    rows["p_base"] = p_base.to_numpy()
    rows["p_new"] = p_new.to_numpy()
    cluster_indices = [
        group.index.to_numpy()
        for _, group in rows.reset_index(drop=True).groupby("ID", sort=False)
    ]

    nri_values = []
    idi_values = []

    for _ in range(reps):
        sampled_clusters = rng.choice(len(cluster_indices), size=len(cluster_indices), replace=True)
        sampled_idx = np.concatenate([cluster_indices[i] for i in sampled_clusters])
        sampled = rows.iloc[sampled_idx]
        if sampled["kr_5yr"].nunique() < 2:
            continue
        metrics = continuous_reclassification(
            sampled["kr_5yr"], sampled["p_base"], sampled["p_new"]
        )
        nri_values.append(metrics["nri"])
        idi_values.append(metrics["idi"])

    return {
        "nri": tuple(np.percentile(nri_values, [2.5, 97.5])),
        "idi": tuple(np.percentile(idi_values, [2.5, 97.5])),
    }


def base_row(data: pd.DataFrame, p_base: pd.Series) -> dict[str, object]:
    y = data["kr_5yr"]
    auc = roc_auc_score(y, p_base)
    _, hl_p = hosmer_lemeshow_test(y, p_base)
    kr_mask = y.eq(1)
    no_kr_mask = y.eq(0)
    return {
        "Model": "Base Model RF",
        "AUC": auc,
        "Hosmer-Lemeshow p-value": hl_p,
        "Odds Ratio for New Variable (95% CI)": "n/a (RF)",
        "% KR Correctly Reclassified": "n/a",
        "% No KR Correctly Reclassified": "n/a",
        "NRI (95% CI)": "n/a",
        "Mean Probability for KR": p_base[kr_mask].mean(),
        "Mean Probability for No KR": p_base[no_kr_mask].mean(),
        "IDI (95% CI)": "n/a",
    }


def experiment_row(
    label: str,
    data: pd.DataFrame,
    p_base: pd.Series,
    p_new: pd.Series,
) -> dict[str, object]:
    y = data["kr_5yr"]
    auc = roc_auc_score(y, p_new)
    _, hl_p = hosmer_lemeshow_test(y, p_new)
    reclass = continuous_reclassification(y, p_base, p_new)
    print(f"Bootstrapping NRI/IDI CI for {label} ({BOOTSTRAP_REPS} reps)...")
    reclass_ci = bootstrap_reclassification_ci(data, p_base, p_new)

    kr_mask = y.eq(1)
    no_kr_mask = y.eq(0)
    return {
        "Model": label,
        "AUC": auc,
        "Hosmer-Lemeshow p-value": hl_p,
        "Odds Ratio for New Variable (95% CI)": "n/a (RF)",
        "% KR Correctly Reclassified": f"{100 * reclass['kr_correct']:.0f}%",
        "% No KR Correctly Reclassified": f"{100 * reclass['no_kr_correct']:.0f}%",
        "NRI (95% CI)": format_ci(
            reclass["nri"], reclass_ci["nri"][0], reclass_ci["nri"][1]
        ),
        "Mean Probability for KR": p_new[kr_mask].mean(),
        "Mean Probability for No KR": p_new[no_kr_mask].mean(),
        "IDI (95% CI)": format_ci(
            reclass["idi"], reclass_ci["idi"][0], reclass_ci["idi"][1]
        ),
    }


def main() -> None:
    print("Loading baseline data...")
    analytic = prepare_baseline_data()

    feature_sets = {
        "base": ["age", "sex", "race", "bmi"],
        "kl": ["age", "sex", "race", "bmi", "kl_ordinal"],
        "eskoa_original": ["age", "sex", "race", "bmi", "eskoa_original"],
        "eskoa_alternative": ["age", "sex", "race", "bmi", "eskoa_alternative"],
    }

    print("Fitting Base Model RF...")
    p_base = grouped_cv_predict_proba(analytic, feature_sets["base"])
    print("Fitting Base + KL (ordinal) RF...")
    p_kl = grouped_cv_predict_proba(analytic, feature_sets["kl"])
    print("Fitting Base + esKOA (original) RF...")
    p_eskoa_orig = grouped_cv_predict_proba(analytic, feature_sets["eskoa_original"])
    print("Fitting Base + esKOA (alternative) RF...")
    p_eskoa_alt = grouped_cv_predict_proba(analytic, feature_sets["eskoa_alternative"])

    summary = pd.DataFrame(
        [
            base_row(analytic, p_base),
            experiment_row("Base + KL (ordinal) RF", analytic, p_base, p_kl),
            experiment_row("Base + esKOA (original) RF", analytic, p_base, p_eskoa_orig),
            experiment_row("Base + esKOA (alternative) RF", analytic, p_base, p_eskoa_alt),
        ]
    )

    predictions = analytic.assign(
        predicted_probability_base_rf=p_base,
        predicted_probability_base_plus_kl_rf=p_kl,
        predicted_probability_base_plus_eskoa_original_rf=p_eskoa_orig,
        predicted_probability_base_plus_eskoa_alternative_rf=p_eskoa_alt,
        prediction_change_kl=p_kl - p_base,
        prediction_change_eskoa_original=p_eskoa_orig - p_base,
        prediction_change_eskoa_alternative=p_eskoa_alt - p_base,
    )

    summary.to_csv(OUT_SUMMARY, index=False)
    predictions.to_csv(OUT_PREDICTIONS, index=False)

    print("\nRandom Forest Table 2 baseline experiments")
    print("=" * 50)
    print(f"Knees in analytic sample: {len(analytic):,}")
    print(f"Participants: {analytic['ID'].nunique():,}")
    print(f"Knee replacements within 5 years: {analytic['kr_5yr'].sum():,}")
    print(f"CV folds grouped by participant: {min(N_SPLITS, analytic['ID'].nunique())}")
    print(f"Random Forest trees: {N_ESTIMATORS}")
    print(f"Original esKOA threshold: disease_activity_orig >= {ESKOA_ORIGINAL_THRESHOLD}")
    print(f"Alternative esKOA threshold: disease_activity_new >= {ESKOA_ALTERNATIVE_THRESHOLD}")
    print("\nSummary")
    printable = summary.copy()
    for col in ["AUC", "Hosmer-Lemeshow p-value", "Mean Probability for KR", "Mean Probability for No KR"]:
        printable[col] = printable[col].map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
    print(printable.to_string(index=False))
    print(f"\nSaved summary to: {OUT_SUMMARY.resolve()}")
    print(f"Saved row-level predictions to: {OUT_PREDICTIONS.resolve()}")


if __name__ == "__main__":
    main()
