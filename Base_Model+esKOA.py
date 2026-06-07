"""
Repeat the Table 2 baseline experiments:
    Base Model
    Base Model + esKOA (original)
    Base Model + esKOA (alternative)
Output:
    1. Terminal summary for Base + esKOA rows
    2. table2_base_plus_eskoa_summary.csv
    3. table2_base_plus_eskoa_predictions.csv
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
    from sklearn.metrics import roc_auc_score
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

DATA_PATH = Path("/Users/jiangxiaohan/Desktop/materials of summer project/combined data.xlsx")
SHEET_NAME = "COMPARABLE"

OUT_SUMMARY = Path("table2_base_plus_eskoa_summary.csv")
OUT_PREDICTIONS = Path("table2_base_plus_eskoa_predictions.csv")

ESKOA_ORIGINAL_THRESHOLD = 0
ESKOA_ALTERNATIVE_THRESHOLD = 0

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


def format_or_ci(or_value: float, ci_low: float, ci_high: float) -> str:
    return f"{or_value:.2f} ({ci_low:.2f}-{ci_high:.2f})"


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

    orig_score = pd.to_numeric(df["disease_activity_orig"], errors="coerce")
    alt_score = pd.to_numeric(df["disease_activity_new"], errors="coerce")
    df["eskoa_original"] = (orig_score >= ESKOA_ORIGINAL_THRESHOLD).astype("float")
    df.loc[orig_score.isna(), "eskoa_original"] = np.nan
    df["eskoa_alternative"] = (alt_score >= ESKOA_ALTERNATIVE_THRESHOLD).astype("float")
    df.loc[alt_score.isna(), "eskoa_alternative"] = np.nan

    analytic = df[
        [
            "ID",
            "side",
            "kr_5yr",
            "age",
            "sex",
            "race",
            "bmi",
            "eskoa_original",
            "eskoa_alternative",
        ]
    ].dropna().copy()
    analytic["ID"] = analytic["ID"].astype("string")
    return analytic


def fit_gee(data: pd.DataFrame, formula: str):
    model = smf.gee(
        formula=formula,
        groups="ID",
        data=data,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    result = model.fit()
    pred = pd.Series(result.predict(data), index=data.index)
    auc = roc_auc_score(data["kr_5yr"], pred)
    hl_stat, hl_p = hosmer_lemeshow_test(data["kr_5yr"], pred)
    return result, pred, auc, hl_stat, hl_p


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
    """Participant-cluster bootstrap CI for NRI and IDI."""
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


def make_experiment_row(
    label: str,
    new_variable: str,
    data: pd.DataFrame,
    p_base: pd.Series,
) -> tuple[dict[str, object], pd.Series]:
    formula = f"kr_5yr ~ age + C(sex) + C(race) + bmi + {new_variable}"
    print(f"Fitting {label}...")
    result, p_new, auc, hl_stat, hl_p = fit_gee(data, formula)

    new_or = float(np.exp(result.params[new_variable]))
    new_ci = np.exp(result.conf_int().loc[new_variable])

    reclass = continuous_reclassification(data["kr_5yr"], p_base, p_new)
    print(f"Bootstrapping NRI/IDI confidence intervals for {label} ({BOOTSTRAP_REPS} reps)...")
    reclass_ci = bootstrap_reclassification_ci(data, p_base, p_new)

    kr_mask = data["kr_5yr"].eq(1)
    no_kr_mask = data["kr_5yr"].eq(0)
    row = {
        "Model": label,
        "AUC": auc,
        "Hosmer-Lemeshow p-value": hl_p,
        "Odds Ratio for New Variable (95% CI)": format_or_ci(
            new_or, float(new_ci[0]), float(new_ci[1])
        ),
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
    return row, p_new


def main() -> None:
    print("Loading baseline data...")
    analytic = prepare_baseline_data()

    base_formula = "kr_5yr ~ age + C(sex) + C(race) + bmi"
    print("Fitting reference base model...")
    base_result, p_base, base_auc, base_hl_stat, base_hl_p = fit_gee(analytic, base_formula)

    kr_mask = analytic["kr_5yr"].eq(1)
    no_kr_mask = analytic["kr_5yr"].eq(0)
    base_row = {
        "Model": "Base Model",
        "AUC": base_auc,
        "Hosmer-Lemeshow p-value": base_hl_p,
        "Odds Ratio for New Variable (95% CI)": "n/a",
        "% KR Correctly Reclassified": "n/a",
        "% No KR Correctly Reclassified": "n/a",
        "NRI (95% CI)": "n/a",
        "Mean Probability for KR": p_base[kr_mask].mean(),
        "Mean Probability for No KR": p_base[no_kr_mask].mean(),
        "IDI (95% CI)": "n/a",
    }

    orig_row, p_orig = make_experiment_row(
        "Base + esKOA (original)", "eskoa_original", analytic, p_base
    )
    alt_row, p_alt = make_experiment_row(
        "Base + esKOA (alternative)", "eskoa_alternative", analytic, p_base
    )

    summary = pd.DataFrame([base_row, orig_row, alt_row])
    predictions = analytic.assign(
        predicted_probability_base=p_base,
        predicted_probability_base_plus_eskoa_original=p_orig,
        predicted_probability_base_plus_eskoa_alternative=p_alt,
        prediction_change_eskoa_original=p_orig - p_base,
        prediction_change_eskoa_alternative=p_alt - p_base,
    )

    summary.to_csv(OUT_SUMMARY, index=False)
    predictions.to_csv(OUT_PREDICTIONS, index=False)

    print("\nTable 2 baseline: Base Model + esKOA")
    print("=" * 44)
    print(f"Base formula: {base_formula}")
    print("Original formula:    kr_5yr ~ age + C(sex) + C(race) + bmi + eskoa_original")
    print("Alternative formula: kr_5yr ~ age + C(sex) + C(race) + bmi + eskoa_alternative")
    print(f"Original esKOA threshold: disease_activity_orig >= {ESKOA_ORIGINAL_THRESHOLD}")
    print(f"Alternative esKOA threshold: disease_activity_new >= {ESKOA_ALTERNATIVE_THRESHOLD}")
    print(f"Knees in analytic sample: {len(analytic):,}")
    print(f"Participants: {analytic['ID'].nunique():,}")
    print(f"Knee replacements within 5 years: {analytic['kr_5yr'].sum():,}")
    print("\nSummary")
    printable = summary.copy()
    for col in ["AUC", "Hosmer-Lemeshow p-value", "Mean Probability for KR", "Mean Probability for No KR"]:
        printable[col] = printable[col].map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
    print(printable.to_string(index=False))
    print(f"\nSaved summary to: {OUT_SUMMARY.resolve()}")
    print(f"Saved row-level predictions to: {OUT_PREDICTIONS.resolve()}")


if __name__ == "__main__":
    main()
