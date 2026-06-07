"""
Repeat the Table 2 baseline "Base Model" from:
Driban et al. The prognostic potential of end-stage knee osteoarthritis
and its components to predict knee replacement: data from the OAI.

Paper base model:
    knee replacement over 5 years ~ age + sex + race + BMI

The model is fit as a knee-level logistic regression with generalized
estimating equations (GEE), clustered by participant ID, to account for
two knees from the same participant.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

try:
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This replication needs scipy, scikit-learn, and statsmodels "
        "(statsmodels is required for logistic GEE).\n"
        "Install it in your Python environment, for example:\n"
        "    pip install statsmodels openpyxl pandas scipy scikit-learn\n"
        "Then rerun this script."
    ) from exc


DATA_PATH = Path("/Users/jiangxiaohan/Desktop/materials of summer project/combined data.xlsx")
SHEET_NAME = "COMPARABLE"

INJURY_SURGERY_COLS: dict[int, list[str]] | list[str] = []


def yes_no_to_binary(x: pd.Series) -> pd.Series:

    if pd.api.types.is_numeric_dtype(x):
        # OAI yes/no formats are commonly 1=yes, 0=no or 1=yes, 2=no.
        vals = set(x.dropna().unique())
        if vals <= {0, 1}:
            return x.astype("float")
        if vals <= {1, 2}:
            return x.map({1: 1, 2: 0}).astype("float")
        return (x > 0).astype("float")

    normalized = x.astype("string").str.strip().str.lower()
    return normalized.map(
        {
            "yes": 1,
            "y": 1,
            "1": 1,
            "true": 1,
            "no": 0,
            "n": 0,
            "0": 0,
            "2": 0,
            "false": 0,
        }
    ).astype("float")


def build_injury_surgery(df: pd.DataFrame) -> pd.Series:
    if not INJURY_SURGERY_COLS:
        msg = (
            "No injury/surgery columns were configured. The paper's base model "
            "includes history of knee injury or surgery, but these variables "
            "were not obvious in the supplied Codebook.xlsx."
        )
        if REQUIRE_INJURY_SURGERY:
            raise ValueError(msg + " Add them to INJURY_SURGERY_COLS or set "
                             "REQUIRE_INJURY_SURGERY = False for a sensitivity model.")
        warnings.warn(msg + " Running without this covariate.", stacklevel=2)
        return pd.Series(np.nan, index=df.index, name="injury_surgery")

    if isinstance(INJURY_SURGERY_COLS, dict):
        out = pd.Series(np.nan, index=df.index, dtype="float")
        for side, cols in INJURY_SURGERY_COLS.items():
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise KeyError(f"Missing injury/surgery columns for side {side}: {missing}")
            side_mask = df["side"].eq(side)
            out.loc[side_mask] = (
                pd.concat([yes_no_to_binary(df.loc[side_mask, c]) for c in cols], axis=1)
                .max(axis=1, skipna=True)
            )
        return out.rename("injury_surgery")

    missing = [c for c in INJURY_SURGERY_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing injury/surgery columns: {missing}")
    return (
        pd.concat([yes_no_to_binary(df[c]) for c in INJURY_SURGERY_COLS], axis=1)
        .max(axis=1, skipna=True)
        .rename("injury_surgery")
    )


def hosmer_lemeshow_test(y_true: pd.Series, y_prob: pd.Series, groups: int = 10) -> tuple[float, float]:
    """Hosmer-Lemeshow calibration test based on deciles of predicted risk."""
    tmp = pd.DataFrame({"y": y_true.to_numpy(), "p": y_prob.to_numpy()}).dropna()
    tmp["bin"] = pd.qcut(tmp["p"], q=groups, duplicates="drop")
    grouped = tmp.groupby("bin", observed=False)
    obs = grouped["y"].sum()
    exp = grouped["p"].sum()
    n = grouped.size()
    denom = exp * (1 - exp / n)
    hl = (((obs - exp) ** 2) / denom.replace(0, np.nan)).sum()
    dof = max(len(obs) - 2, 1)
    p_value = stats.chi2.sf(hl, dof)
    return float(hl), float(p_value)


def prepare_baseline_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    # Table 2 is the OAI baseline analysis.
    df = df[df["visit"].astype("string").str.lower().eq("v00")].copy()

    # Future KR over the next 5 years. v99KRstatus == 3 is adjudicated KR in
    # this workbook, and v99KRmonths gives months from baseline to KR.
    df["kr_5yr"] = (
        df["v99KRstatus"].eq(3)
        & df["v99KRmonths"].notna()
        & df["v99KRmonths"].le(60)
    ).astype(int)

    df["age"] = pd.to_numeric(df["V00AGE"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["v00bmi"], errors="coerce")
    df["sex"] = df["P02SEX"].astype("category")
    df["race"] = df["P02RACE"].astype("category")
    df["injury_surgery"] = build_injury_surgery(df)

    model_cols = ["ID", "side", "kr_5yr", "age", "sex", "race", "bmi"]
    if not df["injury_surgery"].isna().all():
        model_cols.append("injury_surgery")

    analytic = df[model_cols].dropna().copy()
    analytic["ID"] = analytic["ID"].astype("string")
    return analytic


def fit_base_model(analytic: pd.DataFrame):
    rhs = "age + C(sex) + C(race) + bmi"
    if "injury_surgery" in analytic.columns:
        rhs += " + injury_surgery"
    formula = f"kr_5yr ~ {rhs}"

    model = smf.gee(
        formula=formula,
        groups="ID",
        data=analytic,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    result = model.fit()
    pred = result.predict(analytic)

    auc = roc_auc_score(analytic["kr_5yr"], pred)
    hl_stat, hl_p = hosmer_lemeshow_test(analytic["kr_5yr"], pred, groups=10)

    or_table = pd.DataFrame(
        {
            "OR": np.exp(result.params),
            "CI_low": np.exp(result.conf_int()[0]),
            "CI_high": np.exp(result.conf_int()[1]),
            "p_value": result.pvalues,
        }
    )

    return formula, result, pred, auc, hl_stat, hl_p, or_table


def main() -> None:
    analytic = prepare_baseline_data()
    formula, result, pred, auc, hl_stat, hl_p, or_table = fit_base_model(analytic)

    print("\nTable 2 baseline Base Model")
    print("=" * 38)
    print(f"Formula: {formula}")
    print(f"Knees in analytic sample: {len(analytic):,}")
    print(f"Participants: {analytic['ID'].nunique():,}")
    print(f"Knee replacements within 5 years: {analytic['kr_5yr'].sum():,}")
    print(f"AUC: {auc:.3f}")
    print(f"Hosmer-Lemeshow chi-square: {hl_stat:.3f}")
    print(f"Hosmer-Lemeshow p-value: {hl_p:.3f}")
    print(f"Mean predicted probability for KR: {pred[analytic['kr_5yr'].eq(1)].mean():.3f}")
    print(f"Mean predicted probability for no KR: {pred[analytic['kr_5yr'].eq(0)].mean():.3f}")

    print("\nOdds ratios for base-model covariates")
    print(or_table.to_string(float_format=lambda x: f"{x:.3f}"))

    out = Path("table2_base_model_predictions.csv")
    analytic.assign(predicted_probability=pred).to_csv(out, index=False)
    print(f"\nSaved row-level predictions to: {out.resolve()}")


if __name__ == "__main__":
    main()
