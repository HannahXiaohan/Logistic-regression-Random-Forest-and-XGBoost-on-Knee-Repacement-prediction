from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = Path("/Users/jiangxiaohan/Desktop/materials of summer project/combined data.xlsx")

OUT_TABLE_CSV = SCRIPT_DIR / "table2_baseline.csv"
OUT_PREDICTIONS = SCRIPT_DIR / "table2_baseline_predictions.csv"

def pick_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Could not find a column for {label}. Tried: {candidates}")


def prepare_baseline_data() -> pd.DataFrame:
    raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    df = raw[raw["visit"].astype("string").str.lower().eq("v00")].copy()

    age_col = pick_column(df, ["age", "V00AGE"], "baseline age")
    bmi_col = pick_column(df, ["bmi", "v00bmi", "V00BMI"], "baseline BMI")
    kl_col = pick_column(df, ["xrkl", "V00XRKL"], "baseline KL grade")
    womac_pain_col = pick_column(df, ["womkp", "v00womkp"], "baseline WOMAC pain")
    womac_function_col = pick_column(df, ["womadl", "v00womadl"], "baseline WOMAC function")
    persistent_pain_col = pick_column(df, ["kp12cv", "V00KP12CV"], "baseline persistent pain")

    df["kr_5yr"] = (
        df["v99KRstatus"].eq(3)
        & df["v99KRmonths"].notna()
        & df["v99KRmonths"].le(60)
    ).astype(int)
    df["age"] = pd.to_numeric(df[age_col], errors="coerce")
    df["bmi"] = pd.to_numeric(df[bmi_col], errors="coerce")
    df["sex"] = df["P02SEX"].astype("category")
    df["race"] = df["P02RACE"].astype("category")
    df["kl_ordinal"] = pd.to_numeric(df[kl_col], errors="coerce")
    df["kl_eq_4"] = df["kl_ordinal"].eq(4).astype(int)
    df["kl_ge_3"] = df["kl_ordinal"].ge(3).astype(int)
    df["kl_ge_2"] = df["kl_ordinal"].ge(2).astype(int)
    df["symptoms"] = (
        pd.to_numeric(df[womac_pain_col], errors="coerce")
        + pd.to_numeric(df[womac_function_col], errors="coerce")
    )
    df["symptoms_ordinal"] = np.select(
        [df["symptoms"] > 33, df["symptoms"] > 23, df["symptoms"] > 12],
        [3, 2, 1],
        default=0,
    )
    df["symptoms_gt_33"] = df["symptoms"].gt(33).astype(int)
    df["symptoms_gt_23"] = df["symptoms"].gt(23).astype(int)
    df["symptoms_gt_12"] = df["symptoms"].gt(12).astype(int)
    df["persistent_pain"] = pd.to_numeric(df[persistent_pain_col], errors="coerce").eq(1).astype(int)
    df["eskoa_original_proxy"] = pd.to_numeric(
        df.get("disease_activity_orig", np.nan), errors="coerce"
    ).gt(0).astype(int)
    df["eskoa_alternative_proxy"] = pd.to_numeric(
        df.get("disease_activity_new", np.nan), errors="coerce"
    ).gt(0).astype(int)

    keep = [
        "ID",
        "side",
        "kr_5yr",
        "age",
        "bmi",
        "sex",
        "race",
        "kl_ordinal",
        "kl_eq_4",
        "kl_ge_3",
        "kl_ge_2",
        "symptoms_ordinal",
        "symptoms_gt_33",
        "symptoms_gt_23",
        "symptoms_gt_12",
        "persistent_pain",
        "eskoa_original_proxy",
        "eskoa_alternative_proxy",
    ]
    base_required = ["ID", "side", "kr_5yr", "age", "bmi", "sex", "race"]
    analytic = df[keep].replace([np.inf, -np.inf], np.nan).dropna(subset=base_required).copy()
    analytic["ID"] = analytic["ID"].astype("string")
    return analytic

def make_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=1e12,
                    solver="lbfgs",
                    max_iter=5000,
                ),
            ),
        ]
    )


def fit_predict(data: pd.DataFrame, predictors: list[str]) -> tuple[Pipeline, pd.Series]:
    numeric = ["age", "bmi"] + predictors
    categorical = ["sex", "race"]
    model = make_model(numeric, categorical)
    x = data[numeric + categorical]
    y = data["kr_5yr"]
    model.fit(x, y)
    prob = pd.Series(model.predict_proba(x)[:, 1], index=data.index, name="predicted_probability")
    return model, prob


def model_design_matrix(model: Pipeline, data: pd.DataFrame, predictors: list[str]) -> np.ndarray:
    numeric = ["age", "bmi"] + predictors
    categorical = ["sex", "race"]
    transformed = model.named_steps["preprocess"].transform(data[numeric + categorical])
    return np.column_stack([np.ones(transformed.shape[0]), transformed])


def coefficient_or_ci(model: Pipeline, data: pd.DataFrame, predictor: str) -> tuple[float, float, float]:
    """Approximate OR and 95% CI for one added numeric predictor.

    The model standardizes numeric predictors, so this converts the coefficient
    back to the original predictor unit before exponentiating.
    """
    beta_scaled = float(model.named_steps["model"].coef_[0][2])
    scale = float(model.named_steps["preprocess"].named_transformers_["num"].scale_[2])
    beta = beta_scaled / scale

    predictors = [predictor]
    x_design = model_design_matrix(model, data, predictors)
    p = model.predict_proba(data[["age", "bmi", predictor, "sex", "race"]])[:, 1]
    w = np.clip(p * (1 - p), 1e-9, None)
    info = x_design.T @ (x_design * w[:, None])
    cov_scaled = np.linalg.pinv(info)
    se_scaled = float(np.sqrt(max(cov_scaled[3, 3], 0)))
    se = se_scaled / scale
    return (
        float(np.exp(beta)),
        float(np.exp(beta - 1.96 * se)),
        float(np.exp(beta + 1.96 * se)),
    )


def hosmer_lemeshow_test(y_true: pd.Series, y_prob: pd.Series, groups: int = 10) -> tuple[float, float]:
    tmp = pd.DataFrame({"y": y_true.to_numpy(), "p": y_prob.to_numpy()}).dropna()
    tmp["bin"] = pd.qcut(tmp["p"].rank(method="first"), q=groups, duplicates="drop")
    grouped = tmp.groupby("bin", observed=False)
    obs = grouped["y"].sum()
    exp = grouped["p"].sum()
    n = grouped.size()
    denom = exp * (1 - exp / n)
    hl_stat = (((obs - exp) ** 2) / denom.replace(0, np.nan)).sum()
    dof = max(len(obs) - 2, 1)
    return float(hl_stat), float(stats.chi2.sf(hl_stat, dof))


def continuous_reclassification(y: pd.Series, p_base: pd.Series, p_new: pd.Series) -> dict[str, float]:
    diff = p_new - p_base
    event = y.eq(1)
    nonevent = y.eq(0)

    kr_correct = diff[event].gt(0).mean() - diff[event].lt(0).mean()
    no_kr_correct = diff[nonevent].lt(0).mean() - diff[nonevent].gt(0).mean()
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
    rng = np.random.default_rng(seed)
    rows = data[["ID", "kr_5yr"]].copy()
    rows["p_base"] = p_base.to_numpy()
    rows["p_new"] = p_new.to_numpy()
    clusters = [g.index.to_numpy() for _, g in rows.reset_index(drop=True).groupby("ID", sort=False)]

    out = {"nri": [], "idi": []}
    for _ in range(reps):
        sampled_clusters = rng.choice(len(clusters), size=len(clusters), replace=True)
        idx = np.concatenate([clusters[i] for i in sampled_clusters])
        sampled = rows.iloc[idx]
        if sampled["kr_5yr"].nunique() < 2:
            continue
        metrics = continuous_reclassification(
            sampled["kr_5yr"], sampled["p_base"], sampled["p_new"]
        )
        out["nri"].append(metrics["nri"])
        out["idi"].append(metrics["idi"])

    return {
        key: tuple(np.percentile(values, [2.5, 97.5])) if values else (np.nan, np.nan)
        for key, values in out.items()
    }

def fmt_num(value: float) -> str:
    return f"{value:.2f}"


def fmt_pct(value: float) -> str:
    return f"{100 * value:.0f}%"


def fmt_ci(value: float, ci: tuple[float, float]) -> str:
    return f"{value:.2f} ({ci[0]:.2f}-{ci[1]:.2f})"


def fmt_or_ci(or_ci: tuple[float, float, float]) -> str:
    odds, low, high = or_ci
    return f"{odds:.2f} ({low:.2f}-{high:.2f})"

def summarize_model(
    label: str,
    data: pd.DataFrame,
    predictors: list[str] | None,
    p_base: pd.Series | None,
    n_events_label: str,
    bootstrap: bool = False,
) -> tuple[dict[str, str], pd.Series]:
    predictors = [] if predictors is None else predictors
    required = ["kr_5yr", "age", "bmi", "sex", "race"] + predictors
    model_data = data.dropna(subset=required).copy()
    model, model_prob = fit_predict(model_data, predictors)
    prob = pd.Series(np.nan, index=data.index, name="predicted_probability")
    prob.loc[model_data.index] = model_prob
    y = model_data["kr_5yr"]
    _, hl_p = hosmer_lemeshow_test(y, model_prob)

    event = y.eq(1)
    nonevent = y.eq(0)
    row = {
        "n/events": n_events_label,
        "Model": label,
        "AUC": fmt_num(roc_auc_score(y, model_prob)),
        "Hosmer-Lemeshow Test (p-value)": fmt_num(hl_p),
        "Odds Ratio for New Variable (95% CI)": "n/a",
        "% KR Correctly Reclassified": "n/a",
        "% No KR Correctly Reclassified": "n/a",
        "Net Reclassification Index (95% CI)": "n/a",
        "Mean Probability for KR": fmt_num(model_prob[event].mean()),
        "Mean Probability for No KR": fmt_num(model_prob[nonevent].mean()),
        "Integrated discrimination improvement (95% CI)": "n/a",
    }

    if predictors and p_base is not None:
        if len(predictors) == 1:
            row["Odds Ratio for New Variable (95% CI)"] = fmt_or_ci(
                coefficient_or_ci(model, model_data, predictors[0])
            )
        else:
            row["Odds Ratio for New Variable (95% CI)"] = "Not shown"
        base_aligned = p_base.loc[model_data.index]
        reclass = continuous_reclassification(y, base_aligned, model_prob)
        row["% KR Correctly Reclassified"] = fmt_pct(reclass["kr_correct"])
        row["% No KR Correctly Reclassified"] = fmt_pct(reclass["no_kr_correct"])
        if bootstrap:
            ci = bootstrap_reclassification_ci(model_data, base_aligned, model_prob)
            row["Net Reclassification Index (95% CI)"] = fmt_ci(reclass["nri"], ci["nri"])
            row["Integrated discrimination improvement (95% CI)"] = fmt_ci(reclass["idi"], ci["idi"])
        else:
            row["Net Reclassification Index (95% CI)"] = fmt_num(reclass["nri"])
            row["Integrated discrimination improvement (95% CI)"] = fmt_num(reclass["idi"])

    return row, prob


def write_excel(table: pd.DataFrame) -> None:
    title = (
        "Table 2. Osteoarthritis Initiative's baseline visit: prognostic potential "
        "to predict knee replacement (KR) of participant characteristics, components "
        "of end-stage knee osteoarthritis (esKOA), and esKOA."
    )
    with pd.ExcelWriter(OUT_TABLE_XLSX, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Table 2 Baseline", index=False, startrow=2)
        notes.to_excel(writer, sheet_name="Notes", index=False)
        ws = writer.book["Table 2 Baseline"]
        ws["A1"] = title
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(table.columns))
        title_font = copy(ws["A1"].font)
        title_font.bold = True
        title_font.size = 13
        ws["A1"].font = title_font
        title_alignment = copy(ws["A1"].alignment)
        title_alignment.wrap_text = True
        ws["A1"].alignment = title_alignment
        ws.freeze_panes = "A4"
        widths = [22, 38, 10, 18, 28, 18, 22, 26, 18, 22, 30]
        for idx, width in enumerate(widths, start=1):
            ws.column_dimensions[chr(64 + idx)].width = width
        for row in ws.iter_rows(min_row=3, max_row=3):
            for cell in row:
                header_font = copy(cell.font)
                header_font.bold = True
                cell.font = header_font
                header_alignment = copy(cell.alignment)
                header_alignment.horizontal = "center"
                header_alignment.wrap_text = True
                cell.alignment = header_alignment


def main() -> None:
    print("Loading baseline data...")
    data = prepare_baseline_data()
    print(f"Analytic sample: {len(data):,} knees, {data['ID'].nunique():,} participants")
    print(f"Knee replacements within 5 years: {data['kr_5yr'].sum():,}")
    n_events_label = f"{len(data)} knees / {int(data['kr_5yr'].sum())} KR"

    print("Fitting base model...")
    base_row, p_base = summarize_model("Base Model", data, None, None, n_events_label)

    specs = [
        ("Base + KL (ordinal)", ["kl_ordinal"]),
        ("    Base + KL=4 (yes/no)", ["kl_eq_4"]),
        ("    Base + KL>=3 (yes/no)", ["kl_ge_3"]),
        ("    Base + KL>=2 (yes/no)", ["kl_ge_2"]),
        ("    Base + Symptoms (ordinal)", ["symptoms_ordinal"]),
        ("Base + Severe Symptoms (>33)", ["symptoms_gt_33"]),
        ("    Base + Symptoms>=Intense (>23)", ["symptoms_gt_23"]),
        ("Base + Symptoms>=Moderate (>12)", ["symptoms_gt_12"]),
        ("Base + Persistent Pain", ["persistent_pain"]),
        ("Base + esKOA (original proxy)", ["eskoa_original_proxy"]),
        ("Base + esKOA (alternative proxy)", ["eskoa_alternative_proxy"]),
    ]
    rows = [base_row]
    predictions = data.copy()
    predictions["predicted_probability_base"] = p_base

    for label, predictors in specs:
        print(f"Fitting {label}...")
        row, prob = summarize_model(label, data, predictors, p_base, n_events_label)
        rows.append(row)
        prediction_name = "_".join(predictors)
        predictions[f"predicted_probability_{prediction_name}"] = prob
        predictions[f"prediction_change_{prediction_name}"] = prob - p_base

    table = pd.DataFrame(rows)
    table.to_csv(OUT_TABLE_CSV, index=False)
    predictions.to_csv(OUT_PREDICTIONS, index=False)
    write_excel(table)

    print("\nTable 2 baseline rows")
    print("=" * 90)
    print(table.to_string(index=False))
    print(f"\nSaved table CSV: {OUT_TABLE_CSV}")
    print(f"Saved table XLSX: {OUT_TABLE_XLSX}")
    print(f"Saved predictions: {OUT_PREDICTIONS}")


if __name__ == "__main__":
    main()
