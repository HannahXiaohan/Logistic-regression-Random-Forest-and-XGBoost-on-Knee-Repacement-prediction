import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
DATA_PATH = Path("/Users/jiangxiaohan/Desktop/materials of summer project/combined data.xlsx")
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "random_forest_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_baseline_data() -> pd.DataFrame:
    raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    df = raw[raw["visit"].astype("string").str.lower().eq("v00")].copy()

    df["kr_5yr"] = (
        df["v99KRstatus"].eq(3)
        & df["v99KRmonths"].notna()
        & df["v99KRmonths"].le(60)
    ).astype(int)

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["sex"] = df["P02SEX"].astype("category")
    df["race"] = df["P02RACE"].astype("category")
    df["kl_grade"] = pd.to_numeric(df["xrkl"], errors="coerce")
    df["womac_pain"] = pd.to_numeric(df["womkp"], errors="coerce")
    df["womac_function"] = pd.to_numeric(df["womadl"], errors="coerce")
    df["womac_total"] = df["womac_pain"] + df["womac_function"]
    df["persistent_pain"] = pd.to_numeric(df["kp12cv"], errors="coerce")
    df["kl_ge_2"] = df["kl_grade"].ge(2).astype(float)
    df["kl_ge_3"] = df["kl_grade"].ge(3).astype(float)
    df["kl_eq_4"] = df["kl_grade"].eq(4).astype(float)
    df["symptoms_gt_12"] = df["womac_total"].gt(12).astype(float)
    df["symptoms_gt_23"] = df["womac_total"].gt(23).astype(float)
    df["symptoms_gt_33"] = df["womac_total"].gt(33).astype(float)
    df["eskoa_original_proxy"] = pd.to_numeric(df["disease_activity_orig"], errors="coerce")
    df["eskoa_alternative_proxy"] = pd.to_numeric(df["disease_activity_new"], errors="coerce")

    keep = [
        "ID",
        "side",
        "kr_5yr",
        "age",
        "bmi",
        "sex",
        "race",
        "kl_grade",
        "kl_ge_2",
        "kl_ge_3",
        "kl_eq_4",
        "womac_pain",
        "womac_function",
        "womac_total",
        "symptoms_gt_12",
        "symptoms_gt_23",
        "symptoms_gt_33",
        "persistent_pain",
        "eskoa_original_proxy",
        "eskoa_alternative_proxy",
    ]
    return df[keep].copy()


FEATURES = [
    "age",
    "bmi",
    "sex",
    "race",
    "kl_grade",
    "kl_ge_2",
    "kl_ge_3",
    "kl_eq_4",
    "womac_pain",
    "womac_function",
    "womac_total",
    "symptoms_gt_12",
    "symptoms_gt_23",
    "symptoms_gt_33",
    "persistent_pain",
    "eskoa_original_proxy",
    "eskoa_alternative_proxy",
]
NUMERIC_FEATURES = [c for c in FEATURES if c not in ["sex", "race"]]
CATEGORICAL_FEATURES = ["sex", "race"]


def split_by_participant(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant_outcome = data.groupby("ID", as_index=False)["kr_5yr"].max()
    train_ids, test_ids = train_test_split(
        participant_outcome["ID"],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=participant_outcome["kr_5yr"],
    )
    train = data[data["ID"].isin(train_ids)].copy()
    test = data[data["ID"].isin(test_ids)].copy()
    return train, test


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", rf_model)])


def evaluate_model(
    model: Pipeline, test: pd.DataFrame, y_prob: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    y_true = test["kr_5yr"]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "threshold": 0.50,
                "test_knees": len(test),
                "test_participants": test["ID"].nunique(),
                "test_events": int(y_true.sum()),
                "roc_auc": roc_auc_score(y_true, y_prob),
                "average_precision": average_precision_score(y_true, y_prob),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
                "specificity": specificity,
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "brier_score": brier_score_loss(y_true, y_prob),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        ]
    )


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    rf = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": rf.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_plots(test: pd.DataFrame, y_prob: np.ndarray, importance: pd.DataFrame) -> None:
    y_true = test["kr_5yr"]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=8, strategy="quantile"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc_score(y_true, y_prob):.2f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    axes[1].plot(
        recall,
        precision,
        linewidth=2,
        label=f"AP = {average_precision_score(y_true, y_prob):.2f}",
    )
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")

    axes[2].plot(prob_pred, prob_true, marker="o", linewidth=2)
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[2].set_title("Calibration Curve")
    axes[2].set_xlabel("Mean Predicted Probability")
    axes[2].set_ylabel("Observed KR Rate")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "random_forest_evaluation_plots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    plot_data = importance.head(15).iloc[::-1]
    ax.barh(plot_data["feature"], plot_data["importance"])
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Mean decrease in impurity")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "random_forest_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = load_baseline_data()
    train, test = split_by_participant(data)

    print(f"Baseline knees: {len(data):,}")
    print(f"Baseline participants: {data['ID'].nunique():,}")
    print(f"Baseline KR within 5 years: {int(data['kr_5yr'].sum()):,}")
    print(f"Train knees/events: {len(train):,} / {int(train['kr_5yr'].sum()):,}")
    print(f"Test knees/events: {len(test):,} / {int(test['kr_5yr'].sum()):,}")

    model = build_pipeline()
    model.fit(train[FEATURES], train["kr_5yr"])

    y_prob = model.predict_proba(test[FEATURES])[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    metrics = evaluate_model(model, test, y_prob, y_pred)
    predictions = test[["ID", "side", "kr_5yr"]].copy()
    predictions["predicted_probability_kr_5yr"] = y_prob
    predictions["predicted_class_threshold_0_50"] = y_pred
    importance = get_feature_importance(model)

    metrics.to_csv(OUTPUT_DIR / "random_forest_test_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "random_forest_test_predictions.csv", index=False)
    importance.to_csv(OUTPUT_DIR / "random_forest_feature_importance.csv", index=False)
    save_plots(test, y_prob, importance)

    print("\nTest metrics")
    print(metrics.round(4).to_string(index=False))
    print("\nTop 15 feature importances")
    print(importance.head(15).to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
