import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def train_calibrated_classifier():
    print("==========================================")
    print(" TRAINING HIGH-AUC ROTATION CLASSIFIER")
    print("==========================================\n")

    data_path = "data/processed/features.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run export_features.py first!")

    df = pd.read_parquet(data_path)

    # 1. High-Signal Floor Features
    curated_features = [
        "age_at_draft",
        "rec_rank_log",
        "bpm_sos_adj",
        "college_bpm",
        "FT_per_shrunk",
        "TP_per_shrunk",
        "height_in_z",
        "stl_per_z",
        "blk_per_z",
        "AST_per_z",
        "usg_x_eff",
        "touch_divergence",
    ]

    feature_cols = [c for c in curated_features if c in df.columns]

    train_df = df[df["is_training_cohort"] == True].copy()
    unlabeled_df = df[df["is_training_cohort"] == False].copy()

    # 2. Out-of-Fold Cross Validation (GroupKFold by Draft Year)
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(train_df))
    
    X_train_full = train_df[feature_cols].fillna(0.0)
    y_train_full = train_df["reached_min_threshold_5y"].astype(int)
    groups = train_df["draft_year"]

    # Base Estimator: Standardized Logistic Regression Pipeline (Silences warnings & normalizes feature scales)
    base_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.2, solver="lbfgs", max_iter=1000, random_state=42)
    )

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_train_full, y_train_full, groups)):
        X_tr, y_tr = X_train_full.iloc[trn_idx], y_train_full.iloc[trn_idx]
        X_va, y_va = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]

        calibrator = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
        calibrator.fit(X_tr, y_tr)
        oof_preds[val_idx] = calibrator.predict_proba(X_va)[:, 1]

    cv_auc = roc_auc_score(y_train_full, oof_preds)
    print(f"  [5-Fold Out-Of-Fold CV ROC-AUC]: {cv_auc:.3f}")

    # 3. Save Diagnostic Calibration Plot
    os.makedirs("visualizations", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    CalibrationDisplay.from_predictions(y_train_full, oof_preds, n_bins=5, ax=ax, name="Calibrated Floor Model")
    ax.set_title(f"Calibration Curve - Rotation Prob (OOF AUC: {cv_auc:.3f})")
    plt.tight_layout()
    plt.savefig("visualizations/calibration_curve.png", dpi=300)
    plt.close()
    print("  -> Saved 'visualizations/calibration_curve.png'")

    # 4. Retrain Calibrated Model on Full Historical Data (2008-2019)
    print("\n--- RETRAINING CLASSIFIER ON FULL HISTORICAL DATA (2008-2019) ---")
    final_calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=5)
    final_calibrated_clf.fit(X_train_full, y_train_full)

    # 5. Score Unlabeled Prospects (2020+) & Update predictions.parquet
    if len(unlabeled_df) > 0:
        predictions_path = "data/processed/predictions.parquet"
        if not os.path.exists(predictions_path):
            raise FileNotFoundError(f"Missing {predictions_path}. Run train.py first!")

        pred_df = pd.read_parquet(predictions_path)

        X_unlabeled = unlabeled_df[feature_cols].fillna(0.0)
        unlabeled_probs = final_calibrated_clf.predict_proba(X_unlabeled)[:, 1]

        prob_map = dict(zip(unlabeled_df["draft_player_name"], unlabeled_probs))
        pred_df["rotation_prob"] = pred_df["draft_player_name"].map(prob_map).fillna(0.0)

        pred_df.to_parquet(predictions_path, index=False)
        print("  -> Successfully updated 'rotation_prob' in 'data/processed/predictions.parquet'!")


if __name__ == "__main__":
    train_calibrated_classifier()