import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def train_calibrated_classifier():
    print("==========================================")
    print(" TRAINING CALIBRATED ROTATION CLASSIFIER")
    print("==========================================\n")

    df = pd.read_parquet("data/processed/features.parquet")
    
    ignore_cols = [
        "draft_player_name", "draft_year", "drafted_team", "is_training_cohort",
        "vorp_5y", "reached_min_threshold_5y", "player_tier_5y", "overall_pick", "pos_group"
    ]
    
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    train_df = df[df["is_training_cohort"] == True].copy()
    unlabeled_df = df[df["is_training_cohort"] == False].copy()

    X_train = train_df[train_df["draft_year"] <= 2016][feature_cols].fillna(0.0)
    y_train = train_df[train_df["draft_year"] <= 2016]["reached_min_threshold_5y"].astype(int)

    X_val = train_df[train_df["draft_year"].isin([2017, 2018])][feature_cols].fillna(0.0)
    y_val = train_df[train_df["draft_year"].isin([2017, 2018])]["reached_min_threshold_5y"].astype(int)

    # Base Logistic Regression
    base_clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    
    # Calibrated Classifier (Platt Scaling)
    calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=3)
    calibrated_clf.fit(X_train, y_train)

    val_probs = calibrated_clf.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, val_probs)
    print(f"  [Validation ROC-AUC]: {auc_score:.3f}")

    # Plot Calibration Curve
    os.makedirs("visualizations", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    CalibrationDisplay.from_predictions(y_val, val_probs, n_bins=5, ax=ax, name="Rotation Classifier")
    ax.set_title("Calibration Curve - Rotation Probability")
    plt.tight_layout()
    plt.savefig("visualizations/calibration_curve.png", dpi=300)
    plt.close()
    print("  -> Saved 'visualizations/calibration_curve.png'")

    # Save Probabilities to predictions.parquet
    if len(unlabeled_df) > 0:
        predictions_path = "data/processed/predictions.parquet"
        pred_df = pd.read_parquet(predictions_path)
        
        X_unlabeled = unlabeled_df[feature_cols].fillna(0.0)
        pred_df["rotation_prob"] = calibrated_clf.predict_proba(X_unlabeled)[:, 1]
        pred_df.to_parquet(predictions_path, index=False)
        print("  -> Added 'rotation_prob' column to predictions.parquet!")

if __name__ == "__main__":
    train_calibrated_classifier()