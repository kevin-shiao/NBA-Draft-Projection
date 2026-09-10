import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr
import mlflow
import mlflow.lightgbm
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Handle DATABRICKS_SERVER_HOSTNAME mapping
if "DATABRICKS_SERVER_HOSTNAME" in os.environ and "DATABRICKS_HOST" not in os.environ:
    host = os.environ["DATABRICKS_SERVER_HOSTNAME"]
    os.environ["DATABRICKS_HOST"] = host if host.startswith("https://") else f"https://{host}"

if "DATABRICKS_TOKEN" not in os.environ and "DATABRICKS_PERSONAL_ACCESS_TOKEN" in os.environ:
    os.environ["DATABRICKS_TOKEN"] = os.environ["DATABRICKS_PERSONAL_ACCESS_TOKEN"]

def compute_ranking_metrics(y_true, y_pred, draft_years, overall_picks=None):
    """
    Computes Spearman Rho, NDCG@10, NDCG@30, and Top-10 Hit Rate overall and
    against the actual draft order baseline.
    """
    df_res = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'draft_year': draft_years,
        'overall_pick': overall_picks if overall_picks is not None else np.nan
    })

    rhos, ndcg10s, ndcg30s, hit_rates = [], [], [], []
    draft_rhos, draft_ndcg10s, draft_ndcg30s, draft_hit_rates = [], [], [], []

    for yr, group in df_res.groupby('draft_year'):
        if len(group) < 10:
            continue
        yt = group['y_true'].values
        yp = group['y_pred'].values

        # Make targets non-negative for NDCG (sub-replacement = 0 relevance)
        yt_ndcg = np.maximum(yt, 0)

        # Model Metrics
        rhos.append(spearmanr(yt, yp)[0])
        ndcg10s.append(ndcg_score([yt_ndcg], [yp], k=10))
        ndcg30s.append(ndcg_score([yt_ndcg], [yp], k=30))

        top30_true_idx = set(np.argsort(yt)[::-1][:30])
        top10_pred_idx = set(np.argsort(yp)[::-1][:10])
        hit_rates.append(len(top10_pred_idx.intersection(top30_true_idx)) / 10.0)

        # Draft Order Baseline Metrics
        if overall_picks is not None and not group['overall_pick'].isna().all():
            yd = -group['overall_pick'].values  # Negated so pick #1 is highest rank
            draft_rhos.append(spearmanr(yt, yd)[0])
            draft_ndcg10s.append(ndcg_score([yt_ndcg], [yd], k=10))
            draft_ndcg30s.append(ndcg_score([yt_ndcg], [yd], k=30))
            top10_draft_idx = set(np.argsort(yd)[::-1][:10])
            draft_hit_rates.append(len(top10_draft_idx.intersection(top30_true_idx)) / 10.0)

    res = {
        'spearman_rho': float(np.nanmean(rhos)),
        'ndcg_10': float(np.nanmean(ndcg10s)),
        'ndcg_30': float(np.nanmean(ndcg30s)),
        'hit_rate': float(np.nanmean(hit_rates)),
    }
    if draft_rhos:
        res.update({
            'draft_spearman_rho': float(np.nanmean(draft_rhos)),
            'draft_ndcg_10': float(np.nanmean(draft_ndcg10s)),
            'draft_ndcg_30': float(np.nanmean(draft_ndcg30s)),
            'draft_hit_rate': float(np.nanmean(draft_hit_rates)),
        })
    return res


def run_training_pipeline():
    print("==========================================")
    print(" PRE-DRAFT MODEL TRAINING PIPELINE")
    print("==========================================\n")

    # 1. MLflow Tracking Setup
    try:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
    except Exception as e:
        print(f"Notice setting MLflow URIs: {e}")

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/nba-draft")
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            exp_id = mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_id=exp_id)
        else:
            mlflow.set_experiment(experiment_name)
        print(f"  [MLflow] Logging to experiment: '{experiment_name}'")
    except Exception as e:
        print(f"Warning setting MLflow experiment '{experiment_name}': {e}")

    # 2. Load Processed Parquet Data
    data_path = "data/processed/features.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run export_features.py first!")

    df = pd.read_parquet(data_path)

    # 3. Explicit Leakage Prevention: Exclude post-draft, non-feature, and UI bucket columns
    ignore_cols = [
        "draft_player_name",
        "draft_year",
        "drafted_team",
        "is_training_cohort",
        "vorp_5y",
        "reached_min_threshold_5y",
        "player_tier_5y",
        "overall_pick",  # Excluded from training features
        "nba_position",  # Excluded from training features
        "pos_group",
        "pos_bucket",    # Used for frontend UI filtering only
        "season",
    ]

    target_col = "vorp_5y"
    feature_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]

    # Split Cohorts
    train_df = df[df["is_training_cohort"] == True].copy().reset_index(drop=True)
    unlabeled_df = df[df["is_training_cohort"] == False].copy().reset_index(drop=True)

    X_train_full = train_df[feature_cols].fillna(0.0)
    y_train_full = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0.0).values
    groups = train_df["draft_year"].values

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 15,
        "min_child_samples": 15,       # Lowered slightly to capture nuanced archetype interactions
        "learning_rate": 0.02,         # Slowed down for better generalization
        "colsample_bytree": 0.8,       # Force trees to use different features (prevents over-reliance on usage)
        "subsample": 0.8,              # Bagging to prevent overfitting to specific players
        "verbosity": -1,
        "random_state": 42,
    }

    # 4. GroupKFold Out-Of-Fold Cross-Validation (2008-2019)
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(train_df))

    print("\n--- RUNNING GROUP-KFOLD CROSS-VALIDATION (BY DRAFT YEAR) ---")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_full, y_train_full, groups=groups)):
        X_tr, y_tr = X_train_full.iloc[train_idx], y_train_full[train_idx]
        X_va, y_va = X_train_full.iloc[val_idx], y_train_full[val_idx]

        lgb_m = lgb.LGBMRegressor(**lgb_params, n_estimators=300)
        lgb_m.fit(X_tr, y_tr)

        ridge_m = Ridge(alpha=20.0, random_state=42)
        ridge_m.fit(X_tr, y_tr)

        oof_preds[val_idx] = (0.70 * lgb_m.predict(X_va)) + (0.30 * ridge_m.predict(X_va))

    train_df["oof_pred"] = oof_preds
    train_df["residual"] = y_train_full - oof_preds

    # 5. Out-Of-Fold Residual Diagnostics Table
    print("\n==========================================")
    print(" OUT-OF-FOLD RESIDUAL DIAGNOSTICS")
    print("==========================================")

    # Subgroup bins (Removed explicit labels so pandas uses the actual interval ranges safely)
    age_col = "age_at_draft" if "age_at_draft" in train_df.columns else "age"
    usg_col = "usg" if "usg" in train_df.columns else "usage_pct"
    rec_col = "rec_rank_clean" if "rec_rank_clean" in train_df.columns else "rec_rank_log"

    if age_col in train_df.columns:
        train_df["age_band"] = pd.qcut(train_df[age_col], q=3, duplicates="drop")
    if usg_col in train_df.columns:
        train_df["usg_tercile"] = pd.qcut(train_df[usg_col], q=3, duplicates="drop")
    if rec_col in train_df.columns:
        train_df["rec_rank_tier"] = pd.qcut(train_df[rec_col], q=3, duplicates="drop")

    for grp in ["pos_bucket", "age_band", "usg_tercile", "rec_rank_tier"]:
        if grp in train_df.columns:
            print(f"\n--- Residuals by {grp} ---")
            diag = train_df.groupby(grp, observed=False)["residual"].agg(["mean", "std", "count"])
            diag["t_stat"] = diag["mean"] / (diag["std"] / np.sqrt(diag["count"]) + 1e-6)
            diag["bias_flag"] = diag["t_stat"].apply(lambda x: "*** (BIAS)" if abs(x) > 2.0 else "")
            print(diag[["mean", "std", "count", "t_stat", "bias_flag"]])

    # 6. Evaluation vs Draft Order Baseline
    overall_picks_val = train_df["overall_pick"].values if "overall_pick" in train_df.columns else None
    eval_metrics = compute_ranking_metrics(y_train_full, oof_preds, train_df["draft_year"].values, overall_picks_val)

    print("\n==========================================")
    print(" MODEL PERFORMANCE VS DRAFT ORDER BASELINE")
    print("==========================================")
    print(f"Spearman Rho   : Model = {eval_metrics['spearman_rho']:.3f} | Draft Order = {eval_metrics.get('draft_spearman_rho', 0):.3f}")
    print(f"NDCG@10        : Model = {eval_metrics['ndcg_10']:.3f} | Draft Order = {eval_metrics.get('draft_ndcg_10', 0):.3f}")
    print(f"NDCG@30        : Model = {eval_metrics['ndcg_30']:.3f} | Draft Order = {eval_metrics.get('draft_ndcg_30', 0):.3f}")
    print(f"Top-10 Hit Rate: Model = {eval_metrics['hit_rate']:.3f} | Draft Order = {eval_metrics.get('draft_hit_rate', 0):.3f}")

    # 7. Start MLflow Run
    with mlflow.start_run(run_name="70_30_GroupKFold_Ensemble") as run:
        print(f"\n  [MLflow] Active Run ID: {run.info.run_id}")

        mlflow.log_params(lgb_params)
        mlflow.log_params({
            "model_type": "70/30 Ensemble (LightGBM + Ridge)",
            "ridge_alpha": 20.0,
            "lgb_weight": 0.70,
            "ridge_weight": 0.30,
            "num_features": len(feature_cols),
        })

        # Log Metrics
        mlflow.log_metric("oof_spearman_rho", eval_metrics["spearman_rho"])
        mlflow.log_metric("oof_ndcg_10", eval_metrics["ndcg_10"])
        mlflow.log_metric("oof_ndcg_30", eval_metrics["ndcg_30"])
        mlflow.log_metric("oof_hit_rate_top10_in_30", eval_metrics["hit_rate"])

        if "draft_spearman_rho" in eval_metrics:
            mlflow.log_metric("draft_spearman_rho", eval_metrics["draft_spearman_rho"])
            mlflow.log_metric("draft_ndcg_10", eval_metrics["draft_ndcg_10"])

        # 8. Retrain Final Ensemble on All Historical Data (2008-2019)
        print("\n--- RETRAINING ENSEMBLE ON ALL HISTORICAL DATA (2008-2019) ---")
        final_lgb = lgb.LGBMRegressor(**lgb_params, n_estimators=300)
        final_lgb.fit(X_train_full, y_train_full)

        final_ridge = Ridge(alpha=20.0, random_state=42)
        final_ridge.fit(X_train_full, y_train_full)

        # 9. Save Local Model Artifact
        os.makedirs("models", exist_ok=True)
        model_artifact_path = "models/model.joblib"
        artifact = {
            "lgb_model": final_lgb,
            "ridge_model": final_ridge,
            "lgb_weight": 0.70,
            "ridge_weight": 0.30,
            "features": feature_cols,
            "version": "3.0_groupkfold_archetype_shrunk_ensemble",
        }
        joblib.dump(artifact, model_artifact_path)
        print(f"  -> Saved local model artifact to '{model_artifact_path}'")

        # 10. Score Unlabeled Target Set (2020+)
        if len(unlabeled_df) > 0:
            X_unlabeled = unlabeled_df[feature_cols].fillna(0.0)

            lgb_unlabeled_pred = final_lgb.predict(X_unlabeled)
            ridge_unlabeled_pred = final_ridge.predict(X_unlabeled)

            unlabeled_df["pred_vorp_5y"] = (0.70 * lgb_unlabeled_pred) + (0.30 * ridge_unlabeled_pred)

            # Class Rank
            unlabeled_df["model_rank"] = unlabeled_df.groupby("draft_year")["pred_vorp_5y"].rank(
                ascending=False, method="min"
            )

            output_predictions_path = "data/processed/predictions.parquet"
            unlabeled_df.to_parquet(output_predictions_path, index=False)
            print(f"  -> [SUCCESS] Exported {len(unlabeled_df)} pre-draft predictions to '{output_predictions_path}'")


if __name__ == "__main__":
    run_training_pipeline()