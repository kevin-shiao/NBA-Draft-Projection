import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
import mlflow
import mlflow.lightgbm
from mlflow.models import infer_signature
from dotenv import load_dotenv

from metrics import compute_ranking_metrics

# 1. Load Environment Variables from .env
load_dotenv()

# Handle DATABRICKS_SERVER_HOSTNAME mapping to DATABRICKS_HOST if needed
if "DATABRICKS_SERVER_HOSTNAME" in os.environ and "DATABRICKS_HOST" not in os.environ:
    host = os.environ["DATABRICKS_SERVER_HOSTNAME"]
    os.environ["DATABRICKS_HOST"] = host if host.startswith("https://") else f"https://{host}"

if "DATABRICKS_TOKEN" not in os.environ and "DATABRICKS_PERSONAL_ACCESS_TOKEN" in os.environ:
    os.environ["DATABRICKS_TOKEN"] = os.environ["DATABRICKS_PERSONAL_ACCESS_TOKEN"]


def run_training_pipeline():
    print("==========================================")
    print(" PRE-DRAFT MODEL TRAINING PIPELINE (WITH MLFLOW & DATABRICKS UC)")
    print("==========================================\n")

    # 2. Configure MLflow for Databricks Tracking & Unity Catalog Model Registry
    try:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
    except Exception as e:
        print(f"Notice setting MLflow URIs: {e}")

    # Use /Shared/nba-draft as default so it works out-of-the-box on Databricks
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

    # 3. Load Local Parquet Data (Pre-processed with height-gated ape index)
    data_path = "data/processed/features.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run export_features.py first!")

    df = pd.read_parquet(data_path)

    # Exclude metadata, text, and post-draft pick information
    ignore_cols = [
        "draft_player_name",
        "draft_year",
        "drafted_team",
        "is_training_cohort",
        "vorp_5y",
        "reached_min_threshold_5y",
        "player_tier_5y",
        "overall_pick",  # Excluded to eliminate post-draft leakage
        "pos_group",
    ]

    for col in df.columns:
        if col not in ignore_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["vorp_5y"] = pd.to_numeric(df["vorp_5y"], errors="coerce").fillna(0.0)
    df["draft_year"] = pd.to_numeric(df["draft_year"], errors="coerce")

    # Cohort Splits
    train_df = df[df["is_training_cohort"] == True].copy()
    unlabeled_df = df[df["is_training_cohort"] == False].copy()

    train_set = train_df[train_df["draft_year"] <= 2016]
    val_set = train_df[train_df["draft_year"].isin([2017, 2018])]
    test_set = train_df[train_df["draft_year"] == 2019]

    target_col = "vorp_5y"
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    X_train, y_train = train_set[feature_cols], train_set[target_col]
    X_val, y_val = val_set[feature_cols], val_set[target_col]
    X_test, y_test = test_set[feature_cols], test_set[target_col]

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 15,
        "min_child_samples": 20,
        "learning_rate": 0.03,
        "verbosity": -1,
        "random_state": 42,
    }

    # 4. START MLFLOW RUN
    with mlflow.start_run(run_name="70_30_PreDraft_Ensemble") as run:
        print(f"  [MLflow] Active Run ID: {run.info.run_id}")

        # Log Hyperparameters & Model Config
        mlflow.log_params(lgb_params)
        mlflow.log_params({
            "model_type": "70/30 Ensemble (LightGBM + Ridge)",
            "ridge_alpha": 20.0,
            "lgb_weight": 0.70,
            "ridge_weight": 0.30,
            "num_features": len(feature_cols),
        })

        # --- Train Ensemble Components on Training Set ---
        lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=500)
        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        val_pred_lgb = lgb_model.predict(X_val)

        ridge_model = Ridge(alpha=20.0, random_state=42)
        ridge_model.fit(X_train, y_train)
        val_pred_ridge = ridge_model.predict(X_val)

        # 70/30 Validation Blend
        val_pred_ensemble = (0.70 * val_pred_lgb) + (0.30 * val_pred_ridge)
        m_ens = compute_ranking_metrics(y_val.values, val_pred_ensemble, val_set["draft_year"].values)

        # Log Validation Metrics
        mlflow.log_metric("val_spearman_rho", m_ens["spearman_rho"])
        mlflow.log_metric("val_ndcg_10", m_ens["ndcg_10"])
        mlflow.log_metric("val_ndcg_30", m_ens["ndcg_30"])
        mlflow.log_metric("val_hit_rate_top10_in_30", m_ens["hit_rate_top10_in_30"])

        print(f"  [Pre-Draft Ensemble - Val] Spearman: {m_ens['spearman_rho']:.3f} | NDCG@10: {m_ens['ndcg_10']:.3f} | Hit Rate: {m_ens['hit_rate_top10_in_30']:.2f}")

        # --- Evaluate Untouched 2019 Test Cohort ---
        test_pred_lgb = lgb_model.predict(X_test)
        test_pred_ridge = ridge_model.predict(X_test)
        test_pred_ens = (0.70 * test_pred_lgb) + (0.30 * test_pred_ridge)

        test_metrics = compute_ranking_metrics(y_test.values, test_pred_ens, test_set["draft_year"].values)

        # Log Test Metrics
        mlflow.log_metric("test_2019_spearman_rho", test_metrics["spearman_rho"])
        mlflow.log_metric("test_2019_ndcg_10", test_metrics["ndcg_10"])
        mlflow.log_metric("test_2019_ndcg_30", test_metrics["ndcg_30"])
        mlflow.log_metric("test_2019_hit_rate", test_metrics["hit_rate_top10_in_30"])

        print(f"  -> [2019 Test Set] Spearman: {test_metrics['spearman_rho']:.3f} | NDCG@10: {test_metrics['ndcg_10']:.3f} | Hit Rate: {test_metrics['hit_rate_top10_in_30']:.2f}")

        # --- Retrain Ensemble on All Historical Data (2009-2019) ---
        print("\n--- RETRAINING ENSEMBLE ON ALL TRAIN DATA (2009-2019) ---")
        X_full = train_df[feature_cols]
        y_full = train_df[target_col]

        final_lgb = lgb.LGBMRegressor(**lgb_params, n_estimators=lgb_model.best_iteration_)
        final_lgb.fit(X_full, y_full)

        final_ridge = Ridge(alpha=20.0, random_state=42)
        final_ridge.fit(X_full, y_full)

        # Save Local Artifact
        os.makedirs("models", exist_ok=True)
        model_artifact_path = "models/model.joblib"
        artifact = {
            "lgb_model": final_lgb,
            "ridge_model": final_ridge,
            "lgb_weight": 0.70,
            "ridge_weight": 0.30,
            "features": feature_cols,
            "version": "2.2_cubic_height_gated_ape_index_ensemble",
        }
        joblib.dump(artifact, model_artifact_path)
        print(f"  -> Saved local model artifact to '{model_artifact_path}'")

        # Score Unlabeled Prospects (2020+)
        if len(unlabeled_df) > 0:
            X_unlabeled = unlabeled_df[feature_cols]

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