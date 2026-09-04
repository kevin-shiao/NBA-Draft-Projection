import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
import shap

from metrics import compute_ranking_metrics

def run_phase_7_pipeline():
    print("==========================================")
    print(" PHASE 7: MODEL TRAINING & EVALUATION")
    print("==========================================\n")

    # Step 25: Load Local Parquet Data
    data_path = "data/processed/features.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run export_features.py first!")
    
    df = pd.read_parquet(data_path)
    
    # Isolate training set (2009-2019)
    train_df = df[df['is_training_cohort'] == True].copy()
    unlabeled_df = df[df['is_training_cohort'] == False].copy()

    # Step 26: Strict Cohort Splits
    train_set = train_df[train_df['draft_year'] <= 2016]
    val_set = train_df[train_df['draft_year'].isin([2017, 2018])]
    test_set = train_df[train_df['draft_year'] == 2019]

    # Target definition
    target_col = 'vorp_5y'

    # Model A Features (No Pick #)
    ignore_cols = [
        'draft_player_name', 'draft_year', 'drafted_team', 'is_training_cohort',
        'vorp_5y', 'reached_min_threshold_5y', 'player_tier_5y', 'overall_pick'
    ]
    feature_cols_a = [c for c in train_df.columns if c not in ignore_cols]
    
    # Model B Features (Includes Pick # Benchmark)
    feature_cols_b = feature_cols_a + ['overall_pick']

    X_train_a, y_train = train_set[feature_cols_a], train_set[target_col]
    X_val_a, y_val = val_set[feature_cols_a], val_set[target_col]
    X_test_a, y_test = test_set[feature_cols_a], test_set[target_col]

    # --- Step 27: Baseline Models ---
    print("--- 1. EVALUATING BASELINES ON VALIDATION (2017-2018) ---")
    
    # Baseline 1: Training Mean Predictor
    mean_pred = np.full(len(y_val), y_train.mean())
    bm_mean = compute_ranking_metrics(y_val.values, mean_pred, val_set['draft_year'].values)
    print(f"  [Baseline 1 - Mean Predictor] Spearman: {bm_mean['spearman_rho']:.3f} | NDCG@10: {bm_mean['ndcg_10']:.3f}")

    # Baseline 2: Draft Pick Number Alone (The Real Competitor)
    # Inverse overall pick so pick #1 gets highest score
    pick_pred = -val_set['overall_pick'].values
    bm_pick = compute_ranking_metrics(y_val.values, pick_pred, val_set['draft_year'].values)
    print(f"  [Baseline 2 - Pick Number]   Spearman: {bm_pick['spearman_rho']:.3f} | NDCG@10: {bm_pick['ndcg_10']:.3f} | Hit Rate: {bm_pick['hit_rate_top10_in_30']:.2f}")

    # Baseline 3: Ridge Regression (~10 core features)
    core_10 = ['pts_per_40', 'college_bpm', 'age_at_draft', 'ape_index', 'ts_pct', 
               'ast_per_40', 'reb_per_40', 'height_inches', 'usage_pct', 'is_power_5']
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_a[core_10], y_train)
    ridge_pred = ridge.predict(X_val_a[core_10])
    bm_ridge = compute_ranking_metrics(y_val.values, ridge_pred, val_set['draft_year'].values)
    print(f"  [Baseline 3 - Ridge Reg 10]  Spearman: {bm_ridge['spearman_rho']:.3f} | NDCG@10: {bm_ridge['ndcg_10']:.3f}")

    # --- Step 27 & 28: LightGBM Model A vs Model B ---
    print("\n--- 2. TRAINING LIGHTGBM MODELS ---")
    
    # Small-Data Constrained Parameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 15,
        'min_child_samples': 20,
        'learning_rate': 0.03,
        'verbosity': -1,
        'random_state': 42
    }

    # Model A: Pure College & Physical Signal
    model_a = lgb.LGBMRegressor(**lgb_params, n_estimators=500)
    model_a.fit(
        X_train_a, y_train,
        eval_set=[(X_val_a, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    val_pred_a = model_a.predict(X_val_a)
    m_a_val = compute_ranking_metrics(y_val.values, val_pred_a, val_set['draft_year'].values)
    print(f"  [Model A (No Pick) - Val]    Spearman: {m_a_val['spearman_rho']:.3f} | NDCG@10: {m_a_val['ndcg_10']:.3f} | Hit Rate: {m_a_val['hit_rate_top10_in_30']:.2f}")

    # Model B: Includes Draft Pick Benchmark
    X_train_b, X_val_b = train_set[feature_cols_b], val_set[feature_cols_b]
    model_b = lgb.LGBMRegressor(**lgb_params, n_estimators=500)
    model_b.fit(
        X_train_b, y_train,
        eval_set=[(X_val_b, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    val_pred_b = model_b.predict(X_val_b)
    m_b_val = compute_ranking_metrics(y_val.values, val_pred_b, val_set['draft_year'].values)
    print(f"  [Model B (With Pick) - Val]  Spearman: {m_b_val['spearman_rho']:.3f} | NDCG@10: {m_b_val['ndcg_10']:.3f} | Hit Rate: {m_b_val['hit_rate_top10_in_30']:.2f}")

    # --- FINAL TEST EVALUATION (2019 Cohort - Touch Once) ---
    print("\n--- 3. UNTOUCHED 2019 TEST COHORT EVALUATION ---")
    test_pred_a = model_a.predict(X_test_a)
    test_metrics_a = compute_ranking_metrics(y_test.values, test_pred_a, test_set['draft_year'].values)
    print(f"  -> [Model A - 2019 Test Set] Spearman: {test_metrics_a['spearman_rho']:.3f} | NDCG@10: {test_metrics_a['ndcg_10']:.3f} | Hit Rate: {test_metrics_a['hit_rate_top10_in_30']:.2f}")

    # --- Step 29: SHAP Interpretability ---
    print("\n--- 4. GENERATING SHAP EXPLANATIONS ---")
    explainer = shap.TreeExplainer(model_a)
    shap_values = explainer(X_val_a)
    print("  -> SHAP values calculated successfully for Model A.")

    # --- Step 31: Retrain on Full Historical Cohort (<=2019) & Save ---
    print("\n--- 5. RETRAINING ON ALL TRAIN DATA (2009-2019) & EXPORTING ---")
    X_full_a = train_df[feature_cols_a]
    y_full = train_df[target_col]

    final_model_a = lgb.LGBMRegressor(**lgb_params, n_estimators=model_a.best_iteration_)
    final_model_a.fit(X_full_a, y_full)

    os.makedirs("models", exist_ok=True)
    model_artifact_path = "models/model.joblib"
    
    artifact = {
        "model": final_model_a,
        "features": feature_cols_a,
        "version": "1.0"
    }
    joblib.dump(artifact, model_artifact_path)
    print(f"  -> [SUCCESS] Saved final model artifact to '{model_artifact_path}'")

    # Score Unlabeled prospects (2020+) for the Web Application
    if len(unlabeled_df) > 0:
        X_unlabeled = unlabeled_df[feature_cols_a]
        unlabeled_df['pred_vorp_5y'] = final_model_a.predict(X_unlabeled)
        
        # Rank within draft class
        unlabeled_df['model_rank'] = unlabeled_df.groupby('draft_year')['pred_vorp_5y'].rank(ascending=False, method='min')
        
        output_predictions_path = "data/processed/predictions.parquet"
        unlabeled_df.to_parquet(output_predictions_path, index=False)
        print(f"  -> [SUCCESS] Exported {len(unlabeled_df)} future prospect predictions to '{output_predictions_path}'")

if __name__ == "__main__":
    run_phase_7_pipeline()