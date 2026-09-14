import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
import optuna
import json

def objective(trial):
    df = pd.read_parquet("data/processed/features.parquet")
    train_df = df[df["is_training_cohort"] == True].copy().reset_index(drop=True)

    ignore_cols = [
        "draft_player_name", "draft_year", "drafted_team", "is_training_cohort",
        "vorp_5y", "reached_min_threshold_5y", "player_tier_5y", "overall_pick",
        "pos_group", "pos_bucket", "season", "mp_5y", "seasons_played_5y"
    ]
    target_col = "vorp_5y"
    feature_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]

    X = train_df[feature_cols].fillna(0.0)
    y = train_df[target_col].fillna(0.0).values
    groups = train_df["draft_year"].values

    # 1. Suggest LightGBM Hyperparameters
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "random_state": 42,
        "n_estimators": 300,
        "num_leaves": trial.suggest_int("num_leaves", 10, 40),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
    }

    # --- NEW: APPLY MONOTONIC CONSTRAINTS FOR AGE ---
    constraints = []
    for col in feature_cols:
        if col == "age_at_draft":
            constraints.append(-1)  # Force strict age penalty during Optuna trials
        else:
            constraints.append(0)

    lgb_params["monotone_constraints"] = tuple(constraints)
    lgb_params["monotone_constraints_method"] = "advanced"
    # ------------------------------------------------

    # 2. Suggest Ridge & Ensemble Weights
    ridge_alpha = trial.suggest_float("ridge_alpha", 5.0, 50.0)
    lgb_weight = trial.suggest_float("lgb_weight", 0.3, 0.9)
    ridge_weight = 1.0 - lgb_weight

    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(train_df))
    ndcg30s = []

    # 3. GroupKFold CV Loop
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(X_tr, y_tr)

        model_ridge = Ridge(alpha=ridge_alpha, random_state=42)
        model_ridge.fit(X_tr, y_tr)

        fold_preds = (lgb_weight * model_lgb.predict(X_va)) + (ridge_weight * model_ridge.predict(X_va))
        oof_preds[val_idx] = fold_preds
        
        # Calculate NDCG@30 for this fold
        y_va_ndcg = np.maximum(y_va, 0)
        if len(y_va) >= 30:
            fold_ndcg = ndcg_score([y_va_ndcg], [fold_preds], k=30)
            ndcg30s.append(fold_ndcg)

    # Optuna will try to MAXIMIZE the average Out-Of-Fold NDCG@30
    return np.mean(ndcg30s)

if __name__ == "__main__":
    print("Starting Optuna Hyperparameter Search...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    best_params = study.best_trial.params
    lgb_weight = float(best_params["lgb_weight"])
    ridge_weight = 1.0 - lgb_weight

    config = {
        "lgb_params": {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": int(best_params["num_leaves"]),
            "min_child_samples": int(best_params["min_child_samples"]),
            "learning_rate": float(best_params["learning_rate"]),
            "colsample_bytree": float(best_params["colsample_bytree"]),
            "subsample": float(best_params["subsample"]),
            "verbosity": -1,
            "random_state": 42,
        },
        "ridge_alpha": float(best_params["ridge_alpha"]),
        "lgb_weight": lgb_weight,
        "ridge_weight": ridge_weight,
    }

    os.makedirs("models", exist_ok=True)
    config_path = "models/best_params.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n✅ [SUCCESS] Saved best hyperparameters to '{config_path}'!")