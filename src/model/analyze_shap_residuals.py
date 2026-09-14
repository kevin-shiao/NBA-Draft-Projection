import joblib
import pandas as pd
import numpy as np
import shap

def run_residual_analysis():
    print("Running SHAP Residual Analysis...")
    
    df = pd.read_parquet("data/processed/features.parquet")
    train_df = df[(df["is_training_cohort"] == True) & (df["vorp_5y"].notna())].copy()
    
    artifact = joblib.load("models/model.joblib")
    lgb_model = artifact.get("lgb_model")
    model_features = artifact.get("features")
    
    X_train = train_df[model_features].fillna(0.0).astype(float)
    
    # Calculate Residuals (Actual - Predicted)
    # Note: We use purely the LightGBM predictions here to align with the TreeExplainer
    train_df["lgb_pred"] = lgb_model.predict(X_train)
    train_df["residual"] = train_df["vorp_5y"] - train_df["lgb_pred"]
    
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_train)
    
    shap_df = pd.DataFrame(shap_values, columns=model_features)
    shap_df["draft_player_name"] = train_df["draft_player_name"].values
    shap_df["residual"] = train_df["residual"].values
    
    # 1. Isolate the Model's Biggest Misses
    # Overvalued (Residual < -1.0): Model predicted high VORP, actual was terrible.
    overvalued = shap_df[shap_df["residual"] < -1.0].copy()
    
    # Undervalued (Residual > 2.0): Model predicted low VORP, actual was a star.
    undervalued = shap_df[shap_df["residual"] > 2.0].copy()
    
    print(f"\n--- FEATURES THE MODEL OVERVALUES (Drives Busts) ---")
    print(f"Analyzed {len(overvalued)} players the model projected highly who failed.")
    # Calculate average SHAP contribution for these busts
    over_means = overvalued[model_features].mean().sort_values(ascending=False).head(5)
    for feat, val in over_means.items():
        print(f"  {feat}: +{val:.3f} average SHAP push")

    print(f"\n--- FEATURES THE MODEL UNDERVALUES (Missed Steals) ---")
    print(f"Analyzed {len(undervalued)} players the model hated who became stars.")
    # Calculate what features dragged these stars down in the model
    under_means = undervalued[model_features].mean().sort_values(ascending=True).head(5)
    for feat, val in under_means.items():
        print(f"  {feat}: {val:.3f} average SHAP penalty")

if __name__ == "__main__":
    run_residual_analysis()