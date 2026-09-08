import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

def generate_shap_plots():
    print("Loading ensemble model and data for SHAP analysis...")
    
    # 1. Load the finalized model artifact
    model_path = "models/model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find {model_path}. Run train.py first!")
    
    artifact = joblib.load(model_path)
    
    # Updated key to match the ensemble artifact
    lgb_model = artifact["lgb_model"]
    feature_cols = artifact["features"]

    # 2. Load feature data
    df = pd.read_parquet("data/processed/features.parquet")
    
    # Target 2021 draft class for SHAP explanations
    explain_df = df[df['draft_year'] == 2021].copy()
    
    for col in feature_cols:
        explain_df[col] = pd.to_numeric(explain_df[col], errors="coerce").fillna(0.0)
    
    X_explain = explain_df[feature_cols]

    print(f"Calculating SHAP values for {len(X_explain)} prospects in 2021...")
    
    # 3. Calculate SHAP values on the LightGBM component
    explainer = shap.TreeExplainer(lgb_model)
    shap_explanation = explainer(X_explain)

    os.makedirs("visualizations", exist_ok=True)

    # 4. Generate Global Summary Plot (Beeswarm)
    print("Generating Global SHAP Summary Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_explanation.values, 
        X_explain, 
        max_display=15, 
        show=False
    )
    plt.tight_layout()
    plot_path = "visualizations/shap_summary.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved '{plot_path}'")

    # 5. Generate Individual Waterfall Plots for Key Players (Step 29)
    target_players = ["Cade Cunningham", "Sharife Cooper", "Miles McBride"]
    
    for player in target_players:
        player_mask = explain_df['draft_player_name'].str.contains(player, case=False, na=False)
        if player_mask.any():
            player_idx = explain_df[player_mask].index[0]
            loc_idx = explain_df.index.get_loc(player_idx)

            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_explanation[loc_idx], max_display=10, show=False)
            
            clean_name = player.lower().replace(" ", "_")
            waterfall_path = f"visualizations/waterfall_{clean_name}.png"
            plt.tight_layout()
            plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  -> Saved waterfall plot for {player}: '{waterfall_path}'")

if __name__ == "__main__":
    generate_shap_plots()