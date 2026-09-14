import os
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap


def sanitize_filename(name):
    """Sanitize player names to create valid file paths."""
    return re.sub(r'[^\w\s-]', '', str(name)).strip().replace(' ', '_')


def generate_all_shap_plots():
    features_path = "data/processed/features.parquet"
    model_path = "models/model.joblib"
    base_output_dir = "data/shap_plots"

    if not os.path.exists(features_path):
        print(f"Error: {features_path} not found.")
        return

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    df = pd.read_parquet(features_path)
    artifact = joblib.load(model_path)

    # Resolve LightGBM model and explicit feature list from artifact
    if isinstance(artifact, dict):
        lgb_model = artifact.get("lgb_model")
        model_features = artifact.get("features")
    else:
        lgb_model = artifact
        model_features = None

    if lgb_model is None:
        print("Error: Could not retrieve LightGBM model from artifact.")
        return

    # Fallback only if artifact didn't store the feature list
    if not model_features:
        ignore_cols = [
            "draft_player_name", "draft_year", "drafted_team", "is_training_cohort",
            "vorp_5y", "reached_min_threshold_5y", "player_tier_5y", "overall_pick",
            "pos_group", "nba_position", "pos_bucket", "season"
        ]
        model_features = [
            c for c in df.columns 
            if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

    # Initialize SHAP Tree Explainer
    explainer = shap.TreeExplainer(lgb_model)

    # Clean draft_year for safe filtering
    if "draft_year" in df.columns:
        df["draft_year"] = pd.to_numeric(df["draft_year"], errors="coerce").fillna(0).astype(int)

    # Filter for target draft classes (2020 through 2026)
    target_years = range(2020, 2027)
    df_targets = df[df["draft_year"].isin(target_years)].copy()

    total_players = len(df_targets)
    print(f"Starting SHAP plot generation for {total_players} prospects (2020-2026)...")

    saved_count = 0

    for year in target_years:
        year_df = df_targets[df_targets["draft_year"] == year]
        if year_df.empty:
            continue

        year_dir = os.path.join(base_output_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        for _, row in year_df.iterrows():
            player_name = row["draft_player_name"]
            file_safe_name = sanitize_filename(player_name)
            output_file = os.path.join(year_dir, f"{file_safe_name}.png")

            # Extract exact feature vector and force to float to prevent LightGBM dtype errors
            X_player = pd.DataFrame([row[model_features]]).fillna(0.0).astype(float)

            # Calculate SHAP values for the player
            shap_values = explainer(X_player)

            # Initialize clean Matplotlib figure
            fig = plt.figure(figsize=(8, 5))

            # Render SHAP waterfall plot
            shap.plots.waterfall(
                shap_values[0], 
                max_display=8, 
                show=False
            )

            plt.title(f"SHAP Feature Drivers: {player_name} ({year})", fontsize=12, pad=15)
            plt.tight_layout()

            # Save high-resolution PNG
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close(fig)

            saved_count += 1

        print(f" Saved {len(year_df)} SHAP plots for Class of {year} in '{year_dir}'")

    print(f"\nCompleted! Successfully saved {saved_count} SHAP plots to '{base_output_dir}/'.")


if __name__ == "__main__":
    generate_all_shap_plots()