import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

def generate_shap_plots():
    print("Loading model and data for SHAP analysis...")
    
    # 1. Load the finalized model artifact
    model_path = "models/model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find {model_path}. Run train.py first!")
    
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["features"]

    # 2. Load the features dataset
    df = pd.read_parquet("data/processed/features.parquet")
    
    # Isolate a recent draft class to explain (e.g., the 2021 class)
    # Using real prospect data makes the SHAP summary much more interpretable
    recent_class = df[df['draft_year'] == 2021].copy()
    
    # Ensure columns match training exactly
    for col in feature_cols:
        recent_class[col] = pd.to_numeric(recent_class[col], errors="coerce").fillna(0.0)
    
    X_explain = recent_class[feature_cols]

    print(f"Calculating SHAP values for {len(X_explain)} prospects in the 2021 class...")
    
    # 3. Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_explain)

    # 4. Generate the Summary Plot (Beeswarm)
    print("Generating SHAP Summary Plot...")
    
    # Adjust plot size and style for readability
    plt.figure(figsize=(12, 8))
    
    shap.summary_plot(
        shap_values, 
        X_explain, 
        max_display=15,  # Show the top 15 most important features
        show=False       # Prevent the plot from blocking the script
    )
    
    # 5. Save the plot to an image file
    os.makedirs("visualizations", exist_ok=True)
    plot_path = "visualizations/shap_summary.png"
    
    # Tight layout ensures labels don't get cut off
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SUCCESS] SHAP feature importance plot saved to: '{plot_path}'")
    print("Open this image file to see exactly what drives the model's predictions!")

if __name__ == "__main__":
    generate_shap_plots()