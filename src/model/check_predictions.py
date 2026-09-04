import pandas as pd

def inspect_predictions():
    df = pd.read_parquet("data/processed/predictions.parquet")

    print(f"Total Future/Unlabeled Predictions Loaded: {len(df)}")
    print(f"Draft Years Covered: {sorted(df['draft_year'].dropna().unique())}\n")

    # Sample top 5 model predictions for the 2021 draft class
    sample_year = 2021
    year_df = df[df['draft_year'] == sample_year].sort_values("pred_vorp_5y", ascending=False)

    print(f"=== MODEL A TOP 10 PROSPECTS FOR {sample_year} DRAFT ===")
    cols_to_show = ["draft_player_name", "overall_pick", "pred_vorp_5y", "model_rank", "college_bpm", "pts_per_40", "age_at_draft"]
    existing_cols = [c for c in cols_to_show if c in year_df.columns]
    
    print(year_df[existing_cols].head(10).to_string(index=False))

    # Show biggest model disagreements (Model A high rank vs lower actual draft pick)
    if "overall_pick" in year_df.columns:
        year_df["pick_minus_rank"] = year_df["overall_pick"] - year_df["model_rank"]
        sleepers = year_df.sort_values("pick_minus_rank", ascending=False).head(5)
        print(f"\n=== MODEL A SLEEPERS FOR {sample_year} (Model Loved, Drafted Later) ===")
        print(sleepers[existing_cols].to_string(index=False))

if __name__ == "__main__":
    inspect_predictions()