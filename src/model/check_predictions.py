import pandas as pd

def inspect_predictions():
    df = pd.read_parquet("data/processed/predictions.parquet")

    sample_year = 2021
    year_df = df[df['draft_year'] == sample_year].sort_values("model_rank", ascending=True)

    print(f"=== PRE-DRAFT MODEL TOP 10 PROSPECTS FOR {sample_year} ===")
    cols_to_show = ["draft_player_name", "pos_group", "overall_pick", "pred_vorp_5y", "model_rank", "college_bpm", "pts_per_40", "age_at_draft"]
    existing_cols = [c for c in cols_to_show if c in year_df.columns]
    
    print(year_df[existing_cols].head(10).to_string(index=False))

    if "overall_pick" in year_df.columns:
        year_df["pick_minus_rank"] = year_df["overall_pick"] - year_df["model_rank"]
        sleepers = year_df.sort_values("pick_minus_rank", ascending=False).head(5)
        print(f"\n=== MODEL SLEEPERS FOR {sample_year} (Model Loved, Drafted Later) ===")
        print(sleepers[existing_cols].to_string(index=False))

if __name__ == "__main__":
    inspect_predictions()