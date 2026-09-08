import pandas as pd

def find_cade():
    df = pd.read_parquet("data/processed/predictions.parquet")
    cade = df[df['draft_player_name'].str.contains("Cade Cunningham", case=False, na=False)]
    
    cols = ["draft_player_name", "draft_year", "overall_pick", "pred_vorp_5y", "model_rank", "college_bpm", "pts_per_40", "age_at_draft"]
    existing_cols = [c for c in cols if c in cade.columns]
    
    print("=== CADE CUNNINGHAM PRE-DRAFT PREDICTION RECORD ===")
    print(cade[existing_cols].to_string(index=False))

if __name__ == "__main__":
    find_cade()