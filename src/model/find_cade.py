import pandas as pd

def find_cade():
    df = pd.read_parquet("data/processed/predictions.parquet")
    
    # Filter for Cade Cunningham
    cade = df[df['draft_player_name'].str.contains("Cade", case=False, na=False)]
    
    if len(cade) > 0:
        print("=== CADE CUNNINGHAM PREDICTION RECORD ===")
        cols_to_show = [
            "draft_player_name", "draft_year", "overall_pick", 
            "pred_vorp_5y", "model_rank", "college_bpm", 
            "pts_per_40", "age_at_draft", "is_non_college_prospect"
        ]
        existing = [c for c in cols_to_show if c in cade.columns]
        print(cade[existing].to_string(index=False))
    else:
        print("Cade Cunningham not found in predictions.parquet!")

if __name__ == "__main__":
    find_cade()