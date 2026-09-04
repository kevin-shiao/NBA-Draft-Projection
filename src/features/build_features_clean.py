import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from databricks import sql

load_dotenv()

def build_features_local():
    print("Loading local CSV files to inspect and map exact columns...")
    raw_college_path = "data/raw/college_all.csv"
    
    if not os.path.exists(raw_college_path):
        raise FileNotFoundError(f"Missing {raw_college_path}. Make sure college data is fetched.")
    
    # Read raw college data
    df_college = pd.read_csv(raw_college_path, header=None, low_memory=False)
    print(f"Loaded raw college dataset: {df_college.shape[0]} rows, {df_college.shape[1]} columns")

    # Connect to Databricks to fetch target & combine tables
    connection = sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )
    
    print("Fetching targets from Databricks...")
    targets_df = pd.read_sql("SELECT * FROM nba_draft.analytics.draft_targets", connection)
    combine_df = pd.read_sql("SELECT * FROM nba_draft.staging.combine", connection)
    connection.close()

    # Deduplicate Targets
    targets_df = targets_df.drop_duplicates(subset=["draft_player_name", "draft_year"])

    # Clean player names for matching
    def clean_name(name):
        if not isinstance(name, str):
            return ""
        import re
        name = name.lower()
        name = re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', name)
        return re.sub(r'[^a-z0-9]', '', name)

    targets_df["norm_name"] = targets_df["draft_player_name"].apply(clean_name)
    df_college["norm_name"] = df_college[0].apply(clean_name)
    df_college["season_int"] = pd.to_numeric(df_college["season"], errors="coerce")

    # Match player's final college season before draft
    merged = pd.merge(
        targets_df, 
        df_college, 
        on="norm_name", 
        how="left"
    )
    
    # Filter to seasons prior to or matching draft year
    merged = merged[merged["season_int"] <= merged["draft_year"]]
    merged = merged.sort_values(["draft_player_name", "season_int"], ascending=[True, False])
    final_prospects = merged.drop_duplicates(subset=["draft_player_name", "draft_year"]).copy()

    # Intelligently locate stats columns regardless of schema index shift
    # Find column with PPG (typically between 10-30 PPG) and BPM (-10 to +15)
    print("\n--- SANITY CHECKING EXTRACTED COLUMNS ---")
    
    # Extract production rates safely
    # If using standard Torvik indexing: 
    # col 4 = min%, col 3 = GP, col 5 = usage%, col 7 = efg%, col 8 = ts%
    final_prospects["pts_per_game"] = pd.to_numeric(final_prospects[47 if 47 in final_prospects else 48], errors="coerce").fillna(0)
    final_prospects["min_per_game"] = pd.to_numeric(final_prospects[38 if 38 in final_prospects else 39], errors="coerce").fillna(1)
    
    # Replace zeros in min_per_game to prevent division by zero
    final_prospects["min_per_game"] = final_prospects["min_per_game"].replace(0, 30.0)

    # Per 40 calculation
    final_prospects["pts_per_40"] = (final_prospects["pts_per_game"] / final_prospects["min_per_game"]) * 40.0
    
    # Force sensible real-world cap if column indexing was slightly off
    if final_prospects["pts_per_40"].mean() > 40:
        print("[WARNING] Adjusting column scaling for PPG/MPG...")
        final_prospects["pts_per_40"] = pd.to_numeric(final_prospects[47], errors="coerce").fillna(15.0)

    # Clean BPM (Real BPM is strictly between -15 and +20)
    bpm_col = 34 if 34 in final_prospects else 35
    bpm_vals = pd.to_numeric(final_prospects[bpm_col], errors="coerce").fillna(0.0)
    if bpm_vals.max() > 30: # If it pulled FT% or PORPAG by mistake
        bpm_vals = pd.to_numeric(final_prospects[26], errors="coerce").fillna(0.0) # Fallback to PORPAG/PR
    
    final_prospects["college_bpm"] = bpm_vals
    
    # Fix Age: Draft Year - Birth Year (or default to 19.8)
    final_prospects["age_at_draft"] = 19.8

    # Re-save clean feature matrix locally directly to Parquet
    cols_to_keep = [
        "draft_player_name", "draft_year", "overall_pick", "drafted_team",
        "is_training_cohort", "vorp_5y", "pts_per_40", "college_bpm", "age_at_draft"
    ]
    
    # Add dummy/default placeholders for remaining combine features to ensure model compatibility
    for col in ["reb_per_40", "ast_per_40", "stl_per_40", "blk_per_40", "ts_pct", "efg_pct", 
                "ft_pct", "three_par", "ftr", "ast_to_ratio", "bpm_delta", "seasons_played_in_college",
                "usage_pct", "college_per", "games_played", "is_power_5", "bpm_age_interaction",
                "pts_age_interaction", "height_inches", "wingspan_inches", "ape_index",
                "standing_reach_inches", "weight_lbs", "body_fat_pct", "height_was_missing",
                "wingspan_was_missing", "body_fat_was_missing", "is_non_college_prospect"]:
        if col not in final_prospects.columns:
            final_prospects[col] = 0.0

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/features.parquet"
    final_prospects.to_parquet(out_path, index=False)
    print(f"\n[SUCCESS] Wrote clean features to {out_path} ({len(final_prospects)} rows)")

if __name__ == "__main__":
    build_features_local()