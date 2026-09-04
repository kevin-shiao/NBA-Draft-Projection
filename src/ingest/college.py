import os
import time
import requests
import pandas as pd
from datetime import datetime

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

START_YEAR = 2008
CURRENT_YEAR = datetime.now().year

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Explicit Column Mapping for getadvstats.php (65 columns)
PLAYER_ADV_COLUMNS = [
    "player_name", "team", "conf", "games_played", "min_pct", "ortg", "usage_pct",
    "efg_pct", "ts_pct", "oreb_pct", "dreb_pct", "ast_pct", "turnover_pct",
    "ftr", "ft_pct", "two_pt_made", "two_pt_att", "two_pt_pct",
    "three_pt_made", "three_pt_att", "three_pt_pct", "blk_pct", "stl_pct",
    "ft_made", "ft_att", "rec_rank", "porpag", "adj_oe", "per", "year_in_school",
    "height_raw", "num", "dporpag", "stops", "bpm", "obpm", "dbpm",
    "gbpm", "mp_per_game", "ogbpm", "dgbpm", "oreb_per_game", "dreb_per_game",
    "reb_per_game", "ast_per_game", "stl_per_game", "blk_per_game", "pts_per_game",
    "pos_possession", "bpm_pure", "ast_to_ratio", "two_pm_per_game", "two_pa_per_game",
    "three_pm_per_game", "three_pa_per_game", "ftm_per_game", "fta_per_game",
    "pts_raw", "reb_raw", "ast_raw", "stl_raw", "blk_raw", "games_started",
    "birthdate", "player_id"
]

def fetch_torvik_data(start_year: int, end_year: int) -> pd.DataFrame:
    all_seasons = []

    for year in range(start_year, end_year + 1):
        file_path = os.path.join(RAW_DIR, f"college_{year}.csv")
        
        if os.path.exists(file_path):
            print(f"[CACHE] Loading {year} college stats from {file_path}")
            df = pd.read_csv(file_path, header=None)
        else:
            url = f"https://barttorvik.com/getadvstats.php?year={year}&csv=1"
            print(f"[FETCH] Requesting {year} college stats from Bart Torvik...")
            
            try:
                response = requests.get(url, headers=HEADERS, timeout=30)
                response.raise_for_status()
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                
                df = pd.read_csv(file_path, header=None)
                time.sleep(1.5)
                
            except Exception as e:
                print(f"[ERROR] Failed to fetch data for year {year}: {e}")
                continue

        # If columns match player schema, apply column names directly
        if df.shape[1] == len(PLAYER_ADV_COLUMNS):
            df.columns = PLAYER_ADV_COLUMNS
        else:
            # Fallback for historical seasons with missing fields
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

        df["season"] = year
        all_seasons.append(df)

    if not all_seasons:
        raise ValueError("No data was retrieved or loaded.")

    combined_df = pd.concat(all_seasons, ignore_index=True)
    combined_path = os.path.join(RAW_DIR, "college_all.csv")
    
    # Export with full header row included
    combined_df.to_csv(combined_path, index=False)
    print(f"\n[SUCCESS] Saved combined dataset with HEADERS ({len(combined_df)} rows) to {combined_path}")
    
    return combined_df

if __name__ == "__main__":
    fetch_torvik_data(START_YEAR, CURRENT_YEAR)