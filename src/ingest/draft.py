import os
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import drafthistory

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

START_YEAR = 2008
CURRENT_YEAR = datetime.now().year

def fetch_draft_history(start_year: int, end_year: int) -> pd.DataFrame:
    file_path = os.path.join(RAW_DIR, "draft_history.csv")
    
    if os.path.exists(file_path):
        print(f"[CACHE] Loading draft history from {file_path}")
        return pd.read_csv(file_path)

    print("[FETCH] Fetching draft history via nba_api...")
    try:
        endpoint = drafthistory.DraftHistory()
        df = endpoint.get_data_frames()[0]
        
        df.columns = [col.lower() for col in df.columns]
        
        df["season"] = df["season"].astype(int)
        df = df[(df["season"] >= start_year) & (df["season"] <= end_year)].copy()
        
        target_cols = [
            "person_id", 
            "player_name", 
            "season", 
            "round_number", 
            "overall_pick", 
            "organization"
        ]
        
        df = df[[col for col in target_cols if col in df.columns]]

        df.to_csv(file_path, index=False)
        print(f"[SUCCESS] Saved draft history ({len(df)} rows) to {file_path}")
        return df

    except Exception as e:
        print(f"[ERROR] Failed to fetch draft history: {e}")
        raise e

if __name__ == "__main__":
    fetch_draft_history(START_YEAR, CURRENT_YEAR)