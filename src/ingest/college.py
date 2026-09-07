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

def fetch_torvik_data(start_year: int, end_year: int) -> pd.DataFrame:
    all_seasons = []

    for year in range(start_year, end_year + 1):
        file_path = os.path.join(RAW_DIR, f"college_{year}.csv")
        
        if os.path.exists(file_path):
            print(f"[CACHE] Loading {year} college stats from {file_path}")
            # Skip the first row if it's already a col_X header from a previous run
            first_val = pd.read_csv(file_path, nrows=1, header=None).iloc[0, 0]
            df = pd.read_csv(file_path, header=None, skiprows=1 if first_val == 'col_0' else 0)
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

        # Force safe col_X headers for Databricks compatibility
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
        df["season"] = year
        all_seasons.append(df)

    if not all_seasons:
        raise ValueError("No data was retrieved.")

    combined_df = pd.concat(all_seasons, ignore_index=True)
    combined_path = os.path.join(RAW_DIR, "college_all.csv")
    
    # Export with col_X headers included
    combined_df.to_csv(combined_path, index=False)
    print(f"\n[SUCCESS] Saved clean dataset with safe col_X HEADERS ({len(combined_df)} rows) to {combined_path}")
    
    return combined_df

if __name__ == "__main__":
    fetch_torvik_data(START_YEAR, CURRENT_YEAR)