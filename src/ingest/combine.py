import os
import time
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import draftcombinestats

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

START_YEAR = 2008
CURRENT_YEAR = datetime.now().year

def fetch_combine_stats(start_year: int, end_year: int) -> pd.DataFrame:
    combined_file_path = os.path.join(RAW_DIR, "combine_stats.csv")
    
    if os.path.exists(combined_file_path):
        print(f"[CACHE] Loading combine measurements from {combined_file_path}")
        return pd.read_csv(combined_file_path)

    all_combine_data = []

    for year in range(start_year, end_year + 1):
        yearly_file_path = os.path.join(RAW_DIR, f"combine_{year}.csv")
        
        if os.path.exists(yearly_file_path):
            print(f"[CACHE] Loading {year} combine stats from {yearly_file_path}")
            df_year = pd.read_csv(yearly_file_path)
        else:
            print(f"[FETCH] Requesting {year} combine stats from nba_api...")
            try:
                season_str = f"{year-1}-{str(year)[-2:]}"
                
                try:
                    endpoint = draftcombinestats.DraftCombineStats(season_all_time=season_str)
                except Exception:
                    endpoint = draftcombinestats.DraftCombineStats(season_all_time=str(year))

                df_year = endpoint.get_data_frames()[0]
                
                df_year.columns = [col.lower() for col in df_year.columns]
                df_year["season"] = year
                
                df_year.to_csv(yearly_file_path, index=False)
                
                time.sleep(1.5)
                
            except Exception as e:
                print(f"[WARNING] Could not fetch combine data for year {year}: {e}")
                continue

        if not df_year.empty:
            all_combine_data.append(df_year)

    if not all_combine_data:
        print("[WARNING] No combine data was fetched or loaded.")
        return pd.DataFrame()

    combined_df = pd.concat(all_combine_data, ignore_index=True)
    combined_df.to_csv(combined_file_path, index=False)
    print(f"\n[SUCCESS] Saved combine measurements ({len(combined_df)} rows) to {combined_file_path}")
    
    return combined_df

if __name__ == "__main__":
    fetch_combine_stats(START_YEAR, CURRENT_YEAR)