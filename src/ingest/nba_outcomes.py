import os
import time
import io
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

START_YEAR = 2009
CURRENT_YEAR = datetime.now().year

def clean_advanced_table(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df[df["Player"] != "Player"].copy()
    df = df.dropna(subset=["Player"])
    df.columns = [col.strip() for col in df.columns]
    df["season"] = year
    
    desired_cols = ["Player", "season", "Age", "Tm", "G", "MP", "PER", "TS%", "3PAr", "FTr", "USG%", "WS/48", "BPM", "VORP"]
    existing_cols = [col for col in desired_cols if col in df.columns]
    df = df[existing_cols].copy()
    
    df = df.drop_duplicates(subset=["Player", "season"], keep="first")
    
    numeric_cols = ["Age", "G", "MP", "PER", "TS%", "3PAr", "FTr", "USG%", "WS/48", "BPM", "VORP"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    return df

def fetch_nba_outcomes(start_year: int, end_year: int):
    combined_file_path = os.path.join(RAW_DIR, "nba_outcomes.csv")
    if os.path.exists(combined_file_path):
        print(f"[CACHE] Loading master file from {combined_file_path}")
        return pd.read_csv(combined_file_path)

    all_seasons = []
    
    for year in range(start_year, end_year + 1):
        yearly_file_path = os.path.join(RAW_DIR, f"nba_advanced_{year}.csv")
        
        if os.path.exists(yearly_file_path):
            try:
                print(f"[CACHE] Reading {yearly_file_path}...")
                df_year = pd.read_csv(yearly_file_path)
                if df_year.empty:
                    print(f"  [BAD FILE] {yearly_file_path} is empty! Please delete or re-download it.")
                    continue
                all_seasons.append(df_year)
                continue
            except Exception as e:
                print(f"\n[CORRUPTED FILE DETECTED] Unable to read {yearly_file_path}: {e}")
                print(f"Please delete or re-download {yearly_file_path}\n")
                return

        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
        print(f"[FETCH] Requesting {year} NBA advanced stats...")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            cleaned_html = response.text.replace("<!--", "").replace("-->", "")
            
            if "Just a moment..." in cleaned_html or response.status_code in [403, 429]:
                print(f"[RATE LIMIT] Blocked by Cloudflare on year {year}.")
                break
                
            soup = BeautifulSoup(cleaned_html, "html.parser")
            table = soup.find("table", {"id": "advanced_stats"})
            
            if table is None:
                print(f"[WARNING] Table not found for {year}.")
                break
                
            df_year = pd.read_html(io.StringIO(str(table)))[0]
            df_year = clean_advanced_table(df_year, year)
            
            df_year.to_csv(yearly_file_path, index=False)
            all_seasons.append(df_year)
            
            time.sleep(5.5)
            
        except Exception as e:
            print(f"[ERROR] Failed on year {year}: {e}")
            break

    if all_seasons:
        combined_df = pd.concat(all_seasons, ignore_index=True)
        combined_df.to_csv(combined_file_path, index=False)
        print(f"\n[SUCCESS] Saved {len(combined_df)} rows to {combined_file_path}")

if __name__ == "__main__":
    fetch_nba_outcomes(START_YEAR, CURRENT_YEAR)