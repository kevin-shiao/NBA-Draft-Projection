import os
import time
import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

def fetch_birthdates():
    output_file = os.path.join(RAW_DIR, "player_birthdates.csv")
    
    if os.path.exists(output_file):
        print(f"[CACHE] Loading player birthdates from {output_file}")
        return pd.read_csv(output_file)

    draft_file = os.path.join(RAW_DIR, "draft_history.csv")
    if not os.path.exists(draft_file):
        raise FileNotFoundError(f"Missing {draft_file}. Please run src/ingest/draft.py first.")

    draft_df = pd.read_csv(draft_file)
    player_ids = draft_df["person_id"].dropna().unique()

    birthdates = []
    print(f"[FETCH] Fetching birthdate info for {len(player_ids)} drafted players...")

    for i, person_id in enumerate(player_ids, start=1):
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=int(person_id))
            df_info = info.get_data_frames()[0]
            
            if not df_info.empty:
                df_info.columns = [col.lower() for col in df_info.columns]
                
                player_data = {
                    "person_id": person_id,
                    "display_first_last": df_info.get("display_first_last", [None])[0],
                    "birthdate": df_info.get("birthdate", [None])[0]
                }
                birthdates.append(player_data)
                
            # Sleep 0.6s to respect NBA API rate limits
            time.sleep(0.6)

            if i % 50 == 0:
                print(f"  Processed {i}/{len(player_ids)} players...")

        except Exception as e:
            print(f"[WARNING] Could not fetch info for player ID {person_id}: {e}")
            continue

    result_df = pd.DataFrame(birthdates)
    result_df.to_csv(output_file, index=False)
    print(f"\n[SUCCESS] Saved birthdates for {len(result_df)} players to {output_file}")
    
    return result_df

if __name__ == "__main__":
    fetch_birthdates()