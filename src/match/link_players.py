import os
import re
import pandas as pd
from dotenv import load_dotenv
from unidecode import unidecode
from rapidfuzz import fuzz
from databricks import sql

load_dotenv()

def normalize_name(name):
    if not name or pd.isna(name):
        return ""
    # Strip accents
    name = unidecode(str(name))
    # Lowercase
    name = name.lower()
    # Strip suffixes
    name = re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', name)
    # Strip punctuation and extra whitespace
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def fetch_dataframe(cursor, query):
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    return pd.DataFrame(data, columns=columns)

def generate_player_crosswalk():
    print("Connecting to Databricks...")
    conn = sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )
    cursor = conn.cursor()

    print("[FETCH] Pulling Draft History...")
    draft_df = fetch_dataframe(cursor, """
        SELECT 
            player_name AS draft_name,
            CAST(season AS INT) AS draft_year,
            overall_pick
        FROM nba_draft.staging.draft_history
    """)

    print("[FETCH] Pulling College Data...")
    college_df = fetch_dataframe(cursor, """
        SELECT DISTINCT
            `0` AS college_name,
            `1` AS college_team,
            CAST(season AS INT) AS college_season
        FROM nba_draft.raw.college
    """)

    print(f"[PROCESS] Normalizing names across {len(draft_df)} draft records and {len(college_df)} college records...")
    draft_df['norm_draft_name'] = draft_df['draft_name'].apply(normalize_name)
    college_df['norm_college_name'] = college_df['college_name'].apply(normalize_name)

    matched_records = []
    manual_review_queue = []

    print("[MATCHING] Running Blocking + Fuzzy Scoring...")
    
    for _, d_row in draft_df.iterrows():
        d_name = d_row['norm_draft_name']
        d_year = d_row['draft_year']
        
        candidates = college_df[
            (college_df['college_season'] == d_year) | 
            (college_df['college_season'] == d_year - 1)
        ]

        best_score = 0
        best_match = None

        for _, c_row in candidates.iterrows():
            c_name = c_row['norm_college_name']
            
            score = fuzz.token_sort_ratio(d_name, c_name)
            
            if score > best_score:
                best_score = score
                best_match = c_row

        if best_match is not None:
            if best_score >= 92:
                matched_records.append({
                    'draft_player_name': d_row['draft_name'],
                    'college_player_name': best_match['college_name'],
                    'draft_year': d_year,
                    'college_team': best_match['college_team'],
                    'match_score': float(best_score),
                    'match_method': 'AUTO_ACCEPT'
                })
            elif 80 <= best_score < 92:
                manual_review_queue.append({
                    'draft_player_name': d_row['draft_name'],
                    'college_player_name': best_match['college_name'],
                    'draft_year': d_year,
                    'college_team': best_match['college_team'],
                    'match_score': float(best_score),
                    'match_method': 'MANUAL_REVIEW_NEEDED'
                })

    crosswalk_df = pd.DataFrame(matched_records)
    review_df = pd.DataFrame(manual_review_queue)

    if not review_df.empty:
        os.makedirs("data/review", exist_ok=True)
        review_df.to_csv("data/review/review_queue.csv", index=False)
        print(f"  -> [QUEUE] Saved {len(review_df)} rows to data/review/review_queue.csv for manual inspection.")

    print(f"\n[SUMMARY]")
    print(f"  -> Total Drafted Players Processed: {len(draft_df)}")
    if len(draft_df) > 0:
        match_pct = round(len(crosswalk_df) / len(draft_df) * 100, 2)
        print(f"  -> Auto-Matched (Score >= 92): {len(crosswalk_df)} ({match_pct}%)")
    print(f"  -> Needs Review (80-91): {len(review_df)}")

    print("\n[WRITE] Writing permanent nba_draft.staging.player_crosswalk table...")
    
    cursor.execute("""
        CREATE OR REPLACE TABLE nba_draft.staging.player_crosswalk (
            draft_player_name STRING,
            college_player_name STRING,
            draft_year INT,
            college_team STRING,
            match_score DOUBLE,
            match_method STRING
        )
    """)

    if not crosswalk_df.empty:
        rows_to_insert = [
            f"('{str(row.draft_player_name).replace('\'', '\'\'')}', '{str(row.college_player_name).replace('\'', '\'\'')}', {row.draft_year}, '{str(row.college_team).replace('\'', '\'\'')}', {row.match_score}, '{row.match_method}')"
            for _, row in crosswalk_df.iterrows()
        ]
        
        chunk_size = 500
        for i in range(0, len(rows_to_insert), chunk_size):
            chunk = rows_to_insert[i:i + chunk_size]
            insert_sql = f"INSERT INTO nba_draft.staging.player_crosswalk VALUES {','.join(chunk)}"
            cursor.execute(insert_sql)

    cursor.close()
    conn.close()
    print("  -> [SUCCESS] nba_draft.staging.player_crosswalk persisted successfully!")

if __name__ == "__main__":
    generate_player_crosswalk()