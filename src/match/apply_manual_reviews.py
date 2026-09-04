import os
import pandas as pd
from dotenv import load_dotenv
from databricks import sql

load_dotenv()

def apply_reviews():
    review_path = "data/review/review_queue_corrected.csv"
    if not os.path.exists(review_path):
        print(f"[ERROR] File not found: {review_path}")
        return

    df = pd.read_csv(review_path)
    accepted_df = df[df['match_method'] == 'ACCEPTED'].copy()
    
    if accepted_df.empty:
        print("[INFO] No rows marked as 'ACCEPTED' to insert.")
        return

    print(f"[PROCESS] Found {len(accepted_df)} manually accepted matches. Merging into Databricks...")

    conn = sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )
    cursor = conn.cursor()

    rows = [
        f"('{str(row.draft_player_name).replace('\'', '\'\'')}', '{str(row.college_player_name).replace('\'', '\'\'')}', {row.draft_year}, '{str(row.college_team).replace('\'', '\'\'')}', {row.match_score}, 'MANUAL_ACCEPT')"
        for _, row in accepted_df.iterrows()
    ]

    cursor.execute("""
        CREATE OR REPLACE TEMPORARY VIEW review_updates AS
        SELECT * FROM VALUES 
    """ + ",".join(rows) + " AS t(draft_player_name, college_player_name, draft_year, college_team, match_score, match_method)")

    cursor.execute("""
        MERGE INTO nba_draft.staging.player_crosswalk AS target
        USING review_updates AS source
        ON target.draft_player_name = source.draft_player_name 
       AND target.draft_year = source.draft_year
        WHEN NOT MATCHED THEN
            INSERT (draft_player_name, college_player_name, draft_year, college_team, match_score, match_method)
            VALUES (source.draft_player_name, source.college_player_name, source.draft_year, source.college_team, source.match_score, source.match_method)
    """)

    cursor.close()
    conn.close()
    print("  -> [SUCCESS] Manually reviewed matches merged into player_crosswalk!")

if __name__ == "__main__":
    apply_reviews()