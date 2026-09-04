import os
import pandas as pd
from dotenv import load_dotenv
from databricks import sql

load_dotenv()

def check_feature_values():
    print("Connecting to Databricks...\n")
    connection = sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )
    cursor = connection.cursor()

    query = """
        SELECT 
            draft_player_name AS Player,
            draft_year AS Year,
            seasons_played_in_college AS Seasons,
            ROUND(college_bpm, 1) AS BPM,
            ROUND(bpm_delta, 1) AS BPM_Delta,
            ROUND(pts_per_40, 1) AS PTS_40,
            ROUND(ts_pct, 3) AS TS_Pct,
            ROUND(age_at_draft, 2) AS Age,
            ROUND(ape_index, 1) AS Ape_Index,
            height_was_missing AS Missing_Combine
        FROM nba_draft.analytics.prospect_features
        WHERE overall_pick = 1 AND draft_year BETWEEN 2009 AND 2019
        ORDER BY draft_year DESC
        LIMIT 5
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    if rows:
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        print("=== FEATURE VERIFICATION: RECENT #1 PICKS ===")
        print(df.to_string(index=False))
    else:
        print("No data found.")

    cursor.close()
    connection.close()

if __name__ == "__main__":
    check_feature_values()