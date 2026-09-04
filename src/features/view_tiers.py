import os
import pandas as pd
from dotenv import load_dotenv
from databricks import sql

load_dotenv()

def view_players_by_tier():
    print("Connecting to Databricks...\n")
    connection = sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )
    cursor = connection.cursor()

    tiers = ['All-Star', 'Starter', 'Rotation', 'Bust']
    
    for tier in tiers:
        print(f"=== TOP 10 PLAYERS IN '{tier.upper()}' TIER ===")
        query = f"""
            SELECT draft_player_name, draft_year, ROUND(vorp_5y, 2) AS vorp_5y
            FROM nba_draft.analytics.draft_targets
            WHERE is_training_cohort = TRUE AND player_tier_5y = '{tier}'
            ORDER BY vorp_5y DESC
            LIMIT 10
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if rows:
            df = pd.DataFrame(rows, columns=['Player', 'Draft Year', '5-Year VORP'])
            print(df.to_string(index=False))
        else:
            print("No players found in this tier.")
        print("\n")

    cursor.close()
    connection.close()

if __name__ == "__main__":
    view_players_by_tier()