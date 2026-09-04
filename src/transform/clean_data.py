import os
from dotenv import load_dotenv
from databricks import sql

load_dotenv()

def get_clean_name_sql(col_name):
    """SQL snippet to normalize names: lowercase, remove punctuation, strip suffixes."""
    return f"""
        TRIM(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    REGEXP_REPLACE(LOWER(`{col_name}`), '[.,''-]', ''), 
                    '( jr| sr| ii| iii| iv)$', ''
                ),
                '\\s+', ' '
            )
        )
    """

def run_entity_resolution():
    print("Connecting to Databricks...")
    connection = sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )
    cursor = connection.cursor()

    print("[PROCESS] Ensuring staging schema exists...")
    cursor.execute("CREATE SCHEMA IF NOT EXISTS nba_draft.staging;")

    table_player_cols = {
        "draft_history": "player_name",
        "college": "0",
        "combine": "player_name",
        "nba_outcomes": "Player",
        "birthdates": "display_first_last"
    }

    for table, player_col in table_player_cols.items():
        print(f"\n[PROCESS] Staging table: nba_draft.staging.{table} using column `{player_col}`...")
        
        try:
            clean_sql = get_clean_name_sql(player_col)
            
            select_clause = "*,"
            if table == "college":
                select_clause = "*, `0` AS player_name,"

            query = f"""
            CREATE OR REPLACE TABLE nba_draft.staging.{table} AS
            SELECT 
                {select_clause}
                {clean_sql} AS join_name
            FROM nba_draft.raw.{table}
            """
            
            cursor.execute(query)
            print(f"  -> [SUCCESS] nba_draft.staging.{table} created.")
        except Exception as e:
            print(f"  -> [ERROR] Failed to stage {table}: {e}")

    cursor.close()
    connection.close()
    print("\n[FINISHED] Staging and entity resolution name normalization complete.")

if __name__ == "__main__":
    run_entity_resolution()