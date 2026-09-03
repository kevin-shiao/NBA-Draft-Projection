import os
from dotenv import load_dotenv
from databricks import sql

# Load credentials from .env
load_dotenv()

SERVER_HOSTNAME = os.getenv("DATABRICKS_SERVER_HOSTNAME")
HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
ACCESS_TOKEN = os.getenv("DATABRICKS_TOKEN")

def load_raw_tables():
    print("Connecting to Databricks SQL Warehouse...")
    
    try:
        connection = sql.connect(
            server_hostname=SERVER_HOSTNAME,
            http_path=HTTP_PATH,
            access_token=ACCESS_TOKEN
        )
        cursor = connection.cursor()
    except Exception as e:
        print(f"[ERROR] Connection failed. Check your .env file: {e}")
        return

    # Map target raw table name to file in landing volume
    raw_tables = {
        "college": "college_all.csv",
        "draft_history": "draft_history.csv",
        "combine": "combine_stats.csv",
        "nba_outcomes": "nba_outcomes.csv",
        "birthdates": "player_birthdates.csv"
    }

    for table_name, file_name in raw_tables.items():
        print(f"[PROCESS] Creating raw table: nba_draft.raw.{table_name} from {file_name}...")
        
        # Load raw files directly into nba_draft.raw schema
        query = f"""
        CREATE OR REPLACE TABLE nba_draft.raw.{table_name}
        AS SELECT * FROM read_files(
            '/Volumes/nba_draft/raw/landing/{file_name}',
            format => 'csv',
            header => true,
            inferSchema => true
        )
        """
        
        try:
            cursor.execute(query)
            print(f"  -> [SUCCESS] nba_draft.raw.{table_name} created.")
        except Exception as e:
            print(f"  -> [ERROR] Failed to create nba_draft.raw.{table_name}: {e}")

    cursor.close()
    connection.close()
    print("\n[FINISHED] All raw tables successfully created in nba_draft.raw.")

if __name__ == "__main__":
    load_raw_tables()