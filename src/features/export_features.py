import os
import sys
import pandas as pd
from dotenv import load_dotenv
from databricks import sql

load_dotenv()

def export_features_to_parquet():
    print("Connecting to Databricks...")
    connection = sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )
    cursor = connection.cursor()

    print("[PROCESS] Querying nba_draft.analytics.prospect_features...")
    cursor.execute("SELECT * FROM nba_draft.analytics.prospect_features")
    rows = cursor.fetchall()
    
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    
    # Create local directories if they don't exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    output_path = "data/processed/features.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  -> [SUCCESS] Exported {len(df)} rows and {len(df.columns)} columns to '{output_path}'")

    # Display training vs unlabeled split summary
    training_count = df[df['is_training_cohort'] == True].shape[0]
    unlabeled_count = df[df['is_training_cohort'] == False].shape[0]
    print(f"\n[SUMMARY] Data Split:")
    print(f"  -> Training Cohort (2009-2019): {training_count} players")
    print(f"  -> Unlabeled Set (2020+): {unlabeled_count} players")

    cursor.close()
    connection.close()

if __name__ == "__main__":
    export_features_to_parquet()