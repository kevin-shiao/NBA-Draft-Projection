import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from databricks import sql

load_dotenv()


def coalesce_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Permanently fills missing combine measurements with college fallbacks 
    so the frontend and model don't receive 0.0 for missing physicals.
    """
    df = df.copy()

    # 1. Fallback for Height
    if "height_wo_shoes_inches" in df.columns:
        fallback_series = df["height_wo_shoes_inches"]
        if "height_inches" in df.columns:
            fallback_series = fallback_series.fillna(df["height_inches"])
        if "height" in df.columns:
            fallback_series = fallback_series.fillna(df["height"])

        # If completely missing, default to 6'6" (78.0 inches)
        df["height_wo_shoes_inches"] = pd.to_numeric(fallback_series, errors="coerce").fillna(78.0)

    # 2. Fallback for Wingspan
    if "wingspan_inches" in df.columns and "height_wo_shoes_inches" in df.columns:
        # If wingspan is missing, assume a standard +1.0 inch ape index
        df["wingspan_inches"] = pd.to_numeric(df["wingspan_inches"], errors="coerce").fillna(
            df["height_wo_shoes_inches"] + 1.0
        )

    return df


def apply_height_gated_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dampens ape_index for undersized prospects (< 6'2" / 74 inches without shoes)
    using a cubic penalty (** 3) so short guards don't receive artificial SHAP feature 
    boosts intended for wings/bigs.
    """
    df = df.copy()

    height_col = None
    for possible_col in ["height_wo_shoes_inches", "height_inches", "height"]:
        if possible_col in df.columns:
            height_col = possible_col
            break

    if height_col and "ape_index" in df.columns:
        heights = pd.to_numeric(df[height_col], errors="coerce").fillna(78.0)
        ape_vals = pd.to_numeric(df["ape_index"], errors="coerce").fillna(0.0)

        # Baseline multiplier: 1.0 for prospects >= 74 inches, scaling down CUBICALLY (** 3) below 74 inches
        height_scale = (heights / 74.0).clip(upper=1.0) ** 3

        # Rescale ape_index proportionally to standing height
        df["ape_index"] = ape_vals * height_scale

    return df


def fix_positional_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overrides noisy college position scrapes with hard physical thresholds.
    >= 6'9" (81 inches) -> Big
    < 6'4" (76 inches) -> Guard
    """
    df = df.copy()
    height_col = None
    for possible_col in ["height_wo_shoes_inches", "height_inches", "height"]:
        if possible_col in df.columns:
            height_col = possible_col
            break

    if height_col and "pos_group" in df.columns:
        heights = pd.to_numeric(df[height_col], errors="coerce")
        df.loc[heights >= 81.0, "pos_group"] = "Big"
        df.loc[heights < 76.0, "pos_group"] = "Guard"

    return df


def apply_super_senior_dampener(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a cubic penalty to College BPM for older prospects (>21.0 years old) 
    to prevent physically mature seniors from tricking the model.
    """
    df = df.copy()
    if "age_at_draft" in df.columns and "college_bpm" in df.columns:
        ages = pd.to_numeric(df["age_at_draft"], errors="coerce").fillna(20.0)
        bpm = pd.to_numeric(df["college_bpm"], errors="coerce").fillna(0.0)

        # 1.0 for age <= 21.0, scales down cubically for ages > 21.0
        age_scale = (21.0 / np.maximum(ages, 21.0)) ** 3
        df["college_bpm"] = bpm * age_scale

    return df


def export_features_to_parquet():
    print("Connecting to Databricks...")
    connection = sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN"),
    )
    cursor = connection.cursor()

    print("[PROCESS] Querying nba_draft.analytics.prospect_features...")
    cursor.execute("SELECT * FROM nba_draft.analytics.prospect_features")
    rows = cursor.fetchall()

    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)

    # --- SET HISTORICAL TRAINING COHORT (2008-2019) ---
    if "draft_year" in df.columns:
        df["draft_year"] = pd.to_numeric(df["draft_year"], errors="coerce")
        df["is_training_cohort"] = df["draft_year"].between(2008, 2019)

    # --- APPLY DATA CORRECTIONS & TRANSFORMATIONS ---
    df = coalesce_measurements(df)
    df = apply_height_gated_adjustments(df)
    df = fix_positional_labels(df)
    df = apply_super_senior_dampener(df)

    # Create local directories if they don't exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    output_path = "data/processed/features.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  -> [SUCCESS] Exported {len(df)} rows and {len(df.columns)} columns to '{output_path}'")

    # Display training vs unlabeled split summary
    training_count = df[df["is_training_cohort"] == True].shape[0]
    unlabeled_count = df[df["is_training_cohort"] == False].shape[0]
    print(f"\n[SUMMARY] Data Split:")
    print(f"  -> Training Cohort (2008-2019): {training_count} players")
    print(f"  -> Unlabeled Set (2020+): {unlabeled_count} players")

    cursor.close()
    connection.close()


if __name__ == "__main__":
    export_features_to_parquet()