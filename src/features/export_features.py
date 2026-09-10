import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from dotenv import load_dotenv
from databricks import sql

load_dotenv()


def parse_height(h):
    """
    Parses height values, handling string formats like '6-9' or '6-9.5'
    and numeric floats/ints cleanly into total inches.
    """
    if pd.isna(h):
        return np.nan
    try:
        if isinstance(h, str) and '-' in h:
            parts = h.split('-')
            ft = float(parts[0])
            inc = float(parts[1])
            return ft * 12.0 + inc
        return float(h)
    except Exception:
        return np.nan


def e_bayes_shrinkage(made, att, k):
    """
    Applies Empirical Bayes shrinkage to stabilize low-sample shooting rates:
    (made + k * prior) / (att + k)
    """
    total_att = att.sum()
    prior = (made.sum() / total_att) if total_att > 0 else 0.0
    return (made + k * prior) / (att + k)


def process_position_and_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Height Parsing & Missingness Reporting
    raw_height_col = None
    for col in ["height", "height_inches", "height_wo_shoes_inches", "ht"]:
        if col in df.columns:
            raw_height_col = col
            break

    if raw_height_col:
        nan_before = df[raw_height_col].isna().sum()
        df["height_in"] = df[raw_height_col].apply(parse_height)
        nan_after = df["height_in"].isna().sum()
        print(f"  -> [HEIGHT DIAGNOSTIC] Height missingness dropped from {nan_before} to {nan_after} using parse_height().")
    else:
        df["height_in"] = 78.0

    # Fill remaining missing height defaults to 6'6" (78 inches)
    df["height_in"] = df["height_in"].fillna(78.0)

    # 2. Three Rate Calculation: TPA / (TPA + twoPA)
    if "TPA" in df.columns and "twoPA" in df.columns:
        tpa = pd.to_numeric(df["TPA"], errors="coerce").fillna(0.0)
        twopa = pd.to_numeric(df["twoPA"], errors="coerce").fillna(0.0)
        df["three_rate"] = np.where((tpa + twopa) > 0, tpa / (tpa + twopa), 0.0)
    elif "fg3a_per_40" in df.columns and "fg2a_per_40" in df.columns:
        fg3a = pd.to_numeric(df["fg3a_per_40"], errors="coerce").fillna(0.0)
        fg2a = pd.to_numeric(df["fg2a_per_40"], errors="coerce").fillna(0.0)
        df["three_rate"] = np.where((fg3a + fg2a) > 0, fg3a / (fg3a + fg2a), 0.0)
    else:
        df["three_rate"] = 0.0

    # Season mapping for within-season Z-score scaling
    season_col = "draft_year" if "draft_year" in df.columns else ("season" if "season" in df.columns else None)

    # Map target statistical columns
    stats_map = {
        "height_in": "height_in",
        "blk_per": "blk_per_40" if "blk_per_40" in df.columns else ("blk_per" if "blk_per" in df.columns else None),
        "ORB_per": "orb_per_40" if "orb_per_40" in df.columns else ("ORB_per" if "ORB_per" in df.columns else None),
        "DRB_per": "drb_per_40" if "drb_per_40" in df.columns else ("DRB_per" if "DRB_per" in df.columns else None),
        "three_rate": "three_rate",
        "AST_per": "ast_per_40" if "ast_per_40" in df.columns else ("AST_per" if "AST_per" in df.columns else None),
        "stl_per": "stl_per_40" if "stl_per_40" in df.columns else ("stl_per" if "stl_per" in df.columns else None),
    }

    # Within-season Z-score Standardization
    for key, col in stats_map.items():
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            if season_col:
                df[f"{key}_z"] = df.groupby(season_col)[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-6)
                ).fillna(0.0)
            else:
                df[f"{key}_z"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)
        else:
            df[f"{key}_z"] = 0.0

    # Archetype Scoring Weights
    df["score_Big"] = (
        1.1 * df["height_in_z"]
        + 1.0 * df["blk_per_z"]
        + 0.9 * df["ORB_per_z"]
        + 0.5 * df["DRB_per_z"]
        - 0.8 * df["three_rate_z"]
        - 0.6 * df["AST_per_z"]
    )

    df["score_Wing"] = (
        0.3 * df["height_in_z"]
        + 0.3 * df["three_rate_z"]
        + 0.2 * df["DRB_per_z"]
        - 0.2 * df["AST_per_z"]
        - 0.2 * df["ORB_per_z"]
        - 0.1 * df["blk_per_z"]
    )

    df["score_Guard"] = (
        1.1 * df["AST_per_z"]
        + 0.6 * df["three_rate_z"]
        + 0.4 * df["stl_per_z"]
        - 0.5 * df["height_in_z"]
        - 0.7 * df["ORB_per_z"]
        - 0.5 * df["blk_per_z"]
    )

    # Argmax for UI filter bucket
    df["pos_bucket"] = (
        df[["score_Big", "score_Wing", "score_Guard"]]
        .idxmax(axis=1)
        .str.replace("score_", "")
    )

    return df


def process_feature_fixes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Recruiting Rank (explicit unranked imputation = 500)
    rec_col = None
    for col in ["Rec Rank", "rec_rank", "recruiting_rank", "rank"]:
        if col in df.columns:
            rec_col = col
            break

    if rec_col:
        df["rec_rank_clean"] = pd.to_numeric(df[rec_col], errors="coerce").fillna(500.0)
    else:
        df["rec_rank_clean"] = 500.0
    
    df["rec_rank_log"] = np.log1p(df["rec_rank_clean"])

    # 2. Empirical Bayes Shrinkage
    if "TP_made" in df.columns and "TPA" in df.columns:
        made_3p = pd.to_numeric(df["TP_made"], errors="coerce").fillna(0.0)
        att_3p = pd.to_numeric(df["TPA"], errors="coerce").fillna(0.0)
        df["TP_per_shrunk"] = e_bayes_shrinkage(made_3p, att_3p, k=100)
    elif "fg3m_per_40" in df.columns and "fg3a_per_40" in df.columns:
        made_3p = pd.to_numeric(df["fg3m_per_40"], errors="coerce").fillna(0.0)
        att_3p = pd.to_numeric(df["fg3a_per_40"], errors="coerce").fillna(0.0)
        df["TP_per_shrunk"] = e_bayes_shrinkage(made_3p, att_3p, k=100)
    elif "fg3_pct" in df.columns:
        df["TP_per_shrunk"] = pd.to_numeric(df["fg3_pct"], errors="coerce").fillna(0.0)
    else:
        df["TP_per_shrunk"] = 0.0

    if "FT_made" in df.columns and "FTA" in df.columns:
        made_ft = pd.to_numeric(df["FT_made"], errors="coerce").fillna(0.0)
        att_ft = pd.to_numeric(df["FTA"], errors="coerce").fillna(0.0)
        df["FT_per_shrunk"] = e_bayes_shrinkage(made_ft, att_ft, k=250)
    elif "ftm_per_40" in df.columns and "fta_per_40" in df.columns:
        made_ft = pd.to_numeric(df["ftm_per_40"], errors="coerce").fillna(0.0)
        att_ft = pd.to_numeric(df["fta_per_40"], errors="coerce").fillna(0.0)
        df["FT_per_shrunk"] = e_bayes_shrinkage(made_ft, att_ft, k=250)
    elif "ft_pct" in df.columns:
        df["FT_per_shrunk"] = pd.to_numeric(df["ft_pct"], errors="coerce").fillna(0.0)
    else:
        df["FT_per_shrunk"] = 0.0

    # 3. Touch Divergence
    df["touch_divergence"] = df["FT_per_shrunk"] - df["TP_per_shrunk"]

    # 4. Residualize Turnovers on Usage
    to_col = "TO_per" if "TO_per" in df.columns else ("tov_per_40" if "tov_per_40" in df.columns else None)
    usg_col = "usg" if "usg" in df.columns else ("usage_pct" if "usage_pct" in df.columns else None)

    if to_col and usg_col:
        valid_mask = df[to_col].notna() & df[usg_col].notna()
        if valid_mask.sum() > 10:
            to_model = smf.ols(f"{to_col} ~ {usg_col}", data=df[valid_mask]).fit()
            df["TO_res"] = 0.0
            df.loc[valid_mask, "TO_res"] = to_model.resid
        else:
            df["TO_res"] = 0.0
    else:
        df["TO_res"] = 0.0

    # 5. Base Interaction Terms
    bpm_col = "college_bpm" if "college_bpm" in df.columns else ("bpm" if "bpm" in df.columns else None)
    age_col = "age_at_draft" if "age_at_draft" in df.columns else ("age" if "age" in df.columns else None)
    ts_col = "ts_pct" if "ts_pct" in df.columns else ("ts" if "ts" in df.columns else None)

    age_val = pd.to_numeric(df[age_col], errors="coerce").fillna(20.0) if age_col else 20.0
    bpm_val = pd.to_numeric(df[bpm_col], errors="coerce").fillna(0.0) if bpm_col else 0.0
    usg_val = pd.to_numeric(df[usg_col], errors="coerce").fillna(20.0) if usg_col else 20.0
    ts_val = pd.to_numeric(df[ts_col], errors="coerce").fillna(0.5) if ts_col else 0.5

    df["age_x_prod"] = age_val * bpm_val
    df["usg_x_eff"] = usg_val * ts_val

    # 6. Apply Strength of Schedule (SOS) Scaling
    if bpm_col and "sos" in df.columns:
        df["sos_numeric"] = pd.to_numeric(df["sos"], errors="coerce").fillna(0.0)
        df["sos_z"] = (df["sos_numeric"] - df["sos_numeric"].mean()) / (df["sos_numeric"].std() + 1e-6)
        df["bpm_sos_adj"] = bpm_val * (1.0 + (df["sos_z"] * 0.10))
    else:
        df["bpm_sos_adj"] = bpm_val

    # 7. PURE ML CONTINUOUS INTERACTIONS (No Hard Rules)
    # Height Z-score acts as a multiplier: tall players get positive values, short players get negative values.
    # Shooting & Assist stats are multiplied by this continuous physical score.
    df["ast_height_multiplier"] = df["AST_per_z"] * df["height_in_z"]
    df["stretch_factor"] = df["height_in_z"] * df["TP_per_shrunk"]
    
    mean_ts = df["ts_pct"].mean() if "ts_pct" in df.columns else 0.53
    df["usage_efficiency_burden"] = usg_val * (ts_val - mean_ts)

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

    if "draft_year" in df.columns:
        df["draft_year"] = pd.to_numeric(df["draft_year"], errors="coerce")
        df["is_training_cohort"] = df["draft_year"].between(2008, 2019)

    print("[PROCESS] Applying position archetypes and parsing heights...")
    df = process_position_and_archetypes(df)

    print("[PROCESS] Calculating shrinkage, turnover residualization, and interactions...")
    df = process_feature_fixes(df)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    output_path = "data/processed/features.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  -> [SUCCESS] Exported {len(df)} rows and {len(df.columns)} columns to '{output_path}'")

    training_count = df[df["is_training_cohort"] == True].shape[0]
    unlabeled_count = df[df["is_training_cohort"] == False].shape[0]
    print(f"\n[SUMMARY] Data Split:")
    print(f"  -> Training Cohort (2008-2019): {training_count} players")
    print(f"  -> Unlabeled Set (2020+): {unlabeled_count} players")

    cursor.close()
    connection.close()


if __name__ == "__main__":
    export_features_to_parquet()