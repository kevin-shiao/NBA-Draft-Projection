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
    for col in ["height_inches", "height", "height_wo_shoes_inches", "ht"]:
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

    # 2. Three Rate Calculation
    if "three_par" in df.columns:
        df["three_rate"] = pd.to_numeric(df["three_par"], errors="coerce").fillna(0.0)
    elif "TPA" in df.columns and "twoPA" in df.columns:
        tpa = pd.to_numeric(df["TPA"], errors="coerce").fillna(0.0)
        twopa = pd.to_numeric(df["twoPA"], errors="coerce").fillna(0.0)
        df["three_rate"] = np.where((tpa + twopa) > 0, tpa / (tpa + twopa), 0.0)
    else:
        df["three_rate"] = 0.0

    # Season mapping for within-season Z-score scaling
    season_col = "draft_year" if "draft_year" in df.columns else ("season" if "season" in df.columns else None)

    # Map target statistical columns (Priority: Per-100 -> Per-40 -> Raw)
    def resolve_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    stats_map = {
        "height_in": "height_in",
        "blk_per": resolve_col(["blk_per_100", "blk_per_40", "blk_per"]),
        "ORB_per": resolve_col(["orb_per_100", "reb_per_100", "orb_per_40", "ORB_per"]),
        "DRB_per": resolve_col(["drb_per_100", "reb_per_100", "drb_per_40", "DRB_per"]),
        "three_rate": "three_rate",
        "AST_per": resolve_col(["ast_per_100", "ast_per_40", "AST_per"]),
        "stl_per": resolve_col(["stl_per_100", "stl_per_40", "stl_per"]),
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

    # Fallback Positional Archetype Scores
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

    if "pos_group" not in df.columns:
        df["pos_group"] = (
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
    elif "fg3_pct" in df.columns:
        df["TP_per_shrunk"] = pd.to_numeric(df["fg3_pct"], errors="coerce").fillna(0.0)
    else:
        df["TP_per_shrunk"] = 0.0

    if "FT_made" in df.columns and "FTA" in df.columns:
        made_ft = pd.to_numeric(df["FT_made"], errors="coerce").fillna(0.0)
        att_ft = pd.to_numeric(df["FTA"], errors="coerce").fillna(0.0)
        df["FT_per_shrunk"] = e_bayes_shrinkage(made_ft, att_ft, k=250)
    elif "ft_pct" in df.columns:
        df["FT_per_shrunk"] = pd.to_numeric(df["ft_pct"], errors="coerce").fillna(0.0)
    else:
        df["FT_per_shrunk"] = 0.0

    # 3. Touch Divergence
    df["touch_divergence"] = df["FT_per_shrunk"] - df["TP_per_shrunk"]

    # 4. Residualize Turnovers on Usage
    to_col = "TO_per" if "TO_per" in df.columns else ("tov_per_100" if "tov_per_100" in df.columns else None)
    usg_col = "usage_pct" if "usage_pct" in df.columns else ("usg" if "usg" in df.columns else None)

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

    return df


def export_features_to_parquet():
    print("Connecting to Databricks...")
    
    hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    token = os.getenv("DATABRICKS_TOKEN")

    if hostname and hostname.startswith("https://"):
        hostname = hostname.replace("https://", "")

    connection = sql.connect(
        server_hostname=hostname,
        http_path=http_path,
        access_token=token,
    )
    cursor = connection.cursor()

    # --- FILTER OUT NON-COLLEGE PROSPECTS ---
    print("[PROCESS] Querying nba_draft.analytics.prospect_features (filtering out non-college prospects)...")
    cursor.execute("""
        SELECT * 
        FROM nba_draft.analytics.prospect_features 
        WHERE is_non_college_prospect = 0;
    """)
    rows = cursor.fetchall()

    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)

    if "draft_year" in df.columns:
        df["draft_year"] = pd.to_numeric(df["draft_year"], errors="coerce")
        df["is_training_cohort"] = df["draft_year"].between(2008, 2019)

    print("[PROCESS] Applying position archetypes and parsing heights...")
    df = process_position_and_archetypes(df)

    print("[PROCESS] Calculating shrinkage and turnover residualization...")
    df = process_feature_fixes(df)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # --- APPLY MANUAL POSITION OVERRIDES ---
    override_path = "data/interim/position_review.csv"
    if os.path.exists(override_path):
        overrides = pd.read_csv(override_path)
        
        overrides = overrides[
            overrides["manual_override_position"].notna() 
            & (overrides["manual_override_position"].astype(str).str.strip() != "")
        ]
        
        override_map = dict(zip(
            overrides["draft_player_name"], 
            overrides["manual_override_position"].astype(str).str.strip()
        ))
        
        if "draft_player_name" in df.columns:
            applied_count = 0
            for player, pos in override_map.items():
                mask = df["draft_player_name"] == player
                if mask.any():
                    df.loc[mask, "pos_group"] = pos
                    applied_count += 1
            print(f"✅ Successfully applied {applied_count} manual position overrides from '{override_path}'.")

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