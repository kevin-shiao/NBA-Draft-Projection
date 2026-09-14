import os
import pandas as pd


def flag_anomalies():
    print("Scanning for positional edge cases...")

    data_path = "data/processed/features.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}")

    df = pd.read_parquet(data_path)

    # 1. Resolve Height Column
    height_col = None
    for col in ["height_inches", "height_in", "height_wo_shoes_inches", "height"]:
        if col in df.columns:
            height_col = col
            break

    if not height_col:
        print("Error: Could not find a height column in features.parquet")
        return

    # Filter out invalid height rows
    valid_df = df[df[height_col].notna() & (df[height_col] > 0.0)].copy()

    # 2. Resolve Assist Column for Playmaking Checks (Updated for Per-100)
    ast_col = None
    for col in ["ast_per_100", "ast_per_40", "ast_pct", "AST_per_z", "assists_per_40"]:
        if col in df.columns:
            ast_col = col
            break

    # 3. Define Edge Case Criteria
    
    # CASE A: Short Bigs (6'9" and under classified as Bigs)
    short_big = (valid_df["pos_group"] == "Big") & (valid_df[height_col] <= 81.0)

    # CASE B: Tall Guards (Explicitly classified as Guard but 6'5" or taller)
    tall_guard_explicit = (valid_df["pos_group"] == "Guard") & (valid_df[height_col] >= 77.0)
    
    # CASE C: Playmaking Wings / Jumbo Creators (e.g., Haliburton, Cade, Giddey)
    # Checks for anyone 6'5" (77.0) or taller who generates point-guard level assists
    playmaking_wing = pd.Series(False, index=valid_df.index)
    if ast_col:
        # Dynamically set threshold based on the metric scale
        if "100" in ast_col:
            ast_threshold = 6.0  # > 6 ast per 100 poss
        elif "40" in ast_col:
            ast_threshold = 4.5  # > 4.5 ast per 40 min
        else:
            ast_threshold = 1.5  # Z-score fallback

        playmaking_wing = (
            (valid_df["pos_group"].isin(["Wing", "Big"])) & 
            (valid_df[height_col] >= 77.0) & 
            (valid_df[ast_col] >= ast_threshold)
        )

    # CASE D: Small Wings (6'4" and under classified as Wings)
    small_wing = (valid_df["pos_group"] == "Wing") & (valid_df[height_col] <= 76.0)

    # Combine Masks
    anomalies = valid_df[short_big | tall_guard_explicit | playmaking_wing | small_wing].copy()

    if anomalies.empty:
        print("No positional anomalies found.")
        return

    # 4. Format Output Table
    display_cols = ["draft_player_name", "draft_year", "pos_group", height_col]
    if ast_col:
        display_cols.append(ast_col)
    if "college_bpm" in anomalies.columns:
        display_cols.append("college_bpm")

    review_df = anomalies[display_cols].copy()

    # Height to Feet/Inches string
    review_df["height_formatted"] = review_df[height_col].apply(
        lambda x: f"{int(x // 12)}'{int(round(x % 12))}\""
    )

    review_df["manual_override_position"] = ""

    def label_reason(row):
        h = row[height_col]
        pos = row["pos_group"]
        
        # Check playmaking first so Haliburton gets the right tag
        if ast_col and pos in ["Wing", "Big"] and h >= 77.0:
            if row[ast_col] >= (6.0 if "100" in ast_col else 4.5):
                return "Playmaking Wing (Review Guard)"
                
        if pos == "Big" and h <= 81.0:
            return "Short Big"
        elif pos == "Wing" and h <= 76.0:
            return "Small Wing"
        elif pos == "Guard" and h >= 77.0:
            return "Tall Guard"
            
        return "Positional Outlier"

    review_df["review_reason"] = anomalies.apply(label_reason, axis=1)

    # Sort output so high-profile recent classes appear at the top
    review_df = review_df.sort_values(by=["draft_year", "review_reason"], ascending=[False, True])

    # Save to CSV
    os.makedirs("data/interim", exist_ok=True)
    out_path = "data/interim/position_review.csv"
    
    # If a review file already exists, don't overwrite manual work—merge it
    if os.path.exists(out_path):
        existing_df = pd.read_csv(out_path)
        merged_df = review_df.merge(
            existing_df[["draft_player_name", "manual_override_position"]], 
            on="draft_player_name", 
            how="left", 
            suffixes=('', '_old')
        )
        merged_df["manual_override_position"] = merged_df["manual_override_position_old"].fillna("")
        merged_df = merged_df.drop(columns=["manual_override_position_old"])
        merged_df.to_csv(out_path, index=False)
    else:
        review_df.to_csv(out_path, index=False)

    print(f"Success! Flagged {len(review_df)} players for manual review.")
    print(f"File updated: '{out_path}'")


if __name__ == "__main__":
    flag_anomalies()