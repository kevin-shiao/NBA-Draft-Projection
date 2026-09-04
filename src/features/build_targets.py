import os
import sys
from dotenv import load_dotenv
from databricks import sql

load_dotenv()

def build_target_table():
    print("Connecting to Databricks...")
    connection = sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )
    cursor = connection.cursor()

    print("[PROCESS] Ensuring analytics schema exists...")
    cursor.execute("CREATE SCHEMA IF NOT EXISTS nba_draft.analytics;")

    target_table_sql = """
    CREATE OR REPLACE TABLE nba_draft.analytics.draft_targets AS
    WITH draft_base AS (
        SELECT 
            d.player_name AS draft_player_name,
            CAST(d.season AS INT) AS draft_year,
            CAST(d.overall_pick AS INT) AS overall_pick,
            d.organization AS drafted_team,
            cw.college_player_name,
            -- Clean name: lower -> remove jr/sr/ii/iii -> remove all non-alphanumeric
            REGEXP_REPLACE(
                REGEXP_REPLACE(LOWER(d.player_name), '\\\\b(jr|sr|ii|iii|iv)\\\\b', ''), 
                '[^a-z0-9]', ''
            ) AS norm_draft_name
        FROM nba_draft.staging.draft_history d
        LEFT JOIN nba_draft.staging.player_crosswalk cw
            ON d.player_name = cw.draft_player_name
           AND CAST(d.season AS INT) = cw.draft_year
    ),
    
    nba_deduped AS (
        SELECT 
            join_name,
            REGEXP_REPLACE(
                REGEXP_REPLACE(LOWER(join_name), '\\\\b(jr|sr|ii|iii|iv)\\\\b', ''), 
                '[^a-z0-9]', ''
            ) AS norm_join_name,
            CAST(Age AS INT) AS age_int,
            CAST(VORP AS DOUBLE) AS vorp,
            CAST(MP AS DOUBLE) AS mp,
            ROW_NUMBER() OVER (
                PARTITION BY join_name, CAST(Age AS INT) 
                ORDER BY CAST(G AS INT) DESC
            ) as trade_rank
        FROM nba_draft.staging.nba_outcomes
    ),
    
    nba_ranked_outcomes AS (
        SELECT 
            norm_join_name,
            vorp,
            mp,
            ROW_NUMBER() OVER (PARTITION BY norm_join_name ORDER BY age_int ASC) AS career_season_num
        FROM nba_deduped
        WHERE trade_rank = 1
    ),

    nba_5yr_outcomes AS (
        SELECT 
            norm_join_name,
            SUM(vorp) AS vorp_5y,
            SUM(mp) AS mp_5y,
            COUNT(career_season_num) AS seasons_played_5y
        FROM nba_ranked_outcomes
        WHERE career_season_num <= 5
        GROUP BY norm_join_name
    )

    SELECT 
        d.draft_player_name,
        d.draft_year,
        d.overall_pick,
        d.drafted_team,
        d.college_player_name,
        
        -- Target 1: Continuous 5-Year VORP 
        COALESCE(o.vorp_5y, 0.0) AS vorp_5y,
        COALESCE(o.mp_5y, 0.0) AS mp_5y,
        COALESCE(o.seasons_played_5y, 0) AS seasons_played_5y,
        
        -- Target 2: Hurdle Classifier
        CASE 
            WHEN COALESCE(o.mp_5y, 0.0) >= 2000 THEN 1 
            ELSE 0 
        END AS reached_min_threshold_5y,

        -- Target 3: UI Tiers (Adjusted for Rookie Scale Reality)
        CASE 
            WHEN COALESCE(o.vorp_5y, 0.0) < 0.5 THEN 'Bust'
            WHEN COALESCE(o.vorp_5y, 0.0) BETWEEN 0.5 AND 3.999 THEN 'Rotation'
            WHEN COALESCE(o.vorp_5y, 0.0) BETWEEN 4.0 AND 9.999 THEN 'Starter'
            WHEN COALESCE(o.vorp_5y, 0.0) >= 10.0 THEN 'All-Star'
            ELSE 'Bust'
        END AS player_tier_5y,

        -- Cohort Selection
        CASE 
            WHEN d.draft_year BETWEEN 2009 AND 2019 THEN TRUE 
            ELSE FALSE 
        END AS is_training_cohort

    FROM draft_base d
    LEFT JOIN nba_5yr_outcomes o
        ON d.norm_draft_name = o.norm_join_name;
    """

    print("[PROCESS] Building nba_draft.analytics.draft_targets table...")
    try:
        cursor.execute(target_table_sql)
        print("  -> [SUCCESS] Target table built successfully!")
    except Exception as e:
        print(f"  -> [ERROR] Failed to build target table: {e}")
        cursor.close()
        connection.close()
        sys.exit(1)

    print("\n[INSPECT] Training Cohort (2009-2019) Target Tier Distribution:")
    cursor.execute("""
        SELECT 
            player_tier_5y,
            COUNT(*) AS player_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct_of_cohort
        FROM nba_draft.analytics.draft_targets
        WHERE is_training_cohort = TRUE
        GROUP BY player_tier_5y
        ORDER BY 
            CASE player_tier_5y
                WHEN 'All-Star' THEN 1
                WHEN 'Starter' THEN 2
                WHEN 'Rotation' THEN 3
                WHEN 'Bust' THEN 4
            END
    """)
    for row in cursor.fetchall():
        print(f"  -> {row[0]}: {row[1]} players ({row[2]}%)")

    cursor.close()
    connection.close()

if __name__ == "__main__":
    build_target_table()