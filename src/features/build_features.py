import os
import sys
from dotenv import load_dotenv
from databricks import sql

load_dotenv()

def build_prospect_features():
    print("Connecting to Databricks...")
    connection = sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )
    cursor = connection.cursor()

    print("[PROCESS] Building deduplicated & cleaned analytics.prospect_features...")

    feature_table_sql = """
    CREATE OR REPLACE TABLE nba_draft.analytics.prospect_features AS
    WITH target_base AS (
        SELECT DISTINCT
            draft_player_name,
            draft_year,
            overall_pick,
            drafted_team,
            college_player_name,
            vorp_5y,
            reached_min_threshold_5y,
            player_tier_5y,
            is_training_cohort
        FROM nba_draft.analytics.draft_targets
    ),

    college_dedup AS (
        SELECT *,
            REGEXP_REPLACE(
                REGEXP_REPLACE(LOWER(player_name), '\\\\b(jr|sr|ii|iii|iv)\\\\b', ''), 
                '[^a-z0-9]', ''
            ) AS norm_college_name,
            ROW_NUMBER() OVER (
                PARTITION BY REGEXP_REPLACE(
                    REGEXP_REPLACE(LOWER(player_name), '\\\\b(jr|sr|ii|iii|iv)\\\\b', ''), 
                    '[^a-z0-9]', ''
                ), season 
                ORDER BY CAST(`3` AS INT) DESC
            ) as rank_per_season
        FROM nba_draft.staging.college
    ),

    college_history AS (
        SELECT 
            c.*,
            CAST(c.season AS INT) AS season_int,
            CAST(c.`34` AS DOUBLE) AS bpm,
            COUNT(*) OVER (PARTITION BY c.norm_college_name) AS total_seasons,
            LAG(CAST(c.`34` AS DOUBLE)) OVER (
                PARTITION BY c.norm_college_name ORDER BY CAST(c.season AS INT) ASC
            ) AS prev_bpm
        FROM college_dedup c
        WHERE c.rank_per_season = 1
    ),

    final_college AS (
        SELECT 
            t.*,
            c.season_int,
            c.`47`, c.`43`, c.`44`, c.`45`, c.`46`, c.`38`, c.`8`, c.`7`, c.`14`, c.`20`, c.`16`, c.`13`, c.`50`,
            c.`6`, c.`28`, c.`3`, c.`2`, c.`63`,
            c.bpm AS college_bpm,
            c.total_seasons,
            COALESCE(c.bpm - c.prev_bpm, 0.0) AS bpm_delta,
            
            CASE WHEN c.norm_college_name IS NULL THEN 1 ELSE 0 END AS is_non_college_prospect,

            ROW_NUMBER() OVER (
                PARTITION BY t.draft_player_name, t.draft_year 
                ORDER BY c.season_int DESC
            ) as draft_match_rn
        FROM target_base t
        LEFT JOIN college_history c
            ON REGEXP_REPLACE(
                REGEXP_REPLACE(LOWER(t.draft_player_name), '\\\\b(jr|sr|ii|iii|iv)\\\\b', ''), 
                '[^a-z0-9]', ''
            ) = c.norm_college_name
           AND c.season_int <= t.draft_year
    ),

    combine_dedup AS (
        SELECT *,
            REGEXP_REPLACE(
                REGEXP_REPLACE(LOWER(player_name), '\\\\b(jr|sr|ii|iii|iv)\\\\b', ''), 
                '[^a-z0-9]', ''
            ) AS norm_combine_name,
            ROW_NUMBER() OVER (
                PARTITION BY REGEXP_REPLACE(
                    REGEXP_REPLACE(LOWER(player_name), '\\\\b(jr|sr|ii|iii|iv)\\\\b', ''), 
                    '[^a-z0-9]', ''
                ) 
                ORDER BY season DESC
            ) as cb_rn
        FROM nba_draft.staging.combine
    ),

    raw_joined AS (
        SELECT 
            c.*,
            -- Per-40 Rate Calculations: (PPG / MPG) * 40.0
            (CAST(c.`47` AS DOUBLE) / NULLIF(CAST(c.`38` AS DOUBLE), 0.0)) * 40.0 AS pts_per_40,
            (CAST(c.`43` AS DOUBLE) / NULLIF(CAST(c.`38` AS DOUBLE), 0.0)) * 40.0 AS reb_per_40,
            (CAST(c.`44` AS DOUBLE) / NULLIF(CAST(c.`38` AS DOUBLE), 0.0)) * 40.0 AS ast_per_40,
            (CAST(c.`45` AS DOUBLE) / NULLIF(CAST(c.`38` AS DOUBLE), 0.0)) * 40.0 AS stl_per_40,
            (CAST(c.`46` AS DOUBLE) / NULLIF(CAST(c.`38` AS DOUBLE), 0.0)) * 40.0 AS blk_per_40,
            
            CAST(c.`8` AS DOUBLE) AS ts_pct,
            CAST(c.`7` AS DOUBLE) AS efg_pct,
            CAST(c.`14` AS DOUBLE) AS ft_pct,
            CAST(c.`20` AS DOUBLE) / NULLIF((CAST(c.`20` AS DOUBLE) + CAST(c.`16` AS DOUBLE)), 0.0) AS three_par,
            CAST(c.`13` AS DOUBLE) AS ftr,
            CAST(c.`50` AS DOUBLE) AS ast_to_ratio,

            CAST(c.`6` AS DOUBLE) AS usage_pct,
            CAST(c.`28` AS DOUBLE) AS college_per, 
            CAST(c.`3` AS INT) AS games_played,
            CASE WHEN c.`2` IN ('ACC', 'B10', 'B12', 'SEC', 'P12', 'BE', 'Pac12') THEN 1 ELSE 0 END AS is_power_5,
            
            -- Fault-tolerant Age Calculation using TRY_TO_DATE & TRY_CAST
            COALESCE(
                (c.draft_year - YEAR(TRY_TO_DATE(CAST(TRY_CAST(c.`63` AS BIGINT) AS STRING), 'yyyyMMdd')) - (MONTH(TRY_TO_DATE(CAST(TRY_CAST(c.`63` AS BIGINT) AS STRING), 'yyyyMMdd'))/12.0)),
                (c.draft_year - 19.5)
            ) AS age_at_draft,

            CAST(cb.height_wo_shoes AS DOUBLE) AS height_inches,
            CAST(cb.wingspan AS DOUBLE) AS wingspan_inches,
            CAST(cb.standing_reach AS DOUBLE) AS standing_reach_inches,
            CAST(cb.weight AS DOUBLE) AS weight_lbs,
            CAST(cb.body_fat_pct AS DOUBLE) AS body_fat_pct

        FROM final_college c
        LEFT JOIN combine_dedup cb
            ON REGEXP_REPLACE(
                REGEXP_REPLACE(LOWER(c.draft_player_name), '\\\\b(jr|sr|ii|iii|iv)\\\\b', ''), 
                '[^a-z0-9]', ''
            ) = cb.norm_combine_name
           AND cb.cb_rn = 1
        WHERE c.draft_match_rn = 1
    ),

    combine_medians AS (
        SELECT 
            MEDIAN(height_inches) AS med_height,
            MEDIAN(wingspan_inches) AS med_wingspan,
            MEDIAN(standing_reach_inches) AS med_standing_reach,
            MEDIAN(weight_lbs) AS med_weight,
            MEDIAN(body_fat_pct) AS med_body_fat,
            MEDIAN(age_at_draft) AS med_age
        FROM raw_joined
    )

    SELECT 
        r.draft_player_name,
        r.draft_year,
        r.overall_pick,  
        r.drafted_team,
        r.is_training_cohort,
        r.is_non_college_prospect,
        
        -- Targets
        r.vorp_5y,
        r.reached_min_threshold_5y,
        r.player_tier_5y,

        -- Production Features
        COALESCE(r.pts_per_40, 0.0) AS pts_per_40,
        COALESCE(r.reb_per_40, 0.0) AS reb_per_40,
        COALESCE(r.ast_per_40, 0.0) AS ast_per_40,
        COALESCE(r.stl_per_40, 0.0) AS stl_per_40,
        COALESCE(r.blk_per_40, 0.0) AS blk_per_40,
        COALESCE(r.ts_pct, 0.0) AS ts_pct,
        COALESCE(r.efg_pct, 0.0) AS efg_pct,
        COALESCE(r.ft_pct, 0.0) AS ft_pct,
        COALESCE(r.three_par, 0.0) AS three_par,
        COALESCE(r.ftr, 0.0) AS ftr,
        COALESCE(r.ast_to_ratio, 0.0) AS ast_to_ratio,

        -- Advanced, Context, Trajectory
        COALESCE(r.college_bpm, 0.0) AS college_bpm,
        COALESCE(r.bpm_delta, 0.0) AS bpm_delta,
        COALESCE(r.total_seasons, 0) AS seasons_played_in_college,
        COALESCE(r.usage_pct, 0.0) AS usage_pct,
        COALESCE(r.college_per, 0.0) AS college_per,
        COALESCE(r.games_played, 0) AS games_played,
        r.is_power_5,

        -- Age & High-Signal Interactions
        COALESCE(r.age_at_draft, m.med_age) AS age_at_draft,
        (COALESCE(r.college_bpm, 0.0) / COALESCE(r.age_at_draft, m.med_age)) AS bpm_age_interaction,
        (COALESCE(r.pts_per_40, 0.0) / COALESCE(r.age_at_draft, m.med_age)) AS pts_age_interaction,

        -- Imputed Physicals
        COALESCE(r.height_inches, m.med_height) AS height_inches,
        COALESCE(r.wingspan_inches, m.med_wingspan) AS wingspan_inches,
        (COALESCE(r.wingspan_inches, m.med_wingspan) - COALESCE(r.height_inches, m.med_height)) AS ape_index,
        COALESCE(r.standing_reach_inches, m.med_standing_reach) AS standing_reach_inches,
        COALESCE(r.weight_lbs, m.med_weight) AS weight_lbs,
        COALESCE(r.body_fat_pct, m.med_body_fat) AS body_fat_pct,

        -- Missingness Flags
        CASE WHEN r.height_inches IS NULL THEN 1 ELSE 0 END AS height_was_missing,
        CASE WHEN r.wingspan_inches IS NULL THEN 1 ELSE 0 END AS wingspan_was_missing,
        CASE WHEN r.body_fat_pct IS NULL THEN 1 ELSE 0 END AS body_fat_was_missing

    FROM raw_joined r
    CROSS JOIN combine_medians m;
    """

    cursor.execute(feature_table_sql)
    print("  -> [SUCCESS] Rebuilt clean analytics.prospect_features table!")
    cursor.close()
    connection.close()

if __name__ == "__main__":
    build_prospect_features()