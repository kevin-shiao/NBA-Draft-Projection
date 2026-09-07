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

    print("[PROCESS] Building clean analytics.prospect_features with college height fallbacks...")

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
                REGEXP_REPLACE(LOWER(col_0), '\\\\b(jr|sr|ii|iii|iv)\\\\b', ''), 
                '[^a-z0-9]', ''
            ) AS norm_college_name,
            ROW_NUMBER() OVER (
                PARTITION BY REGEXP_REPLACE(
                    REGEXP_REPLACE(LOWER(col_0), '\\\\b(jr|sr|ii|iii|iv)\\\\b', ''), 
                    '[^a-z0-9]', ''
                ), season 
                ORDER BY CAST(col_3 AS INT) DESC
            ) as rank_per_season
        FROM nba_draft.staging.college
    ),

    college_history AS (
        SELECT 
            c.*,
            CAST(c.season AS INT) AS season_int,
            CAST(c.col_50 AS DOUBLE) AS true_bpm,
            COUNT(*) OVER (PARTITION BY c.norm_college_name) AS total_seasons,
            LAG(CAST(c.col_50 AS DOUBLE)) OVER (
                PARTITION BY c.norm_college_name ORDER BY CAST(c.season AS INT) ASC
            ) AS prev_bpm
        FROM college_dedup c
        WHERE c.rank_per_season = 1
    ),

    final_college AS (
        SELECT 
            t.*,
            c.season_int,
            
            CAST(c.col_63 AS DOUBLE) AS pts_per_game,
            CAST(c.col_59 AS DOUBLE) AS reb_per_game,
            CAST(c.col_60 AS DOUBLE) AS ast_per_game,
            CAST(c.col_61 AS DOUBLE) AS stl_per_game,
            CAST(c.col_62 AS DOUBLE) AS blk_per_game,
            CAST(c.col_54 AS DOUBLE) AS mp_per_game,
            
            CAST(c.col_8 AS DOUBLE) / 100.0 AS ts_pct_val,
            CAST(c.col_7 AS DOUBLE) / 100.0 AS efg_pct_val,
            CAST(c.col_15 AS DOUBLE) AS ft_pct_val,
            CAST(c.col_20 AS DOUBLE) AS three_pt_att,
            CAST(c.col_17 AS DOUBLE) AS two_pt_att,
            CAST(c.col_24 AS DOUBLE) / 100.0 AS ftr_val,
            CAST(c.col_35 AS DOUBLE) AS ast_to_ratio_val,

            CAST(c.col_6 AS DOUBLE) AS usage_pct_val,
            CAST(c.col_3 AS INT) AS games_played_val, 
            c.col_2 AS conf, 
            c.col_66 AS birthdate_str,
            c.col_26 AS height_str,                   -- col_26 = Listed Height string (e.g. '7-0', '6-11')
            c.true_bpm AS college_bpm,
            c.total_seasons,
            COALESCE(c.true_bpm - c.prev_bpm, 0.0) AS bpm_delta,
            
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
            (c.pts_per_game / NULLIF(c.mp_per_game, 0.0)) * 40.0 AS pts_per_40,
            (c.reb_per_game / NULLIF(c.mp_per_game, 0.0)) * 40.0 AS reb_per_40,
            (c.ast_per_game / NULLIF(c.mp_per_game, 0.0)) * 40.0 AS ast_per_40,
            (c.stl_per_game / NULLIF(c.mp_per_game, 0.0)) * 40.0 AS stl_per_40,
            (c.blk_per_game / NULLIF(c.mp_per_game, 0.0)) * 40.0 AS blk_per_40,
            
            c.ts_pct_val AS ts_pct,
            c.efg_pct_val AS efg_pct,
            c.ft_pct_val AS ft_pct,
            c.three_pt_att / NULLIF((c.three_pt_att + c.two_pt_att), 0.0) AS three_par,
            c.ftr_val AS ftr,
            c.ast_to_ratio_val AS ast_to_ratio,

            c.usage_pct_val AS usage_pct,
            c.games_played_val AS games_played,
            CASE WHEN c.conf IN ('ACC', 'B10', 'B12', 'SEC', 'P12', 'BE', 'Pac12') THEN 1 ELSE 0 END AS is_power_5,
            
            -- Guard / Creator Interactions
            (c.usage_pct_val * ((c.ast_per_game / NULLIF(c.mp_per_game, 0.0)) * 40.0)) AS usg_ast_interaction,
            (c.usage_pct_val * c.college_bpm) AS usg_bpm_interaction,
            (c.ft_pct_val * c.ftr_val) AS ft_volume_touch,

            COALESCE(
                (c.draft_year - YEAR(TO_DATE(c.birthdate_str, 'yyyy-MM-dd')) - 
                ((MONTH(TO_DATE(c.birthdate_str, 'yyyy-MM-dd')) - 6.0) / 12.0)),
                19.5
            ) AS age_at_draft,

            -- Height with fallback to Torvik listed height string parsing
            COALESCE(
                CAST(cb.height_wo_shoes AS DOUBLE),
                CASE 
                    WHEN c.height_str LIKE '%-%' THEN 
                        (TRY_CAST(SPLIT(TRIM(c.height_str), '-')[0] AS DOUBLE) * 12.0) + 
                         TRY_CAST(SPLIT(TRIM(c.height_str), '-')[1] AS DOUBLE)
                    ELSE NULL
                END,
                78.0
            ) AS height_inches,
            
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

    pos_assigned AS (
        SELECT 
            *,
            -- OBJECTIVE HEIGHT-BASED POSITION TIERS
            CASE 
                WHEN height_inches < 77.0 THEN 'Guard'
                WHEN height_inches >= 77.0 AND height_inches < 81.0 THEN 'Wing'
                ELSE 'Big'
            END AS pos_group
        FROM raw_joined
    ),

    pos_stats AS (
        SELECT 
            *,
            CASE WHEN pos_group = 'Guard' THEN 1 ELSE 0 END AS is_guard,
            CASE WHEN pos_group = 'Wing' THEN 1 ELSE 0 END AS is_wing,
            CASE WHEN pos_group = 'Big' THEN 1 ELSE 0 END AS is_big,

            COALESCE(
                (college_bpm - AVG(college_bpm) OVER(PARTITION BY pos_group)) 
                / NULLIF(STDDEV(college_bpm) OVER(PARTITION BY pos_group), 0.0), 
                0.0
            ) AS bpm_pos_zscore,

            COALESCE(
                (pts_per_40 - AVG(pts_per_40) OVER(PARTITION BY pos_group)) 
                / NULLIF(STDDEV(pts_per_40) OVER(PARTITION BY pos_group), 0.0), 
                0.0
            ) AS pts_pos_zscore,

            COALESCE(
                (ast_per_40 - AVG(ast_per_40) OVER(PARTITION BY pos_group)) 
                / NULLIF(STDDEV(ast_per_40) OVER(PARTITION BY pos_group), 0.0), 
                0.0
            ) AS ast_pos_zscore
        FROM pos_assigned
    ),

    combine_medians AS (
        SELECT 
            MEDIAN(height_inches) AS med_height,
            MEDIAN(wingspan_inches) AS med_wingspan,
            MEDIAN(standing_reach_inches) AS med_standing_reach,
            MEDIAN(weight_lbs) AS med_weight,
            MEDIAN(body_fat_pct) AS med_body_fat,
            MEDIAN(age_at_draft) AS med_age
        FROM pos_stats
    )

    SELECT 
        r.draft_player_name,
        r.draft_year,
        r.overall_pick,  
        r.drafted_team,
        r.is_training_cohort,
        r.is_non_college_prospect,
        r.pos_group,
        
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
        0.0 AS college_per,
        COALESCE(r.games_played, 0) AS games_played,
        r.is_power_5,

        -- Positional Features & Relative Z-Scores
        r.is_guard,
        r.is_wing,
        r.is_big,
        COALESCE(r.bpm_pos_zscore, 0.0) AS bpm_pos_zscore,
        COALESCE(r.pts_pos_zscore, 0.0) AS pts_pos_zscore,
        COALESCE(r.ast_pos_zscore, 0.0) AS ast_pos_zscore,

        -- Creator Interactions
        COALESCE(r.usg_ast_interaction, 0.0) AS usg_ast_interaction,
        COALESCE(r.usg_bpm_interaction, 0.0) AS usg_bpm_interaction,
        COALESCE(r.ft_volume_touch, 0.0) AS ft_volume_touch,

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

    FROM pos_stats r
    CROSS JOIN combine_medians m;
    """

    cursor.execute(feature_table_sql)
    print("  -> [SUCCESS] Rebuilt clean analytics.prospect_features table!")
    cursor.close()
    connection.close()

if __name__ == "__main__":
    build_prospect_features()