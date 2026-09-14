import os
import re
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="NBA Draft Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- CACHED DATA LOADERS ---
@st.cache_data
def load_predictions():
    path = "data/processed/predictions.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_features():
    path = "data/processed/features.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_resource
def load_model():
    path = "models/model.joblib"
    if os.path.exists(path):
        return joblib.load(path)
    return None


def assign_tier(pred_vorp):
    if pred_vorp >= 3.0:
        return "Franchise Cornerstone"
    elif pred_vorp >= 2.0:
        return "All-Star Potential"
    elif pred_vorp >= 1.0:
        return "High-Value Starter"
    elif pred_vorp >= 0.2:
        return "Rotation Contributor"
    else:
        return "Fringe / Developmental"

def sanitize_filename(name):
    return re.sub(r'[^\w\s-]', '', str(name)).strip().replace(' ', '_')

# Load Artifacts
predictions_df = load_predictions()
features_df = load_features()
model_artifact = load_model()

# Navigation Sidebar
st.sidebar.title("Draft Intelligence")
page = st.sidebar.radio(
    "Navigation",
    ["Big Board", "Player Card", "Model vs. Draft", "Backtest", "About"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by LightGBM & Ridge Ensemble | Databricks & MLflow")


# ==========================================
# PAGE 1: BIG BOARD
# ==========================================
if page == "Big Board":
    st.title("Draft Big Board")
    st.markdown(
        "Explore sortable and filterable projected rankings for recent draft classes based on pre-draft predictive VORP and calibrated rotation probability."
    )

    if predictions_df.empty:
        st.error(
            "Predictions dataset not found. Please run `export_features.py` and `train.py` first."
        )
    else:
        available_years = sorted(
            predictions_df["draft_year"].dropna().unique().astype(int),
            reverse=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            selected_year = st.selectbox(
                "Select Draft Class", available_years, index=0
            )
        with col2:
            positions = ["All"] + list(
                predictions_df["pos_group"].dropna().unique()
            )
            selected_pos = st.selectbox("Position Filter", positions)
        with col3:
            search_query = st.text_input("Search Player Name", "")

        # Filter Data
        df_class = predictions_df[
            predictions_df["draft_year"] == selected_year
        ].copy()
        if selected_pos != "All":
            df_class = df_class[df_class["pos_group"] == selected_pos]
        if search_query:
            df_class = df_class[
                df_class["draft_player_name"].str.contains(
                    search_query, case=False, na=False
                )
            ]

        df_class["Tier"] = df_class["pred_vorp_5y"].apply(assign_tier)
        df_class["Model Rank"] = df_class["model_rank"].astype(int)

        # Format rotation_prob as 0-100 percentage if present
        if "rotation_prob" in df_class.columns:
            df_class["Rotation Prob %"] = df_class["rotation_prob"].apply(
                lambda x: x * 100 if pd.notna(x) and x <= 1.0 else x
            )

        # Columns Display Config
        display_cols = [
            "Model Rank",
            "draft_player_name",
            "pos_group",
            "pred_vorp_5y",
            "Rotation Prob %",
            "Tier",
            "overall_pick",
            "college_bpm",
            "pts_per_100",
            "ts_pct",
            "age_at_draft",
        ]
        col_rename = {
            "draft_player_name": "Player Name",
            "pos_group": "Position",
            "pred_vorp_5y": "Projected 5Y VORP",
            "overall_pick": "Actual Draft Pick",
            "college_bpm": "College BPM",
            "pts_per_100": "PTS / 100",
            "ts_pct": "True Shooting %",
            "age_at_draft": "Draft Age",
        }

        df_display = (
            df_class[[c for c in display_cols if c in df_class.columns]]
            .rename(columns=col_rename)
            .sort_values(by="Model Rank")
        )

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Projected 5Y VORP": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Rotation Prob %": st.column_config.NumberColumn(
                    format="%.1f%%"
                ),
                "College BPM": st.column_config.NumberColumn(format="%.1f"),
                "PTS / 100": st.column_config.NumberColumn(format="%.1f"),
                "True Shooting %": st.column_config.NumberColumn(format="%.3f"),
                "Draft Age": st.column_config.NumberColumn(format="%.1f"),
            },
        )

# ==========================================
# PAGE 2: PLAYER CARD
# ==========================================
elif page == "Player Card":
    st.title("Player Profile & Similarity Analysis")

    if features_df.empty:
        st.error("Features dataset not found. Run pipeline steps first.")
    else:
        # --- RESTRICT SEARCH TO 2020+ PROSPECTS ---
        prospect_df = features_df[features_df["draft_year"] >= 2020].copy()

        # --- FILTER & SEARCH BAR SECTION ---
        col_y, col_p, col_s = st.columns([1, 1, 2])

        with col_y:
            available_years = ["All"] + sorted(
                prospect_df["draft_year"].dropna().unique().astype(int),
                reverse=True,
            )
            selected_year = st.selectbox("Filter Draft Year", available_years, index=0)

        with col_p:
            available_positions = ["All"] + sorted(
                prospect_df["pos_group"].dropna().unique()
            )
            selected_pos = st.selectbox("Filter Position", available_positions, index=0)

        # Apply UI filters to build selectable player list
        filtered_df = prospect_df.copy()
        if selected_year != "All":
            filtered_df = filtered_df[filtered_df["draft_year"] == selected_year]
        if selected_pos != "All":
            filtered_df = filtered_df[filtered_df["pos_group"] == selected_pos]

        filtered_player_list = sorted(filtered_df["draft_player_name"].dropna().unique())

        with col_s:
            if not filtered_player_list:
                st.warning("No prospects match the selected Year & Position filters.")
                selected_player = None
            else:
                selected_player = st.selectbox(
                    "Search / Select Prospect (2020+)",
                    filtered_player_list,
                    help="Type a player's name directly in the box to search!"
                )

        st.markdown("---")

        if selected_player:
            player_row = features_df[
                features_df["draft_player_name"] == selected_player
            ].iloc[0]

            # Fetch prediction details if available
            pred_val = "N/A"
            rot_prob_val = "N/A"
            tier_val = "N/A"

            if not predictions_df.empty:
                p_match = predictions_df[predictions_df["draft_player_name"] == selected_player]
                if not p_match.empty:
                    p_pred = p_match.iloc[0]
                    
                    # Fetch VORP and Tier
                    vorp_raw = p_pred.get("pred_vorp_5y")
                    if pd.notna(vorp_raw):
                        pred_val = f"{float(vorp_raw):.2f}"
                        tier_val = assign_tier(float(vorp_raw))

                    # Fetch Rotation Probability
                    rot_raw = p_pred.get("rotation_prob")
                    if pd.notna(rot_raw):
                        rot_float = float(rot_raw)
                        rot_prob_val = f"{rot_float * 100:.1f}%" if rot_float <= 1.0 else f"{rot_float:.1f}%"

            # Top Metric Highlights
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Position", str(player_row.get("pos_group", "N/A")))
            c2.metric("Draft Year", int(player_row.get("draft_year", 0)))
            c3.metric("Projected 5Y VORP", pred_val)
            c4.metric("Rotation Prob", rot_prob_val)
            c5.metric("Tier", tier_val)

            st.markdown("---")

            # Stat Line & Physicals
            col_stat, col_phys = st.columns(2)
            with col_stat:
                st.subheader("College Production Metrics")
                st.write(f"**College BPM:** {player_row.get('college_bpm', 0.0):.2f}")
                st.write(f"**Points / 100:** {player_row.get('pts_per_100', 0.0):.1f}")
                st.write(f"**Rebounds / 100:** {player_row.get('reb_per_100', 0.0):.1f}")
                st.write(f"**Assists / 100:** {player_row.get('ast_per_100', 0.0):.1f}")
                st.write(f"**True Shooting %:** {player_row.get('ts_pct', 0.0):.3f}")

            with col_phys:
                st.subheader("Anthropometrics")

                # Converts raw inches to standard feet/inches format
                def format_feet_inches(val_inches):
                    if not val_inches or val_inches <= 0.0:
                        return "N/A (No Combine Data)"
                    feet = int(val_inches // 12)
                    inches = val_inches % 12
                    if inches.is_integer() or abs(inches - round(inches)) < 0.01:
                        return f"{feet}'{int(round(inches))}\""
                    return f"{feet}'{inches:.1f}\""

                def get_phys_metric(row, keys):
                    for k in keys:
                        if k in row and pd.notna(row[k]) and float(row[k]) > 0.0:
                            return float(row[k])
                    return None

                height_val = get_phys_metric(player_row, ["height_inches", "height_in", "height_wo_shoes_inches", "height"])
                wingspan_val = get_phys_metric(player_row, ["wingspan_inches", "wingspan_in", "wingspan"])
                ape_val = get_phys_metric(player_row, ["ape_index", "ape_index_adj"])
                body_fat_val = get_phys_metric(player_row, ["body_fat_pct", "body_fat"])

                st.write(f"**Height:** {format_feet_inches(height_val)}")
                st.write(f"**Wingspan:** {format_feet_inches(wingspan_val)}")
                st.write(f"**Ape Index (Adjusted):** {f'{ape_val:.2f}' if ape_val else 'N/A'}")
                st.write(f"**Body Fat %:** {f'{body_fat_val:.1f}%' if body_fat_val else 'N/A'}")

            st.markdown("---")
            st.subheader("3-5 Historical Player Comparisons (Cosine Similarity)")

            # Historical Comps Calculation against 2009-2019 baseline
            ignore_cols = [
                "draft_player_name", "draft_year", "drafted_team", "is_training_cohort",
                "vorp_5y", "reached_min_threshold_5y", "player_tier_5y", "overall_pick", "pos_group",
            ]
            feat_cols = [
                c for c in features_df.columns
                if c not in ignore_cols and pd.api.types.is_numeric_dtype(features_df[c])
            ]

            hist_df = features_df[
                (features_df["is_training_cohort"] == True)
                & (features_df["draft_player_name"] != selected_player)
            ].copy()

            if not hist_df.empty and len(feat_cols) > 0:
                X_hist = hist_df[feat_cols].fillna(0.0)
                X_target = (
                    pd.DataFrame([player_row[feat_cols]])
                    .fillna(0.0)
                    .values.reshape(1, -1)
                )

                scaler = StandardScaler()
                X_hist_scaled = scaler.fit_transform(X_hist)
                X_target_scaled = scaler.transform(X_target)

                sims = cosine_similarity(X_target_scaled, X_hist_scaled).flatten()
                hist_df["similarity_score"] = sims

                top_comps = hist_df.sort_values(
                    by="similarity_score", ascending=False
                ).head(5)

                comp_display = top_comps[
                    [
                        "draft_player_name",
                        "draft_year",
                        "pos_group",
                        "overall_pick",
                        "vorp_5y",
                        "similarity_score",
                    ]
                ].rename(
                    columns={
                        "draft_player_name": "Historical Player",
                        "draft_year": "Draft Year",
                        "pos_group": "Position",
                        "overall_pick": "Actual Pick",
                        "vorp_5y": "Actual 5Y VORP",
                        "similarity_score": "Similarity Match",
                    }
                )

                st.dataframe(
                    comp_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Similarity Match": st.column_config.NumberColumn(format="%.3f"),
                        "Actual 5Y VORP": st.column_config.NumberColumn(format="%.2f"),
                    },
                )

        if selected_player:
            player_year = int(player_row.get("draft_year", 0))
            safe_name = sanitize_filename(selected_player)
            
            shap_image_path = f"data/shap_plots/{player_year}/{safe_name}.png"
            
            st.markdown("---")
            st.subheader("🔍 Model Drivers (SHAP Plot)")
            
            if os.path.exists(shap_image_path):
                st.image(shap_image_path, use_container_width=True)
            else:
                st.info(f"No pre-rendered SHAP plot found for {selected_player}.")
    

# ==========================================
# PAGE 3: MODEL VS. DRAFT
# ==========================================
elif page == "Model vs. Draft":
    st.title("Model vs. Draft Board Disagreements")
    st.markdown(
        "Evaluate where the model differed most from actual NBA draft order. Uncover historical steals and overvalued reaches."
    )

    if predictions_df.empty:
        st.error("Predictions dataset not found.")
    else:
        df_eval = predictions_df[
            predictions_df["overall_pick"].notna()
            & (predictions_df["overall_pick"] > 0)
        ].copy()

        df_eval["draft_disagreement"] = (
            df_eval["overall_pick"] - df_eval["model_rank"]
        )

        # Clip negative VORP values to 0 and add an offset so all points render with a valid size
        df_eval["plot_size"] = df_eval["pred_vorp_5y"].clip(lower=0) + 2.0

        fig = px.scatter(
            df_eval,
            x="overall_pick",
            y="model_rank",
            hover_name="draft_player_name",
            color="pos_group",
            size="plot_size",
            size_max=15,
            title="Model Rank vs. Actual Draft Pick",
            labels={
                "overall_pick": "Actual Draft Pick",
                "model_rank": "Model Projected Rank",
            },
        )
        # Add 1:1 Reference Line
        fig.add_trace(
            go.Scatter(
                x=[1, 60],
                y=[1, 60],
                mode="lines",
                name="Perfect Consensus (1:1)",
                line=dict(dash="dash", color="gray"),
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        col_steals, col_reaches = st.columns(2)

        with col_steals:
            st.subheader("Best Steals (Model Loved, Drafted Late)")
            steals_df = df_eval.sort_values(
                by="draft_disagreement", ascending=False
            ).head(5)
            st.dataframe(
                steals_df[
                    [
                        "draft_player_name",
                        "draft_year",
                        "overall_pick",
                        "model_rank",
                        "pred_vorp_5y",
                    ]
                ].rename(
                    columns={
                        "draft_player_name": "Player",
                        "overall_pick": "Draft Pick",
                        "model_rank": "Model Rank",
                        "pred_vorp_5y": "Pred VORP",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

        with col_reaches:
            st.subheader("Biggest Reaches (Drafted Early, Model Hated)")
            reaches_df = df_eval.sort_values(
                by="draft_disagreement", ascending=True
            ).head(5)
            st.dataframe(
                reaches_df[
                    [
                        "draft_player_name",
                        "draft_year",
                        "overall_pick",
                        "model_rank",
                        "pred_vorp_5y",
                    ]
                ].rename(
                    columns={
                        "draft_player_name": "Player",
                        "overall_pick": "Draft Pick",
                        "model_rank": "Model Rank",
                        "pred_vorp_5y": "Pred VORP",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )


# ==========================================
# PAGE 4: BACKTEST
# ==========================================
elif page == "Backtest":
    st.title("Historical Model Backtest (2008-2019)")
    st.markdown(
        "Out-of-fold cross-validation performance evaluating the model's ranking ability against historical baseline draft order."
    )

    import json
    metrics_path = "models/metrics.json"
    
    if not os.path.exists(metrics_path):
        st.warning("Metrics file not found. Run `train.py` to generate backtest scores.")
    else:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
        # Safely extract metrics (default to 0 if missing)
        rho = metrics.get("spearman_rho", 0)
        ndcg10 = metrics.get("ndcg_10", 0)
        ndcg30 = metrics.get("ndcg_30", 0)
        hit_rate = metrics.get("hit_rate", 0)
        
        # Draft baseline metrics
        draft_rho = metrics.get("draft_spearman_rho", 0)
        draft_ndcg10 = metrics.get("draft_ndcg_10", 0)
        draft_ndcg30 = metrics.get("draft_ndcg_30", 0)
        draft_hit_rate = metrics.get("draft_hit_rate", 0)

        # Calculate deltas (Model - Draft Order)
        delta_rho = rho - draft_rho
        delta_ndcg10 = ndcg10 - draft_ndcg10
        delta_ndcg30 = ndcg30 - draft_ndcg30
        delta_hit_rate = hit_rate - draft_hit_rate

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("OOF Hit Rate (Top 10 in 30)", f"{hit_rate * 100:.1f}%", delta=f"{delta_hit_rate * 100:.1f}% vs Draft")
        m2.metric("OOF NDCG@10", f"{ndcg10:.3f}", delta=f"{delta_ndcg10:+.3f} vs Draft")
        m3.metric("OOF NDCG@30", f"{ndcg30:.3f}", delta=f"{delta_ndcg30:+.3f} vs Draft")
        m4.metric("OOF Spearman Rho", f"{rho:.3f}", delta=f"{delta_rho:+.3f} vs Draft")

        st.markdown("---")
        st.subheader("Performance Summary vs. Actual Draft Order")

        backtest_data = pd.DataFrame(
            [
                {
                    "Metric": "Spearman Rho (Rank Correlation)",
                    "Model Score": f"{rho:.3f}",
                    "Draft Order Baseline": f"{draft_rho:.3f}",
                    "Advantage": f"{delta_rho:+.3f}",
                },
                {
                    "Metric": "NDCG@10 (Top 10 Accuracy)",
                    "Model Score": f"{ndcg10:.3f}",
                    "Draft Order Baseline": f"{draft_ndcg10:.3f}",
                    "Advantage": f"{delta_ndcg10:+.3f}",
                },
                {
                    "Metric": "NDCG@30 (First Round Accuracy)",
                    "Model Score": f"{ndcg30:.3f}",
                    "Draft Order Baseline": f"{draft_ndcg30:.3f}",
                    "Advantage": f"{delta_ndcg30:+.3f}",
                },
                {
                    "Metric": "Top 10 Hit Rate (Found in Top 30)",
                    "Model Score": f"{hit_rate * 100:.1f}%",
                    "Draft Order Baseline": f"{draft_hit_rate * 100:.1f}%",
                    "Advantage": f"{delta_hit_rate * 100:+.1f}%",
                },
            ]
        )

        st.dataframe(backtest_data, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📚 Metrics Glossary")
        st.markdown(
            """
            * **Spearman Rho (Rank Correlation):** Evaluates how well the model's overall player rankings mirror reality, ignoring absolute VORP predictions. A score of 1.0 means perfect ranking; 0.0 means completely random.
            * **NDCG (Normalized Discounted Cumulative Gain):** A search-engine metric adapted for the draft. It heavily penalizes misses at the very top of the board. 
                * **NDCG@10:** Measures accuracy strictly within the top 10 picks. Whiffing on the #2 overall player hurts the score significantly more than whiffing on the #9 player.
                * **NDCG@30:** Applies the same sliding-scale penalty logic across the entire first round.
            * **Top 10 Hit Rate (Found in Top 30):** A straightforward hit-or-miss metric. Out of the 10 most productive NBA players in reality, how many did the model successfully project *somewhere* within its first-round board (top 30)? 
            """
        )


# ==========================================
# PAGE 5: ABOUT
# ==========================================
elif page == "About":
    st.title("Methodology & Architecture")

    st.markdown(
        """
        ### System Overview
        This NBA Draft Intelligence System evaluates NCAA prospects by isolating fundamental basketball skills from pace, scheme, and sample-size noise. It uses a dual-pipeline approach to independently project a player's **ceiling** (Value Over Replacement Player) and **floor** (Rotation Probability) using purely pre-draft inputs.

        ---

        ### 1. Data Ingestion & Hygiene
        The foundation of the model relies on three disparate data sources joined and processed via Databricks Unity Catalog:
        * **Production:** NCAA box score statistics and advanced metrics sourced from Bart Torvik.
        * **Anthropometrics:** Official NBA Draft Combine measurements (Height, Wingspan, Standing Reach, Body Fat %).
        * **Outcomes (The Target):** Career VORP and Minutes Played via Basketball-Reference, capped at a strict 5-year window to prevent older veterans from skewing the target variable.
        * **Exclusions:** To maintain a clean feature space, prospects lacking NCAA data (e.g., International, Overtime Elite, G-League Ignite) are explicitly filtered out prior to training.

        ### 2. Feature Engineering
        Raw box score numbers are heavily transformed to maximize predictive signal and stabilize variance:
        * **Pace Normalization:** All counting stats (Points, Rebounds, Assists) are converted to a standardized **Per-100 Possessions** baseline (~1.70 possessions/minute) to evaluate slow-paced bigs and run-and-gun guards on equal footing.
        * **Composite Efficiency Metrics:** Linear stats are replaced with composite vectors like `usage_efficiency_index` (Usage % × True Shooting %) to reward high-volume scorers who maintain efficiency.
        * **Empirical Bayes Shrinkage:** Low-volume shooting metrics (3P% and FT%) are shrunk toward the mean to prevent small sample sizes from tricking the model.
        * **Positional Z-Scores & Overrides:** Players are algorithmically bucketed into Guard, Wing, or Big based on combine heights, with production evaluated relative to their peers (`bpm_pos_zscore`). Known outliers (e.g., tall playmakers like Tyrese Haliburton or Cade Cunningham) are managed via a localized CSV override dictionary.

        ### 3. Model Architecture
        The pipeline avoids relying on a single "black box" by treating ceiling and floor as separate machine learning problems:

        **The Ceiling Model (Projected 5Y VORP)**
        * **Algorithm:** A 70/30 Ensemble of LightGBM (Gradient Boosted Trees) and Ridge Regression. 
        * **Why:** LightGBM excels at discovering non-linear interactions (e.g., high assist rates scaling exponentially with height). The Ridge Regression anchor (Alpha = 20.0) forces the ensemble to respect fundamental linear baselines, preventing the trees from overfitting to obscure outlier combinations.
        * **Validation:** GroupKFold cross-validation by `draft_year` prevents the model from peeking at future outcomes.

        **The Floor Model (Rotation Probability)**
        * **Algorithm:** Standardized Logistic Regression wrapped in a Platt Scaling Calibrator.
        * **Why:** The rotation model answers a binary question: *Will this prospect survive 2,000 NBA minutes?* Linear models naturally apply harsh penalties to high-risk profiles (e.g., older prospects or severely undersized guards). The calibration ensures the output percentage (e.g., 40%) strictly matches the real-world statistical probability.

        ### 4. Interpretation & UI
        * **Similarity Matching:** A standard scaler and Cosine Similarity matrix are applied to the feature vector space to find the 5 closest historical statistical matches from the 2009-2019 training block.
        * **Feature Attribution:** SHAP (SHapley Additive exPlanations) TreeExplainer values are pre-rendered during the pipeline run to visually explain exactly how the LightGBM model weighed a prospect's features to arrive at their VORP projection.

        ---

        ### 5. Challenges & Future Roadmap
        While the current architecture provides a robust historical baseline, several enhancements are planned for future iterations:
        
        * **Algorithmic Positional Flagging:** Currently, positional anomalies (like jumbo playmakers) are flagged using rigid height and assist thresholds. The next step is to implement unsupervised clustering (e.g., K-Means or PCA) to classify players into fluid offensive archetypes (e.g., "Primary Initiator," "Slashing Wing," "Stretch Big") based purely on their statistical footprint rather than combine height.
        * **Live In-Season Updates:** The current pipeline relies on static, end-of-season data dumps. By integrating automated scheduling tools (like Airflow or GitHub Actions) with live NCAA data feeds, the model will dynamically update and project prospects in real-time throughout the college basketball season.
        * **International & Non-NCAA Integration:** Prospects from the EuroLeague, NBL, and developmental leagues (G-League Ignite, Overtime Elite) are currently excluded due to structural data differences. Future versions will introduce a League Equivalency Translation layer (similar to NHLE in hockey) to normalize international stats and pace, allowing the model to accurately project global talents alongside NCAA athletes.
        """
    )