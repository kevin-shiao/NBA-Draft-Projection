import os
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
            "pts_per_40",
            "ts_pct",
            "age_at_draft",
        ]
        col_rename = {
            "draft_player_name": "Player Name",
            "pos_group": "Position",
            "pred_vorp_5y": "Projected 5Y VORP",
            "overall_pick": "Actual Draft Pick",
            "college_bpm": "College BPM",
            "pts_per_40": "PTS / 40",
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
                "PTS / 40": st.column_config.NumberColumn(format="%.1f"),
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
        # --- FILTER & SEARCH BAR SECTION ---
        col_y, col_p, col_s = st.columns([1, 1, 2])

        with col_y:
            available_years = ["All"] + sorted(
                features_df["draft_year"].dropna().unique().astype(int),
                reverse=True,
            )
            selected_year = st.selectbox("Filter Draft Year", available_years, index=0)

        with col_p:
            available_positions = ["All"] + sorted(
                features_df["pos_group"].dropna().unique()
            )
            selected_pos = st.selectbox("Filter Position", available_positions, index=0)

        # Apply filters to build filtered player list
        filtered_df = features_df.copy()
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
                # Streamlit selectbox allows direct typing/searching
                selected_player = st.selectbox(
                    "Search / Select Prospect",
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
                st.write(f"**Points / 40:** {player_row.get('pts_per_40', 0.0):.1f}")
                st.write(f"**Rebounds / 40:** {player_row.get('reb_per_40', 0.0):.1f}")
                st.write(f"**Assists / 40:** {player_row.get('ast_per_40', 0.0):.1f}")
                st.write(f"**True Shooting %:** {player_row.get('ts_pct', 0.0):.3f}")

            with col_phys:
                st.subheader("Anthropometrics")

                # Converts raw inches (e.g. 81.0) to standard feet/inches format (e.g. 6'9")
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

            # Historical Comps Calculation
            ignore_cols = [
                "draft_player_name",
                "draft_year",
                "drafted_team",
                "is_training_cohort",
                "vorp_5y",
                "reached_min_threshold_5y",
                "player_tier_5y",
                "overall_pick",
                "pos_group",
            ]
            feat_cols = [
                c for c in features_df.columns
                if c not in ignore_cols and pd.api.types.is_numeric_dtype(features_df[c])
            ]

            # Filter historical training cohort for comparisons
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
    st.title("Historical Model Backtest (2009-2019)")
    st.markdown(
        "Out-of-fold validation and test performance evaluating the model's ranking ability against historical baseline draft order."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("2019 Test Hit Rate (Top 10 in 30)", "80.0%", delta="Baseline: 50.0%")
    m2.metric("2019 Test NDCG@10", "0.573", delta="+0.09 vs Draft")
    m3.metric("2019 Test NDCG@30", "0.709", delta="+0.06 vs Draft")
    m4.metric("Validation Spearman Rho", "0.314", delta="Stable")

    st.markdown("---")
    st.subheader("Cohort Metrics Summary")

    backtest_data = pd.DataFrame(
        [
            {
                "Cohort": "Validation (2017-2018)",
                "Spearman Rho": 0.314,
                "NDCG@10": 0.476,
                "NDCG@30": 0.609,
                "Hit Rate Top 10": "70.0%",
            },
            {
                "Cohort": "Test Set (2019)",
                "Spearman Rho": 0.317,
                "NDCG@10": 0.573,
                "NDCG@30": 0.709,
                "Hit Rate Top 10": "80.0%",
            },
        ]
    )

    st.dataframe(backtest_data, use_container_width=True, hide_index=True)


# ==========================================
# PAGE 5: ABOUT
# ==========================================
elif page == "About":
    st.title("Methodology & Architecture")

    st.markdown(
        """
        ### Methodology
        This NBA Draft Projection System generates 5-Year VORP (Value Over Replacement Player) forecasts 
        and calibrated rotation probabilities for prospects using strictly **pre-draft** inputs.

        #### Key Modeling Components:
        1. **70/30 Hybrid Ensemble:** Blends gradient boosted trees (LightGBM) with regularized linear regression (Ridge) to balance non-linear interaction learning with baseline stability.
        2. **Calibrated Floor Model:** Employs standardized Logistic Regression with Platt Scaling to calculate true rotation probability floor percentages.
        3. **Height-Gated Physical Scaling:** Applies non-linear continuous interaction scaling to physical metrics (`ape_index`) for undersized prospects to manage position penalties naturally.
        4. **Target Variable:** 5-Year Cumulative NBA VORP constructed with strict thresholding to eliminate career-length skew.
        
        #### Data Pipeline & Infrastructure:
        * **Storage & Warehouse:** Databricks Unity Catalog (`nba_draft.analytics`)
        * **Tracking:** Managed MLflow for hyperparameter & metric experiment tracking
        * **Storage Artifacts:** Local Parquet caching for optimized frontend instance performance
        """
    )