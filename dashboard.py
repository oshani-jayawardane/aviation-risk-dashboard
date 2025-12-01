import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import ast
from collections import Counter


filename = "cleanest_aircraft_data_without_NLP.csv"

from PIL import Image

PLANE_IMG_PATH = "plane_top.png"   # <-- put your PNG here

@st.cache_data
def load_plane_image():
    return Image.open(PLANE_IMG_PATH)


# -------------------------------------------------
# 1. Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Aviation Risk Analysis Dashboard",
    layout="wide"
)

st.markdown(
    """
    <style>
        /* remove extra top padding */
        .block-container {
            padding-top: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# 2. Data loading
# -------------------------------------------------

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    # If Year exists but has missing values, try to infer from Incident_Date
    if "Incident_Date" in df.columns:
        df["Incident_Date"] = pd.to_datetime(df["Incident_Date"], errors="coerce")
    if "Year" not in df.columns and "Incident_Date" in df.columns:
        df["Year"] = df["Incident_Date"].dt.year
    return df

df = load_data(filename)

df["Year"] = df["Year"].astype("Int64")
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())

# ---------------------------------------------
# Initialize filter state for use in the title
# ---------------------------------------------
if "filter_mode" not in st.session_state:
    st.session_state["filter_mode"] = "Year Range"

if "selected_year" not in st.session_state:
    st.session_state["selected_year"] = max_year

if "selected_range" not in st.session_state:
    st.session_state["selected_range"] = (min_year, max_year)

fm = st.session_state["filter_mode"]
sy = st.session_state["selected_year"]
sr = st.session_state["selected_range"]

if fm == "Single Year":
    period_label = str(sy)
else:
    period_label = f"{sr[0]} – {sr[1]}"

# -------------------------------------------------
# Dummy Damage_Areas for prototyping
# -------------------------------------------------
damage_patterns = [
    [],                              # no specific area recorded
    ["engine"],
    ["wing"],
    ["tail"],
    ["nose"],
    ["fuselage"],
    ["landing gear"],
    ["engine", "wing"],
    ["engine", "tail"],
    ["wing", "fuselage"],
]

# simple deterministic assignment, cycles through patterns
df["Damage_Areas"] = [
    damage_patterns[i % len(damage_patterns)] for i in range(len(df))
]

# dummy for flags

np.random.seed(42)

cause_cols = [
    "is_engine_cause",
    "is_model_cause",
    "is_human_cause",
    "is_weather_cause",
    "is_wildlife_cause",
    "is_ground_collision",
    "is_unknown_cause",
]

# random 0/1 with ~30% chance of being 1
rand_matrix = np.random.rand(len(df), len(cause_cols))
df[cause_cols] = (rand_matrix < 0.3).astype(int)

# ensure at least one cause per row → if all zero, mark unknown as 1
mask_no_cause = df[cause_cols].sum(axis=1) == 0
df.loc[mask_no_cause, "is_unknown_cause"] = 1




# -------------------------------------------------
# 3. Title
# -------------------------------------------------

# Title
# Title (reacts to current filter state)
st.markdown(
    f"""
    <h2 style='text-align: center; margin-bottom: 0.3rem; margin-top: 0rem;'>
        Aviation Risk Analysis Dashboard ({period_label})
    </h2>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<hr style='margin-top:0.2rem;margin-bottom:0.8rem;'>",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# 4. Compact Filters (Year + Nature)
# -------------------------------------------------

# Group aircraft nature (only once)
def map_nature_group(x: str) -> str:
    if x == "Passenger":
        return "Passenger"
    elif x == "Cargo":
        return "Cargo"
    elif x == "Private":
        return "Private"
    elif x in ["Military", "Special Operations"]:
        return "Military"
    elif x == "Illegal/Unauthorized":
        return "Unauthorized"
    else:
        return "Other"

df["Nature_Group"] = df["Aircaft_Nature"].astype(str).apply(map_nature_group)

# ---------------- Horizontal filter layout ----------------
col3, col1, col2 = st.columns([6, 2, 3])

# Filter mode: single year or range
with col1:
    filter_mode = st.radio(
        "Select Filter:",
        ["Single Year", "Year Range"],
        horizontal=True,
        index=1,
        key="filter_mode",
    )

# Year slider
with col2:
    if filter_mode == "Single Year":
        selected_year = st.slider(
            "Year",
            min_value=min_year,
            max_value=max_year,
            value=max_year,
            step=1,
            key="selected_year",
            # label_visibility="collapsed",
        )
        year_mask = df["Year"] == selected_year
    else:
        selected_range = st.slider(
            "Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            step=1,
            key="selected_range",
            # label_visibility="collapsed",
        )
        year_mask = df["Year"].between(selected_range[0], selected_range[1])

year_filtered_df = df[year_mask].copy()

# Nature filter checkboxes
with col3:
    st.markdown("**Aircraft Nature:**")
    group_values = [
        g for g in [
            "Passenger",
            "Cargo",
            "Private",
            "Military",
            "Unauthorized",
            "Other",
        ]
        if g in year_filtered_df["Nature_Group"].unique()
    ]

    nature_cols = st.columns(len(group_values)) if len(group_values) > 0 else []
    selected_groups = []

    for n_col, g in zip(nature_cols, group_values):
        with n_col:
            if st.checkbox(g, value=True, key=f"group_{g}"):
                selected_groups.append(g)

# Apply group filter
if selected_groups:
    filtered_df = year_filtered_df[
        year_filtered_df["Nature_Group"].isin(selected_groups)
    ].copy()
else:
    filtered_df = year_filtered_df.copy()


# -------------------------------------------------
# View routing (main dashboard vs details page)
# -------------------------------------------------
if "view" not in st.session_state:
    st.session_state["view"] = "main"

if st.session_state["view"] == "details":
    # st.title("Accident Cause Details")

    # st.markdown(
    #     """
    #     Breakdown of incidents by engine / aircraft models and operators
    #     when specific causes are flagged as 1.
    #     """,
    # )

    col1, col2, col3 = st.columns(3)

    # =========================================================
    # 1) Engine models when is_engine_cause = 1
    # =========================================================
    with col1:
        st.subheader("Engine Failure by Engine Model")

        if "Engine_Model_Standardized" in df.columns and "is_engine_cause" in df.columns:
            eng_fail = df[df["is_engine_cause"] == 1].copy()
            if eng_fail.empty:
                st.info("No incidents with engine failure cause flagged.")
            else:
                top_eng = (
                    eng_fail.groupby("Engine_Model_Standardized")
                            .size()
                            .reset_index(name="Count")
                            .sort_values("Count", ascending=False)
                            .head(10)
                )

                fig_eng = px.bar(
                    top_eng,
                    x="Count",
                    y="Engine_Model_Standardized",
                    orientation="h",
                    text="Count",
                )
                fig_eng.update_layout(
                    xaxis_title="Incidents with Engine Failure",
                    yaxis_title="Engine Model",
                    height=350,
                    margin=dict(l=10, r=10, t=40, b=30),
                )
                fig_eng.update_traces(textposition="outside")
                st.plotly_chart(fig_eng, use_container_width=True, config={"staticPlot": True})

                # ---------- Pie chart: Engine Manufacturer ----------
                st.markdown("**Engine manufacturers in engine-failure incidents**")
                if "Engine_Manufacturer" in eng_fail.columns:
                    manu_eng = (
                        eng_fail.groupby("Engine_Manufacturer")
                                .size()
                                .reset_index(name="Count")
                                .sort_values("Count", ascending=False)
                    )
                    # optional: limit to top 8 for readability
                    manu_eng_top = manu_eng.head(8)

                    fig_eng_pie = px.pie(
                        manu_eng_top,
                        names="Engine_Manufacturer",
                        values="Count",
                        hole=0.35,
                    )
                    fig_eng_pie.update_layout(
                        height=300,
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig_eng_pie, use_container_width=True, config={"staticPlot": True})
                else:
                    st.info("Engine_Manufacturer column not available for pie chart.")
        else:
            st.info("Required columns for engine model analysis are missing.")

    # =========================================================
    # 2) Aircraft models when is_model_cause = 1
    # =========================================================
    with col2:
        st.subheader("Aircraft Design Issues by Model")

        if "Aircaft_Model" in df.columns and "is_model_cause" in df.columns:
            model_fail = df[df["is_model_cause"] == 1].copy()
            if model_fail.empty:
                st.info("No incidents with aircraft design cause flagged.")
            else:
                top_models = (
                    model_fail.groupby("Aircaft_Model")
                              .size()
                              .reset_index(name="Count")
                              .sort_values("Count", ascending=False)
                              .head(10)
                )

                fig_model = px.bar(
                    top_models,
                    x="Count",
                    y="Aircaft_Model",
                    orientation="h",
                    text="Count",
                )
                fig_model.update_layout(
                    xaxis_title="Incidents with Aircraft Design Cause",
                    yaxis_title="Aircraft Model",
                    height=350,
                    margin=dict(l=10, r=10, t=40, b=30),
                )
                fig_model.update_traces(textposition="outside")
                st.plotly_chart(fig_model, use_container_width=True, config={"staticPlot": True})

                # ---------- Pie chart: Aircraft Manufacturer ----------
                st.markdown("**Aircraft manufacturers in design-related incidents**")
                if "Aircraft_Manufacturer" in model_fail.columns:
                    manu_air = (
                        model_fail.groupby("Aircraft_Manufacturer")
                                  .size()
                                  .reset_index(name="Count")
                                  .sort_values("Count", ascending=False)
                    )
                    manu_air_top = manu_air.head(8)

                    fig_air_pie = px.pie(
                        manu_air_top,
                        names="Aircraft_Manufacturer",
                        values="Count",
                        hole=0.35,
                    )
                    fig_air_pie.update_layout(
                        height=300,
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig_air_pie, use_container_width=True, config={"staticPlot": True})
                else:
                    st.info("Aircraft_Manufacturer column not available for pie chart.")
        else:
            st.info("Required columns for aircraft model analysis are missing.")

    # =========================================================
    # 3) Operators vs engine / model / human causes
    #    (stacked horizontal bar)
    # =========================================================
    with col3:
        st.subheader("Operators vs Key Causes")

        needed_cols = [
            "Aircaft_Operator",
            "is_engine_cause",
            "is_model_cause",
            "is_human_cause",
        ]
        if all(c in df.columns for c in needed_cols):
            op_group = (
                df.groupby("Aircaft_Operator")
                  .agg(
                      engine_failures=("is_engine_cause", "sum"),
                      model_issues=("is_model_cause", "sum"),
                      human_errors=("is_human_cause", "sum"),
                  )
            )
            op_group["total"] = (
                op_group["engine_failures"]
                + op_group["model_issues"]
                + op_group["human_errors"]
            )

            op_group = op_group[op_group["total"] > 0]
            if op_group.empty:
                st.info("No operators with these causes flagged.")
            else:
                op_top = op_group.sort_values("total", ascending=False).head(10)

                op_melt = (
                    op_top[["engine_failures", "model_issues", "human_errors"]]
                    .reset_index()
                    .melt(
                        id_vars="Aircaft_Operator",
                        var_name="Cause",
                        value_name="Count",
                    )
                )

                cause_label_map = {
                    "engine_failures": "Engine Failure",
                    "model_issues": "Aircraft Design",
                    "human_errors": "Human Error",
                }
                op_melt["Cause"] = op_melt["Cause"].map(cause_label_map)

                fig_op = px.bar(
                    op_melt,
                    x="Count",
                    y="Aircaft_Operator",
                    color="Cause",
                    orientation="h",
                    text="Count",
                )
                fig_op.update_layout(
                    xaxis_title="Incidents (flagged cause = 1)",
                    yaxis_title="Operator",
                    height=350,
                    margin=dict(l=10, r=10, t=40, b=30),
                    legend_title="Cause Type",
                )
                fig_op.update_traces(textposition="inside", insidetextanchor="middle")
                st.plotly_chart(fig_op, use_container_width=True, config={"staticPlot": True})
        else:
            st.info("Required columns for operator analysis are missing.")

        # back button
        if st.button("⬅ Back to main dashboard"):
            st.session_state["view"] = "main"
            st.rerun()

    st.stop()



# -------------------------------------------------
# 5. map and analytics to the right
# -------------------------------------------------

map_col, right_col = st.columns([4, 6])

with map_col:
    st.subheader("Global Incident Locations")

    map_df = filtered_df.copy()
    map_df["Latitude"] = pd.to_numeric(map_df["Latitude"], errors="coerce")
    map_df["Longitude"] = pd.to_numeric(map_df["Longitude"], errors="coerce")
    map_df = map_df.dropna(subset=["Latitude", "Longitude"])

    if not map_df.empty:
        # Color by Aircaft_Damage_Type (severity)
        color_map = {
            "Destroyed":   [220, 20, 60, 180],   # strong red
            "Missing":     [139, 0, 0, 180],     # dark red
            "Substantial": [255, 140, 0, 180],   # orange
            "Minor":       [255, 215, 0, 180],   # yellow
            "Repairable":  [60, 179, 113, 180],  # green
            "Unknown":     [160, 160, 160, 180], # gray
        }

        map_df["color"] = map_df["Aircaft_Damage_Type"].map(color_map)
        map_df["color"] = map_df["color"].apply(
            lambda c: c if isinstance(c, list) else [160, 160, 160, 160]
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[Longitude, Latitude]",
            get_fill_color="color",
            get_radius=35000,
            radius_min_pixels=3,
            radius_max_pixels=25,
            pickable=True,
        )

        center_lat = map_df["Latitude"].mean()
        center_lon = map_df["Longitude"].mean()

        view_state = pdk.ViewState(
            latitude=float(center_lat),
            longitude=float(center_lon),
            zoom=1,
            min_zoom=1,
            max_zoom=8,
        )

        tooltip = {
            "text": "Location: {Incident_Location}\n"
                    "Nature: {Aircaft_Nature}\n"
                    "Damage: {Aircaft_Damage_Type}"
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
            ),
            height=380
        )

        # ------- Color Legend under the map -------
        legend_items = [
            ("Destroyed",   "rgb(220, 20, 60)"),   # strong red
            ("Missing",     "rgb(139, 0, 0)"),     # dark red
            ("Substantial", "rgb(255, 140, 0)"),   # orange
            ("Minor",       "rgb(255, 215, 0)"),   # yellow
            ("Repairable",  "rgb(60, 179, 113)"),  # green
            ("Unknown",     "rgb(160, 160, 160)")  # grey
        ]

        cols = st.columns(len(legend_items))
        for col, (label, color) in zip(cols, legend_items):
            col.markdown(
                f"""
                <div style="
                    display:flex;
                    align-items:center;
                    gap:6px;
                ">
                    <div style="
                        width:14px;
                        height:14px;
                        background-color:{color};
                        border-radius:2px;
                        border:1px solid #444;
                    "></div>
                    <div style="font-size:0.78rem;">
                        {label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No incident locations available for the selected filters.")

    # ==========================
    # Overall statistics + KPI boxes
    # ==========================
    if filter_mode == "Single Year":
        period_label = str(selected_year)
    else:
        start, end = selected_range
        period_label = f"{start} – {end}"

    st.markdown(
        f"<h4 style='margin-top:0.8rem; margin-bottom:0.6rem;'>"
        f"Overall Statistics ({period_label})"
        f"</h4>",
        unsafe_allow_html=True,
    )

    total_events = len(filtered_df)

    total_fatalities = (
        int(filtered_df["Fatalities"].fillna(0).sum())
        if "Fatalities" in filtered_df.columns
        else 0
    )

    total_ground_fatalities = (
        int(filtered_df["Ground_Casualties"].fillna(0).sum())
        if "Ground_Casualties" in filtered_df.columns
        else 0
    )

    kpi1, kpi2, kpi3 = st.columns(3)

    def kpi_box(col, value, main_label, color):
        box_html = f"""
        <div style="
            background-color:{color};
            padding:0.6rem 0.4rem;
            border-radius:0.4rem;
            text-align:center;
        ">
            <div style="font-size:1.9rem; font-weight:700; color:white;">
                {value:,}
            </div>
            <div style="font-size:0.8rem; color:white; text-transform:uppercase; letter-spacing:0.06em;">
                {main_label}
            </div>
        </div>
        """
        with col:
            st.markdown(box_html, unsafe_allow_html=True)

    kpi_box(
        kpi1,
        total_events,
        "EVENTS",
        "#1f77b4",   # light blue
    )
    kpi_box(
        kpi2,
        total_fatalities,
        "FLIGHT FATALITIES",
        "#1864aa",   # mid blue
    )
    kpi_box(
        kpi3,
        total_ground_fatalities,
        "GROUND FATALITIES",
        "#0f4c81",   # dark blue
    )

    # -------- Fatalities per incident sentence --------
    if total_events > 0:
        fatalities_per_incident = (total_fatalities + total_ground_fatalities) / total_events
        st.markdown(
            f"""
            <p style="text-align:center; font-size:0.9rem; margin-top:0.5rem;">
                Approximately
                <b>{fatalities_per_incident:.2f}</b> fatalities per incident in <b>{period_label}</b>.
            </p>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <p style="text-align:center; font-size:0.9rem; margin-top:0.5rem;">
                No incidents recorded for this selection.
            </p>
            """,
            unsafe_allow_html=True,
        )


###############################################################################################
with right_col:
    # Split right_col into 2 sub-columns
    trend_col, phase_col = st.columns([8, 6])

    # ==========================
    # 1) Incidents vs Year (left)
    # ==========================
    with trend_col:
        st.subheader("Incidents Over Time by Severity")

        if "Year" in filtered_df.columns and "Aircaft_Damage_Type" in filtered_df.columns:
            # Map raw damage types to severity bands
            def map_severity(d):
                if d in ["Minor", "Repairable"]:
                    return "Minor / Repairable"
                elif d == "Substantial":
                    return "Substantial"
                elif d in ["Destroyed", "Missing"]:
                    return "Destroyed / Missing"
                else:
                    return "Unknown"

            tmp = filtered_df.dropna(subset=["Year"]).copy()
            tmp["Damage_Severity"] = tmp["Aircaft_Damage_Type"].apply(map_severity)

            damage_trend = (
                tmp.groupby(["Year", "Damage_Severity"])
                   .size()
                   .reset_index(name="Count")
                   .sort_values(["Year", "Damage_Severity"])
            )

            if damage_trend.empty:
                st.info("No incidents for the selected filters.")
            else:
                severity_order = [
                    "Minor / Repairable",
                    "Substantial",
                    "Destroyed / Missing",
                    "Unknown",
                ]

                color_map = {
                    "Minor / Repairable": "#2ca02c",   # green
                    "Substantial": "#ff7f0e",          # orange
                    "Destroyed / Missing": "#d62728",  # red
                    "Unknown": "#7f7f7f",              # grey
                }

                fig_trend = px.area(
                    damage_trend,
                    x="Year",
                    y="Count",
                    color="Damage_Severity",
                    category_orders={"Damage_Severity": severity_order},
                    color_discrete_map=color_map,
                )

                fig_trend.update_layout(
                    margin=dict(l=10, r=10, t=10, b=0),
                    xaxis_title="Year",
                    yaxis_title="Incident Count",
                    height=260,
                )

                # y-axis from 0 with small headroom
                yearly_totals = (
                    damage_trend.groupby("Year")["Count"].sum().reset_index()
                )
                ymax = yearly_totals["Count"].max()
                fig_trend.update_yaxes(range=[0, ymax * 1.1])

                st.plotly_chart(fig_trend, use_container_width=True, config={"staticPlot": True})
        else:
            st.info("No year or damage data available.")

        # ==============================
        # Cause distribution bar chart
        # ==============================
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Likely Accident Causes")

        cause_cols = [
            "is_engine_cause",
            "is_model_cause",
            "is_human_cause",
            "is_weather_cause",
            "is_wildlife_cause",
            "is_ground_collision",
            "is_unknown_cause",
        ]

        # pretty display labels (in the exact order you requested)
        label_map = {
            "is_engine_cause": "Engine Failure",
            "is_model_cause": "Aircraft Design",
            "is_human_cause": "Human Error",
            "is_weather_cause": "Bad Weather",
            "is_wildlife_cause": "Wildlife Strike",
            "is_ground_collision": "Ground Collision",
            "is_unknown_cause": "Unknown Cause",
        }

        existing_cols = [c for c in cause_cols if c in filtered_df.columns]

        if not existing_cols:
            st.info("No cause flags available for the current dataset.")
        else:
            cause_counts = filtered_df[existing_cols].sum().astype(int)

            cause_df = (
                cause_counts
                .reset_index()
                .rename(columns={"index": "Flag", 0: "Count"})
            )
            cause_df["Cause"] = cause_df["Flag"].map(label_map)

            # apply the fixed display order
            cause_df["Cause"] = pd.Categorical(
                cause_df["Cause"],
                categories=[
                    "Engine Failure",
                    "Aircraft Design",
                    "Human Error",
                    "Bad Weather",
                    "Wildlife Strike",
                    "Ground Collision",
                    "Unknown Cause",
                ],
                ordered=True,
            )
            cause_df = cause_df.sort_values("Cause")

            # nice sequential colors
            sequential_colors = px.colors.sequential.Blues[2:]   # skip pale ones

            fig_cause = px.bar(
                cause_df,
                x="Cause",
                y="Count",
                text="Count",
                color="Cause",
                color_discrete_sequence=sequential_colors,
            )
            fig_cause.update_traces(textposition="outside")
            fig_cause.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=5, b=60),
                xaxis_title="",
                yaxis_title="Number of Incidents",
                showlegend=False,
            )

            st.plotly_chart(fig_cause, use_container_width=True, config={"staticPlot": True})

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if st.button("More details on accident causes"):
                    st.session_state["view"] = "details"
                    st.rerun()



    # ==============================
    # 2) Phase arc infographic (right)
    # ==============================
    with phase_col:
        st.subheader("Incident Profile by Flight Phase")

        if "Aircraft_Phase" in filtered_df.columns:
            # Map detailed phases to clean buckets
            def map_phase(p):
                if p in ["Ground Handling", "Taxi", "Standing"]:
                    return "Ground"
                elif p == "Takeoff":
                    return "Takeoff"
                elif p == "Initial Climb":
                    return "Initial Climb"
                elif p == "En Route":
                    return "En Route"
                elif p == "Approach":
                    return "Approach"
                elif p == "Landing":
                    return "Landing"
                else:
                    return "Other"

            phase_series = (
                filtered_df["Aircraft_Phase"]
                .fillna("Unknown")
                .apply(map_phase)
            )

            phase_counts = (
                phase_series.value_counts()
                .rename_axis("Phase")
                .reset_index(name="Count")
            )

            phase_order = [
                "Ground",
                "Takeoff",
                "Initial Climb",
                "En Route",
                "Approach",
                "Landing",
                "Other",
            ]
            phase_counts["Phase"] = pd.Categorical(
                phase_counts["Phase"],
                categories=phase_order,
                ordered=True,
            )
            phase_counts = (
                phase_counts.sort_values("Phase")
                .dropna(subset=["Phase"])
            )

            if phase_counts.empty:
                st.info("No phase-of-flight data for current filters.")
            else:
                phases = phase_counts["Phase"].tolist()
                counts = phase_counts["Count"].tolist()
                n = len(phases)

                # Arc coordinates
                xs_arc = np.linspace(0, n - 1, 200)
                mid = (n - 1) / 2
                height = 3.0
                ys_arc = -((xs_arc - mid) ** 2) + height

                fig = go.Figure()

                # Arc line
                fig.add_trace(
                    go.Scatter(
                        x=xs_arc,
                        y=ys_arc,
                        mode="lines",
                        line=dict(width=3),
                        showlegend=False,
                    )
                )

                # Use the darker half of the Blues palette for better contrast
                palette = px.colors.sequential.YlOrRd[2:]
                c_min, c_max = min(counts), max(counts)

                for i, (phase, count) in enumerate(zip(phases, counts)):
                    y_arc = float(np.interp(i, xs_arc, ys_arc))

                    if c_max == c_min:
                        idx = len(palette) // 2
                    else:
                        norm = (count - c_min) / (c_max - c_min)
                        idx = int(norm * (len(palette) - 1))
                    box_color = palette[idx]

                    # Vertical line from arc to box
                    fig.add_shape(
                        type="line",
                        x0=i,
                        y0=y_arc,
                        x1=i,
                        y1=y_arc + 0.5,
                        line=dict(color="gray", width=2),
                    )

                    # Box
                    fig.add_shape(
                        type="rect",
                        x0=i - 0.35,
                        x1=i + 0.35,
                        y0=y_arc + 0.5,
                        y1=y_arc + 1.2,
                        fillcolor=box_color,
                        line=dict(width=0),
                        opacity=0.95,
                    )

                    # Count text inside box (white – boxes are now darker)
                    fig.add_annotation(
                        x=i,
                        y=y_arc + 0.85,
                        text=f"<b>{count}</b>",
                        showarrow=False,
                        font=dict(size=12, color="black"),
                    )

                    # Phase label below arc
                    fig.add_annotation(
                        x=i,
                        y=y_arc - 0.3,
                        text=phase,
                        showarrow=False,
                        font=dict(size=11, color="white"),
                    )

                fig.update_xaxes(visible=False)
                fig.update_yaxes(visible=False)
                fig.update_layout(
                    height=240,
                    margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})
        else:
            st.info("No phase-of-flight information available.")

        
        # --------------------------------------------
        # Damage hotspot plane heatmap
        # --------------------------------------------

        st.subheader("Frequent Damage Hotspots")

        if "Damage_Areas" not in filtered_df.columns:
            st.info("No damage area information available in the current dataset.")
        else:
            area_counter = Counter()

            # ---------- normalize raw area strings/lists ----------
            def normalize_areas(val):
                if val is None:
                    return []
                if isinstance(val, float) and pd.isna(val):
                    return []
                if isinstance(val, (list, tuple)) and len(val) == 0:
                    return []
                if hasattr(val, "size") and getattr(val, "size", None) == 0:
                    return []

                if isinstance(val, (list, tuple)):
                    raw_list = val
                else:
                    try:
                        raw_list = ast.literal_eval(val)
                    except Exception:
                        raw_list = [str(val)]

                normed = []
                for a in raw_list:
                    s = str(a).strip().lower()
                    if "engine" in s:
                        normed.append("Engine")
                    elif "wing" in s:
                        normed.append("Wing")
                    elif "tail" in s or "empennage" in s:
                        normed.append("Tail")
                    elif "nose" in s or "cockpit" in s:
                        normed.append("Nose")
                    elif "fuselage" in s or "body" in s:
                        normed.append("Fuselage")
                    elif "gear" in s or "landing" in s:
                        normed.append("Landing Gear")
                    else:
                        normed.append("Other")
                return normed

            for v in filtered_df["Damage_Areas"]:
                for area in normalize_areas(v):
                    area_counter[area] += 1

            if not area_counter:
                st.info("No damage areas recorded for the selected filters.")
            else:
                # canonical zones we’ll show
                logical_zones = ["Nose", "Fuselage", "Tail", "Wing", "Engine", "Landing Gear"]

                counts = {z: area_counter.get(z, 0) for z in logical_zones}
                max_c = max(counts.values()) if max(counts.values()) > 0 else 1

                # ---------- plane image as background ----------
                plane_img = load_plane_image()

                fig = go.Figure()
                fig.add_layout_image(
                    dict(
                        source=plane_img,
                        xref="x",
                        yref="y",
                        x=0,
                        y=1,
                        sizex=1,
                        sizey=1,
                        sizing="stretch",
                        layer="below",
                    )
                )

                fig.update_xaxes(visible=False, range=[0, 1])
                fig.update_yaxes(visible=False, range=[0, 1], scaleanchor="x", scaleratio=1)

                fig.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )

                # ---------- define blob positions on the image ----------
                # coordinates in normalized [0,1] plane space
                blob_positions = {
                    "Nose":        (0.5, 0.90),
                    "Fuselage":    (0.5, 0.60),
                    "Tail":        (0.5, 0.15),
                    "Left Wing":   (0.20, 0.55),
                    "Right Wing":  (0.80, 0.55),
                    "Engines":     (0.5, 0.68),
                    "Landing Gear":(0.5, 0.28),
                }

                # map counts to those positions
                zone_counts_for_plot = {
                    "Nose":        counts["Nose"],
                    "Fuselage":    counts["Fuselage"],
                    "Tail":        counts["Tail"],
                    "Left Wing":   counts["Wing"],
                    "Right Wing":  counts["Wing"],
                    "Engines":     counts["Engine"],
                    "Landing Gear":counts["Landing Gear"],
                }

                names = []
                xs = []
                ys = []
                vals = []

                for name, (x, y) in blob_positions.items():
                    names.append(name)
                    xs.append(x)
                    ys.append(y)
                    vals.append(zone_counts_for_plot.get(name, 0))

                max_val = max(vals) if max(vals) > 0 else 1

                # blob sizes: base + scaled
                sizes = [25 + 40 * (v / max_val) if v > 0 else 0 for v in vals]

                # ---------- add blobs ----------
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers+text",
                        text=[f"{n}<br>{v}" for n, v in zip(names, vals)],
                        textposition="middle center",
                        hovertemplate="%{text}<extra></extra>",
                        marker=dict(
                            size=sizes,
                            color=vals,
                            colorscale="Reds",
                            cmin=0,
                            cmax=max_val,
                            opacity=0.75,
                            line=dict(width=0),
                        ),
                        textfont=dict(color="white", size=10),
                    )
                )

                st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})



