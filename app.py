# Streamlit App
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

# === Page Config ===
st.set_page_config(
    page_title="Türkiye Earthquakes (1915-2025)", layout="wide", page_icon="🌍"
)


# === Custom CSS for styling ===
st.markdown(
    """
    <style>
    body { background-color: #f9f9f9; }

    /* Title */
    .custom-title {
        font-size: 40px;
        color: #232f3e;
        text-align: left;
        margin-top: 10px;
        margin-bottom: 20px;
        font-weight: bold;
    }

    /* Tabs aligned right */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: flex-end;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 18px !important;
        padding: 10px 25px !important;
        color: #232f3e;
    }


    .kpi-value {
        font-size: 30px;
        font-weight: bold;
        color: #232f3e;
        margin: 0;
    }
    .kpi-label {
        font-size: 20px;
        color: #555555;
        margin: 4px 0 0 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# # ----------------------------------------------------------------------------------------------------------
# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_csv("data/turkey_earthquakes_historical.csv")
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
    df["Year"] = df["Datetime"].dt.year
    df["Month"] = df["Datetime"].dt.month
    return df


df = load_data()

# # ----------------------------------------------------------------------------------------------------------
# HEADER
st.markdown(
    """
    <div style='display: flex; align-items: center;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/b/b4/Flag_of_Turkey.svg' width='30' style='margin-right:10px;'/>
        <h1 style='margin: 0px; color: #232f3e;'>Türkiye Earthquakes (1915-2025)</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


# # ----------------------------------------------------------------------------------------------------------
# TABS
tabs = st.tabs(
    [
        "Overview & Temporal Analysis",
        "Magnitude & Depth",
        "Geographical Maps",
        "About Dataset",
    ]
)

# # ----------------------------------------------------------------------------------------------------------
# === SIDEBAR FILTERS ===
st.sidebar.header("🎯 Filters")
years = st.sidebar.slider(
    "Select Year Range:", int(df["Year"].min()), int(df["Year"].max()), (1990, 2025)
)
mag = st.sidebar.slider(
    "Select Magnitude Range:",
    float(df["Magnitude"].min()),
    float(df["Magnitude"].max()),
    (3.0, 8.0),
)
depth = st.sidebar.slider(
    "Select Depth Range (km):",
    float(df["Depth"].min()),
    float(df["Depth"].max()),
    (0.0, df["Depth"].max()),
)

# Apply filters
filtered_df = df[
    (df["Year"] >= years[0])
    & (df["Year"] <= years[1])
    & (df["Magnitude"] >= mag[0])
    & (df["Magnitude"] <= mag[1])
    & (df["Depth"] >= depth[0])
    & (df["Depth"] <= depth[1])
]

# # ----------------------------------------------------------------------------------------------------------
# === OVERVIEW TAB ===
with tabs[0]:
    st.markdown(
        """
    <h3 style='color:#232f3e !important; font-weight:bold;'>Overview</h3>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    # --- KPIs existentes ---
    with col1:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown(
            '<p class="kpi-label">Filtered Earthquakes</p>', unsafe_allow_html=True
        )
        st.markdown(
            f'<p class="kpi-value">{filtered_df.shape[0]:,}</p>', unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<p class="kpi-label">Avg Magnitude</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="kpi-value">{round(filtered_df["Magnitude"].mean(),2)}</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<p class="kpi-label">Max Magnitude</p>', unsafe_allow_html=True)
        if filtered_df.shape[0] > 0:
            max_mag = filtered_df["Magnitude"].max()
            max_loc = filtered_df.loc[filtered_df["Magnitude"].idxmax(), "Location"]

            # Limpiar texto (elimina todo entre [])
            max_loc_clean = re.sub(r"\[.*?\]", "", max_loc).strip()

            st.markdown(f'<p class="kpi-value">{max_mag}</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p style="font-size:14px; color:#555555; margin-top:4px;">{max_loc_clean}</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<p class="kpi-value">-</p>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<p class="kpi-label">Avg Depth</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="kpi-value">{round(filtered_df["Depth"].mean(),1)} km</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Quick Insights KPI ---
    with col5:
        yearly_counts = filtered_df.groupby("Year").size().reset_index(name="Count")
        if not yearly_counts.empty:
            max_year = yearly_counts.loc[yearly_counts["Count"].idxmax()]
            min_year = yearly_counts.loc[yearly_counts["Count"].idxmin()]
            avg_quakes = int(yearly_counts["Count"].mean())

            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.markdown(
                '<p class="kpi-label">Quick Insights</p>', unsafe_allow_html=True
            )

            st.markdown(
                f"<p style='font-size:16px; color:#232f3e;'>⭐ <b>Most Active:</b> {int(max_year['Year'])} ({max_year['Count']})</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='font-size:16px; color:#232f3e;'>🔻 <b>Least Active:</b> {int(min_year['Year'])} ({min_year['Count']})</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='font-size:16px; color:#232f3e;'>📊 <b>Avg/Year:</b> {avg_quakes}</p>",
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi-card">No data.</div>', unsafe_allow_html=True)

    # Separador
    st.markdown("<hr style='border:1px solid #cccccc;'>", unsafe_allow_html=True)

    # # ----------------------------------------------------------------------------------------------------------
    # GRAFICO (sin insights ahora)
    st.subheader("📊 Number of Earthquakes per Year (Filtered)")

    yearly_counts = filtered_df.groupby("Year").size().reset_index(name="Count")

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 5))

    # Línea principal
    sns.lineplot(
        data=yearly_counts,
        x="Year",
        y="Count",
        marker="o",
        color="tomato",
        linewidth=2.5,
        ax=ax,
    )

    # Línea de tendencia (regresión lineal)
    sns.regplot(
        data=yearly_counts,
        x="Year",
        y="Count",
        scatter=False,
        color="#232f3e",  # Oscuro, acorde Amazon
        line_kws={"linestyle": "dotted", "linewidth": 2},
        ax=ax,
    )

    # Estética
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Earthquakes", fontsize=12)
    ax.set_title("Trend of Earthquakes per Year", fontsize=14, color="#232f3e")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Más espaciado para evitar cortes
    plt.tight_layout()

    st.pyplot(fig)

# TABI  1 # ----------------------------------------------------------------------------------------------------------
# === MAGNITUDE & DEPTH TAB ===
with tabs[1]:
    st.markdown(
        """
    <h3 style='color:#232f3e !important; font-weight:bold;'>Magnitude & Depth</h3>
    """,
        unsafe_allow_html=True,
    )

    # ---- Primera fila: dos columnas superiores ----
    upper_col1, upper_col2 = st.columns(2)

    # === Scatterplot Magnitude over Time ===
    with upper_col1:
        st.subheader("📌 Earthquake Magnitude Over Time")
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.scatterplot(
            data=filtered_df,
            x="Year",
            y="Magnitude",
            alpha=0.4,
            s=20,
            ax=ax,
            color="tomato",
        )
        ax.set_title("Earthquake Magnitude Over Time", fontsize=12, color="#232f3e")
        ax.set_xlabel("Year")
        ax.set_ylabel("Magnitude")
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)

    # === Lineplot Average Magnitude per Year ===
    with upper_col2:
        st.subheader("📈 Average Magnitude per Year")
        yearly_magnitude = filtered_df.groupby("Year")["Magnitude"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(
            data=yearly_magnitude,
            x="Year",
            y="Magnitude",
            marker="o",
            color="tomato",
            ax=ax,
        )
        ax.set_title(
            "Average Earthquake Magnitude per Year", fontsize=12, color="#232f3e"
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Magnitude")
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)

    st.markdown("---")  # Separador

    # ---- Segunda fila: dos columnas inferiores ----
    lower_col1, lower_col2 = st.columns(2)

    # === Scatterplot Depth vs Magnitude by Decade ===
    with lower_col1:
        st.subheader("🌋 Depth vs Magnitude Density by Decade")

        filtered_df["Decade"] = (filtered_df["Year"] // 10) * 10
        filtered_df["Decade Label"] = filtered_df["Decade"].apply(
            lambda x: f"'{str(x)[2]}0s"
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        hb = ax.hexbin(
            x=filtered_df["Depth"],
            y=filtered_df["Magnitude"],
            gridsize=30,
            cmap="viridis",
            bins="log",
            alpha=0.95,
        )
        ax.set_xlabel("Depth (km)")
        ax.set_ylabel("Magnitude")
        ax.set_title("Depth vs Magnitude Density", fontsize=12, color="#232f3e")
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("log(count)")
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)

    # === Text Descriptions ===
    with lower_col2:
        st.markdown("<br><br>", unsafe_allow_html=True)  # Espacio superior

        st.markdown(
            """
            <div style='max-width:500px; margin-left:60px; text-align:center;'>
                <p style='font-size:16px; font-style:italic; color:#555555;'>
                📌 This scatterplot displays the distribution of earthquake magnitudes over time. It shows that, although lower-magnitude earthquakes are more frequent, significant high-magnitude events occur sporadically throughout the decades.
                </p>
                <p style='font-size:16px; font-style:italic; color:#555555;'>
                📈 The line graph illustrates the yearly evolution of the average earthquake magnitude. While the average remains relatively stable, occasional spikes suggest periods of increased seismic intensity.
                </p>
                <p style='font-size:16px; font-style:italic; color:#555555;'>
                🌋 This density plot explores the relationship between earthquake depth and magnitude. It highlights that most high-magnitude earthquakes are concentrated at shallow to intermediate depths.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================================================
# === GEOGRAPHICAL MAPS TAB ===
with tabs[2]:
    st.markdown(
        """
    <h3 style='color:#232f3e !important; font-weight:bold;'>Geographical Maps</h3>
    """,
        unsafe_allow_html=True,
    )

    # --- Density Mapbox + Fault Lines ---
    # st.subheader("🌍 Earthquake Density Map of Türkiye with Tectonic Faults")

    fig = px.density_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        z="Magnitude",
        radius=8,
        center=dict(lat=39, lon=35),
        zoom=5,
        mapbox_style="carto-positron",
        color_continuous_scale="YlOrRd",
    )

    # Coordenadas fallas
    naf_latitudes = [39.3, 40.0, 40.5, 40.7]
    naf_longitudes = [40.0, 38.5, 28.0, 27.0]
    eaf_latitudes = [37.0, 38.0, 39.0, 39.3]
    eaf_longitudes = [37.0, 38.0, 39.0, 40.0]

    # Añadir fallas
    fig.add_trace(
        go.Scattermapbox(
            mode="lines",
            lon=naf_longitudes,
            lat=naf_latitudes,
            line=dict(width=2, color="blue"),
            name="North Anatolian Fault",
        )
    )
    fig.add_trace(
        go.Scattermapbox(
            mode="lines",
            lon=eaf_longitudes,
            lat=eaf_latitudes,
            line=dict(width=2, color="green"),
            name="East Anatolian Fault",
        )
    )

    fig.update_layout(
        title="🌍 Earthquake Density Map of Türkiye with Tectonic Faults",
        height=900,
        title_font=dict(
            size=26, color="#232f3e", family="Arial"
        ),  # Tamaño y color editable
        title_x=0.0,  # Alineado a la izquierda (0.5 es centrado)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Frase descriptiva debajo
    st.markdown(
        """
        <p style='font-size:16px; font-style:italic; color:#555555; margin-top:-75px;'>
        🌍 This map illustrates the density of earthquakes across Türkiye, highlighting tectonic fault lines such as the North Anatolian and East Anatolian faults.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border:1px solid #cccccc;'>", unsafe_allow_html=True)

    # --- Interactive Map with Animation ---
    # st.subheader("⏳ Earthquake Timeline Animation")

    fig2 = px.scatter_mapbox(
        df.sort_values("Year"),
        lat="Latitude",
        lon="Longitude",
        color="Magnitude",
        size="Magnitude",
        animation_frame="Year",
        center=dict(lat=39, lon=35),
        zoom=5,
        mapbox_style="carto-positron",
        color_continuous_scale="YlOrRd",
        title="⏳ Earthquake Timeline Animation",
    )
    fig2.update_layout(
        height=800,
        title_font=dict(
            size=26, color="#232f3e", family="Arial"
        ),  # Tamaño y color editable
        title_x=0.0,  # Alineado a la izquierda (0.5 es centrado)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        """
        <p style='font-size:16px; font-style:italic; color:#555555;margin-top:-10px;'>
        ⏳ This animated map shows how earthquake events unfolded across Türkiye over the decades.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border:1px solid #cccccc;'>", unsafe_allow_html=True)

    # --- K-Means Cluster Map ---

    # st.markdown(
    #     """
    #     <h4 style='color:#232f3e; margin-bottom: 1px;'>📍 Geographical Clusters of Earthquakes</h4>
    #     """,
    #     unsafe_allow_html=True,
    # )

    X = df[["Latitude", "Longitude"]]
    k = 6
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X)

    fig3 = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color="Cluster",
        color_continuous_scale="Plasma",
        center=dict(lat=39, lon=35),
        zoom=5,
        mapbox_style="carto-positron",
        title="📍 Geographical Clusters of Earthquakes",
    )
    fig3.update_layout(
        height=800,
        title_font=dict(
            size=26, color="#232f3e", family="Arial"
        ),  # Tamaño y color editable
        title_x=0.0,  # Alineado a la izquierda (0.5 es centrado)
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        """
        <p style='font-size:16px; font-style:italic; color:#555555;margin-top:-75px;'>
        📍 This map identifies key geographical clusters of seismic activity in Türkiye using K-Means clustering.
        </p>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================================================
# === ABOUT DATASET TAB ===
with tabs[3]:
    st.markdown(
        """
        <h3 style='color:#232f3e !important; font-weight:bold;'>About Dataset</h3>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        This dataset contains historical records of earthquakes that occurred in Türkiye between **1915 and 2025**, compiled primarily from the **Kandilli Observatory and Earthquake Research Institute**.

        **Key Columns:**
        - 📅 **Date and Time** of each earthquake.
        - 🌍 **Latitude & Longitude** coordinates.
        - 📊 **Magnitude** and **Depth**.
        - 🏷️ **Location** description.
        
        **Data Processing Steps:**
        - Parsing of date and time fields.
        - Removal of inconsistent and null entries.
        - Feature extraction: Year, Month, and Decade categorization.
        - Addition of derived columns for enhanced temporal and geospatial analysis.
        """
    )

    st.markdown("---")

    st.markdown("### 🌍 Tectonic Context of Türkiye")

    # Crear columnas
    col1, col2 = st.columns([1, 3])  # Proporción 1:3 (imagen más pequeña)

    with col1:
        st.image("images/Turkey-faults-platenames.png", width=300)

    with col2:
        st.markdown(
            """
            Türkiye is located on the **Anatolian Plate**, bordered by the **Eurasian Plate** to the north,
            the **Arabian Plate** to the southeast, and the **African Plate** to the southwest.  
            This tectonic setting makes Türkiye highly prone to seismic activity, particularly along the
            **North Anatolian Fault (NAF)** and **East Anatolian Fault (EAF)**, which traverse the country.
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- Extra Info Box below tectonic map ---

    st.markdown(
        """
        <div style='background-color:rgba(255, 99, 71, 0.05); padding:15px; border-radius:10px;'>
            <h4 style='color:#232f3e;'>🎯 Project Purpose & Methodology</h4>
            <ul style='color:#232f3e;'>
                <li><strong>Objectives:</strong> Provide an exploratory analysis of Türkiye's seismic activity and offer interactive, public-friendly insights.</li>
                <li><strong>Data Treatment:</strong> Cleaned and standardized fields (dates, depth, locations), extracted features (year, month, decade), integrated tectonic faults.</li>
                <li><strong>Tools:</strong> Python, Pandas, Seaborn, Plotly, Streamlit, Machine Learning clustering techniques.</li>
                <li><strong>Educational Purpose:</strong> This dashboard was developed as part of a Data Analytics learning project, aiming to raise awareness while reinforcing technical skills.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
