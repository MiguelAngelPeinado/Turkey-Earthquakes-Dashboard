# === README.md ===

"""
# Turkey Earthquakes Dashboard (1915-2025) 🌍

An interactive Streamlit dashboard visualizing historical earthquake data in Türkiye from **1915 to 2025**.

## Overview

This project provides a comprehensive analysis of Türkiye's seismic activity, highlighting patterns, magnitude evolution, depth distribution, and geospatial clusters. The dashboard is designed for both educational and public awareness purposes.

## Features

- **Overview KPIs:** Total earthquakes, average magnitude, max magnitude, average depth, quick insights.
- **Temporal Analysis:** Yearly evolution of earthquakes, magnitude trends.
- **Magnitude & Depth Analysis:** Scatterplots, average trends, density maps.
- **Geographical Maps:**
  - Density map with tectonic fault lines (NAF & EAF).
  - Earthquake animation timeline.
  - Geospatial clustering using K-Means.
- **Dataset Description:** Detailed info on the dataset, tectonic context, and project objectives.

## Dataset Source

Data compiled mainly from the **Kandilli Observatory and Earthquake Research Institute**.

## Tech Stack

- **Python, Pandas, NumPy**
- **Seaborn, Matplotlib, Plotly**
- **Scikit-Learn (K-Means Clustering)**
- **Streamlit**

## Deployment

The app is deployed using **Streamlit Cloud**.

## Folder Structure

```
Turkey_Earthquakes_Dashboard/
├── app.py
├── data/
│   └── turkey_earthquakes_historical.csv
├── images/
│   └── Turkey-faults-platenames.png
├── requirements.txt
├── README.md
└── .gitignore
```

## Project Objectives

- 🎯 Provide an exploratory, interactive analysis of Türkiye's seismic activity.
- 📊 Apply geospatial clustering & visualization techniques.
- 📚 Serve as an academic project combining data science, machine learning & public data.

---