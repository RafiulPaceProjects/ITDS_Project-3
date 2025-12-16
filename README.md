# NYC Energy Pulse: 2026 Forecast

A high-end interactive dashboard for visualizing and forecasting New York City's energy consumption.

## Features
- **Geospatial Intelligence**: Animated map of energy consumption over time.
- **3D Visuals**: PyDeck 3D “skyline” columns and Plotly 3D surface/bar views.
- **EDA Quick-Hits**: Correlation heatmap, seasonality boxplots, and borough trend-slope ranking.
- **Citywide Trends**: Analysis of monthly consumption trends.
- **Borough AI Forecaster**: Prophet-based forecasting for individual boroughs.
- **Model Lab**: Comparison of forecasting models.

### Optional: Borough polygon choropleth
If you place a borough boundary GeoJSON at `data/nyc_boroughs.geojson`, the Geospatial module enables a polygon choropleth tab.

## Setup
1. Install dependencies: `python -m pip install -r requirements.txt`
2. Run the app: `python -m streamlit run app.py`

## Data
This repo does not commit the main CSV because it exceeds GitHub's file size limit.

The app now auto-downloads the dataset from NYC Open Data on startup (and caches it locally):

- API CSV export: https://data.cityofnewyork.us/api/v3/views/jr24-e7cr/query.csv
- Local cache path: `data/Electric_Consumption_And_Cost_2010_-_May_2025_.csv`

If you prefer to download manually, save the dataset to the same local cache path above.

### Fixing 403 (Forbidden)
NYC Open Data's SODA API v3 may require an app token.

Option A (recommended for Streamlit): create `.streamlit/secrets.toml`:

```toml
SODA_APP_TOKEN = "<your token>"
```

Option B: set an environment variable before running the app:

- `export SODA_APP_TOKEN="<your token>"`

The token is not committed to git (and `.streamlit/secrets.toml` is ignored).
