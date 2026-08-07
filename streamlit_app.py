# streamlit_app.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import xarray as xr

st.set_page_config(page_title="Renewables Cost & Carbon Explorer", layout="wide")
st.title("Renewable Cost & Potential Explorer")
st.caption("Explore spatial variation in potential, LCOE and mitigation effects worldwide from solar and wind.")

# =========================
# Secrets / URLs
# =========================
SOLAR_URL = os.getenv("SOLAR_URL", "")
WIND_URL = os.getenv("WIND_URL", "")

SOLAR_URL = "https://www.dropbox.com/scl/fi/rv6lbuk00yryfznkkkdwg/SOLAR_TOTAL_RESULTS.nc?rlkey=cs71w2nni36w4x5dp6srtsnv3&st=y8dew6wi&dl=0&dl=1"
WIND_URL = "https://www.dropbox.com/scl/fi/gjf8f4iko67wvufuphptm/WIND_TOTAL_RESULTS.nc?rlkey=vjny9o28wndwit8scirkb7rme&st=6trhmznn&dl=0&dl=1"

try:
    SOLAR_URL = st.secrets.get("SOLAR_URL", SOLAR_URL)
    WIND_URL = st.secrets.get("WIND_URL", WIND_URL)
except Exception:
    pass

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Professional variable labels
# =========================
VAR_LABELS = {
    "Calculated_LCOE": "LCOE (USD/MWh, national)",
    "Uniform_LCOE": "LCOE (USD/MWh, uniform)",
    "Electricity_CI": "Grid carbon intensity (kgCO2/kWh)",
    "technical_potential": "Renewable technical potential (TPA)",
    "electricity_production": "Electricity production (kWH per annum per MW)",
    "Calculated_CAPEX": "Total capital expenditure (USD)",
    "Estimated_WACC": "Estimated cost of capital (%)",
    "Uniform_WACC": "Uniform cost of capital (%)",
}
LABEL_TO_VAR = {v: k for k, v in VAR_LABELS.items()}

# =========================
# Download + load helpers
# =========================
def _download_if_missing(url: str, out_path: Path) -> Path:
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    if not url:
        raise ValueError(f"Missing URL for {out_path.name}. Set in st.secrets or env vars.")

    with requests.get(url, stream=True, timeout=300, allow_redirects=True) as r:
        r.raise_for_status()
        ctype = r.headers.get("content-type", "").lower()
        if "text/html" in ctype:
            raise ValueError(f"URL returned HTML, not NetCDF: {url}")
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    with open(out_path, "rb") as f:
        sig = f.read(4)
    if sig[:3] != b"CDF" and sig != b"\x89HDF":
        raise ValueError(f"Downloaded file is not NetCDF/HDF5: {out_path}")

    return out_path


@st.cache_resource(show_spinner=False)
def _load_nc(tech: str) -> xr.Dataset:
    tech = tech.lower()
    if tech == "solar":
        fp = _download_if_missing(SOLAR_URL, DATA_DIR / "SOLAR_TOTAL_RESULTS.nc")
    else:
        fp = _download_if_missing(WIND_URL, DATA_DIR / "WIND_TOTAL_RESULTS.nc")

    return xr.open_dataset(
        fp,
        engine="netcdf4",
        chunks={"latitude": 180, "longitude": 180},
    )

@st.cache_data(show_spinner=False)
def load_country_mapping() -> pd.DataFrame:
    fp = DATA_DIR / "country_mapping_regions.csv"
    if not fp.exists():
        return pd.DataFrame(columns=["country_id", "country_name", "region"])

    cm = pd.read_csv(fp)

    # flexible column names
    cols = {c.lower(): c for c in cm.columns}
    id_col = cols.get("index") or cols.get("Country") or cols.get("country_id") or cols.get("country_number")
    name_col = cols.get("country_name") or cols.get("name")
    region_col = cols.get("region")

    if not id_col or not name_col or not region_col:
        return pd.DataFrame(columns=["country_id", "country_name", "region"])

    cm = cm[[id_col, name_col, region_col]].copy()
    cm.columns = ["country_id", "country_name", "region"]

    cm["country_id"] = pd.to_numeric(cm["country_id"], errors="coerce").astype("Int64")
    cm["country_name"] = cm["country_name"].astype("string")
    cm["region"] = cm["region"].astype("string")

    cm = cm.dropna(subset=["country_id"]).drop_duplicates(subset=["country_id"])
    return cm

@st.cache_data(show_spinner=False)
def load_cost_mapping() -> pd.DataFrame:
    fp = DATA_DIR / "IRENA_Country_Costs_2024.csv"
    if not fp.exists():
        return pd.DataFrame(columns=["index", "solar_costs_usd_kw", "wind_costs_usd_kw"])

    cm = pd.read_csv(fp)

    # flexible column names
    cols = {c.lower(): c for c in cm.columns}
    id_col = cols.get("index") 
    solar_col = cols.get("solar_costs_usd_kw")
    wind_col = cols.get("wind_costs_usd_kw")

    if not id_col or not solar_col or not wind_col:
        return pd.DataFrame(columns=["index", "solar_costs_usd_kw", "wind_costs_usd_kw"])

    cm = cm[[id_col, solar_col, wind_col]].copy()
    cm.columns = ["country_id", "solar_costs_usd_kw", "wind_costs_usd_kw"]
    inflation_2025 = 1.05

    cm["country_id"] = pd.to_numeric(cm["country_id"], errors="coerce").astype("Int64")
    cm["solar_costs_usd_kw"] = pd.to_numeric(cm["solar_costs_usd_kw"], errors="coerce").astype("Int64")
    cm["wind_costs_usd_kw"] = pd.to_numeric(cm["wind_costs_usd_kw"], errors="coerce").astype("Int64")

    # Convert to 2025 terms
    cm["solar_costs_usd_kw"] = cm["solar_costs_usd_kw"]*inflation_2025
    cm["wind_costs_usd_kw"] = cm["wind_costs_usd_kw"]*inflation_2025

    cm = cm.dropna(subset=["country_id"]).drop_duplicates(subset=["country_id"])
    return cm

def regional_LCOE_component_breakdown(
    df: pd.DataFrame,
    lifetime_years: int = 20,
    solar_om_frac: float = 0.023,
    wind_om_frac: float = 0.026,
) -> pd.DataFrame:
    """
    Regional-average LCOE breakdown into:
      - capex_LCOE
      - opex_LCOE
      - residual_finance_other_LCOE

    Assumptions:
      - solar_costs and wind_costs are CAPEX terms per site basis
      - annual O&M = fraction * CAPEX
      - lifetime = 20 years
      - electricity_production is annual production (same basis as the LCOE denominator)
    """

    lcoe_col = "Calculated_LCOE"
    if lcoe_col not in df.columns:
        lcoe_col = "levelised_cost" if "levelised_cost" in df.columns else None

    if lcoe_col is None:
        raise ValueError("Need a LCOE column such as 'Calculated_LCOE' or 'levelised_cost'")

    req = ["region", lcoe_col, "electricity_production"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df.copy()
    d = d.dropna(subset=["region", lcoe_col, "electricity_production"])
    d = d[d["electricity_production"] > 0]

    for c in ["solar_costs", "wind_costs", "renewables_costs"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    if "renewables_costs" in d.columns:
        d["renewables_capex"] = d["renewables_costs"]
    else:
        d["renewables_capex"] = d.get("solar_costs", 0.0) + d.get("wind_costs", 0.0)

    d["renewables_annual_opex"] = (
        solar_om_frac * d.get("solar_costs", 0.0) + wind_om_frac * d.get("wind_costs", 0.0)
    )

    reg = (
        d.groupby("region", as_index=False)
        .agg(
            LCOE_mean=(lcoe_col, "mean"),
            electricity_production_mean=("electricity_production", "mean"),
            capex_mean=("renewables_capex", "mean"),
            annual_opex_mean=("renewables_annual_opex", "mean"),
        )
    )

    reg["capex_LCOE"] = reg["capex_mean"] / (lifetime_years * reg["electricity_production_mean"])
    reg["opex_LCOE"] = reg["annual_opex_mean"] / reg["electricity_production_mean"]
    reg["undiscounted_sum_LCOE"] = reg["capex_LCOE"] + reg["opex_LCOE"]
    reg["residual_finance_other_LCOE"] = reg["LCOE_mean"] - reg["undiscounted_sum_LCOE"]

    denom = reg["LCOE_mean"].replace(0, np.nan)
    reg["capex_share"] = reg["capex_LCOE"] / denom
    reg["opex_share"] = reg["opex_LCOE"] / denom
    reg["residual_share"] = reg["residual_finance_other_LCOE"] / denom

    return reg.sort_values("LCOE_mean")

def apply_country_region_mapping(df: pd.DataFrame, cm: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["country_name"] = pd.Series(dtype="string")
        out["region"] = pd.Series(dtype="string")
        return out

    # country index from NetCDF expected in 'Country'
    if "Country" not in out.columns:
        out["country_id"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    else:
        out["country_id"] = pd.to_numeric(out["Country"], errors="coerce").astype("Int64")

    if cm.empty:
        out["country_name"] = "Unknown"
        out["region"] = "Unknown"
        return out

    out = out.merge(cm, on="country_id", how="left")
    out["country_name"] = out["country_name"].fillna("Unknown").astype("string")
    out["region"] = out["region"].fillna("Unknown").astype("string")
    return out

def _grid_resolution(vals: pd.Series, default: float = 0.5) -> float:
    v = np.sort(pd.to_numeric(vals.dropna().unique(), errors="coerce"))
    if len(v) < 2:
        return default
    d = np.diff(v)
    d = d[d > 0]
    return float(np.median(d)) if len(d) else default

def build_cell_geojson(
    df: pd.DataFrame,
    value_col: str = "Calculated_LCOE",
    id_col: str = "cell_id",
) -> tuple[dict, pd.DataFrame]:
    d = df.dropna(subset=["latitude", "longitude", value_col]).copy()
    if d.empty:
        return {"type": "FeatureCollection", "features": []}, d

    lat_step = _grid_resolution(d["latitude"], default=0.5)
    lon_step = _grid_resolution(d["longitude"], default=0.5)
    half_lat = lat_step / 2.0
    half_lon = lon_step / 2.0

    d[id_col] = (
        d["latitude"].round(6).astype(str) + "_" + d["longitude"].round(6).astype(str)
    )

    features = []
    for _, r in d.iterrows():
        lat = float(r["latitude"])
        lon = float(r["longitude"])

        # bounds
        lat_min = max(-90.0, lat - half_lat)
        lat_max = min(90.0, lat + half_lat)
        lon_min = max(-180.0, lon - half_lon)
        lon_max = min(180.0, lon + half_lon)

        poly = [
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min],
        ]

        features.append(
            {
                "type": "Feature",
                "id": r[id_col],
                "properties": {
                    id_col: r[id_col],
                    "region": str(r.get("region", "")),
                    "country_name": str(r.get("country_name", "")),
                    value_col: None if pd.isna(r[value_col]) else float(r[value_col]),
                },
                "geometry": {"type": "Polygon", "coordinates": [poly]},
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    return geojson, d

def regional_LCOE_component_breakdown(
    df: pd.DataFrame,
    lifetime_years: int = 20,
    solar_om_frac: float = 0.023,
    wind_om_frac: float = 0.026,
) -> pd.DataFrame:
    """
    Regional-average LCOE breakdown into:
      - capex_LCOE
      - opex_LCOE
      - residual_finance_other_LCOE

    Assumptions:
      - solar_costs and wind_costs are CAPEX terms per site basis
      - annual O&M = fraction * CAPEX
      - lifetime = 20 years
      - electricity_production is annual production (same basis as the LCOE denominator)
    """

    lcoe_col = "Calculated_LCOE"
    if lcoe_col not in df.columns:
        lcoe_col = "levelised_cost" if "levelised_cost" in df.columns else None

    if lcoe_col is None:
        raise ValueError("Need a LCOE column such as 'Calculated_LCOE' or 'levelised_cost'")

    req = ["region", lcoe_col, "electricity_production"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df.copy()
    d = d.dropna(subset=["region", lcoe_col, "electricity_production"])
    d = d[d["electricity_production"] > 0]

    for c in ["solar_costs", "wind_costs", "renewables_costs"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    if "renewables_costs" in d.columns:
        d["renewables_capex"] = d["renewables_costs"]
    else:
        d["renewables_capex"] = d.get("solar_costs", 0.0) + d.get("wind_costs", 0.0)

    d["renewables_annual_opex"] = (
        solar_om_frac * d.get("solar_costs", 0.0) + wind_om_frac * d.get("wind_costs", 0.0)
    )

    reg = (
        d.groupby("region", as_index=False)
        .agg(
            LCOE_mean=(lcoe_col, "mean"),
            electricity_production_mean=("electricity_production", "mean"),
            capex_mean=("renewables_capex", "mean"),
            annual_opex_mean=("renewables_annual_opex", "mean"),
        )
    )

    reg["capex_LCOE"] = reg["capex_mean"] / (lifetime_years * reg["electricity_production_mean"])
    reg["opex_LCOE"] = reg["annual_opex_mean"] / reg["electricity_production_mean"]
    reg["undiscounted_sum_LCOE"] = reg["capex_LCOE"] + reg["opex_LCOE"]
    reg["residual_finance_other_LCOE"] = reg["LCOE_mean"] - reg["undiscounted_sum_LCOE"]

    denom = reg["LCOE_mean"].replace(0, np.nan)
    reg["capex_share"] = reg["capex_LCOE"] / denom
    reg["opex_share"] = reg["opex_LCOE"] / denom
    reg["residual_share"] = reg["residual_finance_other_LCOE"] / denom

    return reg.sort_values("LCOE_mean")

def apply_country_mapping(df: pd.DataFrame, cm: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    """
    Map numeric country ID in NetCDF to country_name + region + costs from CSV.
    """
    out = df.copy()
    if out.empty or cm.empty:
        # ensure expected columns exist
        if "country_name" not in out.columns:
            out["country_name"] = pd.Series(["Unknown"] * len(out), dtype="string")
        return out

    # Find country-id column in data
    candidate_cols = ["Country", "country_id", "country_number"]
    data_country_col = next((c for c in candidate_cols if c in out.columns), None)

    if data_country_col is None:
        if "country_name" not in out.columns:
            out["country_name"] = pd.Series(["Unknown"] * len(out), dtype="string")
        return out

    out["country_id"] = pd.to_numeric(out[data_country_col], errors="coerce").astype("Int64")

    out = out.merge(cm, on="country_id", how="left", suffixes=("", "_map"))

    # region priority: mapped region, else existing region
    if "region_map" in out.columns:
        out["region"] = out["region_map"].combine_first(out.get("region"))
        out = out.drop(columns=["region_map"])

    # country_name final
    if "country_name" not in out.columns:
        out["country_name"] = "Unknown"
    out["country_name"] = out["country_name"].fillna("Unknown").astype("string")

    # Merge cost onto output
    out = out.merge(costs[["country_id", "solar_costs_usd_kw", "wind_costs_usd_kw"]], how="left", suffixes=("", "_costs"))

    return out


@st.cache_data(show_spinner=False)
def load_points_slice_from_nc(tech: str, columns: list[str]) -> pd.DataFrame:
    ds_sf = _load_nc(tech)

    available = [c for c in columns if (c in ds_sf.data_vars or c in ds_sf.coords)]
    if len(available) == 0:
        return pd.DataFrame()

    df = ds_sf[available].to_dataframe().reset_index()

    defaults = {
        "Country": "Unknown",
        "carbon_intensity": np.nan,
        "carbon_intensity_low": np.nan,
        "carbon_intensity_high": np.nan,
        "technical_potential": np.nan,
        "electricity_production": np.nan,
        "renewable_electricity": np.nan,
        "electrolyser_capacity": np.nan,
        "solar_costs": np.nan,
        "electrolyser_costs": np.nan,
        "renewables_costs": np.nan,
        "wind_costs": np.nan,
        "levelised_cost": np.nan,
        "levelised_cost_ren": np.nan,
        "levelised_cost_elec": np.nan,
    }
    for c, d in defaults.items():
        if c not in df.columns:
            df[c] = d

    # Convert electrolyser ratio
    df["electrolyser_capacity"] = df["electrolyser_capacity"] * 100

    numeric_cols = [
        "latitude", "longitude", "levelised_cost", "levelised_cost_ren", "levelised_cost_elec",
        "solar_costs", "wind_costs", "renewables_costs", "carbon_intensity", "carbon_intensity_low",
        "carbon_intensity_high", "technical_potential", "electrolyser_capacity", "renewable_electricity", "electricity_production",
        "solar_costs_usd_kw", "wind_costs_usd_kw",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    for c in ["region", "Country"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df


def change_capex_absolute_df(
    df: pd.DataFrame,
    solar_capex: float,
    tech: str,
    wind_capex: float,
    opex_assumption: float,
    estimated_wacc: float,
    wacc_reduction: float,
    lifetime_years: int = 20,
) -> pd.DataFrame:
    out = df.copy()

    if "Calculated_LCOE" not in out.columns:
        raise ValueError("Missing Calculated_LCOE column for recalculation")

    if "electricity_production" not in out.columns:
        raise ValueError("Missing electricity_production column for recalculation")

    electricity_production = pd.to_numeric(out["electricity_production"], errors="coerce").astype(float)

    if tech.lower() == "wind":
        recalc_capex_kw = float(wind_capex)
    else:
        recalc_capex_kw = float(solar_capex)

    capex_per_mw = recalc_capex_kw * 1000.0
    annual_generation_mwh = np.where(electricity_production > 0, electricity_production / 1000.0, np.nan)

    discount_rate = max(float(estimated_wacc) - float(wacc_reduction), 0.0) / 100.0
    if discount_rate > 0:
        capital_recovery_factor = discount_rate / (1 - (1 + discount_rate) ** (-lifetime_years))
    else:
        capital_recovery_factor = 1.0 / lifetime_years

    annualized_capex = capex_per_mw * capital_recovery_factor
    annual_opex = capex_per_mw * (float(opex_assumption) / 100.0)

    recalc_capex_LCOE = np.where(
        annual_generation_mwh > 0,
        annualized_capex / annual_generation_mwh,
        np.nan,
    )
    recalc_opex_LCOE = np.where(
        annual_generation_mwh > 0,
        annual_opex / annual_generation_mwh,
        np.nan,
    )

    out["Recalculated_LCOE"] = recalc_capex_LCOE + recalc_opex_LCOE
    out["delta_LCOE"] = out["Recalculated_LCOE"] - out["Calculated_LCOE"]
    return out


def summarize_group(df: pd.DataFrame, group_col: str, LCOE_col: str = "levelised_cost") -> pd.DataFrame:
    valid = df.dropna(subset=[group_col, LCOE_col]).copy()
    if valid.empty:
        return pd.DataFrame()

    # average CAPEX in USD/kW at group level
    summary = (
        valid.groupby(group_col, as_index=False)
        .agg(
            n_points=(group_col, "size"),
            LCOE_mean=(LCOE_col, "mean"),
            LCOE_p10=(LCOE_col, lambda x: x.quantile(0.1)),
            LCOE_p90=(LCOE_col, lambda x: x.quantile(0.9)),
            solar_capex_usd_kw_mean=("solar_costs_usd_kw", "mean"),
            wind_capex_usd_kw_mean=("wind_costs_usd_kw", "mean"),
        )
        .sort_values("LCOE_mean")
    )
    return summary


# ---------------------------
# Sidebar controls (global)
# ---------------------------
with st.sidebar:
    st.header("Global Controls")
    tech = st.selectbox("Renewable Technology", ["Wind", "Solar"], index=0)
    scenario = st.selectbox("Cost of capital scenario", ["Uniform", "National"], index=0)

needed_cols = [
    "latitude",
    "longitude",
    "Country",   # numeric code from NetCDF
    "Calculated_LCOE",
    "Estimated_WACC",
    "OPEX_LCOE_Uniform",
    "OPEX_LCOE_Country",
    "CAPEX_LCOE_Uniform",
    "CAPEX_LCOE_Country",
    "WACC_LCOE_Uniform",
    "WACC_LCOE_Country",
    "Electricity_CI",
    "Uniform_LCOE",
    "electricity_production",
    "technical_potential",
]

with st.spinner("Downloading/loading underlying data..."):
    df_sf = load_points_slice_from_nc(tech=tech, columns=needed_cols)

country_map = load_country_mapping()
cost_mapping = load_cost_mapping()
df_sf = apply_country_mapping(df_sf, country_map, cost_mapping)
if scenario == "Uniform":
    df_sf["Selected_LCOE"] = df_sf["Uniform_LCOE"]
else: 
    df_sf["Selected_LCOE"] = df_sf["Calculated_LCOE"]

if not df_sf.empty:
    df_sf = df_sf.dropna(subset=["Calculated_LCOE"], how="all")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🗺️ Map Explorer", "🕹 CAPEX Sensitivity (Regional)", "🌍 Regional Summary", "🔬 Country Heatmap", "📈 Country Potentials"]
)

# ==========================================================
# TAB 1: MAP EXPLORER
# ==========================================================
with tab1:
    st.subheader("Spatial Explorer")

    if df_sf.empty:
        st.warning("No data found for this technology / solar fraction selection.")
    else:
        metric_label = st.selectbox(
            "Map metric",
            options=[
                VAR_LABELS["Calculated_LCOE"],
                VAR_LABELS["Uniform_LCOE"],
                VAR_LABELS["electricity_production"],
                VAR_LABELS["technical_potential"],
                VAR_LABELS["Electricity_CI"],
                VAR_LABELS["technical_potential"],
            ],
            index=0,
        )
        metric = LABEL_TO_VAR[metric_label]

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            color_scale = st.selectbox("Color scale", ["Viridis", "Turbo", "Plasma", "Cividis"], index=0)
        with c2:
            # default to maximum points
            max_points = st.slider("Max points on map", 2000, 50000, 50000, step=1000)
        with c3:
            point_size = st.slider("Point size", 1, 18, 7)

        df_map = df_sf.dropna(subset=[metric]).copy()
        if len(df_map) > max_points:
            df_map = df_map.sample(max_points, random_state=42)

        if df_map.empty:
            st.warning("No data available for this metric.")
        else:
            q_low = df_map[metric].quantile(0.02)
            if metric == "Calculated_LCOE" or metric == "Uniform_LCOE":
                q_hi = 250
            else:
                q_hi = df_map[metric].quantile(0.75)
            if np.isclose(q_low, q_hi):
                q_low, q_hi = df_map[metric].min(), df_map[metric].max()

            fig = px.scatter_mapbox(
                df_map,
                lat="latitude",
                lon="longitude",
                color=metric,
                color_continuous_scale=color_scale,
                range_color=(0, q_hi) if np.isfinite(q_low) and np.isfinite(q_hi) else None,
                labels={metric: metric_label},
                hover_data={
                    "region": True,
                    "Country": True if "Country" in df_map.columns else False,
                     metric: ":.3f",
                    "latitude": ":.2f",
                    "longitude": ":.2f",
                },
                zoom=1,
                height=650,
                size_max=max(point_size, 1),
            )
            fig.update_traces(marker=dict(size=point_size, opacity=0.8, sizemode='area'))
            metric_label_wrapped = metric_label.replace(" ", "<br>")
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title=dict(
                        text=metric_label_wrapped
                    )))
            fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# TAB 2: CAPEX SENSITIVITY (REGIONAL ONLY)
# ==========================================================
with tab2:
    st.subheader("Regional CAPEX Sensitivity")
    st.caption("Exploration is restricted to a selected region, with defaults from average regional CAPEX.")

    if df_sf.empty:
        st.warning("No data found for this technology / solar fraction selection.")
    else:
        regions = sorted([r for r in df_sf["region"].dropna().unique().tolist() if r != "Unknown"])
        if not regions:
            st.warning("No region data available.")
        else:
            selected_region = st.selectbox("Select region", regions, index=0)
            df_reg = df_sf[df_sf["region"] == selected_region].copy()

            if df_reg.empty:
                st.warning("No data for selected region.")
            else:
                # defaults from regional averages (USD/kW for solar/wind)
                reg_solar_kw = float(df_reg["solar_costs_usd_kw"].mean()) if "solar_costs_usd_kw" in df_reg else 990
                reg_wind_kw = float(df_reg["wind_costs_usd_kw"].mean()) if "wind_costs_usd_kw" in df_reg else 1500

                # convert back to USD/MW for existing CAPEX function baseline math
                reg_solar_mw = reg_solar_kw
                reg_wind_mw = reg_wind_kw

                # proxy regional electrolyser default from elec component (or tech default if preferred)
                default_elec = 2.3 if tech == "Wind" else 2.6
                estimated_wacc = df_reg["Estimated_WACC"]
                default_wacc = estimated_wacc.mean()

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    solar_capex = st.number_input("Solar CAPEX (USD/kW)", min_value=100.0, max_value=6000.0, value=float(np.nan_to_num(reg_solar_mw, nan=990.0)), step=10.0)
                with c2:
                    wind_capex = st.number_input("Wind CAPEX (USD/kW)", min_value=100.0, max_value=8000.0, value=float(np.nan_to_num(reg_wind_mw, nan=1500.0)), step=10.0)
                with c3:
                    wacc_reduction = st.number_input(f"Reduction on WACC baseline (regional mean of {default_wacc:.1f}%)", min_value=0.0, max_value=25.0, value=0.0, step=1.0)
                with c4:
                    opex = st.number_input("Initial OPEX baseline (%, CAPEX)", min_value=0.0, max_value=10.0, value=default_elec, step=10.0)


                st.markdown(
                    f"Regional baseline CAPEX averages: "
                    f"Solar = **{reg_solar_kw:.0f} USD/kW**, Wind = **{reg_wind_kw:.0f} USD/kW**"
                )

                if st.button("Run Regional CAPEX Recalculation", type="primary"):
                    out = change_capex_absolute_df(
                        df=df_reg,
                        tech=tech,
                        solar_capex=solar_capex,
                        wind_capex=wind_capex,
                        opex_assumption=opex,
                        estimated_wacc=float(np.nan_to_num(default_wacc, nan=0.0)),
                        wacc_reduction=wacc_reduction,
                        lifetime_years=20,
                    )
                    #st.metric("Regional mean ΔLCOE", f"{out['delta_LCOE'].mean():.3f}")

                    # --- mini regional map of recalculated LCOE ---
                    # ---------------------------------------
                    # ---------------------------------------
                    VAR_MAP = {
                        "Recalculated_LCOE": {
                            "label": "Recalculated LCOE (USD/MWh)",
                            "fmt": ":.3f",
                        },
                        "delta_LCOE": {
                            "label": "ΔLCOE (USD/MWh)",
                            "fmt": ":.3f",
                        },
                    }

                    local_label_to_var = {v["label"]: k for k, v in VAR_MAP.items()}
                    map_metric = "Recalculated_LCOE"
                    metric_label = VAR_MAP[map_metric]["label"]
                    map_df = out.dropna(subset=[map_metric, "latitude", "longitude"]).copy()

                    if map_df.empty:
                        st.warning("No mappable points for selected region.")
                    else:
                        max_region_points = 12000
                        if len(map_df) > max_region_points:
                            map_df = map_df.sample(max_region_points, random_state=42)

                        q_hi = 150

                        hover_data = {
                            "region": True,
                            "country_name": True if "country_name" in map_df.columns else False,
                            map_metric: VAR_MAP[map_metric]["fmt"],
                        }

                        geojson_map, map_df_cells = build_cell_geojson(
                            map_df,
                            value_col=map_metric,
                            id_col="cell_id",
                        )

                        if map_df_cells.empty:
                            st.warning("No mappable points for selected region.")
                        else:
                            center_lat = float(map_df_cells["latitude"].mean())
                            center_lon = float(map_df_cells["longitude"].mean())

                            fig = px.choropleth_mapbox(
                                map_df_cells,
                                geojson=geojson_map,
                                locations="cell_id",
                                featureidkey="properties.cell_id",
                                color=map_metric,
                                color_continuous_scale="Turbo",
                                range_color=(0, q_hi),
                                labels={k: VAR_MAP[k]["label"] for k in VAR_MAP if k in map_df_cells.columns},
                                hover_data=hover_data,
                                center={"lat": center_lat, "lon": center_lon},
                                zoom=3,
                                opacity=0.65,
                                height=420,
                                title=f"{metric_label} — {selected_region}",
                            )

                            fig.update_layout(
                                mapbox_style="carto-positron",
                                coloraxis_colorbar=dict(title=metric_label),
                                margin=dict(l=0, r=0, t=30, b=0),
                            )

                        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# TAB 3: REGIONAL SUMMARY
# ==========================================================
with tab3:
    st.subheader("Regional Comparison")
    if df_sf.empty:
        st.warning("No data found.")
    else:
        LCOE_field = st.selectbox(
            "LCOE field for summary",
            options=[VAR_LABELS["Calculated_LCOE"], VAR_LABELS["Uniform_LCOE"]],
            index=0,
            key="region_LCOE_field",
        )
        LCOE_var = LABEL_TO_VAR[LCOE_field]
        reg_summary = summarize_group(df_sf, group_col="region", LCOE_col=LCOE_var)
        if reg_summary.empty:
            st.warning("No regional summary available.")
        else:
            st.markdown("### Regional LCOE comparison")
            breakdown_suffix = "Country" if LCOE_var == "Calculated_LCOE" else "Uniform"
            breakdown_cols = {
                "Capex costs": f"CAPEX_LCOE_{breakdown_suffix}",
                "Opex costs": f"OPEX_LCOE_{breakdown_suffix}",
                "WACC costs": f"WACC_LCOE_{breakdown_suffix}",
            }

            missing_cols = [col for col in breakdown_cols.values() if col not in df_sf.columns]
            if missing_cols:
                st.warning(
                    f"Missing breakdown columns for selected field: {missing_cols}. Falling back to computed component breakdown."
                )
                reg_break = regional_LCOE_component_breakdown(df_sf)
                abs_vars = ["residual_finance_other_LCOE", "capex_LCOE", "opex_LCOE"]
                share_vars = ["residual_share", "capex_share", "opex_share"]
                comp_map_abs = {
                    "capex_LCOE": "Capex costs",
                    "opex_LCOE": "Opex costs",
                    "residual_finance_other_LCOE": "Residual finance",
                }
                comp_map_share = {
                    "capex_share": "Capex costs",
                    "opex_share": "Opex costs",
                    "residual_share": "Residual finance",
                }
            else:
                comp_df = df_sf.dropna(subset=["region", LCOE_var] + list(breakdown_cols.values())).copy()
                comp_df["CAPEX_LCOE"] = comp_df[breakdown_cols["Capex costs"]].astype(float) * comp_df[LCOE_var].astype(float) / 100
                comp_df["OPEX_LCOE"] = comp_df[breakdown_cols["Opex costs"]].astype(float) * comp_df[LCOE_var].astype(float) / 100
                comp_df["WACC_LCOE"] = comp_df[breakdown_cols["WACC costs"]].astype(float) * comp_df[LCOE_var].astype(float) / 100

                reg_break = (
                    comp_df.groupby("region", as_index=False)
                    .agg(
                        CAPEX_LCOE=("CAPEX_LCOE", "median"),
                        OPEX_LCOE=("OPEX_LCOE", "median"),
                        WACC_LCOE=("WACC_LCOE", "median"),
                    )
                )
                reg_break["LCOE_total_components"] = (
                    reg_break[["CAPEX_LCOE", "OPEX_LCOE", "WACC_LCOE"]].sum(axis=1)
                )
                reg_break["capex_share"] = reg_break["CAPEX_LCOE"] / reg_break["LCOE_total_components"].replace({0: np.nan})
                reg_break["opex_share"] = reg_break["OPEX_LCOE"] / reg_break["LCOE_total_components"].replace({0: np.nan})
                reg_break["wacc_share"] = reg_break["WACC_LCOE"] / reg_break["LCOE_total_components"].replace({0: np.nan})
                abs_vars = ["CAPEX_LCOE", "OPEX_LCOE", "WACC_LCOE"]
                share_vars = ["capex_share", "opex_share", "wacc_share"]
                comp_map_abs = {
                    "CAPEX_LCOE": "Capex costs",
                    "OPEX_LCOE": "Opex costs",
                    "WACC_LCOE": "WACC costs",
                }
                comp_map_share = {
                    "capex_share": "Capex costs",
                    "opex_share": "Opex costs",
                    "wacc_share": "WACC costs",
                }

            # Stacked absolute components
            comp_long = reg_break.melt(
                id_vars="region",
                value_vars=abs_vars,
                var_name="component",
                value_name="LCOE_component",
            )
            comp_long["component"] = comp_long["component"].map(comp_map_abs)

            fig_abs = px.bar(
                comp_long,
                x="region",
                y="LCOE_component",
                color="component",
                range_y=[0, 150],
                barmode="stack",
                title=f"Contributions to the {breakdown_suffix} LCOE (USD/MWh)",
                labels={"LCOE_component": "LCOE component (USD/MWh)", "region": "Region"},
            )
            fig_abs.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_abs, use_container_width=True)

            # 100% stacked shares
            share_long = reg_break.melt(
                id_vars="region",
                value_vars=share_vars,
                var_name="component",
                value_name="share",
            )
            share_long["component"] = share_long["component"].map(comp_map_share)

            fig_share = px.bar(
                share_long,
                x="region",
                y="share",
                color="component",
                barmode="stack",
                title=f"Contributions to the {breakdown_suffix} LCOE (share)",
                labels={"share": "Share of LCOE", "region": "Region"},
            )
            fig_share.update_layout(xaxis_tickangle=-35, yaxis_tickformat=".0%")
            st.plotly_chart(fig_share, use_container_width=True)

# ==========================================================
# TAB 4: COUNTRY SUMMARY (new)
# ==========================================================
with tab4:
    if df_sf.empty:
        st.warning("No data found.")
    else:
        if "Country" not in df_sf.columns or df_sf["Country"].isna().all():
            st.info("Country column not yet available in dataset. Add 'country' variable to enable this tab.")
        else:
            st.markdown("### Country LCOE distribution")

            country_options = sorted(df_sf["country_name"].dropna().unique().tolist())
            selected_country = st.selectbox("Select country", country_options, key="country_select")

            df_country = df_sf[df_sf["country_name"] == selected_country].copy()

            if df_country.empty:
                st.warning("No data for selected country.")
            else:
                # ---- metrics ----
                LCOE_min = df_country["Calculated_LCOE"].min()
                LCOE_p10 = df_country["Calculated_LCOE"].quantile(0.10)
                LCOE_p50 = df_country["Calculated_LCOE"].median()
                LCOE_p90 = df_country["Calculated_LCOE"].quantile(0.90)
                LCOE_max = df_country["Calculated_LCOE"].max()

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Minimum LCOE (National)", f"{LCOE_min:.2f}")
                c2.metric("LCOE 10th percentile (National)", f"{LCOE_p10:.2f}")
                c3.metric("LCOE median (National)", f"{LCOE_p50:.2f}")
                c4.metric("LCOE 90th percentile (National)", f"{LCOE_p90:.2f}")
                c5.metric("Maximum LCOE (National)", f"{LCOE_max:.2f}")

                # ---- metrics ----
                LCOE_min_uniform = df_country["Uniform_LCOE"].min()
                LCOE_p10_uniform = df_country["Uniform_LCOE"].quantile(0.10)
                LCOE_p50_uniform = df_country["Uniform_LCOE"].median()
                LCOE_p90_uniform = df_country["Uniform_LCOE"].quantile(0.90)
                LCOE_max_uniform = df_country["Uniform_LCOE"].max()

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Minimum LCOE (Uniform)", f"{LCOE_min_uniform:.2f}")
                c2.metric("LCOE 10th percentile (Uniform)", f"{LCOE_p10_uniform:.2f}")
                c3.metric("LCOE median (Uniform)", f"{LCOE_p50_uniform:.2f}")
                c4.metric("LCOE 90th percentile (Uniform)", f"{LCOE_p90_uniform:.2f}")
                c5.metric("Maximum LCOE (Uniform)", f"{LCOE_max_uniform:.2f}")

                # ---- subset map (auto-zoom to country extent) ----
                lat_c = float(df_country["latitude"].mean())
                lon_c = float(df_country["longitude"].mean())

                value_col = "Selected_LCOE"

                geojson_cells, df_cells = build_cell_geojson(df_country, value_col=value_col, id_col="cell_id")

                if len(df_cells) == 0:
                    st.warning("No data for cell map.")
                else:
                    lat_c = float(df_cells["latitude"].mean())
                    lon_c = float(df_cells["longitude"].mean())

                    LCOE_MIN = 0.0
                    LCOE_MAX = 150.0

                    fig_cells = px.choropleth_mapbox(
                        df_cells,
                        geojson=geojson_cells,
                        locations="cell_id",
                        featureidkey="properties.cell_id",
                        color=value_col,
                        color_continuous_scale="Viridis",
                        range_color=(LCOE_MIN, LCOE_MAX),
                        hover_data={
                            "latitude": ":.2f",
                            "longitude": ":.2f",
                            "region": True,
                            "country_name": True,
                            "Selected_LCOE": ":.3f",
                        },
                        center={"lat": lat_c, "lon": lon_c},
                        zoom=4,
                        opacity=0.65,
                        title=f"{selected_country}: LCOE distribution",
                        height=700,
                    )
                    fig_cells.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig_cells, use_container_width=True)


with tab5:
    if df_sf.empty:
        st.warning("No data found.")
    else:
        if "Country" not in df_sf.columns or df_sf["Country"].isna().all():
            st.info("Country column not yet available in dataset. Add 'country' variable to enable this tab.")
        else:
            st.markdown("### Country LCOE-Potential")
            country_options = sorted(df_sf["country_name"].dropna().unique().tolist())
            selected_country = st.selectbox("Select country", country_options, key="country_select_2")

            df_country = df_sf[df_sf["country_name"] == selected_country].copy()

            if df_country.empty:
                st.warning("No data for selected country.")
            else:
                # ---- metrics ----
                LCOE_min = df_country["Calculated_LCOE"].min()
                LCOE_p10 = df_country["Calculated_LCOE"].quantile(0.10)
                LCOE_p50 = df_country["Calculated_LCOE"].median()
                LCOE_p90 = df_country["Calculated_LCOE"].quantile(0.90)
                LCOE_max = df_country["Calculated_LCOE"].max()

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Minimum LCOE (National)", f"{LCOE_min:.2f}")
                c2.metric("LCOE 10th percentile (National)", f"{LCOE_p10:.2f}")
                c3.metric("LCOE median (National)", f"{LCOE_p50:.2f}")
                c4.metric("LCOE 90th percentile (National)", f"{LCOE_p90:.2f}")
                c5.metric("Maximum LCOE (National)", f"{LCOE_max:.2f}")

                # ---- metrics ----
                LCOE_min_uniform = df_country["Uniform_LCOE"].min()
                LCOE_p10_uniform = df_country["Uniform_LCOE"].quantile(0.10)
                LCOE_p50_uniform = df_country["Uniform_LCOE"].median()
                LCOE_p90_uniform = df_country["Uniform_LCOE"].quantile(0.90)
                LCOE_max_uniform = df_country["Uniform_LCOE"].max()

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Minimum LCOE (Uniform)", f"{LCOE_min_uniform:.2f}")
                c2.metric("LCOE 10th percentile (Uniform)", f"{LCOE_p10_uniform:.2f}")
                c3.metric("LCOE median (Uniform)", f"{LCOE_p50_uniform:.2f}")
                c4.metric("LCOE 90th percentile (Uniform)", f"{LCOE_p90_uniform:.2f}")
                c5.metric("Maximum LCOE (Uniform)", f"{LCOE_max_uniform:.2f}")
            # ---- cumulative cost-technical potential curve ----
            curve = df_country.dropna(subset=["Calculated_LCOE", "technical_potential"]).copy()
            curve = curve[curve["technical_potential"] > 0].copy()

            if curve.empty:
                st.info("No positive technical potential data available for cumulative curve.")
            else:
                curve = curve.sort_values("Calculated_LCOE")
                curve["cum_potential"] = curve["technical_potential"].cumsum()
                curve["cum_potential_share"] = curve["cum_potential"] / curve["technical_potential"].sum()
                curve["Calculated_LCOE"] = pd.to_numeric(curve["Calculated_LCOE"], errors="coerce")

                lcoe_series = ["Calculated_LCOE"]
                if "Uniform_LCOE" in curve.columns:
                    curve["Uniform_LCOE"] = pd.to_numeric(curve["Uniform_LCOE"], errors="coerce")
                    lcoe_series.append("Uniform_LCOE")

                curve_long = curve.melt(
                    id_vars=["cum_potential", "cum_potential_share"],
                    value_vars=lcoe_series,
                    var_name="lcoe_metric",
                    value_name="LCOE",
                )
                curve_long["lcoe_metric"] = curve_long["lcoe_metric"].replace({
                    "Calculated_LCOE": "Calculated LCOE",
                    "Uniform_LCOE": "Uniform LCOE",
                })

                fig_curve = px.line(
                    curve_long,
                    x="cum_potential",
                    y="LCOE",
                    color="lcoe_metric",
                    title=f"{selected_country}: cumulative cost-technical potential curve",
                    labels={
                        "cum_potential": "Cumulative technical potential",
                        "LCOE": "LCOE (USD/MWh)",
                        "lcoe_metric": "LCOE series",
                    },
                )
                fig_curve.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig_curve, use_container_width=True)

                fig_curve_share = px.line(
                    curve_long,
                    x="cum_potential_share",
                    y="LCOE",
                    color="lcoe_metric",
                    title=f"{selected_country}: cumulative curve (normalised potential)",
                    labels={
                        "cum_potential_share": "Cumulative potential share",
                        "LCOE": "LCOE (USD/MWh)",
                        "lcoe_metric": "LCOE series",
                    },
                )
                fig_curve_share.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig_curve_share, use_container_width=True)