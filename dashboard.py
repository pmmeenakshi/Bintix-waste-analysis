# === BLOCK 1: Minimal loader (no previews) ===
import re
import pandas as pd
import streamlit as st
from pathlib import Path



from pathlib import Path
import base64, mimetypes

# === ASSETS: load icons as data URIs and tolerate both folder spellings ===
  # safe, we already set page_config

BASE_DIR = Path(__file__).parent.resolve()

# Look in repo root first (your case), then assets/, then assests/
_ASSET_DIR_CANDIDATES = [BASE_DIR, BASE_DIR / "assets", BASE_DIR / "assests"]
ASSETS_DIR = next((p for p in _ASSET_DIR_CANDIDATES if p.exists()), BASE_DIR)

def load_icon_data_uri(filename: str) -> str:
    """Return data: URI for an image; empty string if missing (so popup still renders)."""
    p = ASSETS_DIR / filename
    if not p.exists():
        return ""
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

# your files are at repo root: tree.png, house.png, waste-management.png
TREE_ICON    = load_icon_data_uri("tree.png")
HOUSE_ICON   = load_icon_data_uri("house.png")
RECYCLE_ICON = load_icon_data_uri("waste-management.png")

if not all([TREE_ICON, HOUSE_ICON, RECYCLE_ICON]):
    st.warning(f"Some icons not found in: {ASSETS_DIR}. Popups will show text without icons.")

# fixed file path relative to this script
BASE_DIR = Path(__file__).parent
CSV_DEFAULT = BASE_DIR / "standardized_wide_fy2024_25.csv"

ID_COLS_REQUIRED = ["City", "Community", "Pincode"]
ID_COLS_OPTIONAL = ["Lat", "Lon"]
METRIC_COL_REGEX = re.compile(
    r"^(Impact|Tonnage|CO2_Kgs_Averted|Households_Participating|Segregation_Compliance_Pct)_(\d{4}-\d{2})$"
)

def _detect_metric_month_cols(columns):
    cols, months = [], set()
    for c in columns:
        m = METRIC_COL_REGEX.match(c)
        if m:
            cols.append(c); months.add(m.group(2))
    return cols, sorted(months)

@st.cache_data(show_spinner=False)
def load_and_prepare(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    missing = [c for c in ID_COLS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    metric_month_cols, months = _detect_metric_month_cols(df.columns)
    if not metric_month_cols:
        raise ValueError("No metric-month columns like Impact_2024-04, Tonnage_2025-03 found.")

    for c in metric_month_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    id_cols_present = [c for c in (ID_COLS_REQUIRED + ID_COLS_OPTIONAL) if c in df.columns]
    long_df = df.melt(
        id_vars=id_cols_present, value_vars=metric_month_cols,
        var_name="Metric_Month", value_name="Value"
    )
    parts = long_df["Metric_Month"].str.rsplit("_", n=1, expand=True)
    long_df["Metric"] = parts[0]
    long_df["Date"]   = pd.to_datetime(parts[1] + "-01", format="%Y-%m-%d")
    long_df = long_df.drop(columns=["Metric_Month"]).sort_values(id_cols_present + ["Metric", "Date"])
    return df, long_df, months

# load once at app start; store to session for later blocks
try:
    df_wide, df_long, months = load_and_prepare(CSV_DEFAULT)
    st.session_state["df_wide"] = df_wide
    st.session_state["df_long"] = df_long
    st.session_state["months"]  = months
    st.session_state["data_src"] = str(CSV_DEFAULT.resolve())
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

   
# === BLOCK 2: Final UI + Map + Selection at Bottom + Brand Styling ===
import io, base64
from pathlib import Path
import time
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
import folium
import plotly.express as px
from folium.plugins import MarkerCluster

# Speed settings
ST_MAP_HEIGHT = 560
ST_RETURNED_OBJECTS = []  # don't send all map layers back to Streamlit

# ---------------- Theme and Layout ----------------
st.set_page_config(page_title="Bintix Waste Analytics", layout="wide")

BRAND_PRIMARY = "#36204D"     # purple
SECONDARY_GREEN = "#2E7D32"   # (kept only if you need elsewhere; not used on map)
TEXT_DARK = "#36204D"         # brand purple for all body text
       # readable gray on white

# --- Environmental conversions ---
CO2_PER_KG_DRY = 2.18      # 1 kg dry waste -> 2.18 kg CO2 averted
KG_PER_TREE     = 117.0     # 117 kg dry waste -> 1 tree saved
st.markdown(
    """
    <style>
    /* Dropdown background should be white */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
    }

    /* Text inside dropdown menu items set to white */
    div[data-baseweb="select"] span {
        color: white !important;
    }

    /* The "All" text and selected values */
    div[data-baseweb="select"] input {
        color: #36204D !important; /* Purple for default field text */
    }

    /* Dropdown menu container */
    div[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
    }

    /* Options text color white to be visible on black background */
    div[data-baseweb="menu"] div[role="option"] {
        color: white !important;
    }

    /* Option hover styling */
    div[data-baseweb="menu"] div[role="option"]:hover {
        background-color: #36204D22 !important; /* Light purple hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)




# ---------------- Sidebar ----------------
with st.sidebar:
    uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"],
                                help="Wide format: one row per community; monthly cols like Impact_2024-04, Tonnage_2024-04, ...")
    st.caption("If no upload, the default CSV from the app folder is used.")

# ---------------- Load Data from session (BLOCK 1 already set these) ----------------
df_wide  = st.session_state["df_wide"]
df_long  = st.session_state["df_long"]
months   = st.session_state["months"]
data_src = st.session_state["data_src"]

# --- Normalize key id columns to STRING across both frames ---
for col in ["Pincode", "Community", "City"]:
    if col in df_wide.columns:
        df_wide[col] = df_wide[col].astype(str)
    if col in df_long.columns:
        df_long[col] = df_long[col].astype(str)

# --- Merge pincode centroids (Lat/Lon) if needed ---
PINCODE_LOOKUP = BASE_DIR / "pincode_centroids.csv"
if ("Lat" not in df_wide.columns or "Lon" not in df_wide.columns):
    if PINCODE_LOOKUP.exists():
        try:
            look = pd.read_csv(PINCODE_LOOKUP)
            look.columns = [c.strip() for c in look.columns]
            if {"Pincode","Lat","Lon"}.issubset(look.columns):
                look["Pincode"] = look["Pincode"].astype(str).str.strip()
                look["Lat"] = pd.to_numeric(look["Lat"], errors="coerce")
                look["Lon"] = pd.to_numeric(look["Lon"], errors="coerce")
                df_wide["Pincode"] = df_wide["Pincode"].astype(str).str.strip()
                df_wide = df_wide.merge(look[["Pincode","Lat","Lon"]], on="Pincode", how="left")
                st.caption(f"🗺️ Coordinates merged from `pincode_centroids.csv` "
                           f"(markers available for {(df_wide[['Lat','Lon']].notna().all(axis=1)).sum()} communities).")
            else:
                st.warning("`pincode_centroids.csv` columns must be exactly: Pincode, Lat, Lon.")
        except Exception as e:
            st.warning(f"Could not read/merge `pincode_centroids.csv`: {e}")

# Persist normalized/merged
st.session_state["df_wide"] = df_wide
st.session_state["df_long"] = df_long

# ---------------- Title ----------------
st.markdown("<h1>Smart Waste Analytics — FY 2024–25</h1>", unsafe_allow_html=True)
st.caption(f"Data source: {data_src}")

# ---------------- Global Filters ----------------
st.markdown("### 🔎 Global Filters")
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 2.2])

city_opts = sorted(df_wide["City"].dropna().unique().tolist()) if "City" in df_wide else []
comm_opts = sorted(df_wide["Community"].dropna().unique().tolist()) if "Community" in df_wide else []
pin_opts  = sorted(df_wide["Pincode"].dropna().unique().tolist()) if "Pincode" in df_wide else []

with c1: sel_city = st.multiselect("City", city_opts, placeholder="All")
with c2: sel_comm = st.multiselect("Community", comm_opts, placeholder="All")
with c3: sel_pin  = st.multiselect("Pincode", pin_opts,  placeholder="All")
with c4:
    start_m, end_m = st.select_slider("Date range (month)", options=months, value=(months[0], months[-1]))

def apply_filters(dfw, dfl):
    dfw = dfw.copy(); dfl = dfl.copy()
    for col in ["Pincode", "Community", "City"]:
        if col in dfw: dfw[col] = dfw[col].astype(str)
        if col in dfl: dfl[col] = dfl[col].astype(str)

    sel_city_s = [str(x) for x in sel_city] if sel_city else []
    sel_comm_s = [str(x) for x in sel_comm] if sel_comm else []
    sel_pin_s  = [str(x) for x in sel_pin]  if sel_pin  else []

    mask_w = pd.Series(True, index=dfw.index)
    if sel_city_s: mask_w &= dfw["City"].isin(sel_city_s)
    if sel_comm_s: mask_w &= dfw["Community"].isin(sel_comm_s)
    if sel_pin_s:  mask_w &= dfw["Pincode"].isin(sel_pin_s)
    dfw_f = dfw[mask_w].copy()

    mask_l = pd.Series(True, index=dfl.index)
    if sel_city_s: mask_l &= dfl["City"].isin(sel_city_s)
    if sel_comm_s: mask_l &= dfl["Community"].isin(sel_comm_s)
    if sel_pin_s:  mask_l &= dfl["Pincode"].isin(sel_pin_s)
    d0 = pd.to_datetime(start_m + "-01"); d1 = pd.to_datetime(end_m + "-01")
    mask_l &= (dfl["Date"] >= d0) & (dfl["Date"] <= d1)
    dfl_f = dfl[mask_l].copy()
    return dfw_f, dfl_f

dfw_filt, dfl_filt = apply_filters(df_wide, df_long)

# ---------------- Summary KPIs ----------------
def kpi_value(dfl, metric, agg="sum"):
    s = dfl.loc[dfl["Metric"] == metric, "Value"]
    if s.empty: return 0.0
    return float(s.sum() if agg == "sum" else s.mean())

n_communities = dfw_filt["Community"].nunique() if "Community" in dfw_filt else 0
n_cities      = dfw_filt["City"].nunique()      if "City" in dfw_filt else 0
total_tonnage = kpi_value(dfl_filt, "Tonnage", "sum")
total_co2     = kpi_value(dfl_filt, "CO2_Kgs_Averted", "sum")
avg_comp      = kpi_value(dfl_filt, "Segregation_Compliance_Pct", "mean")
total_hh      = kpi_value(dfl_filt, "Households_Participating", "sum")

st.markdown("### 📊 Summary")
k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
k1.metric("Communities", n_communities)
k2.metric("Cities", n_cities)
k3.metric("Total Tonnage", f"{total_tonnage:,.0f}")
k4.metric("CO₂ Averted (kg)", f"{total_co2:,.0f}")
k5.metric("Avg Segregation (%)", f"{avg_comp:,.1f}")
k6.metric("Active Households", f"{total_hh:,.0f}")
st.caption(f"Period: **{start_m} → {end_m}**")

# ---------------- Tabs ----------------
tab_map, tab_insights = st.tabs(["🗺️ 2D Map & Popups", "🧠 Insights"])

# --- Trend Helper (brand axes & line color) ---
def small_trend(df_long_or_filtered, community_id, metric):
    d = df_long_or_filtered[
        (df_long_or_filtered["Community"] == str(community_id)) &
        (df_long_or_filtered["Metric"] == metric)
    ].sort_values("Date")
    if d.empty:
        return None

    fig = px.line(
        d, x="Date", y="Value", markers=True,
        labels={"Value": metric.replace("_"," "), "Date": "Date"},
        template=None
    )
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=36, b=36),
        title=dict(text=f"{metric.replace('_',' ')} Trend",
                   font=dict(color="#36204D", size=16)),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Poppins", color="#36204D", size=13),
        xaxis=dict(
            title=dict(text="Date", font=dict(color="#36204D")),
            tickfont=dict(color="#36204D"),
            color="#36204D",
            gridcolor="rgba(54,32,77,0.12)",
            zerolinecolor="rgba(54,32,77,0.18)",
            linecolor="rgba(54,32,77,0.25)"
        ),
        yaxis=dict(
            title=dict(text=metric.replace("_"," "), font=dict(color="#36204D")),
            tickfont=dict(color="#36204D"),
            color="#36204D",
            gridcolor="rgba(54,32,77,0.12)",
            zerolinecolor="rgba(54,32,77,0.18)",
            linecolor="rgba(54,32,77,0.25)"
        ),
        showlegend=False,
    )
    fig.update_traces(line=dict(color="#36204D", width=2),
                      marker=dict(color="#36204D"))
    return fig

def _fig_to_base64_png(fig, w=420, h=180, scale=2):
    """
    Render a Plotly fig to PNG and return base64 string.
    Requires `kaleido` in requirements.txt.
    """
    fig.update_layout(width=w, height=h)
    png = fig.to_image(format="png", width=w, height=h, scale=scale)  # needs kaleido
    return base64.b64encode(png).decode("utf-8")


def small_trend_base64(df_long_filtered, community_id, metric):
    """
    Make your brand-styled sparkline (purple) and return <img> tag (base64 PNG).
    """
    fig = small_trend(df_long_filtered, community_id, metric)
    if fig is None:
        return ""
    b64 = _fig_to_base64_png(fig, w=420, h=180, scale=2)
    return f"<img src='data:image/png;base64,{b64}' style='width:100%;border:1px solid #eee;border-radius:8px'/>"




# --- Community summary using filtered frame ---
def summarize_for_community(community_id=None, pincode=None):
    d = dfl_filt.copy()  # already filtered by date slider
    if "Community" in d: d["Community"] = d["Community"].astype(str)
    if "Pincode"   in d: d["Pincode"]   = d["Pincode"].astype(str)
    if community_id is not None:
        d = d[d["Community"] == str(community_id)]
    if pincode is not None:
        d = d[d["Pincode"] == str(pincode)]

    def agg(metric, how="sum"):
        s = d.loc[d["Metric"] == metric, "Value"]
        if s.empty: return 0.0
        return float(s.sum() if how=="sum" else s.mean())

    # Prefer a dedicated dry-waste metric; fallback to total tonnage
    dry_candidates = ["Tonnage_Dry", "Dry_Tonnage", "DryWaste", "Tonnage"]
    dry_kg = 0.0
    for m in dry_candidates:
        s = d.loc[d["Metric"] == m, "Value"]
        if not s.empty:
            dry_kg = float(s.sum())
            break

    # If your Tonnage was stored in tonnes (not kg), uncomment the next line:
    # dry_kg *= 1000.0

    co2_calc   = dry_kg * CO2_PER_KG_DRY
    trees      = dry_kg / KG_PER_TREE

    return {
        "Tonnage_sum": agg("Tonnage", "sum"),
        "CO2_sum": agg("CO2_Kgs_Averted", "sum"),  # if present in data
        "HH_sum": agg("Households_Participating", "sum"),
        "Seg_pct_avg": agg("Segregation_Compliance_Pct", "mean"),
        "Impact_avg": agg("Impact", "mean"),
        # new
        "Dry_kg": dry_kg,
        "CO2_calc": co2_calc,
        "Trees_saved": trees,
    }
import plotly.express as px
import plotly.io as pio
import numpy as np




@st.cache_data(show_spinner=False)
def monthly_series(df_long, community: str, metric: str):
    d = df_long[
        (df_long["Community"].astype(str) == str(community)) &
        (df_long["Metric"] == metric)
    ][["Date", "Value"]].sort_values("Date").copy()
    return d

# --- image helpers for popup cards ---
import io, base64, matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def _to_data_uri(fig, w=340):  # return an <img> HTML tag from a Matplotlib fig
    buf = io.BytesIO()
    plt.tight_layout(pad=0.3)
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=180)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"<img src='data:image/png;base64,{b64}' style='width:{w}px;height:auto;border:0;'/>"

# helper for distinct colors (enough for 12+ months)
def _distinct_colors(n):
    import matplotlib.pyplot as plt
    # tab20 is good up to 20; extend with Set3 then Pastel1 if ever needed
    cmaps = [plt.cm.tab20, plt.cm.Set3, plt.cm.Pastel1]
    colors = []
    i = 0
    while len(colors) < n:
        cmap = cmaps[i % len(cmaps)]
        M = cmap.N
        take = min(n - len(colors), M)
        # sample evenly across the colormap
        for j in range(take):
            colors.append(cmap(j / max(M - 1, 1)))
        i += 1
    return colors[:n]
@st.cache_data(show_spinner=False)

def popup_charts_for_comm(dfl_filtered, community_id):
    """
    Charts for the *current date range* (dfl_filtered is already slider-filtered):
      - BAR (brand purple): Tonnage by month across the selected range
      - DONUT (distinct colors): CO2 averted by month across the selected range
    Returns (bar_img_html, donut_img_html) as <img> tags (base64 PNG).
    """
    BRAND = BRAND_PRIMARY

    dm = dfl_filtered.copy()
    dm["Community"] = dm["Community"].astype(str)
    dm = dm[dm["Community"] == str(community_id)]
    if dm.empty:
        return "", ""

    # --- month key that preserves chronological order across years ---
    dm["MonthKey"] = dm["Date"].dt.to_period("M")  # e.g., 2024-04, 2024-05, ...
    month_order = (dm["MonthKey"].drop_duplicates().sort_values().tolist())
    month_labels = [p.to_timestamp().strftime("%b") for p in month_order]  # Apr, May, ...

    # ---------- TONNAGE BAR: all months in selected range ----------
    bar_img = ""
    d_ton = dm[dm["Metric"] == "Tonnage"][["MonthKey", "Value"]].copy()
    if not d_ton.empty:
        d_ton = (d_ton.groupby("MonthKey", as_index=False)["Value"].sum()
                        .sort_values("MonthKey"))
        # label column in same order as month_order
        d_ton["Month"] = [p.to_timestamp().strftime("%b") for p in d_ton["MonthKey"]]

        fig, ax = plt.subplots(figsize=(3.2, 1.8))
        ax.bar(d_ton["Month"], d_ton["Value"], color=BRAND)
        ax.set_title("Tonnage", fontsize=9, color=BRAND, pad=2)
        ax.tick_params(axis="x", labelsize=8, colors=BRAND, rotation=0)
        ax.tick_params(axis="y", labelsize=8, colors=BRAND)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.grid(alpha=0.15, axis="y")
        bar_img = _to_data_uri(fig, w=300)

    # ---------- CO2 DONUT: all months in selected range ----------
    donut_img = ""
    # Use dry tonnage if exists; otherwise fallback to Tonnage
    dry_candidates = ["Tonnage_Dry", "Dry_Tonnage", "DryWaste", "Tonnage"]
    dry_month = None
    for m in dry_candidates:
        cur = dm[dm["Metric"] == m][["MonthKey", "Value"]].copy()
        if not cur.empty:
            dry_month = cur
            break

    if dry_month is not None and not dry_month.empty:
        d = (dry_month.groupby("MonthKey", as_index=False)["Value"].sum()
                        .sort_values("MonthKey"))
        vals = (d["Value"] * CO2_PER_KG_DRY).clip(lower=0.0).to_numpy()
        labels = [p.to_timestamp().strftime("%b") for p in d["MonthKey"]]
        colors = _distinct_colors(len(labels))  # one distinct color per month

        fig, ax = plt.subplots(figsize=(2.4, 2.4))
        wedges, _ = ax.pie(
            vals,
            wedgeprops=dict(width=0.45),
            startangle=90,
            colors=colors
        )
        ax.set(aspect="equal")
        ax.text(0, 0, "CO₂\nAverted", ha="center", va="center",
                fontsize=9, color=BRAND, fontweight="bold", linespacing=1.1)
        # legend on the right with month labels
        ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=8, frameon=False)
        donut_img = _to_data_uri(fig, w=220)

    return bar_img, donut_img


import numpy as np
  # keep this near your other imports

def jitter_duplicates(df, lat_col="Lat", lon_col="Lon", jitter_deg=0.00025):
    """
    Move markers that share the same (Lat, Lon) into a tiny circle around
    the original location so each one gets a working tooltip/popup.
    jitter_deg ~ 0.00025 ≈ 25–30 meters.
    """
    df = df.copy()
    # bucket by rounded coords so "almost equal" also spreads
    gb = df.groupby([df[lat_col].round(6), df[lon_col].round(6)])

    for _, idx in gb.groups.items():                 # <— idx is an Index of rows in this bucket
        n = len(idx)
        if n > 1:
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            r = jitter_deg
            df.loc[idx, lat_col] = df.loc[idx, lat_col].to_numpy() + r * np.sin(angles)
            df.loc[idx, lon_col] = df.loc[idx, lon_col].to_numpy() + r * np.cos(angles)
    return df

# ---------------- Map Tab ----------------
with tab_map:
    has_latlon = (
        "Lat" in dfw_filt.columns and "Lon" in dfw_filt.columns and
        dfw_filt[["Lat","Lon"]].notna().all(axis=1).any()
    )

    if not has_latlon:
        st.warning("Map needs coordinates. Add **Lat/Lon** columns or merge a `pincode_centroids.csv`.")
        st.info("Click markers to see details here (after coordinates are available).")
        selected_comm, selected_pin = None, None

    else:
        # Center & map
       
        valid = dfw_filt.dropna(subset=["Lat", "Lon"])
        valid = jitter_duplicates(valid)   # <<< add this line

        lat0 = float(valid["Lat"].mean())
        lon0 = float(valid["Lon"].mean())
        fmap = folium.Map(location=[lat0, lon0], zoom_start=11, tiles="cartodbpositron")

        # Cluster all markers (keeps performance while showing everything)
        cluster = MarkerCluster().add_to(fmap)

        # Prepare arrays
        comm_arr = valid["Community"].astype(str).to_numpy()
        pin_arr  = valid["Pincode"].astype(str).to_numpy()
        lat_arr  = valid["Lat"].astype(float).to_numpy()
        lon_arr  = valid["Lon"].astype(float).to_numpy()
        city_arr = valid["City"].astype(str).to_numpy() if "City" in valid else [""]*len(valid)

        # Compute date window once
        start_dt = pd.to_datetime(start_m + "-01")
        end_dt   = pd.to_datetime(end_m + "-01")

        # Loop once
        for comm, pin, lat, lon, city in zip(comm_arr, pin_arr, lat_arr, lon_arr, city_arr):
            stats = summarize_for_community(community_id=comm, pincode=pin)

            # >>> THIS IS THE REPLACED POPUP CONTENT <<<
            # Use image-based charts (safe for Folium popups)
            
            try:
                bar_img, donut_img = popup_charts_for_comm(dfl_filt, comm)
            except Exception as e:
                bar_img, donut_img = "", ""   # fail safe so the map still renders

            popup_html = f"""
            <div style='font-family:Poppins; width:360px;'>
                <h4 style='margin:0 0 4px 0; color:#36204D;'>{comm}</h4>
                <div style='font-size:12px; color:#333;'>City: {city} | Pincode: {pin}</div>
                <hr style='margin:6px 0;'>

                <div style='display:flex; align-items:center; margin-bottom:6px;'>
                    <img src="{TREE_ICON}" width="18">
                    <span style='margin-left:8px;'><b>{stats['Trees_saved']:,.0f} Trees Saved</b></span>
                </div>
                <div style='display:flex; align-items:center; margin-bottom:6px;'>
                    <img src="{HOUSE_ICON}" width="18">
                    <span style='margin-left:8px;'><b>{stats['HH_sum']:,.0f} Households Participating</b></span>
                </div>
                <div style='display:flex; align-items:center; margin-bottom:6px;'>
                    <img src="{RECYCLE_ICON}" width="18">
                    <span style='margin-left:8px;'><b>{stats['Seg_pct_avg']:.1f}% Segregation</b></span>
                </div>

                <hr style='margin:8px 0;'>
                <div style='margin-bottom:8px;'>
                    <b>CO₂ Averted</b>
                    {donut_img}
                </div>

                <div style='margin-top:6px;'>
                    <b>Tonnage</b>
                    {bar_img}
                </div>
            </div>
            """

        
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=BRAND_PRIMARY,
                fill=True,
                fill_color=BRAND_PRIMARY,
                fill_opacity=0.9,
                tooltip=folium.Tooltip(f"{comm} • {pin}"),
                popup=folium.Popup(popup_html, max_width=420),
            ).add_to(cluster)

        # Render map
        st.markdown("##### Map")
        map_event = st_folium(
            fmap,
            height=ST_MAP_HEIGHT,
            use_container_width=True,
            returned_objects=ST_RETURNED_OBJECTS  # don't return all markers to Streamlit
        )

        # Capture selection
        selected_comm, selected_pin = None, None
        if map_event and map_event.get("last_object_clicked_tooltip"):
            tip = map_event["last_object_clicked_tooltip"]  # "COMMUNITY • PINCODE"
            parts = [p.strip() for p in tip.split("•")]
            if len(parts) == 2:
                selected_comm, selected_pin = parts[0], parts[1]
            else:
                selected_comm = parts[0]
        elif not dfw_filt.empty:
            selected_comm = str(dfw_filt.iloc[0]["Community"])
            selected_pin  = str(dfw_filt.iloc[0]["Pincode"])


            
    # ---------------- Marker Loop Ends Here ----------------

    


       
        
    # ---- Selection & Trends moved BELOW the map ----
    # ---- Selection & Trends BELOW the map ----
st.markdown("---")
st.markdown("### Selection & Trends")

if not has_latlon or selected_comm is None:
    st.info("Click a marker to see selection details and trends here.")
else:
    # safer match on string type
    row0 = dfw_filt[dfw_filt["Community"].astype(str) == str(selected_comm)].iloc[0]
    city = str(row0.get("City", ""))

    st.markdown(
        f"<h4 style='margin:0;color:{BRAND_PRIMARY};'>{selected_comm}</h4>"
        f"<div>City: {city} | Pincode: {selected_pin}</div>",
        unsafe_allow_html=True,
    )

    # --- KPIs (CO2 from formula) ---
    cA, cB, cC, cD = st.columns(4)
    stats = summarize_for_community(community_id=selected_comm, pincode=selected_pin)
    cA.metric("Tonnage (kg)", f"{stats['Tonnage_sum']:,.0f}")
    cB.metric("CO₂ Averted (kg)", f"{stats['CO2_calc']:,.0f}")  # ← formula-based
    cC.metric("Households", f"{stats['HH_sum']:,.0f}")
    cD.metric("Segregation % (avg)", f"{stats['Seg_pct_avg']:.1f}")

    # --- Trends (brand color #36204D) ---
    st.markdown("#### Trends (Selected Community)")

    # 1) Tonnage over time (now a LINE chart)
    ton = monthly_series(dfl_filt, selected_comm, "Tonnage")
    if not ton.empty:
        fig_ton = px.line(
            ton, x="Date", y="Value",
            title="Tonnage over Time",
            labels={"Value": "Tonnage (kg)", "Date": "Date"},
            markers=True,
        )
        fig_ton.update_traces(line=dict(color=BRAND_PRIMARY, width=2),
                            marker=dict(color=BRAND_PRIMARY))
        fig_ton.update_layout(
            font=dict(color="#000"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=30, r=20, t=50, b=60),
            xaxis=dict(
                title=dict(text="Date", font=dict(color="#000")),
                tickfont=dict(color="#000"),
                gridcolor="#EEE",
                zerolinecolor="#EEE",
            ),
            yaxis=dict(
                title=dict(text="Tonnage (kg)", font=dict(color="#000")),
                tickfont=dict(color="#000"),
                gridcolor="#EEE",
                zerolinecolor="#EEE",
            ),
        )
        st.plotly_chart(fig_ton, use_container_width=True)
    else:
        st.info("No tonnage data in this date range for the selected community.")

    # 2) CO₂ Averted over time (calculated from dry waste / tonnage)
    dry_candidates = ["Tonnage_Dry", "Dry_Tonnage", "DryWaste", "Tonnage"]
    dry = None
    for m in dry_candidates:
        s = monthly_series(dfl_filt, selected_comm, m)
        if not s.empty:
            dry = s
            break

    if dry is not None and not dry.empty:
        co2 = dry.copy()
        # If your source unit is tonnes (not kg), uncomment next line:
        # co2["Value"] = co2["Value"] * 1000.0
        co2["CO2_kg"] = (co2["Value"] * CO2_PER_KG_DRY).clip(lower=0.0)

        fig_co2 = px.line(
            co2, x="Date", y="CO2_kg",
            markers=True,
            title="CO₂ Averted (Calculated) over Time",
            labels={"CO2_kg": "CO₂ Averted (kg)", "Date": "Date"},
        )
        fig_co2.update_traces(
            line=dict(color=BRAND_PRIMARY, width=2),
            marker=dict(color=BRAND_PRIMARY, size=6)
        )
        fig_co2.update_layout(
            font=dict(color="#000"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=30, r=20, t=50, b=60),
            xaxis=dict(
                title=dict(text="Date", font=dict(color="#000")),
                tickfont=dict(color="#000"),
                gridcolor="#EEE",
                zerolinecolor="#EEE",
            ),
            yaxis=dict(
                title=dict(text="CO₂ Averted (kg)", font=dict(color="#000")),
                tickfont=dict(color="#000"),
                gridcolor="#EEE",
                zerolinecolor="#EEE",
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_co2, use_container_width=True)
    else:
        st.info("No dry/tonnage series available to compute CO₂ for this community.")

        

    
    # 3) (Optional) Segregation % over time
    seg = monthly_series(dfl_filt, selected_comm, "Segregation_Compliance_Pct")
    if not seg.empty:
        fig_seg = px.line(
            seg, x="Date", y="Value",
            markers=True,
            title="Segregation % over Time",
            labels={"Value": "Segregation (%)", "Date": "Date"},
        )
        fig_seg.update_traces(
            line=dict(color=BRAND_PRIMARY, width=2),
            marker=dict(color=BRAND_PRIMARY, size=6)
        )
        fig_seg.update_layout(
            font=dict(color="#000"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=30, r=20, t=50, b=60),
            xaxis=dict(
                title=dict(text="Date", font=dict(color="#000")),
                tickfont=dict(color="#000"),
                gridcolor="#EEE",
                zerolinecolor="#EEE",
            ),
            yaxis=dict(
                title=dict(text="Segregation (%)", font=dict(color="#000")),
                tickfont=dict(color="#000"),
                gridcolor="#EEE",
                zerolinecolor="#EEE",
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_seg, use_container_width=True)



# ---------------- Insights Tab ----------------
with tab_insights:
    # ===================== BLOCK 3 — Auto Insights (cross-city) =====================
    st.markdown("### 🧠 Auto Insights (All Cities in Selected Date Range)")

    # --- Date-only filter (ignore area pickers here so we compare across cities) ---
    d0 = pd.to_datetime(start_m + "-01")
    d1 = pd.to_datetime(end_m + "-01")
    dfl_date = df_long[(df_long["Date"] >= d0) & (df_long["Date"] <= d1)].copy()

    # Ensure text types
    for col in ["City", "Community", "Pincode"]:
        if col in dfl_date.columns:
            dfl_date[col] = dfl_date[col].astype(str)

    # Helper: brand axes to purple everywhere
    def _brand_axes(fig, title=None):
        fig.update_layout(
            title=title or (fig.layout.title.text if fig.layout.title and fig.layout.title.text else None),
            font=dict(family="Poppins", color=BRAND_PRIMARY, size=14),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=30, r=20, t=50, b=60),
        )
        fig.update_xaxes(
            title_font=dict(color=BRAND_PRIMARY, size=12),  # <-- correct
            tickfont=dict(color=BRAND_PRIMARY, size=11),
            gridcolor="#EEE",
            zerolinecolor="#EEE",
        )
        fig.update_yaxes(
            title_font=dict(color=BRAND_PRIMARY, size=12),  # <-- correct
            tickfont=dict(color=BRAND_PRIMARY, size=11),
            gridcolor="#EEE",
            zerolinecolor="#EEE",
        )
        return fig


    # ---- City-level aggregations (sums for volumes, mean for %) ----
    sum_metrics  = ["Tonnage", "CO2_Kgs_Averted", "Households_Participating"]
    mean_metrics = ["Segregation_Compliance_Pct", "Impact"]

    # City totals (sums)
    city_sum = (
        dfl_date[dfl_date["Metric"].isin(sum_metrics)]
        .pivot_table(index="City", columns="Metric", values="Value", aggfunc="sum", fill_value=0.0)
        .reset_index()
    )
    # City means
    city_mean = (
        dfl_date[dfl_date["Metric"].isin(mean_metrics)]
        .pivot_table(index="City", columns="Metric", values="Value", aggfunc="mean", fill_value=0.0)
        .reset_index()
    )

    # --- KPI row: best cities ---
    colA, colB, colC, colD = st.columns(4)
    def _top_city(df, metric, how="max"):
        if df.empty or metric not in df.columns:
            return "—", 0.0
        row = df.loc[df[metric].idxmax()] if how == "max" else df.loc[df[metric].idxmin()]
        return str(row["City"]), float(row[metric])

    t_city, t_val = _top_city(city_sum, "Tonnage", "max")
    c_city, c_val = _top_city(city_sum, "CO2_Kgs_Averted", "max")
    h_city, h_val = _top_city(city_sum, "Households_Participating", "max")
    s_city, s_val = _top_city(city_mean, "Segregation_Compliance_Pct", "max")

    with colA:
        st.caption("Top city by Tonnage")
        st.subheader(t_city)
        st.success(f"↑ {t_val:,.0f}")
    with colB:
        st.caption("Top city by CO₂ averted (kg)")
        st.subheader(c_city)
        st.success(f"↑ {c_val:,.0f}")
    with colC:
        st.caption("Top city by Households")
        st.subheader(h_city)
        st.success(f"↑ {h_val:,.0f}")
    with colD:
        st.caption("Highest Avg Segregation (%)")
        st.subheader(s_city)
        st.success(f"↑ {s_val:,.1f}%")

    st.markdown("---")

    # ---- City charts (purple labels/axes) ----
    import plotly.express as px

    if not city_sum.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                city_sum.sort_values("Tonnage", ascending=False),
                x="City", y="Tonnage",
                text="Tonnage", title="Total Tonnage by City",
            )
            fig.update_traces(marker_color=BRAND_PRIMARY, texttemplate="%{text:,.0f}", textposition="outside")
            st.plotly_chart(_brand_axes(fig), use_container_width=True)
        with c2:
            fig = px.bar(
                city_sum.sort_values("CO2_Kgs_Averted", ascending=False),
                x="City", y="CO2_Kgs_Averted",
                text="CO2_Kgs_Averted", title="CO₂ Averted (kg) by City",
            )
            fig.update_traces(marker_color=BRAND_PRIMARY, texttemplate="%{text:,.0f}", textposition="outside")
            st.plotly_chart(_brand_axes(fig), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.bar(
                city_sum.sort_values("Households_Participating", ascending=False),
                x="City", y="Households_Participating",
                text="Households_Participating", title="Households by City",
            )
            fig.update_traces(marker_color=BRAND_PRIMARY, texttemplate="%{text:,.0f}", textposition="outside")
            st.plotly_chart(_brand_axes(fig), use_container_width=True)
        with c4:
            fig = px.bar(
                city_mean.sort_values("Segregation_Compliance_Pct", ascending=False),
                x="City", y="Segregation_Compliance_Pct",
                text="Segregation_Compliance_Pct", title="Avg Segregation (%) by City",
            )
            fig.update_traces(
                marker_color=BRAND_PRIMARY, texttemplate="%{text:.1f}%", textposition="outside"
            )
            st.plotly_chart(_brand_axes(fig), use_container_width=True)

    st.markdown("---")

    # ---- NEW: Top Communities by Tonnage (across all cities) ----
    topN = 10
    comm_tonn = (
        dfl_date[dfl_date["Metric"] == "Tonnage"]
        .groupby(["City", "Community", "Pincode"], as_index=False)["Value"].sum()
        .rename(columns={"Value": "Tonnage"})
        .sort_values("Tonnage", ascending=False)
        .head(topN)
    )

    st.markdown(f"#### Top {topN} Communities by Tonnage (All Cities)")
    if not comm_tonn.empty:
        fig_comm = px.bar(
            comm_tonn,
            x="Community", y="Tonnage",
            color="City",
            title="Top Communities by Total Tonnage",
            labels={"Value": "Tonnage", "Community": "Community"},
        )
        # make every visual element purple except the series colors (keep default)
        fig_comm.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
        fig_comm = _brand_axes(fig_comm)
        st.plotly_chart(fig_comm, use_container_width=True)
        st.caption("Tip: Hover a bar to see its city and pincode.")
    else:
        st.info("No community tonnage available in this date range.")

    st.markdown("---")

    # ====== Move table + download to the BOTTOM of Insights tab ======
    st.write("The table below reflects the **current filters** (city/community/pincode + date range).")
    filtered_csv = dfl_filt.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Trends (filtered CSV)",
        data=filtered_csv,
        file_name="trends_filtered.csv",
        mime="text/csv",
        key="dl_trends_bottom",  # unique key avoids duplicate-id errors
    )
    st.dataframe(dfl_filt, use_container_width=True, height=420)


