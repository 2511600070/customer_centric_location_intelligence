
import math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Customer-Centric Location Intelligence V2",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

CITY_ORDER = ["Jakarta", "Bandung", "Semarang", "Yogyakarta", "Surabaya"]
CATEGORY_ORDER = ["Sangat Strategis", "Strategis", "Cukup Potensial", "Kurang Potensial"]

CATEGORY_COLORS = {
    "Sangat Strategis": "#1a9850",
    "Strategis": "#91cf60",
    "Cukup Potensial": "#fee08b",
    "Kurang Potensial": "#d73027",
    "Active": "#1a9850",
    "Risk": "#fdae61",
    "Inactive": "#d73027",
}

OBJECTIVE_LABEL = {
    "Customer retention": "Customer Retention",
    "Market share protection": "Market Share Defense",
    "New customer acquisition": "New Customer Acquisition",
    "Balanced potential": "Balanced Potential",
    "Acquisition": "New Customer Acquisition",
    "Defense": "Market Share Defense",
    "Retention": "Customer Retention",
}

@st.cache_data
def load_data():
    data = {}
    files = {
        "customer": "customer_master.csv",
        "branch": "branch_service_point.csv",
        "competitor": "competitor_location.csv",
        "poi": "poi_facility.csv",
        "demographic": "demographic_area.csv",
        "mobility": "mobility_area.csv",
        "candidate": "candidate_location_ranking_proxy_v1.csv",
        "grid": "modeling_location_grid_scored_v1.csv",
        "ranking": "strategic_location_ranking_v1.csv",
        "top25": "top_25_strategic_locations_v1.csv",
        "dictionary": "data_dictionary.csv",
        "fi": "feature_importance_proxy_v2.csv",
        "method": "method_comparison_proxy_v2.csv",
        "executive": "executive_summary_by_city_v2.csv",
    }
    for key, filename in files.items():
        path = DATA_DIR / filename
        if path.exists():
            data[key] = pd.read_csv(path)
        else:
            data[key] = pd.DataFrame()
    return data


def fmt_num(v, digits=0):
    try:
        if pd.isna(v):
            return "-"
        if abs(float(v)) >= 1_000_000:
            return f"{float(v)/1_000_000:.1f}M"
        if abs(float(v)) >= 1_000:
            return f"{float(v):,.0f}"
        return f"{float(v):,.{digits}f}"
    except Exception:
        return str(v)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))


def add_distance(df, lat, lon, lat_col="latitude", lon_col="longitude"):
    if df.empty or lat_col not in df.columns or lon_col not in df.columns:
        return df.copy()
    out = df.copy()
    out["distance_km"] = [haversine_km(lat, lon, r[lat_col], r[lon_col]) for _, r in out.iterrows()]
    return out


def normalize_series(s, inverse=False):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    mn, mx = s.min(), s.max()
    if mx == mn:
        res = pd.Series(np.ones(len(s))*50, index=s.index)
    else:
        res = (s - mn) / (mx - mn) * 100
    if inverse:
        res = 100 - res
    return res


def calc_custom_score(df, weights):
    out = df.copy()
    score = (
        weights["customer_density"] * normalize_series(out["customer_density_score"]) +
        weights["retention_risk"] * normalize_series(out["retention_risk_score"]) +
        weights["market_share"] * normalize_series(out["market_share_defense_score"]) +
        weights["new_customer"] * normalize_series(out["new_customer_potential_score"]) +
        weights["accessibility"] * normalize_series(out["accessibility_score"]) -
        weights["competition_pressure"] * normalize_series(out["competition_pressure_score"])
    )
    # rescale 0-100 after weighted combination
    out["custom_strategic_score"] = normalize_series(score).round(2)
    return out.sort_values("custom_strategic_score", ascending=False)


def filtered(data, cities):
    result = {}
    for key, df in data.items():
        if not df.empty and "city" in df.columns:
            result[key] = df[df["city"].isin(cities)].copy()
        else:
            result[key] = df.copy()
    return result


def map_center(df):
    lat_col = "centroid_latitude" if "centroid_latitude" in df.columns else "latitude"
    lon_col = "centroid_longitude" if "centroid_longitude" in df.columns else "longitude"
    if df.empty or lat_col not in df.columns or lon_col not in df.columns:
        return [-7.2, 110.5], 6
    return [float(df[lat_col].mean()), float(df[lon_col].mean())], 7 if df["city"].nunique() > 1 else 11


def score_color(value):
    if value >= 70:
        return "green"
    if value >= 55:
        return "orange"
    return "red"


def render_folium(m, height=560):
    st_folium(m, width=None, height=height, returned_objects=[])


def header(title, subtitle):
    st.markdown(f"# {title}")
    st.caption(subtitle)


def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


def make_base_map(df, tiles="CartoDB positron"):
    center, zoom = map_center(df)
    return folium.Map(location=center, zoom_start=zoom, tiles=tiles, control_scale=True)

# Load and filter data
raw = load_data()

st.sidebar.title("📍 CCLI Prototype V2")
st.sidebar.caption("Customer-Centric Location Intelligence")
selected_cities = st.sidebar.multiselect(
    "Pilih kota",
    CITY_ORDER,
    default=CITY_ORDER,
)
if not selected_cities:
    selected_cities = CITY_ORDER

data = filtered(raw, selected_cities)

menu = st.sidebar.radio(
    "Menu Prototype",
    [
        "Dashboard Overview",
        "Customer Distribution Map",
        "Customer Density Heatmap",
        "Retention Risk Map",
        "Market Share Defense Map",
        "New Customer Potential Map",
        "Candidate Location Analysis",
        "Radius Analysis",
        "Strategic Location Ranking",
        "Explainability Dashboard",
        "Executive Report",
        "Data Dictionary",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("**Novelty Positioning**")
st.sidebar.info("From ordinary site selection into customer-centric strategic location intelligence.")
st.sidebar.markdown("**Business Objective**")
st.sidebar.write("Customer retention • Market share protection • New customer acquisition")

customer = data["customer"]
grid = data["grid"]
branch = data["branch"]
competitor = data["competitor"]
poi = data["poi"]
candidate = data["candidate"]
fi = data["fi"]
method = data["method"]
executive = data["executive"]

if menu == "Dashboard Overview":
    header("Dashboard Overview", "Ringkasan customer, lokasi, dan skor strategis untuk lima kota pilot.")
    total_customer = len(customer)
    total_area = len(grid)
    avg_score = grid["strategic_location_score"].mean() if not grid.empty else 0
    total_candidate = len(candidate)
    kpi_row([
        ("Total Customer", fmt_num(total_customer), "Jumlah customer existing pada dataset dummy."),
        ("Total Area/Grid", fmt_num(total_area), "Jumlah area/grid analisis."),
        ("Avg Strategic Score", f"{avg_score:.2f}", "Rata-rata Strategic Location Score."),
        ("Candidate Location", fmt_num(total_candidate), "Jumlah kandidat lokasi yang dianalisis."),
    ])

    c1, c2 = st.columns([1.1, 1])
    with c1:
        city_customer = customer.groupby("city").size().reindex(selected_cities).dropna().reset_index(name="customer_count")
        fig = px.bar(city_customer, x="city", y="customer_count", text="customer_count", title="Distribusi Customer per Kota")
        fig.update_layout(height=360, xaxis_title="Kota", yaxis_title="Jumlah Customer")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        cat_count = grid["recommendation_category"].value_counts().reindex(CATEGORY_ORDER).dropna().reset_index()
        cat_count.columns = ["category", "count"]
        fig = px.pie(cat_count, names="category", values="count", title="Distribusi Kategori Rekomendasi")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ringkasan Score per Kota")
    if not executive.empty:
        cols_show = ["city","total_customer","total_area","avg_strategic_score","avg_customer_density_score","avg_retention_risk_score","avg_market_share_defense_score","avg_new_customer_potential_score","total_competitor","total_poi"]
        st.dataframe(executive[cols_show].sort_values("avg_strategic_score", ascending=False), use_container_width=True, hide_index=True)

elif menu == "Customer Distribution Map":
    header("Customer Distribution Map", "Menampilkan sebaran customer existing berdasarkan status customer.")
    status_filter = st.multiselect("Filter status customer", sorted(customer["customer_status"].dropna().unique()), default=sorted(customer["customer_status"].dropna().unique()))
    sample_size = st.slider("Jumlah titik customer yang ditampilkan", 500, min(5500, len(customer)), min(2500, len(customer)), step=250)
    df = customer[customer["customer_status"].isin(status_filter)].sample(min(sample_size, len(customer)), random_state=42) if not customer.empty else customer
    m = make_base_map(df)
    cluster = MarkerCluster(name="Customer existing").add_to(m)
    for _, r in df.iterrows():
        color = CATEGORY_COLORS.get(r.get("customer_status", "Active"), "blue")
        folium.CircleMarker(
            [r["latitude"], r["longitude"]],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.65,
            popup=f"Customer: {r['customer_id']}<br>Status: {r['customer_status']}<br>Segment: {r['customer_segment']}<br>City: {r['city']}",
        ).add_to(cluster)
    for _, b in branch.iterrows():
        folium.Marker([b["latitude"], b["longitude"]], popup=f"{b['branch_name']}<br>{b['branch_type']}", icon=folium.Icon(color="blue", icon="wrench", prefix="fa")).add_to(m)
    folium.LayerControl().add_to(m)
    render_folium(m)

elif menu == "Customer Density Heatmap":
    header("Customer Density Heatmap", "Menampilkan area dengan kepadatan customer existing tinggi.")
    m = make_base_map(customer)
    heat_data = customer[["latitude","longitude"]].dropna().values.tolist()
    HeatMap(heat_data, radius=13, blur=16, min_opacity=0.25, name="Customer density heatmap").add_to(m)
    for _, b in branch.iterrows():
        folium.Marker([b["latitude"], b["longitude"]], popup=b["branch_name"], icon=folium.Icon(color="blue", icon="building", prefix="fa")).add_to(m)
    folium.LayerControl().add_to(m)
    render_folium(m)
    st.subheader("Top Area by Customer Density Score")
    top = grid.sort_values("customer_density_score", ascending=False).head(20)
    st.dataframe(top[["area_id","city","district","subdistrict","customer_count","active_customer_ratio","customer_density_score","strategic_location_score"]], use_container_width=True, hide_index=True)

elif menu == "Retention Risk Map":
    header("Retention Risk Map", "Menampilkan area dengan risiko kehilangan customer berdasarkan inactivity, churn risk, recency, dan jarak layanan.")
    threshold = st.slider("Minimum Retention Risk Score", 0, 100, 50)
    df = grid[grid["retention_risk_score"] >= threshold].copy()
    m = make_base_map(df if not df.empty else grid)
    for _, r in df.iterrows():
        folium.CircleMarker(
            [r["centroid_latitude"], r["centroid_longitude"]],
            radius=5 + r["retention_risk_score"] / 12,
            color=score_color(r["retention_risk_score"]),
            fill=True,
            fill_opacity=0.55,
            popup=f"{r['area_id']} - {r['district']}<br>Retention Risk: {r['retention_risk_score']:.2f}<br>Inactive Ratio: {r['inactive_customer_ratio']:.2f}<br>Avg Recency: {r['avg_recency_days']:.1f} days",
        ).add_to(m)
    render_folium(m)
    st.subheader("Area Prioritas Retention")
    st.dataframe(df.sort_values("retention_risk_score", ascending=False)[["area_id","city","district","subdistrict","customer_count","inactive_customer_ratio","avg_recency_days","distance_to_nearest_branch_km","retention_risk_score"]].head(30), use_container_width=True, hide_index=True)

elif menu == "Market Share Defense Map":
    header("Market Share Defense Map", "Menampilkan area yang perlu dipertahankan dari tekanan kompetitor.")
    df = grid.sort_values("market_share_defense_score", ascending=False).head(60)
    m = make_base_map(df)
    for _, r in df.iterrows():
        folium.CircleMarker(
            [r["centroid_latitude"], r["centroid_longitude"]],
            radius=5 + r["market_share_defense_score"] / 12,
            color="purple",
            fill=True,
            fill_opacity=0.45,
            popup=f"{r['area_id']} - {r['district']}<br>Defense Score: {r['market_share_defense_score']:.2f}<br>Customer: {r['customer_count']}<br>Competitor: {r['competitor_count']}",
        ).add_to(m)
    comp_cluster = MarkerCluster(name="Competitor").add_to(m)
    for _, c in competitor.iterrows():
        folium.Marker([c["latitude"], c["longitude"]], popup=f"{c['competitor_name']}<br>{c['competitor_type']}<br>Strength: {c['estimated_strength']}", icon=folium.Icon(color="red", icon="flag", prefix="fa")).add_to(comp_cluster)
    folium.LayerControl().add_to(m)
    render_folium(m)
    st.subheader("Top Area Market Share Defense")
    st.dataframe(df[["area_id","city","district","subdistrict","customer_count","active_customer_ratio","competitor_count","competition_pressure_index","market_share_defense_score"]], use_container_width=True, hide_index=True)

elif menu == "New Customer Potential Map":
    header("New Customer Potential Map", "Menampilkan area potensial untuk akuisisi customer baru berdasarkan populasi, POI, mobilitas, dan penetrasi customer.")
    df = grid.sort_values("new_customer_potential_score", ascending=False).head(70)
    m = make_base_map(df)
    for _, r in df.iterrows():
        folium.CircleMarker(
            [r["centroid_latitude"], r["centroid_longitude"]],
            radius=5 + r["new_customer_potential_score"] / 11,
            color="green",
            fill=True,
            fill_opacity=0.45,
            popup=f"{r['area_id']} - {r['district']}<br>Acquisition Score: {r['new_customer_potential_score']:.2f}<br>Population: {r['population_count']:,}<br>Penetration: {r['customer_penetration_ratio']:.5f}<br>POI: {r['poi_count']}",
        ).add_to(m)
    render_folium(m)
    st.subheader("Top Area New Customer Potential")
    st.dataframe(df[["area_id","city","district","subdistrict","population_count","customer_penetration_ratio","poi_count","mobility_density","new_customer_potential_score"]], use_container_width=True, hide_index=True)

elif menu == "Candidate Location Analysis":
    header("Candidate Location Analysis", "Menganalisis titik kandidat lokasi dan area/grid terdekat.")
    if candidate.empty:
        st.warning("Candidate dataset tidak tersedia.")
    else:
        selected = st.selectbox("Pilih kandidat lokasi", candidate.sort_values("candidate_rank")["candidate_name"].tolist())
        row = candidate[candidate["candidate_name"] == selected].iloc[0]
        area = grid[grid["area_id"] == row.get("nearest_area_id")]
        cols = st.columns(4)
        cols[0].metric("Rank", int(row["candidate_rank"]))
        cols[1].metric("Proxy Score", f"{row['strategic_location_score_proxy']:.2f}")
        cols[2].metric("Kategori", row["recommendation_category_proxy"])
        cols[3].metric("Fokus Bisnis", row["business_focus_proxy"])
        c1, c2 = st.columns([1.2, 1])
        with c1:
            m = folium.Map(location=[row["latitude"], row["longitude"]], zoom_start=13, tiles="CartoDB positron", control_scale=True)
            folium.Marker([row["latitude"], row["longitude"]], popup=f"{row['candidate_name']}<br>Score: {row['strategic_location_score_proxy']:.2f}", icon=folium.Icon(color="green", icon="star", prefix="fa")).add_to(m)
            if not area.empty:
                a = area.iloc[0]
                folium.CircleMarker([a["centroid_latitude"], a["centroid_longitude"]], radius=12, color="blue", fill=True, fill_opacity=0.4, popup=f"Nearest area: {a['area_id']}<br>Strategic Score: {a['strategic_location_score']:.2f}").add_to(m)
                folium.PolyLine([[row["latitude"], row["longitude"]], [a["centroid_latitude"], a["centroid_longitude"]]], color="gray", weight=2, opacity=0.7).add_to(m)
            render_folium(m, height=500)
        with c2:
            st.subheader("Detail Kandidat")
            st.dataframe(row.to_frame("value"), use_container_width=True)
            if not area.empty:
                st.subheader("Area/Grid Terdekat")
                fields = ["area_id","city","district","subdistrict","customer_count","population_count","competitor_count","poi_count","strategic_location_score","recommendation_category","primary_business_objective"]
                st.dataframe(area[fields].T.rename(columns={area.index[0]: "value"}), use_container_width=True)
        st.subheader("Ranking Kandidat Lokasi")
        st.dataframe(candidate.sort_values("candidate_rank"), use_container_width=True, hide_index=True)

elif menu == "Radius Analysis":
    header("Radius Analysis", "Menghitung customer, POI, kompetitor, dan populasi dalam radius tertentu dari kandidat lokasi.")
    if candidate.empty:
        st.warning("Candidate dataset tidak tersedia.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            selected = st.selectbox("Pilih kandidat lokasi untuk radius analysis", candidate.sort_values("candidate_rank")["candidate_name"].tolist())
        with col2:
            radius = st.slider("Radius analisis (km)", 1, 10, 3)
        row = candidate[candidate["candidate_name"] == selected].iloc[0]
        lat, lon = float(row["latitude"]), float(row["longitude"])
        cust_r = add_distance(customer, lat, lon)
        poi_r = add_distance(poi, lat, lon)
        comp_r = add_distance(competitor, lat, lon)
        grid_r = add_distance(grid.rename(columns={"centroid_latitude":"latitude","centroid_longitude":"longitude"}), lat, lon)
        cust_in = cust_r[cust_r["distance_km"] <= radius]
        poi_in = poi_r[poi_r["distance_km"] <= radius]
        comp_in = comp_r[comp_r["distance_km"] <= radius]
        grid_in = grid_r[grid_r["distance_km"] <= radius]
        kpi_row([
            ("Customer dalam radius", fmt_num(len(cust_in)), "Jumlah customer existing dalam radius."),
            ("POI dalam radius", fmt_num(len(poi_in)), "Jumlah fasilitas umum/POI dalam radius."),
            ("Kompetitor dalam radius", fmt_num(len(comp_in)), "Jumlah kompetitor dalam radius."),
            ("Estimasi Populasi", fmt_num(grid_in["population_count"].sum()), "Total populasi area/grid yang masuk radius."),
        ])
        m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB positron", control_scale=True)
        folium.Marker([lat, lon], popup=row["candidate_name"], icon=folium.Icon(color="green", icon="star", prefix="fa")).add_to(m)
        folium.Circle([lat, lon], radius=radius*1000, color="green", fill=False, weight=2, popup=f"Radius {radius} km").add_to(m)
        for _, r in cust_in.sample(min(len(cust_in), 500), random_state=42).iterrows():
            folium.CircleMarker([r["latitude"], r["longitude"]], radius=2.5, color=CATEGORY_COLORS.get(r["customer_status"], "blue"), fill=True, fill_opacity=0.5).add_to(m)
        for _, r in comp_in.iterrows():
            folium.Marker([r["latitude"], r["longitude"]], popup=r["competitor_name"], icon=folium.Icon(color="red", icon="flag", prefix="fa")).add_to(m)
        for _, r in poi_in.head(120).iterrows():
            folium.CircleMarker([r["latitude"], r["longitude"]], radius=3, color="orange", fill=True, fill_opacity=0.5, popup=f"{r['poi_name']}<br>{r['poi_category']}").add_to(m)
        render_folium(m, height=540)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Customer Status")
            if not cust_in.empty:
                st.dataframe(cust_in["customer_status"].value_counts().reset_index().rename(columns={"index":"status","customer_status":"count"}), use_container_width=True, hide_index=True)
        with c2:
            st.subheader("POI Category")
            if not poi_in.empty:
                st.dataframe(poi_in["poi_category"].value_counts().reset_index().rename(columns={"index":"poi_category","poi_category":"count"}), use_container_width=True, hide_index=True)
        with c3:
            st.subheader("Competitor Type")
            if not comp_in.empty:
                st.dataframe(comp_in["competitor_type"].value_counts().reset_index().rename(columns={"index":"competitor_type","competitor_type":"count"}), use_container_width=True, hide_index=True)

elif menu == "Strategic Location Ranking":
    header("Strategic Location Ranking", "Menampilkan ranking lokasi terbaik berdasarkan Strategic Location Score.")
    category = st.multiselect("Filter kategori rekomendasi", CATEGORY_ORDER, default=CATEGORY_ORDER)
    top_n = st.slider("Top N area", 10, min(100, len(grid)), 25, step=5)
    df = grid[grid["recommendation_category"].isin(category)].sort_values("strategic_location_score", ascending=False).head(top_n)
    fig = px.bar(df.sort_values("strategic_location_score"), x="strategic_location_score", y="area_id", orientation="h", color="recommendation_category", title=f"Top {top_n} Strategic Location Score")
    fig.update_layout(height=650, xaxis_title="Strategic Location Score", yaxis_title="Area ID")
    st.plotly_chart(fig, use_container_width=True)
    cols = ["area_id","city","district","subdistrict","customer_count","population_count","competitor_count","poi_count","customer_density_score","retention_risk_score","market_share_defense_score","new_customer_potential_score","accessibility_score","competition_pressure_score","strategic_location_score","recommendation_category","primary_business_objective"]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)
    st.download_button("Download ranking CSV", df[cols].to_csv(index=False).encode("utf-8"), "strategic_location_ranking_filtered.csv", "text/csv")

elif menu == "Explainability Dashboard":
    header("Explainability Dashboard", "Menampilkan faktor utama yang memengaruhi rekomendasi lokasi.")
    st.info("Pada prototype V2 ini, explainability ditampilkan sebagai feature importance proxy berbasis korelasi terhadap Strategic Location Score. Pada eksperimen final, bagian ini dapat diganti dengan Feature Importance Random Forest/XGBoost atau SHAP.")
    if not fi.empty:
        top_n = st.slider("Jumlah faktor utama", 5, 20, 12)
        top = fi.head(top_n).sort_values("importance_percent")
        fig = px.bar(top, x="importance_percent", y="feature", orientation="h", color="direction", title="Top Feature Importance Proxy")
        fig.update_layout(height=560, xaxis_title="Importance (%)", yaxis_title="Feature")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Interpretasi Faktor")
        interpretation = []
        for _, r in fi.head(8).iterrows():
            if r["direction"] == "positive":
                msg = f"Semakin tinggi **{r['feature']}**, semakin cenderung meningkatkan Strategic Location Score."
            else:
                msg = f"Semakin tinggi **{r['feature']}**, semakin cenderung menurunkan Strategic Location Score."
            interpretation.append(msg)
        for msg in interpretation:
            st.markdown(f"- {msg}")
        st.dataframe(fi, use_container_width=True, hide_index=True)
    st.subheader("Method Comparison Proxy")
    if not method.empty:
        mdf = method.copy()
        show = mdf[["method","method_group","accuracy_proxy","precision_proxy","recall_proxy","f1_proxy","roc_auc_proxy","business_interpretability_proxy","notes"]]
        st.dataframe(show, use_container_width=True, hide_index=True)
        fig = px.bar(mdf.dropna(subset=["f1_proxy"]), x="method", y=["f1_proxy","business_interpretability_proxy"], barmode="group", title="Perbandingan F1 Proxy vs Business Interpretability")
        fig.update_layout(height=420, xaxis_title="Metode", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

elif menu == "Executive Report":
    header("Executive Report", "Ringkasan rekomendasi untuk manajemen.")
    st.markdown("### Executive Summary")
    total_customer = len(customer)
    top_city = executive.sort_values("avg_strategic_score", ascending=False).iloc[0]["city"] if not executive.empty else "-"
    top_area = grid.sort_values("strategic_location_score", ascending=False).iloc[0] if not grid.empty else None
    st.success(f"Prototype ini menganalisis **{fmt_num(total_customer)} customer existing** pada **{len(selected_cities)} kota pilot**. Kota dengan rata-rata Strategic Location Score tertinggi pada filter saat ini adalah **{top_city}**.")
    if top_area is not None:
        st.markdown(f"Area prioritas tertinggi adalah **{top_area['area_id']} - {top_area['district']}, {top_area['city']}** dengan Strategic Location Score **{top_area['strategic_location_score']:.2f}** dan fokus bisnis **{OBJECTIVE_LABEL.get(top_area.get('primary_business_objective'), top_area.get('primary_business_objective'))}**.")
    st.markdown("### Rekomendasi Manajemen")
    st.markdown("""
1. Prioritaskan area **Sangat Strategis** untuk kajian lokasi lanjutan, termasuk validasi lapangan, biaya sewa, dan potensi kapasitas layanan.
2. Gunakan **Retention Risk Map** untuk menentukan area yang membutuhkan campaign retention, mobile service, atau peningkatan coverage layanan.
3. Gunakan **Market Share Defense Map** untuk mempertahankan area customer kuat yang mulai mendapat tekanan kompetitor.
4. Gunakan **New Customer Potential Map** untuk menemukan area populasi tinggi, POI padat, mobilitas tinggi, tetapi penetrasi customer masih rendah.
5. Gunakan **Radius Analysis** sebelum memilih titik final agar customer, POI, kompetitor, dan populasi sekitar kandidat dapat dihitung secara objektif.
""")
    st.subheader("Top 20 Area Prioritas")
    cols = ["area_id","city","district","subdistrict","customer_count","population_count","strategic_location_score","recommendation_category","primary_business_objective"]
    top = grid.sort_values("strategic_location_score", ascending=False).head(20)[cols]
    st.dataframe(top, use_container_width=True, hide_index=True)
    st.subheader("Ringkasan per Kota")
    if not executive.empty:
        st.dataframe(executive.sort_values("avg_strategic_score", ascending=False), use_container_width=True, hide_index=True)
    report_csv = top.to_csv(index=False).encode("utf-8")
    st.download_button("Download Executive Top Area CSV", report_csv, "executive_top_area_recommendation.csv", "text/csv")

elif menu == "Data Dictionary":
    header("Data Dictionary", "Dokumentasi ringkas dataset yang digunakan dalam prototype.")
    dd = data["dictionary"]
    if not dd.empty:
        st.dataframe(dd, use_container_width=True, hide_index=True)
    st.subheader("Daftar Dataset")
    files = sorted([p.name for p in DATA_DIR.glob("*.csv")])
    st.write(files)

# Footer
st.divider()
st.caption("Prototype V2 — Customer-Centric Location Intelligence. Dataset dummy untuk kebutuhan riset, eksperimen model, dan presentasi. Jangan gunakan data dummy ini sebagai keputusan bisnis final tanpa validasi data aktual.")
