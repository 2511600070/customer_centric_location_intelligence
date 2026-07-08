# Customer-Centric Location Intelligence — Streamlit Prototype V2

Prototype ini dibuat untuk project penelitian:

**Customer-Centric Location Intelligence: A Comparative Study of Machine Learning and Decision Support Methods for Strategic Site Selection Using Customer and Geospatial Data**

## Menu Prototype

1. Dashboard Overview
2. Customer Distribution Map
3. Customer Density Heatmap
4. Retention Risk Map
5. Market Share Defense Map
6. New Customer Potential Map
7. Candidate Location Analysis
8. Radius Analysis
9. Strategic Location Ranking
10. Explainability Dashboard
11. Executive Report
12. Data Dictionary

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Atau pada Windows:

```bash
run_streamlit.bat
```

## Catatan

- Dataset yang digunakan adalah dataset dummy V1 untuk lima kota: Jakarta, Bandung, Semarang, Yogyakarta, dan Surabaya.
- Prototype V2 ini sudah berisi menu lengkap sesuai scope penelitian.
- Explainability masih menggunakan proxy feature importance. Pada tahap eksperimen final, bagian ini dapat diganti dengan SHAP atau feature importance dari Random Forest/XGBoost/LightGBM.
- Output prototype diarahkan untuk mendukung tiga tujuan bisnis: customer retention, market share protection, dan new customer acquisition.
