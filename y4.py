import streamlit as st
import pandas as pd
import joblib
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import requests
from tqdm import tqdm
from io import StringIO



st.set_page_config(page_title="🌾 Crop Yield Predictor", layout="wide")
st.title("🌾 Smart Crop Yield Predictor with Map Support")

# Load the trained model
@st.cache_resource

def load_model():
    return joblib.load("yield_model.pkl")

model = load_model()


# ---------- Sidebar Inputs ----------pyt
with st.sidebar:
    st.header("📋 Manual Input Parameters")
    year = st.slider("Year", 2010, 2030, 2025)
    tempmax = st.slider("Max Temperature (°C)", 15.0, 45.0, 27.5)
    tempmin = st.slider("Min Temperature (°C)", 5.0, 25.0, 10.2)
    precip = st.slider("Precipitation (mm)", 0.0, 20.0, 0.35)
    humidity = st.slider("Humidity (%)", 10.0, 100.0, 36.5)
    ndvi = st.slider("NDVI", 0.0, 1.0, 0.48)
    ndmi = st.slider("NDMI", 0.0, 1.0, 0.21)
    msavi = st.slider("MSAVI", 0.0, 1.0, 0.31)

# Prepare input DataFrame
input_df = pd.DataFrame([{
    "Year": year,
    "tempmax": tempmax,
    "tempmin": tempmin,
    "precip": precip,
    "humidity": humidity,
    "NDVI": ndvi,
    "NDMI": ndmi,
    "MSAVI": msavi
}])

# ---------- Manual Prediction ----------
st.subheader("🔍 Manual Prediction Result")

if ndvi < 0.2:
    st.warning("⚠️ NDVI too low (< 0.2). Area might be non-agricultural.")
else:
    prediction = model.predict(input_df)[0]
    st.success(f"🌱 Predicted Yield: **{prediction:.2f} kg per acre**")

# ---------- KML Upload + Map ----------
st.header("🗺️ Upload Field KML")

uploaded_kml = st.file_uploader("📂 Upload your KML file", type="kml")


def get_coord(polygon):

    # Get the exterior coordinates (list of (lon, lat) tuples)
    coords = list(polygon.exterior.coords)

    # Separate into longitudes and latitudes
    longitudes = [pt[0] for pt in coords]
    latitudes = [pt[1] for pt in coords]

    # Calculate the average
    avg_lon = sum(longitudes) / len(longitudes)
    avg_lat = sum(latitudes) / len(latitudes)

    # Print the results
    print("Longitude List:", longitudes)
    print("Latitude List:", latitudes)
    print("Average Longitude:", avg_lon)
    print("Average Latitude:", avg_lat) # List of (lon, lat) tuples
    return (avg_lon, avg_lat)

def get_params(longitude, latitude):
    # Updated parameters to ensure we get precipitation
    parameters = "T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M"  # Using corrected precipitation

    # Optimized date chunks
    year_chunks = [
        ("20000115", "20000315"),
        ("20010115", "20010315"),
        ("20020115", "20020315"),
        ("20030115", "20030315"),
        ("20040115", "20040315"),
        ("20050115", "20050315"),
        ("20060115", "20060315"),
        ("20070101", "20071231"),
        ("20080101", "20081231"),
        ("20090101", "20091231"),
        ("20100101", "20100315"),
        ("20110115", "20110315"),
        ("20120115", "20120315"),
        ("20130101", "20131231"),
        ("20140101", "20141231"),
        ("20150101", "20151231"),
        ("20160101", "20160315"),
        ("20170115", "20170315"),
        ("20180101", "20181231"),
        ("20190101", "20191231"),
        ("20200101", "20201231"),
        ("20210101", "20210315"),
        ("20220101", "20221231"),
        ("20230101", "20230315"),
    ]


    all_data = []

    for start, end in tqdm(year_chunks, desc="Downloading data"):
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": parameters,
            "community": "AG",
            "longitude": longitude,
            "latitude": latitude,
            "start": start,
            "end": end,
            "format": "CSV"
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                # Find data start line more robustly
                lines = [line.strip() for line in response.text.split('\n') if line.strip()]
                header_line = next(i for i, line in enumerate(lines)
                                if line.startswith(('YEAR', 'YYYYMMDD')))

                # Read data with proper headers
                chunk_df = pd.read_csv(StringIO('\n'.join(lines[header_line:])),
                                    skiprows=0)

                # Clean column names
                chunk_df.columns = [col.split('(')[0].strip().upper() for col in chunk_df.columns]
                all_data.append(chunk_df)
            else:
                print(f"Failed for {start}-{end}: Status {response.status_code}")
        except Exception as e:
            print(f"Error for {start}-{end}: {str(e)}")

    if not all_data:
        raise ValueError("No data was downloaded - please check parameters")

    # Combine all data
    full_df = pd.concat(all_data)

    # Standardize columns - handle different API versions
    column_mapping = {
        'T2M_MAX': 'Max_Temp_C',
        'T2M_MIN': 'Min_Temp_C',
        'PRECTOTCORR': 'Precipitation_mm',
        'RH2M': 'Humidity_pct',
        'YEAR': 'Year',
        'DOY': 'Day_of_Year',
        'YYYYMMDD': 'Date_Number'
    }

    # Rename columns
    full_df = full_df.rename(columns={k.upper(): v for k, v in column_mapping.items()
                                    if k.upper() in full_df.columns})

    # Create proper Date column
    if 'Year' in full_df.columns and 'Day_of_Year' in full_df.columns:
        full_df['Date'] = pd.to_datetime(
            full_df['Year'].astype(str) + '-' +
            full_df['Day_of_Year'].astype(str),
            format='%Y-%j'
        )
    elif 'Date_Number' in full_df.columns:
        full_df['Date'] = pd.to_datetime(full_df['Date_Number'], format='%Y%m%d')

    # Select and order our desired columns
    final_columns = [
        'Date',
        'Max_Temp_C',
        'Min_Temp_C',
        'Precipitation_mm',
        'Humidity_pct'
    ]

    # Ensure we only keep columns that exist
    final_df = full_df[[col for col in final_columns if col in full_df.columns]]

    # Fill any missing precipitation with 0 (assuming no rain)
    if 'Precipitation_mm' in final_df.columns:
        final_df['Precipitation_mm'] = final_df['Precipitation_mm'].fillna(0)
    
     # Calculate average of each numeric column (excluding Date)
    avg_values = final_df.drop(columns=['Date']).mean()

    # Return the averages as a dictionary
    return avg_values.to_dict()
    

if uploaded_kml:
    try:
        gdf = gpd.read_file(uploaded_kml, driver="KML")
        st.success("✅ KML Loaded Successfully")

        print(gdf.geometry)

        polygon = gdf.geometry.iloc[0]

        avg_lon, avg_lat = get_coord(polygon)

        Sget_params(avg_lon, avg_lat)
        


        st.subheader("📋 Field Attributes")
        st.dataframe(gdf.drop(columns="geometry"))

        st.subheader("🌍 Field Map View")
        center = gdf.geometry.iloc[0].centroid
        m = folium.Map(location=[center.y, center.x], zoom_start=15)
        folium.GeoJson(gdf, tooltip="KML Layer").add_to(m)
        st_folium(m, width=700, height=500)

        st.info("✅ KML shown on map. You can still use sliders for prediction.")

    except Exception as e:
        st.error(f"❌ Error reading KML: {e}")
else:
    st.info("👆 Upload a `.kml` file to visualize your field on the map.")

# ---------- Model Info ----------
with st.expander("ℹ️ About this App"):
    st.markdown("""
    - This tool uses a **trained Random Forest model** to predict crop yield.
    - Inputs include **remote sensing indices** (NDVI, NDMI, MSAVI) and weather.
    - NDVI < 0.2 is flagged as **residential or non-agricultural**.
    - Upload a `.kml` file to **visualize your field**, but current version does not extract inputs from KML.
    """)

st.caption("🚀 Built by Mohsin Taj")
