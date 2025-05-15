import streamlit as st
import pandas as pd
import joblib
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import ee
import geemap
import datetime
import uuid

# Set page configuration
st.set_page_config(page_title="🌾 Crop Yield Predictor", layout="wide")

# Inject CSS to reduce padding and style prediction box
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .st-emotion-cache-1rtdr02 {
        margin-bottom: 0.5rem;
    }
    .yield-prediction {
        margin-top: 10px;
        margin-bottom: 20px;
        padding: 25px;
        background-color: #e8f4f8;
        border-radius: 10px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        font-size: 1.4rem;
        font-weight: 700;
        border-left: 5px solid #28a745;
    }
    .yield-prediction h3 {
        color: #1a5f2d;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize GEE
try:
    ee.Initialize(project='yeildprediction')
except Exception as e:
    st.error(f"❌ GEE Authentication Failed: {str(e)}")
    st.info("Run `earthengine authenticate --project yeildprediction` in your terminal.")
    st.stop()

st.title("🌾 Smart Crop Yield Predictor with Google Earth Engine")
st.success("✅ GEE Initialized Successfully")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("yield_model.pkl")

model = load_model()

# ---------- Upload KML ----------
st.header("🗺️ Upload Field KML")
uploaded_kml = st.file_uploader("📂 Upload your KML file", type="kml")

coords = None
weather_data = {}

if uploaded_kml:
    try:
        gdf = gpd.read_file(uploaded_kml, driver="KML")
        # st.success("✅ KML Loaded Successfully")

        non_geom_cols = [col for col in gdf.columns if col != "geometry"]
        if non_geom_cols:
            st.header("📋 Field Attributes")
            st.dataframe(gdf[non_geom_cols])
        else:
            st.info("No attribute fields found other than geometry.")

        center = gdf.geometry.iloc[0].centroid
        lat, lon = center.y, center.x
        coords = (lat, lon)

        # ---------- Select Date ----------
        st.sidebar.subheader("📅 Select Date")
        selected_date = st.sidebar.date_input(
            "Choose a date for weather data",
            datetime.date(2025, 1, 15),
            min_value=datetime.date(2010, 1, 1),
            max_value=datetime.date.today()
        )
        selected_date = datetime.datetime.combine(selected_date, datetime.datetime.min.time())
        ee_date = ee.Date(selected_date)
        end_date = ee_date.advance(1, 'day')

        # Define time difference function
        def add_time_diff(image):
            time_diff = ee.Number(image.date().difference(ee_date, 'second')).abs()
            return image.set('time_diff', time_diff)

        # ---------- Weather Data ----------
        point = ee.Geometry.Point([lon, lat])
        dataset = ee.ImageCollection('NOAA/CFSV2/FOR6H') \
            .filterBounds(point) \
            .select([
                'Temperature_height_above_ground',
                'Precipitation_rate_surface_6_Hour_Average',
                'Specific_humidity_height_above_ground',
                'Pressure_surface'
            ])

        # First, try to get data for the exact date
        dataset_exact = dataset.filterDate(ee_date, end_date)
        image_list = dataset_exact.toList(1)
        image_count = image_list.size().getInfo()

        if image_count == 0:
            # If no data on exact date, find nearest available date within ±30 days
            # st.warning(f"⚠️ No weather data available for {selected_date.strftime('%Y-%m-%d')}. Searching for nearest available date.")
            start_date = ee_date.advance(-30, 'day')
            current_date = ee.Date(datetime.datetime.now())
            advanced_date = ee_date.advance(30, 'day')
            end_date_nearest = ee.Algorithms.If(advanced_date.difference(current_date, 'second').gt(0), current_date, advanced_date)
            dataset_nearest = dataset.filterDate(start_date, end_date_nearest)
            dataset_with_time = dataset_nearest.map(add_time_diff)
            image_list = dataset_with_time.limit(1, 'time_diff')
            image_count = image_list.size().getInfo()
            if image_count == 0:
                st.error("❌ No weather data available within ±30 days of the selected date. Try a different location or date.")
                st.stop()
            latest_image = ee.Image(image_list.first())
            image_date = ee.Date(latest_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            st.info(f"📅 Using weather data from nearest available date: {image_date}")
        else:
            latest_image = ee.Image(image_list.get(0))
            image_date = selected_date.strftime('%Y-%m-%d')
            # st.info(f"📅 Using weather data from selected date: {image_date}")

        weather_values = latest_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=1000
        ).getInfo()

        temp = weather_values.get('Temperature_height_above_ground', 300) - 273.15
        precip = weather_values.get('Precipitation_rate_surface_6_Hour_Average', 0.00003) * 86400
        specific_humidity = weather_values.get('Specific_humidity_height_above_ground', 0.008)
        pressure = weather_values.get('Pressure_surface', 101325) / 100

        es = 6.1078 * 10 ** (7.5 * temp / (temp + 237.3))
        q_sat = 0.622 * es / (pressure - es)
        humidity = min((specific_humidity / q_sat) * 100, 100)

        weather_data = {
            "Year": pd.Timestamp.now().year,
            "tempmax": temp + 1.5,
            "tempmin": temp - 1.5,
            "precip": precip,
            "humidity": humidity
        }

        # ---------- Sentinel-2 Indices ----------
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(point) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
            .select(['B4', 'B8', 'B11'])

        # First, try to get imagery for the exact date
        s2_exact = s2.filterDate(ee_date, end_date)
        image_list_s2 = s2_exact.toList(1)
        image_count_s2 = image_list_s2.size().getInfo()

        if image_count_s2 == 0:
            # If no imagery on exact date, find nearest available date within ±30 days
            # st.warning(f"⚠️ No Sentinel-2 imagery available for {selected_date.strftime('%Y-%m-%d')}. Searching for nearest available date.")
            start_date = ee_date.advance(-30, 'day')
            current_date = ee.Date(datetime.datetime.now())
            advanced_date = ee_date.advance(30, 'day')
            end_date_nearest = ee.Algorithms.If(advanced_date.difference(current_date, 'second').gt(0), current_date, advanced_date)
            s2_nearest = s2.filterDate(start_date, end_date_nearest)
            s2_with_time = s2_nearest.map(add_time_diff)
            image_list_s2 = s2_with_time.limit(1, 'time_diff')
            image_count_s2 = image_list_s2.size().getInfo()
            if image_count_s2 == 0:
                st.error("❌ No Sentinel-2 imagery found within ±30 days of the selected date. Try a different location or date.")
                st.stop()
            latest_image = ee.Image(image_list_s2.first())
            s2_image_date = ee.Date(latest_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            # st.info(f"📅 Using Sentinel-2 imagery from nearest available date: {s2_image_date}")
        else:
            latest_image = ee.Image(image_list_s2.get(0))
            s2_image_date = selected_date.strftime('%Y-%m-%d')
            # st.info(f"📅 Using Sentinel-2 imagery from selected date: {s2_image_date}")

        def calculate_indices(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')
            msavi = image.expression(
                '(2 * NIR + 1 - sqrt((2 * NIR + 1)**2 - 8 * (NIR - RED))) / 2',
                {'NIR': image.select('B8'), 'RED': image.select('B4')}
            ).rename('MSAVI')
            return image.addBands([ndvi, ndmi, msavi])

        latest_image_with_indices = calculate_indices(latest_image)
        indices = latest_image_with_indices.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10
        ).getInfo()

        ndvi = indices.get('NDVI', 0.48)
        ndmi = indices.get('NDMI', 0.21)
        msavi = indices.get('MSAVI', 0.31)

        # ---------- Sidebar Controls ----------
        st.sidebar.subheader("⛅ Weather Adjustment")
        weather_data["tempmax"] = st.sidebar.slider("Max Temperature (°C)", 2.0, 50.0, float(weather_data["tempmax"]))
        weather_data["tempmin"] = st.sidebar.slider("Min Temperature (°C)", 2.0, 50.0, float(weather_data["tempmin"]))
        weather_data["precip"] = st.sidebar.slider("Precipitation (mm)", 0.0, 500.0, float(weather_data["precip"]))
        weather_data["humidity"] = st.sidebar.slider("Humidity (%)", 20.0, 100.0, float(weather_data["humidity"]))

        st.sidebar.subheader("🛰️ Indices Adjustment")
        ndvi = st.sidebar.slider("NDVI", -1.0, 1.0, ndvi)
        ndmi = st.sidebar.slider("NDMI", -1.0, 1.0, ndmi)
        msavi = st.sidebar.slider("MSAVI", 0.0, 1.0, msavi)

        # ---------- Yield Prediction (PROMINENTLY DISPLAYED BEFORE MAP) ----------
        st.subheader("🔍 Yield Prediction")
        if coords and weather_data:
            input_df = pd.DataFrame([{
                "Year": weather_data["Year"],
                "tempmax": weather_data["tempmax"],
                "tempmin": weather_data["tempmin"],
                "precip": weather_data["precip"],
                "humidity": weather_data["humidity"],
                "NDVI": ndvi,
                "NDMI": ndmi,
                "MSAVI": msavi
            }])

            if ndvi < 0.2:
                st.warning("⚠️ NDVI too low (< 0.2). Area might be non-agricultural.")
            else:
                prediction = model.predict(input_df)[0]
                st.success(f"🌱 Predicted Yield: **{prediction:.2f} kg per acre**")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Map ----------
        st.subheader("🌍 Field Map View")
        m = folium.Map(
            location=[lat, lon],
            zoom_start=15,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community"
        )
        folium.GeoJson(gdf, tooltip="KML Layer").add_to(m)
        st_folium(m, width=700, height=400)

        # ---------- About ----------
        with st.expander("ℹ️ About this App"):
            st.markdown("""
            - Upload a `.kml` file to locate your farm.
            - Weather data is fetched using Google Earth Engine (CFSv2 dataset) for the selected date, or the nearest available date within ±30 days if not available.
            - NDVI, NDMI, MSAVI are calculated from Sentinel-2 imagery for the selected date, or the nearest available date within ±30 days if not available.
            - Yield is predicted using a trained **Random Forest model**.
            - Built with Streamlit and Google Earth Engine for scalable geospatial intelligence.
            """)

        st.caption("🚀 Built by Team RCAI")

    except Exception as e:
        st.error(f"❌ Error reading KML or fetching data from GEE: {str(e)}")

else:
    st.info("👆 Upload a `.kml` file to visualize your field and get yield prediction.")