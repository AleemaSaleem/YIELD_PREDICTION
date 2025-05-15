import streamlit as st
import pandas as pd
import joblib
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import ee
import geemap
import datetime

# Set page configuration as the FIRST Streamlit command
st.set_page_config(page_title="🌾 Crop Yield Predictor", layout="wide")

# Inject CSS to reduce padding
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
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Google Earth Engine
try:
    ee.Initialize(project='weatherdata06208')
except Exception as e:
    st.error(f"❌ GEE Authentication Failed: {str(e)}")
    st.info("Run `earthengine authenticate --project weatherdata06208` in your terminal.")
    st.stop()

st.title("🌾 Smart Crop Yield Predictor with Google Earth Engine")
st.success("✅ GEE Initialized Successfully")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("yield_model.pkl")

model = load_model()

# ---------- KML Upload ----------
st.header("🗺️ Upload Field KML")
uploaded_kml = st.file_uploader("📂 Upload your KML file", type="kml")

coords = None
weather_data = {}

if uploaded_kml:
    try:
        gdf = gpd.read_file(uploaded_kml, driver="KML")
        st.success("✅ KML Loaded Successfully")

        # Display attributes if any
        non_geom_cols = [col for col in gdf.columns if col != "geometry"]
        if non_geom_cols:
            st.header("📋 Field Attributes")
            st.dataframe(gdf[non_geom_cols])
        else:
            st.info("No attribute fields found other than geometry.")

        # Extract coordinates for weather and indices
        center = gdf.geometry.iloc[0].centroid
        lat, lon = center.y, center.x
        coords = (lat, lon)

        # Display map in a container to control spacing
        # with st.container():
        st.subheader("🌍 Field Map View")
        m = folium.Map(
            location=[lat, lon],
            zoom_start=15,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community"
        )
        folium.GeoJson(gdf, tooltip="KML Layer").add_to(m)
        st_folium(m, width=700, height=400)  # Reduced height

    # ---------- Fetch Weather Data from GEE ----------
    
        
        # Define the point of interest
        point = ee.Geometry.Point([lon, lat])

        st.sidebar.subheader("📅 Select Date")

        selected_date = st.sidebar.date_input(
        "Choose a date for weather data",
        datetime.date.today() - datetime.timedelta(days=7),
        min_value=datetime.date(2010, 1, 1),
        max_value=datetime.date.today()
        )
        print(selected_date)
        # Convert date to datetime
        selected_date = datetime.datetime.combine(selected_date, datetime.datetime.min.time())
        print(selected_date)
        start = ee.Date(selected_date)
        end = start.advance(1, 'day')



        dataset = ee.ImageCollection('NOAA/CFSV2/FOR6H') \
        .filterDate(start.advance(-3, 'day'), end) \
        .select([
        'Temperature_height_above_ground',
        'Precipitation_rate_surface_6_Hour_Average',
        'Specific_humidity_height_above_ground',
        'Pressure_surface'
        ])


        image_list = dataset.sort('system:time_start', False).toList(1)
        if image_list.size().getInfo() == 0:
            st.error("❌ No weather data available for selected date. Try a different date.")
            st.stop()
        latest_image = ee.Image(image_list.get(0))

        weather_values = latest_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=1000
        ).getInfo()

        # Extract weather variables (convert units)
        temp = weather_values.get('Temperature_height_above_ground', 300) - 273.15
        precip = weather_values.get('Precipitation_rate_surface_6_Hour_Average', 0.00003) * 86400
        specific_humidity = weather_values.get('Specific_humidity_height_above_ground', 0.008)
        pressure = weather_values.get('Pressure_surface', 101325) / 100

        # Calculate relative humidity (Tetens formula)
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

            

    # ---------- Weather Adjustments ----------
            
        
        point = ee.Geometry.Point([lon, lat])
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(point) \
        .filterDate(start.advance(-3, 'day'), end) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .select(['B4', 'B8', 'B11'])


        def calculate_indices(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')
            msavi = image.expression(
                '(2 * NIR + 1 - sqrt((2 * NIR + 1)**2 - 8 * (NIR - RED))) / 2',
                {'NIR': image.select('B8'), 'RED': image.select('B4')}
            ).rename('MSAVI')
            return image.addBands([ndvi, ndmi, msavi])

        s2_with_indices = s2.map(calculate_indices)
        image_list_s2 = s2_with_indices.sort('system:time_start', False).toList(1)
        if image_list_s2.size().getInfo() == 0:
            st.error("❌ No Sentinel-2 imagery found for this period/location. Try another date.")
            st.stop()
        latest_image = ee.Image(image_list_s2.get(0))

        indices = latest_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10
        ).getInfo()

        
        ndvi = indices.get('NDVI', 0.48)
        ndmi = indices.get('NDMI', 0.21)
        msavi = indices.get('MSAVI', 0.31)

        

        if weather_data:
            with st.container():
                st.sidebar.subheader("⛅ Fetching Weather from Google Earth Engine...")
                st.sidebar.success(f"✅ Weather fetched for lat={lat:.2f}, lon={lon:.2f}")
                # st.subheader("🌡️ Adjust Weather Parameters")
                weather_data["tempmax"] = st.sidebar.slider("Max Temperature (°C)", 2.0, 50.0, float(weather_data["tempmax"]))
                weather_data["tempmin"] = st.sidebar.slider("Min Temperature (°C)", 2.0, 50.0, float(weather_data["tempmin"]))
                weather_data["precip"] = st.sidebar.slider("Precipitation (mm)", 0.0, 500.0, float(weather_data["precip"]))
                weather_data["humidity"] = st.sidebar.slider("Humidity (%)", 20.0, 100.0, float(weather_data["humidity"]))
            # with st.container:
                st.sidebar.subheader("Calculating Indices from Sentinel-2...")
                ndvi = st.sidebar.slider("NDVI", -1.0, 1.0, ndvi)
                ndmi = st.sidebar.slider("NDMI", -1.0, 1.0, ndmi)
                msavi = st.sidebar.slider("MSAVI", 0.0, 1.0, msavi)

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
        else:
            st.info("Upload KML to get auto-weather and prediction.")

        # ---------- Info ----------
        with st.expander("ℹ️ About this App"):
            st.markdown("""
            - Upload a `.kml` file to locate your farm.
            - Weather data is fetched using Google Earth Engine (CFSv2 dataset) based on field location.
            - NDVI, NDMI, MSAVI can be manually adjusted or calculated from Sentinel-2 imagery via GEE.
            - Yield is predicted using a trained **Random Forest model**.
            - Built with Streamlit and Google Earth Engine for scalable geospatial analysis.
            """)

        st.caption("🚀 Built by Team RCAI")



    except Exception as e:
        st.error(f"❌ Error reading KML: {e} or")
        st.error(f"❌ Error fetching weather from GEE: {e}")



    # ---------- Prediction ----------
   

else:
    st.info("👆 Upload a `.kml` file to visualize your field and auto-fetch weather.")



