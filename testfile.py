import pandas as pd
import geopandas as gpd
import plotly.express as px
import fsspec

# 1. Create a sample DataFrame with your location data
data = {
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Zipcode': ['10001', '90001', '60601', '77001', '85001'],
    'Latitude': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484],
    'Longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740]
}



df = pd.DataFrame(data)

# 2. Define the URL for the Natural Earth shapefile
# This URL points to the 110m cultural countries dataset, which is a good world map base.
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"

# 3. Read the GeoDataFrame directly from the URL using fsspec
with fsspec.open(f"simplecache::{url}") as file:
    world_map = gpd.read_file(file)

# 4. Create a GeoDataFrame from your pandas DataFrame
# The 'geometry' column is created from the lat/lon coordinates.
# The `crs` (Coordinate Reference System) is set to WGS84 (EPSG:4326), a common format for GPS data.
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs='EPSG:4326'
)

# 5. Create the interactive map using Plotly Express
fig = px.scatter_mapbox(
    gdf,
    lat=gdf.geometry.y,
    lon=gdf.geometry.x,
    zoom=2, # Adjust the zoom level as needed
    color_discrete_sequence=['red'],
    mapbox_style="carto-positron", # Use a common, web-friendly map style
    hover_name='City', # The main text to display on hover
    hover_data={'Zipcode': True, 'Latitude': False, 'Longitude': False}
)

# 6. Show the interactive map
fig.show()