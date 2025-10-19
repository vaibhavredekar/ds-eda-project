'''
Central file for dealing with all the hypothesis
'''

from eda_data_cleaning import DataCleaning
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import fsspec
from shapely.geometry import Point
import folium
import math


class Hypothesis:
    def __init__(self):
        self.dc_ob = DataCleaning("data\eda_house_price_details.csv")
        self.df = DataCleaning("data\eda_house_price_details.csv").cleaned_data_and_transformation()
        # print(self.df.head(10))

class Hypothesis3(Hypothesis):
    """
    Homes in central zip codes have higher prices regardless of grade or condition.
    Need to check certain columns within the data.

    id
    yr_renovated
    age_building
    sale date
    location
    grade

    My Take: 
    I need from the data frame columns :

    id    
    zipcode
    lat
    long
    price 
    grade
    condition
    sqft_living
    sqft_lot
    price
    sqft_living15
    sqft_lot15

    """

    def __init__(self):
        super().__init__()
        # print(self.df.head(15))
        self.selective_df = self.filtered_df_cols()
        
    def filtered_df_cols(self):
        self.df.head(15)      
        selective_df = Hypothesis().dc_ob.create_selective_coln_df(self.df, [  'id',
                                                                        'zipcode',
                                                                        'lat',
                                                                        'long',
                                                                        'price',
                                                                        'grade',
                                                                        'condition',
                                                                        'sqft_living',
                                                                        'sqft_lot',
                                                                        'sqft_living15',
                                                                        'sqft_lot15',
                                                                        'price',
                                                                        'date',
                                                                        'yr_renovated'])
        return selective_df
  

    def filter_locations(self):
        df = self.selective_df.copy()
        print(df.columns, df.shape, df.info())
        return df


    def calc_mean_mode_max_min(self, df, df_column):
        '''
        return tuple(min,max,mean,mode)
        '''

        mode = df[df_column].mode()
        max =df[df_column].max()
        min = df[df_column].min()
        mean = df[df_column].mean()

        result = (min,max,mean,mode)
        return result


    def h3_prep_data(self, df):
        '''
        Finding the central location 
        '''

        # get price per sqft:
        df['price_per_sqft'] = df['price'] / df['sqft_living']

        # Define the centeral location:
        central_lat = df['lat'].mean()
        central_long = df['long'].mean()
        
        #Calculate the distance from center for each property:
        # formula for flat map: 
        # d = sqrt((lat2-lat1)**2 + (long2-long1)**2)
        df['distance_from_center'] = ((df['lat'] - central_lat)**2 + (df['long'] - central_long)**2)**0.5
        
        # Based upon the above data split the data-set, considering 25%
        central_loc = df['distance_from_center'].quantile(0.25)
        df['is_central'] = df['distance_from_center'] <= central_loc

        print(central_lat, central_loc, central_long, df['distance_from_center'], df['is_central'])


    def corr_anly(self,df):
        loc_prc_corr = df[['distance_from_center', 'price', 'price_per_sqft']].corr()
        print("Correlation Matrix:")
        print(loc_prc_corr)


if __name__ == "__main__":

    df_filtered= Hypothesis3().filtered_df_cols()
    print(df_filtered.head())

    Hypothesis3().h3_prep_data(df_filtered)
    Hypothesis3().corr_anly(df_filtered)





# Concept

    # def basic_static_plot(self):


    #     # Copy your dataframe
    #     df = self.selective_df.copy()

    #     # Check for required columns
    #     if not {'lat', 'long', 'zipcode'}.issubset(df.columns):
    #         raise ValueError("DataFrame must contain 'lat', 'long', and 'zipcode' columns")

    #     # Load the Natural Earth shapefile from remote URL
    #     url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    #     with fsspec.open(f"simplecache::{url}") as file:
    #         world_map = gpd.read_file(file)

    #     # Create GeoDataFrame for your points
    #     geometry = [Point(xy) for xy in zip(df['long'], df['lat'])]
    #     gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    #     # Plotting
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     world_map.plot(ax=ax, color='lightgray', edgecolor='white')

    #     # Plot your data points
    #     gdf.plot(ax=ax, color='red', markersize=3, alpha=0.3)

    #     # Zoom to data points
    #     margin_x = 5
    #     margin_y = 3
    #     ax.set_xlim(gdf.geometry.x.min() - margin_x, gdf.geometry.x.max() + margin_x)
    #     ax.set_ylim(gdf.geometry.y.min() - margin_y, gdf.geometry.y.max() + margin_y)

    #     # Title and layout
    #     plt.title("Static Map of Locations")
    #     plt.axis("off")
    #     plt.tight_layout()
    #     plt.show()

    # def plot_with_gmplot(self, apikey=None, output_html="map.html", zoom=12):
    #     import gmplot
    #     df = self.selective_df  # expects columns 'lat' and 'long'
    #     if not {'lat', 'long'}.issubset(df.columns):
    #         raise ValueError("Need 'lat' and 'long' columns")

    #     # Center map roughly around mean lat/long
    #     center_lat = df['lat'].mean()
    #     center_lng = df['long'].mean()

    #     # Create gmplot map
    #     if apikey:
    #         gmap = gmplot.GoogleMapPlotter(center_lat, center_lng, zoom, apikey=apikey)
    #     else:
    #         gmap = gmplot.GoogleMapPlotter(center_lat, center_lng, zoom)

    #     # Scatter points (markers)
    #     # You can set marker=True or False
    #     gmap.scatter(df['lat'].tolist(), df['long'].tolist(),
    #                 color='red', size=40, marker=True)

    #     # Optionally, connect them or draw paths
    #     # gmap.plot(df['lat'].tolist(), df['long'].tolist(), 'blue', edge_width=2)

    #     # Write to HTML
    #     gmap.draw(output_html)
    #     print(f"Map saved to {output_html}")

    # def plot_with_folium(self, output_html="mymap.html", zoom_start=12):
    #     df = self.selective_df
    #     if not {'lat', 'long'}.issubset(df.columns):
    #         raise ValueError("Need 'lat' and 'long' columns")
        
    #     center = [df['lat'].mean(), df['long'].mean()]
    #     m = folium.Map(location=center, zoom_start=zoom_start)
        
    #     for _, row in df.iterrows():
    #         folium.Marker([row['lat'], row['long']], popup=str(row.get('zipcode', ''))).add_to(m)
        
    #     # save to html
    #     m.save(output_html)
    #     print(f"Map saved to {output_html}")
