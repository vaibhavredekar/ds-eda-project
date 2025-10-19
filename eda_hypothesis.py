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
from time import sleep
from scipy import stats
import seaborn as sns

# class Hypothesis:
#     def __init__(self):
#         self.dc_ob = DataCleaning("data\eda_house_price_details.csv")
#         self.df = DataCleaning("data\eda_house_price_details.csv").cleaned_data_and_transformation()
#         # print(self.df.head(10))

class Hypothesis3():
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
        # super().__init__()
        # print(self.df.head(15))
        # self.selective_df = self.filtered_df_cols()
        self.df = DataCleaning("data\eda_house_price_details.csv").cleaned_data_and_transformation()
        self.dc_ob = DataCleaning("data\eda_house_price_details.csv")

    # Analysis
    def filtered_df_cols(self, df_columns):
        self.df.head(15)
        selective_df = self.dc_ob.create_selective_coln_df(self.df, df_columns)
        return selective_df
  
    def get_mean_mode_max_min(self, df, df_column):
        '''
        return tuple(min,max,mean,mode)
        '''

        mode = df[df_column].mode()
        max =df[df_column].max()
        min = df[df_column].min()
        mean = df[df_column].mean()

        result = (min,max,mean,mode)
        return result

    def determine_central_loc_an(self, df):
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
        # 1. Location vs Price correlation
        location_price_corr = df[['distance_from_center', 'price', 'price_per_sqft']].corr()
        print("Correlation Matrix:")
        print(location_price_corr)

        # 2. Compare correlation strengths
        correlations = {
            'distance_vs_price': df['distance_from_center'].corr(df['price']),
            'grade_vs_price': df['grade'].corr(df['price']),
            'condition_vs_price': df['condition'].corr(df['price']),
            'sqft_vs_price': df['sqft_living'].corr(df['price'])
        }

        print("\nCorrelation Strength Comparison:")
        for key, value in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"{key:20}: {value:+.3f}")

    def check_prices_for_cen_periph_an(self,df):
        # Do central and peripheral properties have different prices?
        central_prices = df[df['is_central']]['price_per_sqft']
        peripheral_prices = df[~df['is_central']]['price_per_sqft']

        t_stat, p_value = stats.ttest_ind(central_prices, peripheral_prices, equal_var=False)
        print(f"T-test results: t={t_stat:.2f}, p-value={p_value:.6f}")
        print(f"Central avg price/sqft: ${central_prices.mean():.2f}")
        print(f"Peripheral avg price/sqft: ${peripheral_prices.mean():.2f}")


    # Visualizations
    def vis_price_map(self,df):
        # Create a base map centered on your area
        m = folium.Map(location=[df['lat'].mean(), df['long'].mean()], zoom_start=10)

        # Color code by price percentile
        price_75 = df['price_per_sqft'].quantile(0.75)
        price_25 = df['price_per_sqft'].quantile(0.25)

        for idx, row in df.iterrows():
            # Determine color based on price
            if row['price_per_sqft'] >= price_75:
                color = 'red'  # High price
            elif row['price_per_sqft'] <= price_25:
                color = 'blue'   # Low price
            else:
                color = 'green'  # Medium price
            
            # Add circle marker
            folium.CircleMarker(
                location=[row['lat'], row['long']],
                radius=3,
                popup=f"Price/sqft: ${row['price_per_sqft']:.0f}<br>Grade: {row['grade']}<br>Condition: {row['condition']}",
                color=color,
                fill=True
            ).add_to(m)

        # Save the map
        m.save('property_prices_map.html')

    def vis_price_distr_cent(self,df):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='is_central', y='price_per_sqft')
        plt.title('Price per SqFt: Central vs Peripheral Areas')
        plt.xlabel('Central Area')
        plt.ylabel('Price per Square Foot ($)')
        plt.xticks([0, 1], ['Peripheral', 'Central'])
        plt.show()

    def vis_loc_vs_qlty_scatter(self,df):
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(df['distance_from_center'], df['price_per_sqft'], 
                            c=df['grade'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Property Grade')
        plt.xlabel('Distance from Center')
        plt.ylabel('Price per Square Foot ($)')
        plt.title('Price vs Location: Color shows Property Grade\n(Proving location dominates grade)')
        #plt.gca().invert_xaxis()  # So closer to center is on right
        plt.show()

    def vis_prc_by_grade_condition(self,df):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Price by Grade in each area
        sns.boxplot(data=df, x='grade', y='price_per_sqft', hue='is_central', ax=ax1)
        ax1.set_title('Price by Grade: Central vs Peripheral')
        ax1.legend(['Peripheral', 'Central'])

        # Plot 2: Price by Condition in each area  
        sns.boxplot(data=df, x='condition', y='price_per_sqft', hue='is_central', ax=ax2)
        ax2.set_title('Price by Condition: Central vs Peripheral')
        ax2.legend(['Peripheral', 'Central'])

        plt.tight_layout()
        plt.show()




    def main(self):

        df_columns = ['id',
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
            'yr_renovated']
        df_filtered= self.filtered_df_cols(df_columns)
        # print(df_filtered.head())

        # Flow for Hypothesis -3
        df_3 = self.df.copy()
        self.determine_central_loc_an(df_3)
        self.corr_anly(df_3)
        self.check_prices_for_cen_periph_an(df_3)

        # vis
        self.vis_price_map(df_3)
        self.vis_price_distr_cent(df_3)
        self.vis_loc_vs_qlty_scatter(df_3)
        self.vis_prc_by_grade_condition(df_3)


if __name__ == "__main__":

    Hypothesis3().main()





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
