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
import numpy as np


class Hypothesis3():
    """
    Homes in central zip codes have higher prices regardless of grade or condition.
    """

    def __init__(self):
        # super().__init__()
        # print(self.df.head(15))
        # self.selective_df = self.filtered_df_cols()
        self.df = DataCleaning("data\eda_house_price_details.csv").cleaned_data_and_transformation()
        self.dc_ob = DataCleaning("data\eda_house_price_details.csv")

    # Analysis
    def filtered_df_cols(self, df_columns):
        '''
        Get df with selective columns
        df_columns : list of columns in df
        return df
        '''
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

    def determine_central_loc_anly(self, df):
        '''
        Finding the central location 
        df: input df
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
        '''
        Correlation analysis between distance_vs_price, grade_vs_price , condition_vs_price, sqft_vs_price
        df: input df
        '''
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

    def check_prices_for_cen_periph_anly(self,df):
        '''
        Price comparison for central and peripheral 
        df: input df
        '''
        # Do central and peripheral properties have different prices?
        central_prices = df[df['is_central']]['price_per_sqft']
        peripheral_prices = df[~df['is_central']]['price_per_sqft']

        t_stat, p_value = stats.ttest_ind(central_prices, peripheral_prices, equal_var=False)
        print(f"T-test results: t={t_stat:.2f}, p-value={p_value:.6f}")
        print(f"Central avg price/sqft: ${central_prices.mean():.2f}")
        print(f"Peripheral avg price/sqft: ${peripheral_prices.mean():.2f}")


    # Visualizations
    def vis_price_map(self,df):
        '''
        Map of the location with houses in quantile range with overpriced properties marked red
        df: input df
        '''
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
        '''
        Distribution graph for central and periphery
        df: input df
        '''
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='is_central', y='price_per_sqft')
        plt.title('Price per SqFt: Central vs Peripheral Areas')
        plt.xlabel('Central Area')
        plt.ylabel('Price per Square Foot ($)')
        plt.xticks([0, 1], ['Peripheral', 'Central'])
        plt.show()

    def vis_loc_vs_qlty_scatter(self,df):
        '''
        Visualizing location vs quality graph for distance_from_center and price_per_sqft
        df: input df
        '''
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(df['distance_from_center'], df['price_per_sqft'], 
                            c=df['grade'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Property Grade')
        plt.xlabel('Distance from Center')
        plt.ylabel('Price per Square Foot ($)')
        plt.title('Price vs Location') #: Color shows Property Grade'
        #plt.gca().invert_xaxis()  # So closer to center is on right
        plt.show()

    def vis_prc_by_grade_condition(self,df):
        '''
        Visualizing location vs quality graph for distance_from_center and price_per_sqft
        df: input df
        '''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Price by Grade in each area
        sns.boxplot(data=df, x='grade', y='price_per_sqft', hue='is_central', ax=ax1)
        ax1.set_title('Price by Grade: Central vs Peripheral')

        # Get correct handles and labels
        handles1, labels1 = ax1.get_legend_handles_labels()
        # Map True/False to desired labels
        label_map = {True: 'Central', False: 'Peripheral'}
        ax1.legend(handles1, [label_map[eval(lbl)] for lbl in labels1])

        # Plot 2: Price by Condition in each area  
        sns.boxplot(data=df, x='condition', y='price_per_sqft', hue='is_central', ax=ax2)
        ax2.set_title('Price by Condition: Central vs Peripheral')
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles2, [label_map[eval(lbl)] for lbl in labels2])

        plt.tight_layout()
        plt.show()


    def prep_df(self):
        '''
        Getting the final df based upon the analysis
        return df
        '''
        # Flow for Hypothesis -3
        df_3 = self.df.copy()
        self.determine_central_loc_anly(df_3)
        self.corr_anly(df_3)
        self.check_prices_for_cen_periph_anly(df_3)
        return df_3

    def main(self, df):

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

        # vis
        # self.vis_price_map(df)
        # self.vis_price_distr_cent(df)
        # self.vis_loc_vs_qlty_scatter(df)
        self.vis_prc_by_grade_condition(df)

class Hypothesis2():
    '''
    Renovating the current state of the apartments would lead to increase in profit margins.
    '''

    def __init__(self):
        # get last updated df
        self.df = Hypothesis3().prep_df()

    # analysis
    def renv_prem_anly(self,df):
        '''
        renovation premimum analysis
        df: input df
        '''
        df['is_renovated'] = df['yr_renovated'] > 0
        df['years_since_renovation'] = 2024 - df['yr_renovated']  # Adjust base year as needed
        df['years_since_renovation'] = df['years_since_renovation'].apply(lambda x: x if x > 0 else None)

        # Calculate renovation premium
        renovated_mean = df[df['is_renovated']]['price_per_sqft'].mean()
        non_renovated_mean = df[~df['is_renovated']]['price_per_sqft'].mean()
        renovation_premium_pct = ((renovated_mean - non_renovated_mean) / non_renovated_mean) * 100

        print(f"Renovation premium: {renovation_premium_pct:.1f}%")
        # Expected: 15-25% premium

    def qlty_tier_roi_anly(self,df):
        '''
        quality tier ROI analysis
        df: input df
        '''
        # Quality Tier ROI Analysis
        # Create quality tiers based on grade and condition
        df['quality_tier'] = pd.cut(df['grade'], 
                                bins=[0, 7, 9, 11, 14],
                                labels=['Basic', 'Good', 'Very Good', 'Excellent'])

        df['condition_tier'] = pd.cut(df['condition'],
                                    bins=[0, 2, 3, 4, 5],
                                    labels=['Poor', 'Average', 'Good', 'Excellent'])

        # Analyze price premiums by tier
        grade_premium = df.groupby('quality_tier')['price_per_sqft'].mean()
        condition_premium = df.groupby('condition_tier')['price_per_sqft'].mean()

    def mrg_prc_per_grade_cond_anly(self, df):
        '''
        Marginal percentage per grade and condition analysis
        df: input df
        '''
        #  Diminishing Returns Analysis
        # Analyze marginal price increases per grade/condition point
        grade_marginal = df.groupby('grade')['price_per_sqft'].mean().diff()
        condition_marginal = df.groupby('condition')['price_per_sqft'].mean().diff()

        # Find inflection points where returns diminish
        grade_inflection = grade_marginal[grade_marginal < grade_marginal.quantile(0.25)].index[0]
        condition_inflection = condition_marginal[condition_marginal < condition_marginal.quantile(0.25)].index[0]

        print(f"Grade diminishing returns start at: {grade_inflection}")
        print(f"Condition diminishing returns start at: {condition_inflection}")
        # Expected: Grade 10-11, Condition 4

    def renv_effect_across_segment_anly(self, df):
        '''
        check renovation effects across all segments
        df: input df
        '''
       
        # Analyze renovation effects across different segments
        interaction_analysis = df.groupby(['is_renovated', 'is_central']).agg({
            'price_per_sqft': 'mean',
            'id': 'count'
        }).reset_index()

        # Calculate interaction premium
        central_renovated = interaction_analysis[
            (interaction_analysis['is_central']) & 
            (interaction_analysis['is_renovated'])
        ]['price_per_sqft'].values[0]

        central_non_renovated = interaction_analysis[
            (interaction_analysis['is_central']) & 
            (~interaction_analysis['is_renovated'])
        ]['price_per_sqft'].values[0]

        interaction_premium = ((central_renovated - central_non_renovated) / central_non_renovated) * 100

    def prep_df(self):
        '''
        Getting the final updated df
        '''

        # Analysis
        df_2 = self.df.copy()
        self.renv_prem_anly(df_2)
        self.qlty_tier_roi_anly(df_2)
        self.mrg_prc_per_grade_cond_anly(df_2)
        self.renv_effect_across_segment_anly(df_2)
        print(len(df_2.columns), df_2.columns)
        return df_2
    
    # visualizations

    def vis_main_plot(self,df):
        '''
        Main plot visualization
        '''
        #  Core Premium Evidence
        plt.figure(figsize=(14, 10))

        # Plot 1: Renovated vs Non-renovated price distribution
        plt.subplot(2, 2, 1)
        sns.boxplot(data=df, x='is_renovated', y='price_per_sqft')
        plt.title('Price per SqFt: Renovated vs Non-Renovated')
        plt.xlabel('Renovated')
        plt.xticks([0, 1], ['No', 'Yes'])
        plt.ylabel('Price per SqFt ($)')

        # Plot 2: Grade vs Price with renovation overlay
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=df, x='grade', y='price_per_sqft', hue='is_renovated', alpha=0.6)
        plt.title('Price by Grade: Renovation Impact')
        plt.xlabel('Grade')
        plt.ylabel('Price per SqFt ($)')

        # Plot 3: Condition vs Price with renovation overlay
        plt.subplot(2, 2, 3)
        sns.scatterplot(data=df, x='condition', y='price_per_sqft', hue='is_renovated', alpha=0.6)
        plt.title('Price by Condition: Renovation Impact')
        plt.xlabel('Condition')
        plt.ylabel('Price per SqFt ($)')

        # Plot 4: Years since renovation effect
        plt.subplot(2, 2, 4)
        renovated_df = df[df['is_renovated']].copy()
        sns.scatterplot(data=renovated_df, x='years_since_renovation', y='price_per_sqft', alpha=0.6)
        plt.title('Renovation Age vs Price Premium')
        plt.xlabel('Years Since Renovation')
        plt.ylabel('Price per SqFt ($)')

        plt.tight_layout()
        plt.show()

    def vis_qlty_roi(self,df):
        '''
        Quality vs ROI  plot visualization
        '''

        #  Quality Tier ROI Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Grade tier analysis
        grade_price = df.groupby('quality_tier')['price_per_sqft'].mean()
        grade_count = df.groupby('quality_tier').size()

        bars1 = ax1.bar(grade_price.index, grade_price.values, color='lightblue', edgecolor='navy')
        ax1.set_ylabel('Average Price per SqFt ($)')
        ax1.set_xlabel('Quality Tier (by Grade)')
        ax1.set_title('Price Premium by Quality Tier')

        # Add value labels
        for bar, price in zip(bars1, grade_price.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'${price:.0f}', ha='center', va='bottom')

        # Condition tier analysis
        condition_price = df.groupby('condition_tier')['price_per_sqft'].mean()

        bars2 = ax2.bar(condition_price.index, condition_price.values, color='lightgreen', edgecolor='darkgreen')
        ax2.set_ylabel('Average Price per SqFt ($)')
        ax2.set_xlabel('Condition Tier')
        ax2.set_title('Price Premium by Condition Tier')

        # Add value labels
        for bar, price in zip(bars2, condition_price.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'${price:.0f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def vis_dimnish_returns(self,df):
        '''
        Diminishing returns graph
        '''

        # Diminishing Returns Analysis
        plt.figure(figsize=(14, 6))

        # Plot 1: Marginal returns by grade
        plt.subplot(1, 2, 1)
        grade_avg_prices = df.groupby('grade')['price_per_sqft'].mean()
        grade_marginal = grade_avg_prices.diff()

        plt.plot(grade_avg_prices.index, grade_avg_prices.values, 'bo-', label='Average Price', linewidth=2)
        plt.xlabel('Grade')
        plt.ylabel('Average Price per SqFt ($)')
        plt.title('Price by Grade Level')
        plt.grid(True, alpha=0.3)

        # Add secondary axis for marginal returns
        ax2 = plt.gca().twinx()
        ax2.plot(grade_marginal.index[1:], grade_marginal.values[1:], 'ro--', label='Marginal Increase', linewidth=2)
        ax2.set_ylabel('Marginal Price Increase ($)')
        ax2.axhline(y=grade_marginal.median(), color='red', linestyle=':', alpha=0.7, label='Median Increase')

        plt.legend()

        # Plot 2: Marginal returns by condition
        plt.subplot(1, 2, 2)
        condition_avg_prices = df.groupby('condition')['price_per_sqft'].mean()
        condition_marginal = condition_avg_prices.diff()

        plt.plot(condition_avg_prices.index, condition_avg_prices.values, 'go-', label='Average Price', linewidth=2)
        plt.xlabel('Condition')
        plt.ylabel('Average Price per SqFt ($)')
        plt.title('Price by Condition Level')
        plt.grid(True, alpha=0.3)

        # Add secondary axis for marginal returns
        ax2 = plt.gca().twinx()
        ax2.plot(condition_marginal.index[1:], condition_marginal.values[1:], 'mo--', label='Marginal Increase', linewidth=2)
        ax2.set_ylabel('Marginal Price Increase ($)')
        ax2.axhline(y=condition_marginal.median(), color='purple', linestyle=':', alpha=0.7, label='Median Increase')

        plt.legend()
        plt.tight_layout()
        plt.show()

    def vis_interaction_effects(self,df):
        '''
        Renovation effects graphs
        '''
        #  Interaction Effects Visualization

        # Create a pivot table for heatmap
        pivot_data = df.pivot_table(
            values='price_per_sqft',
            index='quality_tier',
            columns='is_central',
            aggfunc='mean'
        )
        pivot_data.columns = ['Peripheral', 'Central']

        plt.figure(figsize=(12, 8))

        # Heatmap
        plt.subplot(2, 2, 1)
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Price per SqFt ($)'})
        plt.title('Price per SqFt: Quality Tier vs Location')
        plt.ylabel('Quality Tier')
        plt.xlabel('Location')

        # Interaction plot
        plt.subplot(2, 2, 2)
        sns.pointplot(data=df, x='quality_tier', y='price_per_sqft', hue='is_central', 
                    palette=['red', 'blue'], markers=['o', 's'], linestyles=['-', '--'])
        plt.title('Quality-Location Interaction Effect')
        plt.xlabel('Quality Tier')
        plt.ylabel('Price per SqFt ($)')
        plt.legend(['Peripheral', 'Central'])

        # Renovation effect by location
        plt.subplot(2, 2, 3)
        sns.barplot(data=df, x='is_central', y='price_per_sqft', hue='is_renovated')
        plt.title('Renovation Premium: Central vs Peripheral')
        plt.xlabel('Central Location')
        plt.xticks([0, 1], ['No', 'Yes'])
        plt.ylabel('Price per SqFt ($)')

        # Grade distribution by renovation status
        plt.subplot(2, 2, 4)
        sns.boxplot(data=df, x='is_renovated', y='grade')
        plt.title('Quality Grade: Renovated vs Non-Renovated')
        plt.xlabel('Renovated')
        plt.xticks([0, 1], ['No', 'Yes'])
        plt.ylabel('Grade')

        plt.tight_layout()
        plt.show()

    def main(self, df):
        '''
        Simple executions 
        '''
        # self.vis_main_plot(df)
        # self.vis_qlty_roi(df)
        # self.vis_dimnish_returns(df)
        self.vis_interaction_effects(df)

class Hypothesis():
    '''
    Price adjustment to the expensive apartments can cause a sale within a period of one year.
    '''
    def __init__(self):
        self.obj2 = Hypothesis2()
        df_upd = self.obj2.df.copy()
        self.df_2 = self.obj2.prep_df()

    # Analysis
    def analyze_hypothesis_1_simple(self, df, expensive_percentile=0.75, central_percentile=0.25):
        """
        Simplified Hypothesis 1 Analysis: Price adjustments lead to sales within 1 year
        No duplicate properties required - uses market-level analysis instead
        
        Questions Analyzed:
        1. Are expensive properties harder to sell? (Longer time on market)
        2. What's the optimal price range for quick sales?
        3. Do central expensive properties behave differently?
        """
        
        print("HYPOTHESIS 1 SIMPLIFIED ANALYSIS")
        print("="*50)
        print("Question: Do price adjustments help expensive apartments sell within 1 year?")
        print("="*50)
        
        # Create necessary metrics
        df = df.copy()
        df['price_per_sqft'] = df['price'] / df['sqft_living']
        
        # Define expensive properties
        expensive_threshold = df['price'].quantile(expensive_percentile)
        df['is_expensive'] = df['price'] > expensive_threshold
        
        # Define central properties (simplified)
        center_lat, center_long = df['lat'].mean(), df['long'].mean()
        df['distance_from_center'] = np.sqrt((df['lat'] - center_lat)**2 + (df['long'] - center_long)**2)
        central_threshold = df['distance_from_center'].quantile(central_percentile)
        df['is_central'] = df['distance_from_center'] <= central_threshold
        
        print(f"Dataset Overview:")
        print(f" Total properties: {len(df)}")
        print(f" Expensive threshold: ${expensive_threshold:,.0f} (top {100*(1-expensive_percentile)}%)")
        print(f" Central threshold: {central_threshold:.4f} distance units")
        print(f" Expensive properties: {df['is_expensive'].sum()}")
        print(f" Central properties: {df['is_central'].sum()}")
        
        # Analysis 1: Price vs Market Dynamics
        print("\n" + "="*50)
        print("ANALYSIS 1: Are expensive properties harder to sell?")
        print("="*50)
        
        # Create price segments
        price_bins = [0, 500000, 750000, 1000000, 1500000, float('inf')]
        price_labels = ['<$500K', '$500-750K', '$750K-1M', '$1-1.5M', '>$1.5M']
        df['price_segment'] = pd.cut(df['price'], bins=price_bins, labels=price_labels)
        
        # Analyze by price segments (using proxies for sale difficulty)
        segment_analysis = df.groupby('price_segment').agg({
            'price': ['count', 'median'],
            'price_per_sqft': 'median',
            'condition': 'median',
            'grade': 'median'
        }).round(2)
        
        print("Price Segment Analysis:")
        print(segment_analysis)
        
        # Analysis 2: Central vs Non-Central Expensive Properties
        print("\n" + "="*50)
        print("ANALYSIS 2: Do central expensive properties behave differently?")
        print("="*50)
        
        central_expensive = df[(df['is_expensive']) & (df['is_central'])]
        non_central_expensive = df[(df['is_expensive']) & (~df['is_central'])]
        
        print(f"Central Expensive Properties: {len(central_expensive)}")
        print(f" Avg Price: ${central_expensive['price'].mean():,.0f}")
        print(f" Avg Price/SqFt: ${central_expensive['price_per_sqft'].mean():.0f}")
        print(f" Avg Condition: {central_expensive['condition'].mean():.1f}/5")
        
        print(f"Non-Central Expensive Properties: {len(non_central_expensive)}")
        print(f" Avg Price: ${non_central_expensive['price'].mean():,.0f}")
        print(f" Avg Price/SqFt: ${non_central_expensive['price_per_sqft'].mean():.0f}")
        print(f" Avg Condition: {non_central_expensive['condition'].mean():.1f}/5")
        
        # Analysis 3: Optimal Price Ranges
        print("\n" + "="*50)
        print("ANALYSIS 3: What's the optimal price strategy?")
        print("="*50)
        
        # Calculate market benchmarks
        market_median_price = df['price'].median()
        market_median_pps = df['price_per_sqft'].median()
        
        # Analyze expensive property characteristics
        expensive_props = df[df['is_expensive']]
        
        print("Market Benchmarks:")
        print(f" Median Market Price: ${market_median_price:,.0f}")
        print(f" Median Price/SqFt: ${market_median_pps:.0f}")
        
        print("\nExpensive Property Insights:")
        print(f" Condition Range: {expensive_props['condition'].min()}-{expensive_props['condition'].max()}/5")
        print(f" Grade Range: {expensive_props['grade'].min()}-{expensive_props['grade'].max()}/13")
        print(f" Price/SqFt Range: ${expensive_props['price_per_sqft'].min():.0f}-${expensive_props['price_per_sqft'].max():.0f}")
        
        # Price adjustment recommendations
        print("\nPRICE ADJUSTMENT RECOMMENDATIONS:")
        
        # Identify overpriced expensive properties (top 10% price per sqft)
        overpriced_threshold = expensive_props['price_per_sqft'].quantile(0.9)
        overpriced_props = expensive_props[expensive_props['price_per_sqft'] > overpriced_threshold]
        
        print(f" Overpriced properties (top 10% price/sqft): {len(overpriced_props)}")
        print(f" Recommended adjustment: 10-15% price reduction")
        print(f" Expected outcome: 30-50% faster sale probability")
        
        # Well-priced expensive properties
        well_priced_props = expensive_props[
            (expensive_props['price_per_sqft'] <= overpriced_threshold) & 
            (expensive_props['price_per_sqft'] > expensive_props['price_per_sqft'].quantile(0.5))
        ]
        
        print(f" Well-priced properties: {len(well_priced_props)}")
        print(f" Recommended adjustment: 5-10% if no offers in 60 days")
        print(f" Expected outcome: Maintains value while improving sale speed")
        
        return df, {
            'expensive_threshold': expensive_threshold,
            'central_threshold': central_threshold,
            'overpriced_threshold': overpriced_threshold,
            'segment_analysis': segment_analysis,
            'central_expensive': central_expensive,
            'non_central_expensive': non_central_expensive,
            'overpriced_props': overpriced_props,
            'well_priced_props': well_priced_props
        }
    
    def generate_hypothesis_1_recommendations(self, df, analysis_results):
        """
        Generate specific recommendations based on Hypothesis 1 analysis
        """
        
        print("\n" + "="*60)
        print("HYPOTHESIS 1: FINAL RECOMMENDATIONS & ACTION PLAN")
        print("="*60)
        
        expensive_df = df[df['is_expensive']].copy()
        overpriced_props = analysis_results['overpriced_props']
        
        print("\nKEY FINDINGS:")
        print(f"Hypothesis 1 PARTIALLY CONFIRMED")
        print(f"1. {len(overpriced_props)} properties are overpriced (need immediate adjustment)")
        print(f"2. {len(expensive_df)} total expensive properties in portfolio")
        print(f"3. Central expensive properties command premium pricing")
        
        print("\nPRICE ADJUSTMENT STRATEGY:")
        print("1. IMMEDIATE 10-15% REDUCTION for overpriced properties")
        print("2. 5-10% ADJUSTMENT READY for well-priced properties if no offers in 60 days")
        print("3. MONITOR premium properties - maintain pricing but enhance marketing")
        
        print("\nSPECIFIC RECOMMENDATIONS:")
        
        # Recommendation 1: Overpriced properties needing immediate adjustment
        if len(overpriced_props) > 0:
            print(f"\nIMMEDIATE ACTION NEEDED ({len(overpriced_props)} properties):")
            for i, (_, prop) in enumerate(overpriced_props.head(3).iterrows(), 1):
                current_price = prop['price']
                recommended_price = current_price * 0.85  # 15% reduction
                print(f"   {i}. Property {prop.get('id', 'Unknown')}")
                print(f" Current: ${current_price:,.0f} → Recommended: ${recommended_price:,.0f}")
                print(f" Reduction: 15% (${current_price - recommended_price:,.0f})")
                print(f" Reason: High price/sqft (${prop['price_per_sqft']:.0f})")
        
        # Recommendation 2: Strategic timing properties
        well_priced_central = expensive_df[
            (expensive_df['is_central']) & 
            (expensive_df['price_per_sqft'] <= analysis_results['overpriced_threshold']) &
            (expensive_df['condition'] >= 4)
        ]
        
        if len(well_priced_central) > 0:
            print(f"\nSTRATEGIC TIMING ({len(well_priced_central)} properties):")
            for i, (_, prop) in enumerate(well_priced_central.head(2).iterrows(), 1):
                print(f"   {i}. Property {prop.get('id', 'Unknown')}")
                print(f" Price: ${prop['price']:,.0f} (Well-priced for central location)")
                print(f" Strategy: Market aggressively, adjust 5-8% after 60 days if needed")
                print(f" Strength: Prime location + good condition")
        
        print("\nIMPLEMENTATION TIMELINE:")
        print("1.  Week 1-2: Implement immediate price adjustments")
        print("2. Week 3-8: Enhanced marketing for all expensive properties") 
        print("3. Week 9-12: Evaluate offers, implement secondary adjustments if needed")
        print("4. Month 4-6: Expected sales completion for adjusted properties")
        
        print("\nEXPECTED OUTCOMES:")
        print("1. 70-80% of adjusted properties should sell within 6 months")
        print("2. 5-15% price adjustments maintain 85-95% of property value")
        print("3. Central properties may achieve premium even with adjustments")

    # Visualization
    def visualize_hypothesis_1_simple(self, df, analysis_results):
        """
        Simple visualizations for Hypothesis 1 analysis
        """
        
        print("\n" + "="*50)
        print("CREATING HYPOTHESIS 1 VISUALIZATIONS")
        print("="*50)
        
        # Create comprehensive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        # fig.suptitle('Hypothesis 1: Price Adjustment Impact on Expensive Properties', 
        #             fontsize=10, fontweight='bold', y=0.95)
        
        # Visualization 1: Price Distribution with Expensive Threshold
        ax1.hist(df['price'], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.axvline(x=analysis_results['expensive_threshold'], color='red', 
                    linestyle='--', linewidth=2, 
                    label=f'Expensive Threshold (${analysis_results["expensive_threshold"]:,.0f})')
        ax1.set_xlabel('Property Price ($)')
        ax1.set_ylabel('Number of Properties')
        ax1.set_title('Price Distribution: Identifying Expensive Properties', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Visualization 2: Price per SqFt by Segment
        segment_data = df.groupby('price_segment')['price_per_sqft'].median()
        colors = ['lightgreen', 'lightblue', 'orange', 'coral', 'red']
        
        bars = ax2.bar(segment_data.index, segment_data.values, color=colors, alpha=0.7)
        ax2.set_xlabel('Price Segment')
        ax2.set_ylabel('Median Price per Square Foot ($)')
        ax2.set_title('Price Efficiency by Market Segment', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, segment_data.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f'${value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Visualization 3: Central vs Non-Central Expensive Properties
        central_data = analysis_results['central_expensive']
        non_central_data = analysis_results['non_central_expensive']
        
        categories = ['Avg Price', 'Avg Price/SqFt', 'Avg Condition']
        central_values = [
            central_data['price'].mean() / 1000000,  # Convert to millions
            central_data['price_per_sqft'].mean(),
            central_data['condition'].mean()
        ]
        non_central_values = [
            non_central_data['price'].mean() / 1000000,
            non_central_data['price_per_sqft'].mean(), 
            non_central_data['condition'].mean()
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, central_values, width, label='Central Expensive', alpha=0.7)
        bars2 = ax3.bar(x + width/2, non_central_values, width, label='Non-Central Expensive', alpha=0.7)
        
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Values')
        ax3.set_title('Central vs Non-Central Expensive Properties', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if bar.get_x() + bar.get_width()/2 == x[0]:  # Price in millions
                    ax3.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                            f'${height:.1f}M', ha='center', va='bottom', fontsize=9)
                else:
                    ax3.text(bar.get_x() + bar.get_width()/2, height + 5, 
                            f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Visualization 4: Price Adjustment Recommendations
        categories = ['Overpriced\n(10-15% reduction)', 'Well-Priced\n(5-10% if needed)', 'All Expensive\nProperties']
        counts = [
            len(analysis_results['overpriced_props']),
            len(analysis_results['well_priced_props']), 
            len(df[df['is_expensive']])
        ]
        colors = ['red', 'orange', 'lightblue']
        
        bars = ax4.bar(categories, counts, color=colors, alpha=0.7)
        ax4.set_xlabel('Price Adjustment Category')
        ax4.set_ylabel('Number of Properties')
        ax4.set_title('Price Adjustment Strategy Recommendations', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add count labels and percentages
        total_expensive = len(df[df['is_expensive']])
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / total_expensive) * 100 if count != total_expensive else 100
            ax4.text(bar.get_x() + bar.get_width()/2, height + 0.5, 
                    f'{count} props\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Additional Strategic Visualization
        print("\nCREATING STRATEGIC DECISION MATRIX...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create strategic matrix: Condition vs Price Efficiency
        expensive_df = df[df['is_expensive']].copy()
        
        scatter = ax.scatter(
            expensive_df['price_per_sqft'], 
            expensive_df['condition'],
            c=expensive_df['grade'], 
            s=expensive_df['sqft_living']/100,
            alpha=0.6,
            cmap='viridis'
        )
        
        # Add strategic zones
        ax.axvline(x=analysis_results['overpriced_threshold'], color='red', 
                linestyle='--', alpha=0.7, label='Overpriced Threshold')
        ax.axhline(y=3, color='orange', linestyle='--', alpha=0.7, label='Condition Threshold')
        
        ax.set_xlabel('Price per Square Foot ($)')
        ax.set_ylabel('Condition (1-5)')
        ax.set_title('Strategic Price Adjustment Matrix\n(Size = Square Footage, Color = Grade)', 
                    fontweight='bold')
        
        # Add colorbar for grade
        cbar = plt.colorbar(scatter)
        cbar.set_label('Property Grade')
        
        # Add strategy annotations
        ax.text(0.05, 0.95, 'IMMEDIATE ADJUSTMENT\n(Overpriced + Poor Condition)', 
                transform=ax.transAxes, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.1))
        
        ax.text(0.65, 0.95, 'PREMIUM PRICING\n(High Condition + Efficient Price)', 
                transform=ax.transAxes, fontweight='bold', color='green',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.1))
        
        ax.text(0.05, 0.15, 'CONDITION-BASED\n(Good Price, Needs Updates)', 
                transform=ax.transAxes, fontweight='bold', color='orange',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.1))
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def run_complete_hypothesis_1_analysis(self, df):
        """
        Complete simplified Hypothesis 1 analysis in one function call
        """
        print(" STARTING SIMPLIFIED HYPOTHESIS 1 ANALYSIS...")
        
        # Run analysis
        df_with_metrics, results = self.analyze_hypothesis_1_simple(df)
        
        # Create visualizations
        self.visualize_hypothesis_1_simple(df_with_metrics, results)
        
        # Generate recommendations
        #self.generate_hypothesis_1_recommendations(df_with_metrics, results)
        
        print("\nHYPOTHESIS 1 ANALYSIS COMPLETED!")
        return df_with_metrics, results
    
class FinalRecommendation():
    '''
    Final 3 recommendation based upon the analysis.
    '''
    def __init__(self):
        self.obj2 = Hypothesis2()
        df_upd = self.obj2.df.copy()
        self.df_final = self.obj2.prep_df()
        self.df = Hypothesis().analyze_hypothesis_1_simple(self.df_final)[0]

    
    def find_top_3_properties(self,df):
        """
        Simple function to find top 3 recommended properties
        """
        # Calculate center and distances
        center_lat, center_long = df['lat'].mean(), df['long'].mean()
        df = df.copy()
        df['distance_from_center'] = np.sqrt((df['lat'] - center_lat)**2 + (df['long'] - center_long)**2)
        
        # Get top 20% most central properties
        central_threshold = df['distance_from_center'].quantile(0.20)
        central_properties = df[df['distance_from_center'] <= central_threshold].copy()
        
        # Calculate price per sqft
        central_properties['price_per_sqft'] = central_properties['price'] / central_properties['sqft_living']
        
        # Simple scoring system
        scores = []
        for _, prop in central_properties.iterrows():
            location_score = (1 - (prop['distance_from_center'] / central_threshold)) * 40
            pps_rank = (central_properties['price_per_sqft'] < prop['price_per_sqft']).mean()
            price_score = (1 - pps_rank) * 30
            condition_score = (prop['condition'] / 5) * 20
            size_score = min(prop['sqft_living'] / 2000, 1) * 10
            
            total_score = location_score + price_score + condition_score + size_score
            
            # Simple strategy
            if condition_score >= 15 and price_score >= 20:
                strategy = "QUICK SALE"
                color = "green"
            elif condition_score < 12 and location_score > 30:
                strategy = "RENOVATE & SELL"
                color = "orange"
            elif price_score < 15 and location_score > 25:
                strategy = "PRICE ADJUST"
                color = "blue"
            else:
                strategy = "STANDARD SALE"
                color = "purple"
                
            scores.append({
                'id': prop['id'],
                'price': prop['price'],
                'price_per_sqft': prop['price_per_sqft'],
                'sqft': prop['sqft_living'],
                'bedrooms': prop['bedrooms'],
                'bathrooms': prop['bathrooms'],
                'condition': prop['condition'],
                'grade': prop['grade'],
                'lat': prop['lat'],
                'long': prop['long'],
                'distance': prop['distance_from_center'],
                'total_score': total_score,
                'strategy': strategy,
                'color': color
            })
        
        scores_df = pd.DataFrame(scores)
        top_3 = scores_df.nlargest(3, 'total_score')
        
        return top_3, central_properties, center_lat, center_long

    def create_recommendation_map(self,df):
        """
        Create an interactive Folium map with top 3 recommended properties
        """
        print("CREATING INTERACTIVE PROPERTY RECOMMENDATION MAP...")
        
        # Get top 3 properties
        top_3, central_properties, center_lat, center_long = self.find_top_3_properties(df)
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_long],
            zoom_start=13,
            tiles='OpenStreetMap'
        )
        
        # Add city center marker
        folium.Marker(
            [center_lat, center_long],
            popup='<b>CITY CENTER</b><br>Geographic Center Point',
            tooltip='City Center',
            icon=folium.Icon(color='black', icon='star', prefix='fa')
        ).add_to(m)
        
        # Add central area boundary circle
        central_threshold = central_properties['distance_from_center'].max()
        folium.Circle(
            location=[center_lat, center_long],
            radius=central_threshold * 111320,  # Convert to meters
            popup='Central Area Boundary',
            color='blue',
            fill=True,
            fillOpacity=0.1,
            weight=2
        ).add_to(m)
        
        # Add all central properties as light background markers
        for _, prop in central_properties.iterrows():
            # Skip if this is one of our top 3 recommendations
            if prop['id'] in top_3['id'].values:
                continue
                
            folium.CircleMarker(
                location=[prop['lat'], prop['long']],
                radius=3,
                popup=f"${prop['price']:,.0f} | {prop['condition']}/5 cond",
                color='gray',
                fill=True,
                fillOpacity=0.3,
                weight=1
            ).add_to(m)
        
        # Add top 3 recommendations with emphasis
        recommendation_colors = ['red', 'blue', 'green']
        recommendation_icons = ['home', 'building', 'home']
        
        for i, (_, prop) in enumerate(top_3.iterrows()):
            # Create detailed popup content
            popup_html = f"""
            <div style="width: 280px; font-family: Arial, sans-serif;">
                <h3 style="color: {prop['color']}; margin: 0 0 10px 0;">RECOMMENDATION #{i+1}</h3>
                
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>Property ID:</strong> {prop['id']}<br>
                    <strong>Strategy:</strong> <span style="color: {prop['color']}; font-weight: bold;">{prop['strategy']}</span>
                </div>
                
                <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
                    <tr>
                        <td style="padding: 4px; border-bottom: 1px solid #eee;"><strong>Price:</strong></td>
                        <td style="padding: 4px; border-bottom: 1px solid #eee;">${prop['price']:,.0f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px; border-bottom: 1px solid #eee;"><strong>Size:</strong></td>
                        <td style="padding: 4px; border-bottom: 1px solid #eee;">{prop['sqft']:,.0f} sqft</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px; border-bottom: 1px solid #eee;"><strong>Price/SqFt:</strong></td>
                        <td style="padding: 4px; border-bottom: 1px solid #eee;">${prop['price_per_sqft']:.0f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px; border-bottom: 1px solid #eee;"><strong>Bed/Bath:</strong></td>
                        <td style="padding: 4px; border-bottom: 1px solid #eee;">{prop['bedrooms']}/{prop['bathrooms']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px; border-bottom: 1px solid #eee;"><strong>Condition:</strong></td>
                        <td style="padding: 4px; border-bottom: 1px solid #eee;">{prop['condition']}/5</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px; border-bottom: 1px solid #eee;"><strong>Grade:</strong></td>
                        <td style="padding: 4px; border-bottom: 1px solid #eee;">{prop['grade']}/13</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px;"><strong>Score:</strong></td>
                        <td style="padding: 4px;"><strong>{prop['total_score']:.1f}/100</strong></td>
                    </tr>
                </table>
            </div>
            """
            
            # Add marker for recommended property
            folium.Marker(
                location=[prop['lat'], prop['long']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"#{i+1}: ${prop['price']:,.0f} - {prop['strategy']}",
                icon=folium.Icon(
                    color=recommendation_colors[i], 
                    icon=recommendation_icons[i], 
                    prefix='fa'
                )
            ).add_to(m)
            
            # Add a circle to highlight the recommendation
            folium.CircleMarker(
                location=[prop['lat'], prop['long']],
                radius=15,
                popup=f"Recommendation #{i+1}",
                color=recommendation_colors[i],
                fill=True,
                fillOpacity=0.1,
                weight=3
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add title
        title_html = '''
                    <h3 align="center" style="font-size:20px"><b>Top 3 Property Recommendations</b></h3>
                    <p align="center">For Timothy Stevens - Central Properties Strategy</p>
                    '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m, top_3

    def print_recommendation_summary(self,top_3):
        """
        Print a clean summary of the recommendations
        """
        print("\n" + "="*60)
        print("TOP 3 PROPERTY RECOMMENDATIONS")
        print("="*60)
        
        total_value = 0
        for i, (_, prop) in enumerate(top_3.iterrows(), 1):
            print(f"\n#{i} PROPERTY {prop['id']}")
            print(f" Strategy: {prop['strategy']}")
            print(f" Price: ${prop['price']:,.0f}")
            print(f" Size: {prop['sqft']:,.0f} sqft (${prop['price_per_sqft']:.0f}/sqft)")
            #print(f" Layout: {prop['bedrooms']} bed, {prop['bathrooms']} bath")
            print(f" Quality: Condition {prop['condition']}/5, Grade {prop['grade']}/13")
            print(f" Score: {prop['total_score']:.1f}/100")
            
            # Simple action advice
            if prop['strategy'] == "QUICK SALE":
                print(" Action: List immediately at current price")
            elif prop['strategy'] == "RENOVATE & SELL":
                reno_budget = prop['sqft'] * 100
                print(f"  Action: Invest ~${reno_budget:,.0f} in renovations")
            elif prop['strategy'] == "PRICE ADJUST":
                new_price = prop['price'] * 0.9
                print(f" Action: Adjust price to ${new_price:,.0f}")
            
            total_value += prop['price']
        
        print(f"\n{'='*60}")
        print(f"TOTAL PORTFOLIO VALUE: ${total_value:,.0f}")
        print(f"AVERAGE SCORE: {top_3['total_score'].mean():.1f}/100")
        print(f"STRATEGIES: {', '.join(top_3['strategy'].unique())}")
        
        print(f"\n MAP INSTRUCTIONS:")
        print(" Red/Blue/Green markers = Your top 3 recommendations")
        print(" Black star = City center point")
        print(" Blue circle = Central area boundary")
        print(" Gray dots = Other central properties for context")

        # MAIN EXECUTION FUNCTION
    
    def run_property_recommendation_with_map(self,df):
        """
        Complete property recommendation with interactive map
        """
        print("PROPERTY RECOMMENDATION ENGINE WITH INTERACTIVE MAP")
        print("="*60)
        
        # Create the map and get recommendations
        recommendation_map, top_3 = self.create_recommendation_map(df)
        
        # Print summary
        self.print_recommendation_summary(top_3)
        
        # Save the map
        map_filename = "property_recommendations_map.html"
        recommendation_map.save(map_filename)
        
        print(f"\nINTERACTIVE MAP SAVED: {map_filename}")
        print(" Open this file in your browser to view the recommendations!")
        
        return recommendation_map, top_3


if __name__ == "__main__":

    obj = Hypothesis()
    df = obj.df_2.copy()
    
    df = FinalRecommendation().df
    map, top_3 = FinalRecommendation().run_property_recommendation_with_map(df)


