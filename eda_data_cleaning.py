'''
Central file for EDA operations
'''

import pandas as pd
import numpy as np
from time import sleep
import missingno as msno
import matplotlib.pyplot as plt


class DataCleaning:

    df = pd.DataFrame([])
    #print('default Dataframe', df.head())

    def __init__(self,dataset_file):
        self.dataset_file = dataset_file
        
    def re_read_df_housing_price(self, dataset_file, delims):
        '''
        Reads the csv file into dataframe

        input
        dataset_file = path to csv
        delims= str

        return df
        '''
        df = pd.read_csv(dataset_file, delimiter= delims)
        return df

    def check_dups_via_csv(self, df,df_column,name='default.csv'):
        '''
        Test the value count are singular/duplicates for a column data via csv files

        df = dataframe of table
        df_column = evaluation column
        name=ouput csv file 
        '''
        df[df_column].value_counts().to_csv(name)
        print(f'Created {name}')

    def ret_df_unique_values(self, df, df_col, text='',display= False):
        '''
        Test the value count are singular/duplicates for a column data via csv files
        Input:
        df = dataframe of table
        df_column = evaluation column
        txt= Print flag text

        Output:
        return unique values within the column
        '''
        unique_values = df[df_col].unique()
        if display:
            print('\n',text, unique_values,'\n', type(unique_values),'\n',type(unique_values[0]),'\n',len(unique_values),'\n',df[df_col].shape ,'\nDuplicated can be with multiples',df[df_col].value_counts())
        return unique_values
    
    def drop_dups_from_dataset(self,df,kep,df_column,*args):
        '''
        If duplicated are found then a common function to drop duplicates from multiple column
        Input:
        df = dataframe of table
        df_column = evaluation column
        keep= Either first, last or False

        Output:
        return df

        '''
        df1 = df.copy()
        df1 = df1.drop_duplicates(subset=[df_column], keep=kep)
        return df1

    def dfcoln_chg_dtyp_operation(self, df,df_coln,ch_type):
        '''
        Changes the data-type of the column by 
        1. Converting nan to 0
        2. Converting the column to ch_type data-type
        Input:
        df = dataframe of table
        df_column = evaluation column
        ch_type= int, float, str

        Output:
        return df

        '''
        df1 = df.copy()
        for i in df_coln:
            df1[i] = df1[i].fillna(0).astype(ch_type)
        return df1

    def remove_str_char(self,df,df_column,remove_char=-3,ch_type=int):
        '''
        Removes the characters from the column records, and then changing the column to a desired datatype
        For the current project is set -3 as default but can be changeable to any desired
        1. Converting nan to 0 and changing to str in-order to remove the characters from the str
        2. Converting the column to ch_type data-type
        Input:
        df = dataframe of table
        df_column = evaluation column
        remove_char = remove the char based upon front + and back with -
        ch_type= int, float, str

        Output:
        return df
        '''
        df1 = df.copy()
        df1[df_column] = df1[df_column].fillna(0).astype(str).str[:remove_char] 
        df1[df_column] = df1[df_column].replace('',0).astype(ch_type)
        return df1

    def convert_to_dt_type(self,df,frmt='%Y-%m-%d',df_column='date',display=False):
        '''
        converting to date-time format suitable for plots
        Input:
        df = dataframe of table
        df_column = default is date str
        frmt = default - %Y-%m-%d ; Enter your data-format here

        Output:
        return df
        '''
        df1 = df.copy()
        if display:
            print('Before:',df1[df_column][0],'\n\t',type(df1[df_column][0]))
        df1[df_column] = pd.to_datetime(df1[df_column], format=frmt)
        if display:
            print('After:',df1.df_column[0],'\n\t',type(df1.df_column[0]))
        return df1
  
    def save_df(self,updated_df, display=False):
        '''
        Saving the last updated df to the class df object
        updated_df = provide the last df after all transformations
        '''
        DataCleaning.df = updated_df.copy()
        if display:
            print(DataCleaning.df.head())
        
    def cleaned_data_and_transformation(self,kp_dups=False):
        '''
        Data cleaning for complete table 
        Saves the updated df to the class df and
        return df
        '''
        df_housing_prices_data = self.re_read_df_housing_price(self.dataset_file,',')
        if not kp_dups:
            df_housing_prices_data = self.drop_dups_from_dataset(df_housing_prices_data,'first','id')
        new = self.dfcoln_chg_dtyp_operation(df_housing_prices_data, ['bedrooms','bathrooms','view','sqft_above','sqft_basement','sqft_living15','sqft_living','sqft_lot','sqft_lot15','waterfront','price','floors'], int)
        new1 = self.remove_str_char(new,'yr_renovated')
        last = self.convert_to_dt_type(new1)

        # Final df
        # print(last.info(),'\n',last.head())
        self.save_df(last)
        return last

    def main(self):
        '''
        Include flows for checking the data
        '''
        new_df = self.cleaned_data_and_transformation()
        # print("Cleaned Data:\n","*"*30)
        # print(DataCleaning.df.head())
        # print("*"*30)

        # msno.bar(new_df,color='orange')
        # plt.show()
        df_housing_prices_data = self.re_read_df_housing_price(self.dataset_file,',')
        msno.matrix(df_housing_prices_data, sparkline=True, figsize=(10,5), fontsize=12, color=(0.5, 0.58, 0.2))
        # plt.show()

        msno.matrix(new_df, sparkline=True, figsize=(10,5), fontsize=12, color=(0.5, 0.58, 0.2))
        # plt.show()


    def create_selective_coln_df(self,df,df_coln):
        '''
        Creates a selective dataframe based upon the columns

        Input:
        df = dataframe of table
        df_coln = Number of columns or list of columns

        Output:
        return selective df
        '''
        # selective_df1 = df.copy()

        try:
            selective_df1 = df.loc[:,df_coln]
            return selective_df1
        except Exception as e:
            print(f"Failed to create a df Error:{e}, type of error: {type(e)}")
            print(f"Failed to create a df by loc method so used fail back to get the data-frame")
            selective_df = df[df_coln]
            return selective_df


if __name__ == "__main__":
    dc = DataCleaning("data\eda_house_price_details.csv")
    df = DataCleaning("data\eda_house_price_details.csv").cleaned_data_and_transformation()
    #dc.check_dups_via_csv(df,'id')
    # dc.ret_df_unique_values(df,'id','Before',True)

    # print(df.columns)
    selective = dc.create_selective_coln_df(df, ['id',
                                                'zipcode',
                                                'lat',
                                                'long',
                                                'price',
                                                'grade',
                                                'condition',
                                                'sqft_living',
                                                'sqft_lot'])
    
    print(selective.head(15))