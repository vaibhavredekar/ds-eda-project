'''
Central file for EDA operations
'''

import pandas as pd
import numpy as np
from time import sleep
import missingno as msno



class DataCleaning:

    df = pd.DataFrame([])
    print('default Dataframe', df.head())

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

    def ret_df_unique_values(self, df, df_col, text=''):
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
        print('\n',text,unique_values,'\n', type(unique_values),'\n',type(unique_values[0]),'\n',len(unique_values),'\n',df[df_col].shape )
        return unique_values
    
    def drop_dups_from_dataset(self,df,keep,df_column,*args):
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
        df1 = df1.drop_duplicates(subset=[df_column], keep='first')
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

    def convert_to_dt_type(self,df,frmt='%Y-%m-%d',df_column='date'):
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
        print('Before:',df1[df_column][0],'\n\t',type(df1[df_column][0]))
        df1[df_column] = pd.to_datetime(df1[df_column], format=frmt)
        # print('After:',df1.df_column[0],'\n\t',type(df1.df_column[0]))
        return df1
  
    def save_df(self,updated_df):
        '''
        Saving the last updated df to the class df object
        updated_df = provide the last df after all transformations
        '''
        DataCleaning.df = updated_df.copy()
        print(DataCleaning.df.head())
        

    def clean_data_considering_dtyp(self):
        df_housing_prices_data = self.re_read_df_housing_price(self.dataset_file,',')
        new = self.dfcoln_chg_dtyp_operation(df_housing_prices_data, ['bedrooms','bathrooms','view','sqft_above','sqft_basement','sqft_living15','sqft_living','sqft_lot','sqft_lot15','waterfront','price','floors'], int)
        new1 = self.remove_str_char(new,'yr_renovated')
        last = self.convert_to_dt_type(new1)

        # Final df
        print(last.info(),'\n',last.head())
        self.save_df(last)


    def main(self):
        self.clean_data_considering_dtyp()
        print("Cleaned Data:\n","*"*30)
        print(DataCleaning.df.head())
        print("*"*30)
        


if __name__ == "__main__":
    DataCleaning("data\eda_house_price_details.csv").main()
