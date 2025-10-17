'''
Central file for dealing with all the hypothesis
'''

from eda_data_cleaning import DataCleaning

class Hypothesis:
    def __init__(self):
        self.df = DataCleaning("data\eda_house_price_details.csv").cleaned_data_and_transformation()
        print(self.df.head(10))



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
    """

    def __init__(self):
        super().__init__()
        print(self.df.head(15))

Hypothesis3()
