# auxiliary functions to clean data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analysis_data(df, name):
    
    # number of register 
    
    total = df.shape[0]
    
    print(f"There are {total} of {name}")
    
    print("-----------------------\n")
    
    # null data
    
    print("Null values: \n")
    
    print(df.isna().sum())
    
    print("-----------------------\n")
     
    # Duplicates
    
    duplicated_data = df.duplicated()
    print(f"Number duplicated rows: {duplicated_data.sum()}\n")
    print("-----------------------\n")
    
    # Clases:
    
    print(df.nunique() )
    print("-----------------------\n")
    
    # Description
    
    print("Description: ")
    print(df.describe())
    
    print("-----------------------\n")
     
    print(df.dtypes)