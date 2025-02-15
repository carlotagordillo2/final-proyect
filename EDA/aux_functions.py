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
    

def remove_outliers(data, col):
    Q3 = np.quantile(data[col], 0.75)
    Q1 = np.quantile(data[col], 0.25)
    IQR = Q3 - Q1
    
    # Define bounds for the outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)
    
    # Filter the DataFrame based on the condition
    filtered_data = data[(data[col] > lower_bound) & (data[col] < upper_bound)]

    return filtered_data


def percentage_outliers(data, col):
    Q3 = np.quantile(data[col], 0.75)
    Q1 = np.quantile(data[col], 0.25)
    IQR = Q3 - Q1
    
    # Define bounds for the outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    #print("Lower Bound:", lower_bound)
    #print("Upper Bound:", upper_bound)
    
    # Filter the DataFrame based on the condition
    filtered_data = data[(data[col] < lower_bound) | (data[col] > upper_bound)]

    
    percentage_outliers = (len(filtered_data) / len(data))*100
    
    percentage = f'The percentage of outliers in {col} is {percentage_outliers}\n'
    
    print(percentage)

    return percentage_outliers

def create_frequency_table(df,name):
    
    # control
    frequency_table_control = df[name].value_counts()
   
    proportion_table_control = df[name].value_counts(normalize=True)
    
    
    return frequency_table_control, proportion_table_control


def piechart(df, name, threshold):
    
    frequency_table, proportion_table = create_frequency_table(df, name)

    print(frequency_table)

    #threshold = 0.001  # minimum percentage to show a region alone 

    # the region with a percentage lower it will be agrupated in 'other' colum
    frequency_table_sorted = frequency_table.sort_values(ascending=False)
    total = frequency_table_sorted.sum()

    frequency_table_grouped = frequency_table_sorted.copy()
    frequency_table_grouped[frequency_table_sorted / total < threshold] = 0
    frequency_table_grouped["Others"] = frequency_table_sorted[frequency_table_sorted / total < threshold].sum()

    frequency_table_grouped = frequency_table_grouped[frequency_table_grouped > 0]

    plt.figure(figsize=(10, 10))
    colors = sns.color_palette("Set3", len(frequency_table_grouped))

    plt.pie(
        frequency_table_grouped, 
        labels=frequency_table_grouped.index, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=colors
    )
    plt.xticks(rotation=45)

    plt.title(f'{name}')
    plt.show()
    
    
    def get_season(date):
        
        month = date.month
        
        if month in [12,1,2]:
            return 'Winter'
        elif month in [3,4,5]:
            return 'Spring'
        elif month in [6,7,8]:
            return 'Summer'
        else: 
            'Automn'
            

def sold_tendecies_season(df, group, top = 10, top_figure = 10):
    
    season_sales = df.groupby(['Season', group])['QuantitySold'].sum().reset_index()
    season_sales = season_sales.sort_values(by=['Season', 'QuantitySold'], ascending=[True, False])
    print(season_sales)
    
    season_sales["Rank"] = season_sales.groupby("Season")["QuantitySold"].rank(method="dense", ascending=False)
    top_sales = season_sales[season_sales["Rank"] <= top]

    print(top_sales)
    
    top_sales_graph = season_sales.groupby("Season").apply(lambda x: x.nlargest(top_figure, "QuantitySold")).reset_index(drop=True)

    plt.figure(figsize=(12,6))
    sns.barplot(data=top_sales_graph, x="Season", y="QuantitySold", hue=group)

    plt.xlabel("Season")
    plt.ylabel("Quantity Sold")
    plt.title("Top 10 best-selling products by season")
    plt.legend(title="Product", bbox_to_anchor=(1,1))

    plt.show()