import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def abc_classification(percentage):
    
    if percentage <= 0.60:
        return 'A'
    elif percentage <= 0.90:
        return 'B'
    else:
        return 'C'
    
    
def demand_classification(row):
    
    if row['AverageDemand'] >=10 and row['NullPercentage'] < 0.30:
        return 'High demand'
    else: 
        return 'Low demand'
    
def plot_class(top_products, name):
    
    class_a_df = top_products[top_products['Classification'] == 'A']
    class_b_df = top_products[top_products['Classification'] == 'B']
    class_c_df = top_products[top_products['Classification'] == 'C']

    class_a_counts = class_a_df[name].value_counts()
    class_b_counts = class_b_df[name].value_counts()
    class_c_counts = class_c_df[name].value_counts()



    # Calculamos los porcentajes
    a_percentages = class_a_counts / class_a_counts.sum() * 100
    b_percentages = class_b_counts / class_b_counts.sum() * 100
    c_percentages = class_c_counts / class_c_counts.sum() * 100


    plt.figure(figsize=(10, 6))
    plt.pie(a_percentages, labels=a_percentages.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2", len(a_percentages)))
    plt.title(f'Distribution of {name} in A')
    plt.axis('equal')  # Asegura que el gráfico sea un círculo perfecto
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.pie(b_percentages, labels=b_percentages.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2", len(b_percentages)))
    plt.title(f'Distribution of {name} in B')
    plt.axis('equal')  
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.pie(c_percentages, labels=c_percentages.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2", len(c_percentages)))
    plt.title(f'Distribution of {name} in C')
    plt.axis('equal')  
    plt.xticks(rotation=45)
    plt.show()

