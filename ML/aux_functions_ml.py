import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox, norm, ks_2samp
from scipy.stats import shapiro, ks_2samp
from scipy.stats import poisson, expon, ks_2samp
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

#read data

df_high_demand_A = pd.read_csv('../Datasets/all_predictions_high_demand_A.csv', index_col=0)
df_products = pd.read_csv('../Datasets/clean_products.csv', index_col=0)
df_orders = pd.read_csv('../Datasets/clean_orders.csv', index_col = 0)
df_order_details = pd.read_csv('../Datasets/clean_order_details.csv', index_col=0)
df_inventory = pd.read_csv('../Datasets/clean_inventory.csv', index_col=0)
df_purchase = pd.read_csv('../Datasets/clean_purchase_orders.csv', index_col=0)

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


def select_products_lead_time(df, id):
    
    df = df[df['ProductID'] == id]
    df = df.merge(df_inventory, on= 'ProductID')
    df = df.merge(df_purchase, on = 'PurchaseOrderID')
    
    # Convert to date time
    
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    
    lead_time = (df['TransactionDate'] - df['OrderDate']).dt.total_seconds() / 86400
    
    return df, lead_time

def metricas_lead_time(lead_time):
    
    lead_time_avg = lead_time.mean()
    lead_time_std = lead_time.std()
    
    return lead_time_avg, lead_time_std

def demanda(df, regimen = 'D'):
    
    df = df.merge(df_order_details, on = 'ProductID')
    df = df.merge(df_orders, on = 'OrderID')
    
    df = df.rename(columns = {'OrderDate_y': 'OrderDate'})
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df  = df[['OrderDate', 'QuantitySold']]
    sales = df.groupby(['OrderDate'])['QuantitySold'].sum().reset_index()
    sales = sales.set_index('OrderDate')
    sales = sales.asfreq('D')
    
    sales['QuantitySold'] = sales['QuantitySold'].interpolate()
    
    mean_value = sales ['QuantitySold'].mean()
    
    #plot
    
    plt.figure(figsize=(18,5))
    plt.plot(sales['QuantitySold'], linestyle="-", label = 'Sales')
    plt.axhline(y=mean_value, color='r', linestyle="--", label=f"Mean: {mean_value:.2f}")
    plt.title('Sales forecasting')
    plt.legend()
    plt.grid()
    plt.show()
    
    return sales

def estudio_outliers(sales, drop_outliers = True):
    
    sns.boxplot(sales['QuantitySold'])


    q1 = sales['QuantitySold'].quantile(0.25)
    q3 = sales['QuantitySold'].quantile(0.75)

    iqr = q3-q1

    lim_inf = q1 - 1.5*iqr
    lim_sup = q3 + 1.5*iqr

    outliers = sales[(sales['QuantitySold'] < lim_inf) | (sales['QuantitySold'] > lim_sup)]

    perc_out = (len(outliers) / len(sales['QuantitySold'])) * 100

    print(f'Percentage outliers:  {perc_out}')  
    
    sales['QuantitySold_no_out'] = np.where((sales['QuantitySold'] < lim_inf) | (sales['QuantitySold'] > lim_sup),
                              sales['QuantitySold'].median(), sales['QuantitySold'])
    
    plt.figure(figsize=(18,5))
    plt.plot(sales['QuantitySold'], linestyle="-", label = 'Sales')
    plt.plot(sales['QuantitySold_no_out'], linestyle="-", label = 'Sales no outliers' )
    plt.title('Sales forecasting')
    plt.legend()
    plt.grid()
    plt.show()
    
    if(drop_outliers):
        
        sales['QuantitySold'] = sales['QuantitySold_no_out']
        sales.drop(columns=['QuantitySold_no_out'], inplace=True)
        
    else:
        
        sales.drop(columns=['QuantitySold_no_out'], inplace=True)
        
    return sales

def normality_study(sales):
    
    residuos = sales['QuantitySold']

    # Test de Kolmogorov-Smirnov
    ks_stat, p_ks = ks_2samp(residuos, norm.rvs(size=len(residuos)))
    # Test de Shapiro-Wilk (para n < 5000)
    shapiro_stat, p_shapiro = shapiro(residuos) if len(residuos) < 5000 else (None, None)

    print(f"p-value Kolmogorov-Smirnov: {p_ks}")
    print(f"p-value Shapiro-Wilk: {p_shapiro}")

    # Histograma y QQ-Plot para ver distribución
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    sns.histplot(residuos, kde=True)
    plt.title("Sales histplot")

    plt.subplot(122)
    sm.qqplot(residuos, line='s', fit=True)
    plt.title("Sales QQ-Plots")
    plt.show()

    # Si p-valor < 0.05 en ambos test, la serie no es normal -> aplicar Box-Cox
    if p_ks < 0.05 or (p_shapiro is not None and p_shapiro < 0.05):
        print("It isn't normal. Applying Box-Cox...")
        sales['QuantitySold'] = sales['QuantitySold'] + 1e-6
        sales['QuantitySold'], lambda_bc = boxcox(sales['QuantitySold'])
        print(f"Lambda de Box-Cox: {lambda_bc}")
    else:
        print("It's normal.")
        
    return sales

def security_stock(sales, lead_time_avg, lead_time_std, confidence=0.99):
    
    # Eliminar valores negativos o nulos
    sales = sales[sales["QuantitySold"] > 0].dropna()
    
    # Descomponer la demanda para extraer la estacionalidad
    descomposicion = seasonal_decompose(sales["QuantitySold"], model="additive", period=30)
    sales["SeasonalDemand"] = sales["QuantitySold"] - descomposicion.seasonal
    
    # Calcular media y desviación estándar
    demanda_prom = sales["SeasonalDemand"].mean()
    sigma_demanda = sales["SeasonalDemand"].std()
    
    # Seleccionar la mejor distribución
    if np.all(sales["SeasonalDemand"] >= 0) and np.issubdtype(sales["SeasonalDemand"].dtype, np.integer):
        # Comparar si Poisson se ajusta mejor que una normal
        poisson_fit = poisson(demanda_prom)
        ks_poisson, p_poisson = ks_2samp(sales["SeasonalDemand"], poisson_fit.rvs(len(sales)))
        
        if p_poisson > 0.05:  # No se rechaza la hipótesis nula de que los datos siguen Poisson
            distribucion = "Poisson"
            sigma_demanda = np.sqrt(demanda_prom)
        else:
            distribucion = "Normal"
    else:
        distribucion = "Exponencial"
        sigma_demanda = demanda_prom  

    # Stock de seguridad con nivel de servicio
    Z = norm.ppf(confidence)
    sigma_LT = np.sqrt((lead_time_avg * sigma_demanda**2) + (demanda_prom**2 * lead_time_std**2))
    security_stock = Z * sigma_LT
    
    # Calcular punto de reorden (ROP)
    ROP = (demanda_prom * lead_time_avg) + security_stock
        
    # Identificar fechas de reorden
    sales["AccumulativeStock"] = ROP - sales["SeasonalDemand"].cumsum()
    sales["TriggerOrder"] = sales["AccumulativeStock"] <= security_stock
    fechas_reposicion = sales[sales["TriggerOrder"]].index
    
    print(f"🔹 Reordered dates:", fechas_reposicion.to_list())
    print(f"🔹 ROP: {ROP:.2f}")
    print(f"🔹 Security Stock: {security_stock:.2f}")
    
    # Graficar demanda y puntos de reorden
    plt.figure(figsize=(10, 4))
    plt.plot(sales.index, sales["QuantitySold"], label="Real demand")
    plt.axhline(ROP, color='r', linestyle='--', label="Reorder point")
    plt.scatter(fechas_reposicion, [ROP] * len(fechas_reposicion), color='red', marker='o', label="Trigger order")
    plt.legend()
    plt.title(f"Demand and Reorder Point - {distribucion}")
    plt.grid()
    plt.show()
    
    return ROP, security_stock, fechas_reposicion

