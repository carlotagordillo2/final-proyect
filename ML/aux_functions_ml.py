import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox, norm, ks_2samp
from scipy.stats import shapiro, ks_2samp
from scipy.stats import poisson, expon, ks_2samp
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from mlxtend.frequent_patterns import apriori, association_rules
import itertools
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import Counter
import math
from pulp import *


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

def similar_products(df, min_support = 0.05, min_lift = 2.0):
    
    df_encoded = df.groupby(['OrderID', 'ProductID'])['ProductID'].count().unstack().fillna(0)
    df_encoded = df_encoded.applymap(lambda x: 1 if x> 0 else 0)
    
    
    frequent_itemsets= apriori(df_encoded, min_support= min_support, use_colnames=True )
    
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
    rules = rules[(rules['support'] >= min_support) & (rules['lift'] >= min_lift)]
    
    
    products_in_rules = set(rules['antecedents'].apply(lambda x: list(x)).sum() + rules['consequents'].apply(lambda x: list(x)).sum())
    
    all_products = set(df['ProductID'].unique())
    
    missing_products = all_products - products_in_rules
    n_missing_products = len(missing_products)
    
    if missing_products:
        print(f"Productos no cubiertos en las reglas: {missing_products}\n")
        print(f"Number of missing products: {n_missing_products} \n")
        residual_rule = {
            'antecedents': [tuple(missing_products)],  # Los productos no cubiertos son los antecedentes
            'consequents': [tuple(missing_products)],  # Los mismos productos como consecuencia
            'support': 1.0,  # Considerarlos como productos agrupados (puedes ajustar este valor)
            'lift': 1.0,     # No tiene lift, ya que no hay una relación fuerte entre ellos
        }
        residual_rule_df = pd.DataFrame([residual_rule])
        rules = pd.concat([rules, residual_rule_df], ignore_index=True)
    
   
    return rules

def calculate_category_percentages(warehouse_map):
    
    flattened = warehouse_map.flatten()   
    category_counts = Counter(flattened)   
    total_elements = flattened.size  
    category_percentages = {category: (count / total_elements) * 100 for category, count in category_counts.items()}
    
    return category_percentages


def visualize_categories(warehouse_map, rows, cols):
    """
    Visualiza cómo están distribuidas las categorías en el warehouse, asignando un color a cada categoría.
    
    warehouse_map: Mapa del warehouse con las categorías asignadas a los slots
    rows: Número de filas del warehouse
    cols: Número de columnas del warehouse
    """
    
    # Crear una lista de categorías únicas, excluyendo los 0 (vacíos)
    unique_categories = set(category for row in warehouse_map for category in row if category != 0)
    
    # Agregar un valor para el vacío (0) si es necesario
    unique_categories.add(0)  # Esto asegura que los huecos también sean representados
    
    # Crear un diccionario que asigna un color a cada categoría (usando una paleta de colores)
    category_to_color = {category: plt.cm.get_cmap("tab20")(i / len(unique_categories))[:3] if category != 0 
                         else (0, 0, 0)  # Color gris para los huecos
                         for i, category in enumerate(unique_categories)}
    
    # Crear la imagen para el warehouse
    img = np.zeros((rows, cols, 3))  
    
    # Asignar colores en la matriz de imagen
    for r in range(rows):
        for c in range(cols):
            category = warehouse_map[r][c]
            img[r, c] = category_to_color.get(category, (0.5, 0.5, 0.5))  
    
    # Crear la visualización
    plt.figure(figsize=(10, 10))
    plt.imshow(img, interpolation='nearest')
    plt.title("Warehouse Distribution")
    plt.axis('off')  

    
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=category_to_color[category], markersize=10) 
               for category in unique_categories if category != 0]  # Añadimos las categorías

    # Añadimos el negro para los huecos con la etiqueta "Available Space"
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0, 0, 0), markersize=10))
    
    # Modificar las etiquetas, reemplazando el 0 por "Available Space"
    labels = [str(category) for category in unique_categories if category != 0]
    labels.append('Available Space')  # Etiqueta para los huecos
    plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

    # Mostrar la visualización
    plt.show()

def distribute_products(warehouse_map, rows, cols, df_categories, rules_by_category, products_by_category, category_slots, df_products):
    """
    Distribuye los productos en el warehouse según las reglas de asociación (por categoría),
    asignando primero productos según las reglas de asociación, luego los productos de 'Cuff-Over the Calf',
    y finalmente los productos no asignados se colocan en los huecos vacíos de cada categoría.
    
    warehouse_map: Mapa del warehouse con las categorías asignadas a los slots
    rows: Número de filas del warehouse
    cols: Número de columnas del warehouse
    df_categories: DataFrame con categorías y slots disponibles
    rules_by_category: Diccionario con las reglas de asociación por categoría
    products_by_category: Diccionario con los productos por categoría
    category_slots: Diccionario con los slots disponibles por categoría
    df_products: DataFrame con todos los productos
    """
    
    row, col = 0, 0
    product_slots = {}  # Diccionario para registrar qué productos se asignan a qué slot
    all_products = set(df_products['ProductID'].unique())  # Todos los productos disponibles
    distributed_products = set()  # Productos que ya han sido distribuidos

    # Paso 1: Distribuir productos según las reglas de asociación por categoría
    for _, row_data in df_categories.iterrows():
        category = row_data["Category"]
        slots = row_data["Slots"]
        
        # Obtener los productos correspondientes a esta categoría
        products = products_by_category.get(category, [])
        if not products:
            continue
        
        # Distribuir productos en los slots de la categoría según reglas de asociación
        assigned_products = 0
        for r in range(rows):
            for c in range(cols):
                # Verificamos si este slot pertenece a la categoría actual y está vacío
                if warehouse_map[r, c] == category and (r, c) not in product_slots.values():
                    if assigned_products < len(products):
                        warehouse_map[r, c] = products[assigned_products]  # Asignar producto al slot
                        product_slots[products[assigned_products]] = (r, c)  # Registrar la posición
                        distributed_products.add(products[assigned_products])  # Marcar como asignado
                        assigned_products += 1
                    else:
                        break
            if assigned_products >= len(products):
                break

    # Paso 2: Distribuir productos de la categoría especial 'Cuff-Over the Calf'
    cuff_category = 'Cuff-Over the Calf'
    cuff_products = products_by_category.get(cuff_category, [])
    if cuff_products:
        assigned_products = 0
        for r in range(rows):
            for c in range(cols):
                if warehouse_map[r, c] == cuff_category and (r, c) not in product_slots.values():
                    if assigned_products < len(cuff_products):
                        warehouse_map[r, c] = cuff_products[assigned_products]  # Asignar producto al slot
                        product_slots[cuff_products[assigned_products]] = (r, c)  # Registrar la posición
                        distributed_products.add(cuff_products[assigned_products])  # Marcar como asignado
                        assigned_products += 1
                    else:
                        break
            if assigned_products >= len(cuff_products):
                break

    # Paso 3: Distribuir los productos no asignados en los huecos vacíos de cada categoría
    not_distributed = all_products - distributed_products
    if not_distributed:
        for _, row_data in df_categories.iterrows():
            category = row_data["Category"]
            products = products_by_category.get(category, [])

            # Identificar los slots vacíos en esta categoría
            empty_slots = []
            for r in range(rows):
                for c in range(cols):
                    if warehouse_map[r, c] == category and (r, c) not in product_slots.values():
                        empty_slots.append((r, c))
            
            # Colocar los productos no asignados en los huecos vacíos de la categoría
            assigned_products = 0
            for (r, c) in empty_slots:
                if assigned_products < len(not_distributed):
                    product = not_distributed.pop()
                    warehouse_map[r, c] = product  # Asignar producto al slot
                    product_slots[product] = (r, c)  # Registrar la posición
                    assigned_products += 1
                else:
                    break

    return warehouse_map, product_slots

def distancia_euclidia(i, j, df):
    x1, y1 = df.loc[i, ['X', 'Y']]
    x2, y2 = df.loc[j, ['X', 'Y']]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)



def distancia_origen(i, df, origen=(0,0)):
    
    x1,y1 = df.loc[i, ['X', 'Y']]
    
    return math.sqrt((x1 - origen[0])+ (y1-origen[1]))




def pickup_products(productos_a_recoger, df_slots):
    
    
    #Filtro el dataset para quedarme con la informacion de los productos que quiero
    productos_disponibles = df_slots[df_slots['ProductName'].isin(productos_a_recoger.keys())]
    print(productos_disponibles)
    
    # verifico si hay suficiente cantidad en stock
    
    productos_disponibles['CantidadRecoger'] = productos_disponibles['ProductName'].apply(
        lambda row: min(productos_a_recoger.get(row, 0), productos_disponibles.loc[productos_disponibles['ProductName'] == row, 'Stock'].values[0])
    )
    
    # filtrar para aquellos que pida mas cantidad de la que hay
    
    productos_insuficientes = productos_disponibles[productos_disponibles['CantidadRecoger'] > productos_disponibles['Stock']]
    
    if not productos_insuficientes.empty:
        print("Productos con cantidad insuficiente:")
        print(productos_insuficientes[['ProductName', 'CantidadRecoger', 'Stock']])
        
    # Actualizar el stock de los productos que si voy a vender
    
    
    
    for index, row in productos_disponibles.iterrows():
        
        df_slots.loc[df_slots['ProductName'] == row['ProductName'], 'Stock'] -= row['CantidadRecoger']
    
    # Crear un nuevo DataFrame con los productos que van a ser recogidos (cantidad > 0)
    productos_ajustados = productos_disponibles[productos_disponibles['CantidadRecoger'] > 0]

    # Ver los productos ajustados
    print("\nProductos que se van a recoger (ajustados):")
    print(productos_ajustados[['ProductName', 'CantidadRecoger', 'Stock']])
        
    print("Stock Actualizado \n")
    
    # creo una lista de los productos que voy a recoger
    
    productos_seleccionados = productos_ajustados[['ProductName', 'X', 'Y', 'CantidadRecoger']].set_index('ProductName')
    print("Productos seleccionados y sus coordenadas: \n")
    print(productos_seleccionados)
    
    productos = productos_seleccionados.index.tolist()
    
    # creamos el modelo de optimizacion
    
    prob = LpProblem("Minimizar_Ruta", LpMinimize)
    
    # creamos la variable para la ruta (x_i_j)
    
    x = LpVariable.dicts("ruta", (productos, productos), lowBound=0, upBound=1, cat='Binary')

    # Función objetivo: minimizar la distancia total
    prob += lpSum(distancia_euclidia(i, j, productos_seleccionados) * x[i][j] for i in productos for j in productos if i != j) + \
                lpSum(distancia_origen(i, productos_seleccionados) * x[productos[0]][i] for i in productos)
                
    # Restricciones del problema 
    # Cada producto debe ser visitado exactamente una vez
    for i in productos:
        prob += lpSum(x[i][j] for j in productos if i != j) == 1
        prob += lpSum(x[j][i] for j in productos if i != j) == 1

    # Resolver el problema
    prob.solve()
    
    # Mostrar la solución
    if LpStatus[prob.status] == "Optimal":
        print(f"Distancia mínima total: {value(prob.objective)}")
        recorrido = []
        for i in productos:
            for j in productos:
                if i != j and x[i][j].varValue == 1:
                    recorrido.append((i, j))
        
        # Mostrar el orden de recogida
        print("Orden de recogida de productos:")
        for i, j in recorrido:
            print(f"Recoger producto {i} -> producto {j}")
    else:
        print("No se pudo encontrar una solución óptima.")

    # Mostrar los productos con la cantidad seleccionada
    print("\nResumen de productos recogidos y su cantidad:")
    for index, row in productos_ajustados.iterrows():
        print(f"Producto {row['ProductName']} - Cantidad a recoger: {row['CantidadRecoger']} - Stock final: {row['Stock']}")
        
    return df_slots

