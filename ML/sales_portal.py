import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, association_rules

st.markdown(
    """
    <style>
        
        .st-at { color: #007bff !important; }
        .st-dx { background-color: #007bff !important; }
    </style>
    """,
    unsafe_allow_html=True
)


def recomendar_productos_cliente(client_id, customer_product_matrix, customer_sim_df, top_n=5, top_n_clientes=5):
    # Verificar si el cliente está en el DataFrame de similitudes
    if client_id not in customer_sim_df.columns:
        st.write(f"El cliente con ID {client_id} no tiene datos de similitud.")
        return [], []

    clientes_similaridad = customer_sim_df[client_id]
    clientes_similares = clientes_similaridad.sort_values(ascending=False)[1:top_n_clientes + 1].index

    productos_recomendados = set()
    
    for cliente in clientes_similares:
        productos_comprados = customer_product_matrix.loc[cliente].where(customer_product_matrix[cliente] > 0).dropna().index
        productos_recomendados.update(productos_comprados)
        
    return list(productos_recomendados)[:top_n], list(clientes_similares)

def recomendar_productos(cliente_id, productos_seleccionados):
    recomendaciones = set()

    # 1️⃣ Recomendaciones basadas en el historial de compras del cliente
    if cliente_id in customer_product_matrix.index:
        productos_cliente = customer_product_matrix.loc[cliente_id]
        productos_frecuentes = productos_cliente[productos_cliente > 0].index
        recomendaciones.update(productos_frecuentes)

    # 2️⃣ Recomendaciones basadas en productos similares
    for producto in productos_seleccionados:
        if producto in product_sim_df.index:
            similares = product_sim_df.loc[producto].sort_values(ascending=False).index[1:3]
            recomendaciones.update(similares)

    # 3️⃣ Recomendaciones basadas en reglas de asociación (productos comprados juntos)
    for producto in productos_seleccionados:
        reglas_filtradas = rules[rules['antecedents'].apply(lambda x: producto in list(x))]
        consequents_validos = reglas_filtradas['consequents'].explode().dropna().values
        recomendaciones.update(consequents_validos)

    # Filtrar los productos recomendados en el DataFrame original
    productos_recomendados = df_products[df_products['ProductName'].isin(recomendaciones - set(productos_seleccionados))]

    return productos_recomendados


def recomendar_productos_similares(productos_seleccionados, df_products, top_n=5):
    recomendaciones = set()
    
    features = df_products[['Category', 'ProductLine', 'Size', 'Weight', 'PurchasePrice', 'Gender', 'PackSize']].fillna(0)

    cosine_sim = cosine_similarity(pd.get_dummies(features))

    # Crear un DataFrame de similitudes
    product_sim_df = pd.DataFrame(cosine_sim, index=df_products['ProductName'], columns=df_products['ProductName'])

    for producto in productos_seleccionados:
        if producto in product_sim_df.index:
            similar_scores = product_sim_df.loc[producto].sort_values(ascending=False)
            for rec in similar_scores.index:
                if rec != producto and rec not in productos_seleccionados:
                    recomendaciones.add(rec)
                    if len(recomendaciones) >= top_n:
                        break

    return list(recomendaciones)[:top_n]

def mostrar_tablas_recomendaciones(recomendaciones_productos, df_products, clientes_similares, df_customer):
    # Mostrar tabla de productos recomendados
    productos_recomendados_df = df_products[df_products['ProductName'].isin(recomendaciones_productos)]
    st.write("Top 5 Product Recommendations based on selected products:")
    st.dataframe(productos_recomendados_df[['ProductName', 'Category', 'ProductLine', 'Gender', 'Size', 'Weight', 'PackSize', 'PurchasePrice']])

    # Mostrar tabla de clientes similares
    if clientes_similares:
        clientes_similares_df = df_customer[df_customer.index.isin(clientes_similares)]
        st.write("Customers similar to the selected customer:")
        st.dataframe(clientes_similares_df[['CustomerName', 'Country', 'Region', 'PriceCategory','CustomerClass','LeadSource','Discontinued']])

        
# Función para calcular distancia euclidiana entre productos
def distancia_euclidia(i, j, productos_seleccionados):
    if i == 'Origen' or j == 'Origen':
        return 0
    x_i, y_i = productos_seleccionados.loc[i, ['X', 'Y']]
    x_j, y_j = productos_seleccionados.loc[j, ['X', 'Y']]
    return ((x_j - x_i)**2 + (y_j - y_i)**2)**0.5

# Distancia del origen a un producto
def distancia_origen(i, productos_seleccionados):
    if i == 'Origen':
        return 0
    x_i, y_i = productos_seleccionados.loc[i, ['X', 'Y']]
    return (x_i**2 + y_i**2)**0.5

# Función para calcular la ruta óptima
def pickup_products(productos_a_recoger, df_slots):
    productos_disponibles = df_slots[df_slots['ProductName'].isin(productos_a_recoger.keys())]

    productos_disponibles['CantidadRecoger'] = productos_disponibles['ProductName'].apply(
        lambda row: min(productos_a_recoger.get(row, 0), 
                        productos_disponibles.loc[productos_disponibles['ProductName'] == row, 'Stock'].values[0])
    )

    # Actualizar stock
    for index, row in productos_disponibles.iterrows():
        df_slots.loc[df_slots['ProductName'] == row['ProductName'], 'Stock'] -= row['CantidadRecoger']

    productos_ajustados = productos_disponibles[productos_disponibles['CantidadRecoger'] > 0]
    
    if productos_ajustados.empty:
        return df_slots, productos_ajustados, [], {}

    productos = ['Origen'] + productos_ajustados['ProductName'].tolist()
    productos_seleccionados = productos_ajustados.set_index('ProductName')

    # Modelo de optimización
    prob = LpProblem("Minimizar_Ruta", LpMinimize)
    x = LpVariable.dicts("ruta", (productos, productos), lowBound=0, upBound=1, cat='Binary')

    # Función objetivo: minimizar la distancia total
    prob += lpSum(distancia_euclidia(i, j, productos_seleccionados) * x[i][j] for i in productos for j in productos if i != j)

    # Restricciones
    for i in productos:
        prob += lpSum(x[i][j] for j in productos if i != j) == 1  # Salida única
    for j in productos:
        prob += lpSum(x[i][j] for i in productos if i != j) == 1  # Entrada única
    for i in productos:
        for j in productos:
            if i != j:
                prob += x[i][j] + x[j][i] <= 1  # Evitar ciclos inversos

    # Resolver
    prob.solve()
    
    if LpStatus[prob.status] == "Optimal":
        print(f"Distancia mínima total: {value(prob.objective)}")
        total_recorrido = value(prob.objective)

        # Construcción del recorrido
        recorrido = []
        visitados = set()
        actual = "Origen"

        while len(visitados) < len(productos) - 1:  # Origen no cuenta como producto
            for j in productos:
                if actual != j and x[actual][j].varValue == 1 and j not in visitados:
                    recorrido.append((actual, j))
                    visitados.add(actual)
                    actual = j
                    break

        # Asegurar que solo el último paso lleva al origen
        if recorrido and recorrido[-1][1] != "Origen":
            recorrido.append((recorrido[-1][1], "Origen"))
    else:
        print("No se puedo encontrar la ruta optima")
        total_recorrido = 0

    return df_slots, productos_ajustados, recorrido, productos_seleccionados, total_recorrido

# Función para mostrar el recorrido
def mostrar_recorrido(recorrido, productos_seleccionados):
    st.write("Optimal route for picking up products (with locations):")

    for i, j in recorrido:
        if i in productos_seleccionados.index:
            coords_i = productos_seleccionados.loc[i, ['X', 'Y']]
            st.write(f"Pick up {i} (Location: {coords_i['X']}, {coords_i['Y']}) → {j}")
        elif i == "Origen":
            st.write(f"Start from Origen → {j}")
        else:
            st.write(f"Product {i} not found in selected products.")

# **Carga de datos**
df = pd.read_csv("../Datasets/final_distribution_warehouse.csv", index_col=0)
df_products = pd.read_csv('../Datasets/clean_products.csv', index_col = 0)
df_orders = pd.read_csv('../Datasets/clean_orders.csv', index_col = 0)
df_order_details = pd.read_csv('../Datasets/clean_order_details.csv', index_col=0)
df_customer = pd.read_csv('../Datasets/clean_customer.csv', index_col=0)
df_products_security_stock = pd.read_csv('../Datasets/product_security_stock.csv', index_col=0)

df_1 = df_order_details.merge(df_products, on = 'ProductID')

#crear la matriz customer-product

customer_product_matrix = df_order_details.pivot_table(index='OrderID', columns='ProductID', values='QuantitySold', aggfunc='sum').fillna(0)

#calcular la similaridad entre los clientes

customer_sim_df = cosine_similarity(customer_product_matrix)
customer_sim_df = pd.DataFrame(customer_sim_df, index=customer_product_matrix.index, columns=customer_product_matrix.index)

product_features = df_products[['Category', 'ProductLine', 'Size', 'Weight', 'PurchasePrice', 'Gender', 'PackSize']].fillna(0)

product_similarity = cosine_similarity(pd.get_dummies(product_features))
product_sim_df = pd.DataFrame(product_similarity, index=df_products['ProductName'], columns=df_products['ProductName'])

basket = df_1.pivot_table(index="OrderID", columns="ProductName", values="QuantitySold", aggfunc="sum").fillna(0)
basket = (basket > 0).astype(int)

frequent_items = apriori(basket, min_support=0.05, use_colnames=True)

rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)




# **Interfaz Streamlit**
st.title("Sales Portal")

opcion = st.sidebar.radio("Choose an option", ("Home", "Pick-up products", "Product Location Finder", "Product Recommendations", "Product Replenishment Check"))

if opcion == "Home":
    st.header("Bienvenido a la página de inicio")
    st.write("¡Hola!")

elif opcion == "Pick-up products":
    st.subheader('Route Optimisation for Product Pick-up')

    categorias = df['Category'].unique()
    categorias_seleccionadas = st.multiselect("Select categories", categorias)

    if categorias_seleccionadas:
        productos_filtrados = df[df['Category'].isin(categorias_seleccionadas)]
        st.write("Products available in selected categories:")
        st.write(productos_filtrados[['ProductName', 'Category', 'ProductLine', 'Gender', 'Weight', 'Size', 'PackSize','Stock']])

        productos_a_recoger = {}
        productos_seleccionados = st.multiselect('Select the products:', productos_filtrados['ProductName'].unique(), default=[])

        for producto in productos_seleccionados:
            stock_disponible = productos_filtrados.loc[productos_filtrados['ProductName'] == producto, 'Stock']
            max_stock = int(stock_disponible.iloc[0]) if not stock_disponible.empty else 0

            cantidad = st.number_input(f"Quantity of {producto}", min_value=0, max_value=max_stock)
            if cantidad > 0:
                productos_a_recoger[producto] = cantidad

        st.write("Selected products:")
        for producto, cantidad in productos_a_recoger.items():
            st.write(f"{producto} - {cantidad} units")

        if st.button('Calculate the optimal route'):
            if productos_a_recoger:
                df, productos_ajustados, recorrido, productos_seleccionados, total_recorrido = pickup_products(productos_a_recoger, df)
                
                st.write("Selected Products:")
                st.dataframe(productos_ajustados[['ProductName', 'CantidadRecoger', 'Stock']])

                mostrar_recorrido(recorrido, productos_seleccionados)
                st.write(f'Minimum distance: {total_recorrido}')
            else:
                st.write("Please select at least one product.")

elif opcion == "Product Location Finder":
    st.subheader("Product Location Finder")
    st.write("Select your products and receive their location!")
    
    # selecciona la categoria
    categorias = df['Category'].unique()
    categorias_seleccionadas = st.multiselect("Select your product categories", categorias)
    
    if not categorias_seleccionadas:
        st.warning("Select at least one category")
    # selecciona la linea de producto
    
    lineas = df[df['Category'].isin(categorias_seleccionadas)]['ProductLine'].unique()
    lineas_seleccionadas = st.multiselect("Select Product Line", lineas)
    
    if not lineas_seleccionadas:
        st.warning("Select at least one product line")
        
    # select products
    
    productos_filtrados = df[(df['Category'].isin(categorias_seleccionadas)) & (df['ProductLine'].isin(lineas_seleccionadas))]
    productos_seleccionados = st.multiselect("Select product(s)", productos_filtrados['ProductName'].unique())
    
    if not productos_seleccionados:
        st.warning("Select at least one product")
        
    if st.button("Show product locations"):
        
        if productos_seleccionados:
            ubicaciones = df[df['ProductName'].isin(productos_seleccionados)][['ProductName', 'X', 'Y']]
            ubicaciones = ubicaciones.rename(columns={'X': 'Row', 'Y': 'Column'})
            st.write("Product locations:")
            st.dataframe(ubicaciones)
        
        else: 
            st.warning("Please select at least one product.")

elif opcion == 'Product Recommendations':
    st.subheader("Product Recommendations for You")

    # Opción de recomendación por clientes o por productos
    ##recomendacion_tipo = st.radio("Choose recommendation type", ("Based on Customers", "Based on Products", "Based on Customer's basket"))

    recomendacion_tipo = st.radio("Choose recommendation type", ("Based on Products", "Based on Customer's basket"))
    
    
    #if recomendacion_tipo == "Based on Customers":
        # Crear un desplegable para seleccionar un cliente
    #   cliente_ids = df_customer['CustomerID'].unique()
    #    cliente_id = st.selectbox("Selecciona tu ID de cliente", cliente_ids)

     #   if cliente_id:
            # Obtener las recomendaciones basadas en el cliente seleccionado
    #        recomendaciones_cliente, clientes_similares = recomendar_productos_cliente(cliente_id, customer_product_matrix, customer_sim_df, top_n=5)
            
            # Mostrar las recomendaciones de productos para el cliente seleccionado
    #        if recomendaciones_cliente:
    #            st.write(f"Top 5 product recommendations for customer {cliente_id}:")
    #           mostrar_tablas_recomendaciones(recomendaciones_cliente, df_products, clientes_similares, df_customer)
    #        else:
    #            st.write(f"No se encontraron recomendaciones para el cliente {cliente_id}.")
    #    else:
     #       st.write("Please select a valid customer ID.")*/

    if recomendacion_tipo == "Based on Products":
        productos_seleccionados = st.multiselect('Select the products you are interested in:', df_products['ProductName'].unique(), default=[])
        if productos_seleccionados:
            recomendaciones_productos = recomendar_productos_similares(productos_seleccionados, df_products, top_n=5)
            #st.write(f"Top 5 product recommendations based on selected products:")
            mostrar_tablas_recomendaciones(recomendaciones_productos, df_products, [], df_customer)
        else:
            st.write("Please select at least one product.")
            
    elif recomendacion_tipo == "Based on Customer's basket":
        
        st.subheader("Personalized recommendations:")
        
        cliente_id = st.selectbox("Select a client:", df_customer["CustomerID"].unique())
        categorias = df_products['Category'].unique()
        categorias_seleccionadas = st.multiselect("Select categories", categorias)

        if  not categorias_seleccionadas:
            
            st.warning("Please select at least one category.")
            
        else:
            
            productos_filtrados = df_products[df_products['Category'].isin(categorias_seleccionadas)]
            st.write("Products available in selected categories:")
            st.write(productos_filtrados[['ProductName', 'Category', 'ProductLine', 'Gender', 'Weight', 'Size', 'PackSize']])

            productos_seleccionados = st.multiselect("Select products:", productos_filtrados["ProductName"].unique())
            

        if st.button("Obtein recommendations"):
                
            if not productos_seleccionados:
                st.warning("Please select at least one product.")
            else:
                recomendaciones = recomendar_productos(cliente_id, productos_seleccionados)
                if not recomendaciones.empty:
                    st.write("✅ Recommended products:")
                    st.dataframe(recomendaciones[['ProductName', 'Category', 'ProductLine', 'Gender', 'Size', 'Weight', 'PackSize', 'PurchasePrice']])
                else:
                    st.write("❌ It doesn't find any recommendation for this combination.")

elif opcion == "Product Replenishment Check":
    
    st.subheader("Product Replenishment Check")
    
    #select categories 
    categorias = df_products_security_stock['Category'].unique()
    categoria_seleccionadas = st.selectbox("Select category:", categorias)
    
    #select products based on category 
    
    products_filters = df_products_security_stock[df_products_security_stock['Category']==categoria_seleccionadas]
    product_selected = st.multiselect("Select products:", products_filters['ProductName'].unique())
    
    # enter the quantity 
    cantidad_seleccionada = {}
    
    if product_selected:
        for prod in product_selected:
            cantidad = st.number_input(f"Enter quantity needed for {prod}:", min_value=1, step=1, key=prod)
            cantidad_seleccionada[prod] = cantidad
    
        if st.button("Check Replenishment Need:"):
            if product_selected:
                for prod in product_selected:
                    
                    product_info = df_products_security_stock[df_products_security_stock['ProductName'] == prod].iloc[0]
                    
                    # data
                    
                    stock_disponible = product_info['Stock']
                    rop = product_info['ROP']
                    security_stock = product_info['SecurityStock']
                    lead_time = (float(product_info['LeadTime(avg)']), float(product_info['LeadTime(std)']))
                    
                    # Mostrar los valores del producto
                    st.write(f"🔹 **Product:** {prod}")
                    st.write(f"🔹 **Available stock:** {stock_disponible}")
                    st.write(f"🔹 **ROP (Reorder Point):** {rop}")
                    st.write(f"🔹 **Security Stock (Min. Required):** {security_stock}")
                    st.write(f"🔹 **Lead Time (average, std):** {lead_time}")
                    
                    cantidad_requerida = cantidad_seleccionada[prod]  # Cantidad ingresada por el usuario
                    st.write(f"🔹 **Quantity requested:** {cantidad_requerida}")
                    
                    # Actualizar el stock disponible restando la cantidad solicitada
                    stock_disponible -= cantidad_requerida
                    st.write(f"🔹 **Updated available stock (after customer request):** {stock_disponible}")
            
                    
                    # Verificar si se necesita hacer un pedido
                    if stock_disponible < security_stock:
                        cantidad_a_pedir = max(0, security_stock - stock_disponible)  # Para asegurar el mínimo de stock
                        st.write(f"⚠️ You need to reorder {cantidad_a_pedir} units to reach the Security Stock!")
                    elif stock_disponible < rop:
                        cantidad_a_pedir = max(0, rop - stock_disponible)  # Para alcanzar el ROP
                        st.write(f"⚠️ You need to reorder {cantidad_a_pedir} units to reach the ROP!")
                    else:
                        st.write("✅ No need to reorder, sufficient stock available.")
        else:
            st.warning("Please select at least one product.")
        