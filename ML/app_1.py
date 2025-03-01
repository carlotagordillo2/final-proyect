import streamlit as st
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value



def distancia_euclidia(i, j, productos_seleccionados):
    if i == 'Origen' or j == 'Origen':  # El origen no tiene coordenadas, no debe usarse en el cálculo
        return 0
    x_i = productos_seleccionados.loc[i, 'X']
    y_i = productos_seleccionados.loc[i, 'Y']
    x_j = productos_seleccionados.loc[j, 'X']
    y_j = productos_seleccionados.loc[j, 'Y']
    return ((x_j - x_i)**2 + (y_j - y_i)**2)**0.5

# Función para calcular la distancia desde el origen (0,0) a un producto
def distancia_origen(i, productos_seleccionados):
    if i == 'Origen':  # El origen no tiene coordenadas
        return 0
    x_i = productos_seleccionados.loc[i, 'X']
    y_i = productos_seleccionados.loc[i, 'Y']
    return (x_i**2 + y_i**2)**0.5

# Función principal para procesar los productos y calcular la ruta
def pickup_products(productos_a_recoger, df_slots):
    # Filtro el dataset para quedarme con los productos seleccionados
    productos_disponibles = df_slots[df_slots['ProductName'].isin(productos_a_recoger.keys())]
    
    # Verificar si hay suficiente stock
    productos_disponibles['CantidadRecoger'] = productos_disponibles['ProductName'].apply(
        lambda row: min(productos_a_recoger.get(row, 0), productos_disponibles.loc[productos_disponibles['ProductName'] == row, 'Stock'].values[0])
    )
    
    # Actualizar el stock
    for index, row in productos_disponibles.iterrows():
        df_slots.loc[df_slots['ProductName'] == row['ProductName'], 'Stock'] -= row['CantidadRecoger']
    
    # Filtrar productos ajustados
    productos_ajustados = productos_disponibles[productos_disponibles['CantidadRecoger'] > 0]
    
    # Crear el modelo de optimización
    productos = ['Origen'] + productos_ajustados['ProductName'].tolist()  # Añadimos el origen
    productos_seleccionados = productos_ajustados.set_index('ProductName')
    
    prob = LpProblem("Minimizar_Ruta", LpMinimize)
    x = LpVariable.dicts("ruta", (productos, productos), lowBound=0, upBound=1, cat='Binary')
    
    # Función objetivo: minimizar la distancia total (distancia entre productos más regreso al origen)
    prob += lpSum(distancia_euclidia(i, j, productos_seleccionados) * x[i][j] for i in productos for j in productos if i != j) + \
             lpSum(distancia_origen(i, productos_seleccionados) * x['Origen'][i] for i in productos[1:])
    
    # Restricciones de entrada y salida (cada producto debe ser visitado una vez)
    for i in productos:
        prob += lpSum(x[i][j] for j in productos if i != j) == 1
    for j in productos:
        prob += lpSum(x[i][j] for i in productos if i != j) == 1
        
    for i in productos:
        for j in productos:
            if i != j:
                prob += x[i][j] + x[j][i] <= 1  # Solo un camino entre i y j (no se puede ir de A -> B y luego de B -> A)
    
    # Resolver el problema
    prob.solve()
    
    # Obtener el recorrido óptimo
    recorrido = []
    if LpStatus[prob.status] == "Optimal":
        for i in productos:
            for j in productos:
                if i != j and x[i][j].varValue == 1:
                    recorrido.append((i, j))
    
    # Aseguramos que el regreso al origen esté al final del recorrido
    if recorrido:
        if len(recorrido) > 1:
            # Obtener el último producto del recorrido
            ultimo_producto = recorrido[-1][1]  # Último producto recogido
            recorrido.append((ultimo_producto, "Origen (0,0)"))  # Conectar el último producto al origen
        else:
            # Si solo hay un producto, tratamos ese caso como un regreso directo
            recorrido.append((recorrido[0][1], "Origen (0,0)"))
            
        print(recorrido)
        recorrido = recorrido[:-1]
        print(recorrido)
    else:
        
        
        recorrido = [(productos[1], "Origen (0,0)")]
        
    
    print(recorrido)

    return df_slots, productos_ajustados, recorrido, productos_seleccionados

def mostrar_recorrido(recorrido, productos_seleccionados):
    st.write("Recorrido óptimo para recoger los productos (con ubicaciones):")
    
    for i, j in recorrido:
       
        # Comprobar si el producto i está en el DataFrame
        if i in productos_seleccionados.index:
            coords_i = productos_seleccionados.loc[i, ['X', 'Y']]
            st.write(f"Recoger {i} (Ubicación: {coords_i['X']}, {coords_i['Y']}) → {j}")
        else:
            st.write(f"Producto {i} no encontrado en los productos seleccionados.")

# Cargar los datos de productos
df = pd.read_csv("../Datasets/final_distribution_warehouse.csv", index_col=0)


st.title ("Portal Ventas")

opcion = st.sidebar.radio("Elige una opcion", 
                          ("Inicio", "Recolección de productos", "Pedidos"))

if opcion == "Inicio":
    
    st.header("Bienvenido a la página de incio")
    st.write("holaaaaa")
    
    
elif opcion == "Recolección de productos":
    # Interfaz de usuario
    st.subheader('Optimización de Ruta para Recoger Productos')

    # Selección de categoría
    categorias = df['Category'].unique()

    categorias_seleccionadas = st.multiselect("Selecciona las categorías", categorias)

    # Filtrar productos según las categorías seleccionadas
    if categorias_seleccionadas:
        productos_filtrados = df[df['Category'].isin(categorias_seleccionadas)]
        st.write(f"Productos disponibles de las categorías seleccionadas ({', '.join(categorias_seleccionadas)}):")
        st.write(productos_filtrados)

        # Inicializar lista de productos seleccionados
        productos_a_recoger = {}

        # Mostrar los productos disponibles y permitir la selección de múltiples productos
        productos_seleccionados = st.multiselect(
            'Selecciona los productos:',
            productos_filtrados['ProductName'].unique(),
            default=[]
        )

        # Permitir la selección de cantidades para los productos seleccionados
        for producto in productos_seleccionados:
            cantidad = st.number_input(f"Cantidad de {producto}", min_value=0, max_value=int(productos_filtrados[productos_filtrados['ProductName'] == producto]['Stock'].values[0]))
            if cantidad > 0:
                productos_a_recoger[producto] = cantidad

        # Mostrar los productos seleccionados y sus cantidades
        st.write("Productos seleccionados:")
        for producto, cantidad in productos_a_recoger.items():
            st.write(f"{producto} - {cantidad} unidades")

        # Botón para ejecutar la función de calcular la ruta óptima
        if st.button('Calcular Ruta Óptima'):
            if productos_a_recoger:
                df, productos_ajustados, recorrido, productos_seleccionados = pickup_products(productos_a_recoger, df)
                
                # Mostrar los productos ajustados
                st.write("Productos seleccionados y ajustados:")
                st.dataframe(productos_ajustados[['ProductName', 'CantidadRecoger', 'Stock']])
                
                # Mostrar el recorrido con ubicaciones
                mostrar_recorrido(recorrido, productos_seleccionados)
            else:
                st.write("Por favor, seleccione al menos un producto.")
            
    else:
        st.write("Por favor, selecciona al menos una categoría para ver los productos disponibles.")
        
elif opcion == "Pestaña 2":
    
    st.subheader("contenido 2")
    st.write("holaaa")
