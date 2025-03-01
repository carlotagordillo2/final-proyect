import streamlit as st
import pandas as pd

#load data

df_slots = pd.read_csv('..Datasets/final_distribution_warehouse.csv', index_col = 0)

def actualizar_stock(productos_a_recoger, df_slots):
    # Filtramos los productos que vamos a recoger
    productos_disponibles = df_slots[df_slots['ProductName'].isin(productos_a_recoger.keys())]

    # Ajustamos las cantidades a recoger según el stock disponible
    productos_disponibles['Cantidad_a_recoger'] = productos_disponibles.apply(
        lambda row: min(productos_a_recoger.get(row['ProductName'], 0), row['Stock']),
        axis=1
    )

    # Verificamos qué productos tienen cantidad insuficiente
    productos_insuficientes = productos_disponibles[productos_disponibles['Cantidad_a_recoger'] > productos_disponibles['Stock']]
    if not productos_insuficientes.empty:
        st.subheader("Productos con cantidad insuficiente")
        st.write(productos_insuficientes[['ProductName', 'Cantidad_a_recoger', 'Stock']])

    # Mostramos los productos que se van a recoger
    productos_ajustados = productos_disponibles[productos_disponibles['Cantidad_a_recoger'] > 0]
    st.subheader("Productos que se van a recoger (ajustados):")
    st.write(productos_ajustados[['ProductName', 'Cantidad_a_recoger', 'Stock']])

    # Actualizamos el stock en df_slots
    for index, row in productos_ajustados.iterrows():
        # Restamos la cantidad a recoger del stock de cada producto
        df_slots.loc[df_slots['ProductName'] == row['ProductName'], 'Stock'] -= row['Cantidad_a_recoger']

    # Mostramos el stock actualizado
    st.subheader("Stock actualizado:")
    st.write(df_slots[['ProductName', 'Stock']])
    
# Titulo de la app

st.title("Gestión de inventario")
opciones = ['Recoger Inventario', 'Mostrar Stock Actualizado']

pagina = st.radio("Selecciona una opción", opciones)

if pagina == 'Recoger Inventario':

    st.header("Recoger Productos del Inventario")
    
    #Filtrar las categorias
    
    categorias = df_slots['Category'].unique()
    categoria_seleccionada = st.selectbox("Selecciona una categoría", categorias)
    
    #Filtrar los productos por categoría
    
    productos_filtrados = df_slots[df_slots['Category'] == categoria_seleccionada]['ProductName'].tolist()
    
    productos_a_recoger = {}
    
    for producto in productos_filtrados:
        cantidad = st.number_input(f"Cantidad a recoger de {producto}", min_value=0, step=1)
        if cantidad > 0:
            productos_a_recoger[producto] = cantidad

    # Botón para ejecutar la actualización del stock
    if st.button("Actualizar Stock"):
        if productos_a_recoger:
            actualizar_stock(productos_a_recoger, df_slots)
        else:
            st.write("No se ha seleccionado ningún producto.")

    elif pagina == 'Mostrar Stock Actualizado':
        st.header("Estado del Stock Actualizado")
        st.write(df_slots[['ProductName', 'Stock']])
    