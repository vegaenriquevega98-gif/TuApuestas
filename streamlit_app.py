import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests

# Configuración inicial de la app
st.set_page_config(page_title="Analizador de Apuestas Pro", layout="wide")

# 1. INTEGRACIÓN DE API Y BASE DE DATOS
def obtener_datos_api(api_key, liga):
    """
    Aquí conectas tu proveedor (ej. API-Football). 
    Retornamos datos simulados con estadísticas avanzadas para que el script funcione.
    """
    # En producción: response = requests.get(f"URL_DE_TU_API/{liga}", headers={"key": api_key})
    datos_simulados = {
        'Equipo': ['Local FC', 'Visitante United'],
        'Goles_Promedio': [1.8, 1.2],
        'xG_Promedio': [2.1, 1.0], # Goles esperados
        'Tiros_Puerta_Promedio': [6.5, 3.2],
        'Fuerza_Defensiva': [0.8, 1.3] # Menor es mejor
    }
    return pd.DataFrame(datos_simulados)

# Inicializar estado para no perder datos al cambiar de mercado
if 'datos_liga' not in st.session_state:
    st.session_state.datos_liga = None

# 2. MOTOR MATEMÁTICO: POISSON CON DATA AVANZADA
def calcular_fuerza_ajustada(xg, tiros, goles, es_local=False):
    # Ponderación para crear una "Fuerza Ofensiva Real"
    fuerza = (xg * 0.5) + (goles * 0.3) + (tiros * 0.05)
    if es_local:
        fuerza = fuerza * 1.15 # Factor de Fuerza de Localía (Home Advantage)
    return fuerza

def matriz_poisson(lambda_local, lambda_visitante, max_goles=6):
    # Genera la matriz de probabilidades de todos los resultados posibles (0-0, 1-0, 2-1, etc.)
    prob_local = [poisson.pmf(i, lambda_local) for i in range(max_goles)]
    prob_visitante = [poisson.pmf(i, lambda_visitante) for i in range(max_goles)]
    matriz = np.outer(prob_local, prob_visitante)
    return matriz

# 3. INTERFAZ Y MERCADOS
st.title("⚽ Analizador Algorítmico de Fútbol")

# Panel lateral: Configuración y API
with st.sidebar:
    st.header("Configuración de Datos")
    api_key_input = st.text_input("Ingresa tu API Key (Opcional por ahora)", type="password")
    if st.button("Cargar Base de Datos"):
        st.session_state.datos_liga = obtener_datos_api(api_key_input, "Premier League")
        st.success("Datos cargados en memoria.")

if st.session_state.datos_liga is not None:
    # Simulación de extracción de métricas del dataframe
    lambda_loc = calcular_fuerza_ajustada(2.1, 6.5, 1.8, es_local=True)
    lambda_vis = calcular_fuerza_ajustada(1.0, 3.2, 1.2, es_local=False)
    
    matriz = matriz_poisson(lambda_loc, lambda_vis)
    
    # Cálculos de Mercados extrayendo áreas de la matriz
    prob_local = np.sum(np.tril(matriz, -1))
    prob_empate = np.sum(np.diag(matriz))
    prob_visitante = np.sum(np.triu(matriz, 1))
    
    prob_over_25 = 1 - np.sum(matriz[np.triu_indices_from(matriz)] * (np.add.outer(np.arange(6), np.arange(6)) <= 2)[np.triu_indices_from(matriz)])
    
    # Pestañas para cambiar de mercado sin perder contexto
    tab1, tab2, tab3 = st.tabs(["1X2 (Match Odds)", "Over/Under", "Asistente IA"])
    
    with tab1:
        st.subheader("Mercado 1X2")
        col1, col2, col3 = st.columns(3)
        col1.metric("Gana Local", f"{prob_local*100:.1f}%")
        col2.metric("Empate", f"{prob_empate*100:.1f}%")
        col3.metric("Gana Visitante", f"{prob_visitante*100:.1f}%")
        
        st.markdown("### Valor Esperado (EV)")
        cuota_casa = st.number_input("Cuota que ofrece la casa de apuestas (Local)", value=2.0)
        ev = (prob_local * cuota_casa) - 1
        st.success(f"EV para el Local: {ev*100:.2f}%") if ev > 0 else st.error(f"EV Negativo: {ev*100:.2f}%")

    with tab2:
        st.subheader("Mercado de Goles Totales")
        st.metric("Probabilidad Over 2.5 Goles", f"{prob_over_25*100:.1f}%")
        st.metric("Probabilidad Under 2.5 Goles", f"{(1-prob_over_25)*100:.1f}%")

    with tab3:
        st.subheader("Generador de Prompt para Gemini")
        prompt_generado = f"""
        Actúa como un analista experto de fútbol. Basado en el siguiente modelo de Poisson ajustado por xG y fuerza de localía:
        - Probabilidad Local: {prob_local*100:.1f}%
        - Probabilidad Empate: {prob_empate*100:.1f}%
        - Probabilidad Visitante: {prob_visitante*100:.1f}%
        - Goles Esperados (xG) Local: 2.1 | Visitante: 1.0
        
        Sabiendo que la casa de apuestas paga una cuota de {cuota_casa} por el Local. ¿Crees que el riesgo justifica la apuesta según la varianza típica de este deporte?
        """
        st.text_area("Copia este texto y pégalo en nuestro chat para un análisis profundo:", value=prompt_generado, height=200)

else:
    st.info("Carga la base de datos en el menú lateral para comenzar el análisis.")
