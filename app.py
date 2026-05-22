import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

from tensorflow.keras.models import load_model

# CONFIGURACIÓN
st.set_page_config(
    page_title="MINALTA Dashboard IA",
    layout="wide"
)

# TÍTULO
st.title("Sistema Predictivo de Fallas - MINALTA S.A.")

st.write(
    """
    Monitoreo inteligente de correas transportadoras mediante
    sensores IoT y modelos de Inteligencia Artificial.
    """
)

# CARGAR MODELO Y SCALER
@st.cache_resource
def cargar_modelos():

    modelo = load_model("modelo_mlp_fallas_correas.h5")

    scaler = joblib.load(
        "scaler_fallas_correas.pkl"
    )

    return modelo, scaler

modelo, scaler = cargar_modelos()

# SIDEBAR
st.sidebar.header("Configuración")

modo = st.sidebar.selectbox(
    "Modo de operación",
    [
        "Simulación automática",
        "Modo manual"
    ]
)

# FUNCIÓN PREDICCIÓN
def predecir_falla(
    temperatura,
    vibracion,
    sonido,
    rpm
):

    entrada = pd.DataFrame({
        "temperatura": [temperatura],
        "vibracion": [vibracion],
        "sonido": [sonido],
        "rpm": [rpm]
    })

    entrada_scaled = scaler.transform(
        entrada
    )

    probabilidad = modelo.predict(
        entrada_scaled,
        verbose=0
    )[0][0]

    prediccion = (
        1 if probabilidad > 0.5 else 0
    )

    return prediccion, probabilidad

# MODO AUTOMÁTICO
if modo == "Simulación automática":

    st.subheader(
        "Simulación IoT en tiempo real"
    )

    temperatura = np.random.normal(
        65,
        12
    )

    vibracion = np.random.normal(
        5,
        2.5
    )

    sonido = np.random.normal(
        72,
        10
    )

    rpm = np.random.normal(
        1500,
        120
    )

    temperatura = np.clip(
        temperatura,
        30,
        110
    )

    vibracion = np.clip(
        vibracion,
        0,
        16
    )

    sonido = np.clip(
        sonido,
        40,
        115
    )

    rpm = np.clip(
        rpm,
        1000,
        2000
    )

# MODO MANUAL
else:

    st.subheader("Modo manual")

    temperatura = st.sidebar.slider(
        "Temperatura (°C)",
        20.0,
        120.0,
        60.0
    )

    vibracion = st.sidebar.slider(
        "Vibración (mm/s)",
        0.0,
        15.0,
        5.0
    )

    sonido = st.sidebar.slider(
        "Sonido (dB)",
        40.0,
        120.0,
        70.0
    )

    rpm = st.sidebar.slider(
        "RPM",
        1000.0,
        2000.0,
        1500.0
    )

# PREDICCIÓN
prediccion, probabilidad = predecir_falla(
    temperatura,
    vibracion,
    sonido,
    rpm
)

# MÉTRICAS
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Temperatura",
    f"{temperatura:.1f} °C"
)

col2.metric(
    "Vibración",
    f"{vibracion:.1f} mm/s"
)

col3.metric(
    "Sonido",
    f"{sonido:.1f} dB"
)

col4.metric(
    "RPM",
    f"{rpm:.0f}"
)

# RESULTADO
st.subheader(
    "Resultado del modelo IA"
)

if prediccion == 1:

    st.error(
        "ALERTA: posible falla mecánica detectada"
    )

else:

    st.success(
        "Operación normal"
    )

# PROBABILIDAD
st.metric(
    "Probabilidad de falla",
    f"{probabilidad * 100:.2f}%"
)

# INTERPRETACIÓN
st.subheader(
    "Interpretación preliminar"
)

if temperatura > 75:

    st.warning(
        "Posible sobrecalentamiento"
    )

elif vibracion > 8:

    st.warning(
        "Posible falla de rodamiento"
    )

elif sonido > 85:

    st.warning(
        "Posible desalineación"
    )

else:

    st.info(
        "Comportamiento operacional normal"
    )

# TABLA
st.subheader(
    "Lecturas actuales"
)

tabla = pd.DataFrame({
    "Variable": [
        "Temperatura",
        "Vibración",
        "Sonido",
        "RPM"
    ],
    "Valor": [
        round(temperatura, 2),
        round(vibracion, 2),
        round(sonido, 2),
        round(rpm, 2)
    ]
})

st.table(tabla)

# GRÁFICO
st.subheader(
    "Visualización sensores"
)

grafico = pd.DataFrame({
    "Sensor": [
        "Temperatura",
        "Vibración",
        "Sonido",
        "RPM"
    ],
    "Valor": [
        temperatura,
        vibracion,
        sonido,
        rpm
    ]
})

st.bar_chart(
    grafico.set_index("Sensor")
)

# REFRESH AUTOMÁTICO
if modo == "Simulación automática":

    time.sleep(3)

    st.rerun()
