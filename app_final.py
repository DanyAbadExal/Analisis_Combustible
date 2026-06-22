import os
import io
import base64
import warnings
import pyodbc
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from scipy.stats import gaussian_kde

# LIBRERÍAS DE INTERACTIVIDAD Y GRÁFICOS
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)

# ===================================================================
# 0. CONFIGURACIÓN GLOBAL Y TEMA CORPORATIVO EXALMAR
# ===================================================================
st.set_page_config(
    page_title="Pesquera Exalmar | Analytics Fleet",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🐟"
)

# ─── Paleta Exalmar (UI y Módulo 2) ────────────────────
C_BG_DARK      = '#0a1628'
C_PANEL_DARK   = '#0f1e38'
C_PANEL_MID    = '#162440'
C_BORDER       = '#1e3258'
C_TEXT_LIGHT   = '#dce8f5'
C_TEXT_MID     = '#8ba3c4'
C_TEAL_MID     = '#3f8d8c'
C_NAVY_DARK    = '#03386f'
C_CYAN_DRK     = '#038f8d'
C_GREEN_XLG    = '#89c04c'
C_BLUE_DEEP    = '#004692'
C_WARN_ORANGE  = '#f59e0b'
C_DANGER_RED   = '#ef4444'

# ─── Paleta Pastel ───────────
P_TEAL         = '#a2e1db'
P_GREEN        = '#baffc9'
P_BLUE         = '#bae1ff'
P_ORANGE       = '#ffb347'
P_RED          = '#ff6961'

# Configuración global de Seaborn para fondo oscuro
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "none",
    "figure.facecolor": "none",
    "grid.color": C_BORDER,
    "text.color": C_TEXT_LIGHT,
    "axes.labelcolor": C_TEXT_MID,
    "xtick.color": C_TEXT_MID,
    "ytick.color": C_TEXT_MID,
    "legend.frameon": True,
    "legend.facecolor": C_PANEL_DARK,
    "legend.edgecolor": C_BORDER
})

# ─── Inyección de CSS Global ────────────────────────────────────────
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp {{ background-color: {C_BG_DARK}; color: {C_TEXT_LIGHT}; font-family: 'Inter', 'Segoe UI', sans-serif; }}
    h1, h2, h3, h4, h5 {{ color: white !important; font-family: 'Inter', 'Segoe UI', sans-serif !important; letter-spacing: -0.3px; }}
    [data-testid="stSidebar"] {{ background-color: {C_PANEL_DARK}; border-right: 1px solid {C_BORDER}; }}
    [data-testid="stSidebar"] * {{ color: {C_TEXT_LIGHT} !important; }}
    [data-testid="stSidebar"] .stRadio label {{ font-size: 0.9rem; padding: 4px 0; }}
    div[data-testid="metric-container"] {{ background: linear-gradient(135deg, {C_PANEL_MID} 0%, {C_PANEL_DARK} 100%); border: 1px solid {C_BORDER}; border-left: 4px solid {C_CYAN_DRK}; border-radius: 10px; padding: 16px 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); transition: transform 0.2s ease; }}
    div[data-testid="metric-container"]:hover {{ transform: translateY(-2px); border-left-color: {C_GREEN_XLG}; }}
    div[data-testid="metric-container"] label {{ color: {C_TEXT_MID} !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 500; }}
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{ color: white !important; font-size: 1.6rem !important; font-weight: 700; }}
    .stTabs [data-baseweb="tab-list"] {{ background-color: transparent; border-bottom: 2px solid {C_BORDER}; gap: 0; }}
    .stTabs [data-baseweb="tab"] {{ color: {C_TEXT_MID} !important; background-color: transparent; border-radius: 8px 8px 0 0; padding: 10px 20px; font-size: 0.85rem; font-weight: 500; letter-spacing: 0.3px; transition: color 0.2s; }}
    .stTabs [data-baseweb="tab"]:hover {{ color: {C_TEXT_LIGHT} !important; }}
    .stTabs [aria-selected="true"] {{ color: {C_GREEN_XLG} !important; border-bottom: 3px solid {C_GREEN_XLG} !important; font-weight: 600; }}
    .stSelectbox > div > div {{ background-color: {C_PANEL_MID} !important; border: 1px solid {C_BORDER} !important; border-radius: 8px; color: {C_TEXT_LIGHT} !important; }}
    .streamlit-expanderHeader {{ background-color: {C_PANEL_MID}; border: 1px solid {C_BORDER}; border-radius: 8px; color: {C_TEXT_LIGHT} !important; }}
    .stDataFrame {{ border-radius: 8px; overflow: hidden; border: 1px solid {C_BORDER}; }}
    .stDownloadButton > button {{ background: linear-gradient(135deg, {C_CYAN_DRK}, {C_BLUE_DEEP}); color: white; border: none; border-radius: 8px; padding: 8px 20px; font-weight: 600; letter-spacing: 0.3px; transition: opacity 0.2s; }}
    .stDownloadButton > button:hover {{ opacity: 0.85; }}
    hr {{ border: none; border-top: 1px solid {C_BORDER}; margin: 8px 0; }}
    .stAlert {{ border-radius: 8px; border-left-width: 4px; }}
    .exalmar-header {{ background: linear-gradient(135deg, {C_NAVY_DARK} 0%, {C_BLUE_DEEP} 50%, {C_TEAL_MID} 100%); border-radius: 12px; padding: 18px 24px; margin-bottom: 20px; border-bottom: 3px solid {C_GREEN_XLG}; display: flex; align-items: center; gap: 16px; box-shadow: 0 6px 25px rgba(0,0,0,0.4); }}
    .exalmar-header-title {{ font-size: 1.4rem; font-weight: 700; color: white; margin: 0; letter-spacing: -0.5px; }}
    .exalmar-header-sub {{ font-size: 0.8rem; color: rgba(255,255,255,0.7); margin: 2px 0 0 0; letter-spacing: 0.5px; text-transform: uppercase; }}
    .module-badge {{ display: inline-block; background: rgba(137, 192, 76, 0.15); border: 1px solid {C_GREEN_XLG}; color: {C_GREEN_XLG}; border-radius: 20px; padding: 3px 12px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase; }}
    .kpi-section {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.2px; color: {C_TEXT_MID}; font-weight: 600; margin-bottom: 8px; }}
    </style>
""", unsafe_allow_html=True)

LAYOUT_DARK = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=C_TEXT_LIGHT, family="Inter, Segoe UI, sans-serif", size=12),
    xaxis=dict(showgrid=True, gridcolor='#1a2d4d', zerolinecolor='#1a2d4d', linecolor=C_BORDER),
    yaxis=dict(showgrid=True, gridcolor='#1a2d4d', zerolinecolor='#1a2d4d', linecolor=C_BORDER),
    hoverlabel=dict(bgcolor=C_PANEL_MID, bordercolor=C_BORDER, font_color=C_TEXT_LIGHT)
)

def layout_dark(**overrides):
    base = {**LAYOUT_DARK, 'margin': dict(t=50, b=40, l=40, r=20)}
    base.update(overrides)
    return base

ESTILOS_TABLA = [
    {'selector': 'thead th', 'props': [('background', f'linear-gradient(135deg, {C_PANEL_MID}, {C_BORDER})'), ('color', '#ffffff'), ('font-size', '12px'), ('font-weight', '600'), ('text-transform', 'uppercase'), ('letter-spacing', '0.5px'), ('padding', '10px 14px'), ('border-bottom', f'2px solid {C_CYAN_DRK}')]},
    {'selector': 'tbody td', 'props': [('font-size', '13px'), ('text-align', 'center'), ('border-bottom', f'1px solid {C_BORDER}'), ('padding', '8px 14px'), ('color', C_TEXT_LIGHT)]},
    {'selector': 'tbody tr:hover', 'props': [('background-color', C_PANEL_MID)]},
    {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%')]}
]

def get_logo_base64(path: str = "logoexa.jpg") -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

LOGO_B64 = get_logo_base64("logoexa.jpg")

def render_header(titulo: str, subtitulo: str = "") -> None:
    logo_html = (
        f'<img src="data:image/jpeg;base64,{LOGO_B64}" style="height:52px; border-radius:6px; background:white; padding:4px 8px;" />'
        if LOGO_B64 else '<span style="font-size:2rem;">🐟</span>'
    )
    st.markdown(f"""
        <div class="exalmar-header">{logo_html}
            <div><p class="exalmar-header-title">{titulo}</p><p class="exalmar-header-sub">{subtitulo}</p></div>
        </div>
    """, unsafe_allow_html=True)

# ===================================================================
# 1. FUNCIONES COMPARTIDAS DE UTILIDAD
# ===================================================================

@st.cache_data
def bootstrapping_media(datos, n_iteraciones=5000, intervalo_confianza=95):
    datos = datos.dropna().values
    n = len(datos)
    if n < 5:
        return None, None, None, None, n
    muestras_bootstrap = np.random.choice(datos, size=(n_iteraciones, n), replace=True)
    medias_bootstrap = np.mean(muestras_bootstrap, axis=1)
    limite_inf = np.percentile(medias_bootstrap, (100 - intervalo_confianza) / 2)
    limite_sup = np.percentile(medias_bootstrap, 100 - (100 - intervalo_confianza) / 2)
    media_original = np.mean(datos)
    return medias_bootstrap, media_original, limite_inf, limite_sup, n

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')
    return output.getvalue()

def Q1(x): return x.quantile(0.25)
def Q3(x): return x.quantile(0.75)

# ===================================================================
# 2. CONEXIONES A BASES DE DATOS Y CARGA DE MODELOS
# ===================================================================

@st.cache_data(show_spinner="Extrayendo datos...", ttl=3600, max_entries=3)
def load_data_metas():
    server   = os.getenv('DB_SERVER_INDICADORES', '10.1.0.4')
    database = 'DB_Indicadores'
    username = os.getenv('DB_USER_INDICADORES')
    password = os.getenv('DB_PASS_INDICADORES')

    if not username or not password:
        st.error("Credenciales de DB_Indicadores no encontradas en el archivo .env")
        return pd.DataFrame()

    conn_str = (f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=yes;")
    query = """
    WITH FaenasBase AS (
        SELECT p.[IND_IdEmbarcacion], p.[idTemporadaRegion], p.[GalonesConsumo], p.[FechaHoraZarpe], p.[FechaHoraArriboReal],
               e.[Embarcacion], e.[CB], e.[Casco], e.[RSW], t.[temporada],
               (DATEDIFF(minute, p.[FechaHoraZarpe], p.[FechaHoraArriboReal]) * 1.0 / 60) AS DuracionFaenaHoras
        FROM [DB_Indicadores].[app].[View_PescaDescargas] p
        LEFT JOIN [dbo].[Embarcacion] e ON p.[IND_IdEmbarcacion] = e.[IdEmbarcacion]
        LEFT JOIN [dbo].[TemporadaRegion] tr ON p.[IdTemporadaRegion] = tr.[idtemporadaregion]
        LEFT JOIN [dbo].[Temporada] t ON tr.[idtemporada] = t.[idtemporada]
        WHERE p.[idTemporadaRegion] >= 57
    )

    SELECT * FROM FaenasBase
    WHERE GalonesConsumo > 0 AND DuracionFaenaHoras > 0 AND DuracionFaenaHoras <= 120
    AND  Embarcacion NOT IN ('ZHENNA 3', 'PONTEVEDRA', 'MARIA MERCEDES 5668', 'CORINTIA');
    """
    try:
        with pyodbc.connect(conn_str) as conn:
            df = pd.read_sql(query, conn)
        df['temporada']  = df['temporada'].astype('category')
        df['Embarcacion'] = df['Embarcacion'].astype(str).str.strip().str.upper().astype('category')
        
        # Parseo explícito de fechas para el filtro temporal
        df['FechaHoraZarpe'] = pd.to_datetime(df['FechaHoraZarpe'], errors='coerce')
        df['FechaHoraArriboReal'] = pd.to_datetime(df['FechaHoraArriboReal'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error de conexión (DB_Indicadores): {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Extrayendo telemetría...", ttl=3600, max_entries=3)
def load_data_telemetria():
    username = os.getenv('DB_USER_QAS')
    password = os.getenv('DB_PASS_QAS')
    server   = os.getenv('DB_SERVER_TELEMETRIA', 'exalmar-rg-qas.database.windows.net,1433')

    if not username or not password:
        st.error("Credenciales de Telemetría no encontradas en el archivo .env")
        return pd.DataFrame()

    conn_str = (f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE=BD_TEL_FLOTA_PRD;UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;")
    queries = {
        "df_mp_m":   "SELECT codigo AS id, ep, fecha_Registro AS fecha, TRY_CAST(latitud AS float) AS Latitud, TRY_CAST(longitud AS float) AS Longitud, TRY_CAST(velocidad AS float) / 10.0 AS Velocidad, rumbo AS Rumbo, TRY_CAST(gal_h AS float) / 10.0 AS gal_h, TRY_CAST(rpm AS float) AS rpm FROM dbo.VW_MP_M_ORDENADO WHERE rpm <> '0' AND gal_h <> '0';",
        "df_mp_e":   "SELECT [codigo] AS id, [ep], [fecha_registro] AS fecha, TRY_CAST([latitud] AS float) AS Latitud, TRY_CAST([longitud] AS float) AS Longitud, TRY_CAST([velocidad] AS float) AS Velocidad, [rumbo] AS Rumbo, TRY_CAST([consumo_gal_h] AS float) AS gal_h, TRY_CAST([rpm] AS float) AS rpm FROM [dbo].[VW_MP_E_ORDENADO] WHERE [rpm] <> '0' AND [consumo_gal_h] <> '0';",
        "df_mp_e_h": "SET NOCOUNT ON; SELECT m.id, e.Embarcacion AS ep, m.fecha, TRY_CAST(m.Latitude AS float) AS Latitud, TRY_CAST(m.Longitude AS float) AS Longitud, TRY_CAST(m.Speed AS float) / 10.0 AS Velocidad, m.header AS Rumbo, TRY_CAST(m.parameter12 AS float) / 3.785 AS gal_h, TRY_CAST(m.parameter02 AS float) AS rpm FROM [dbo].[MP_E_H] m INNER JOIN dbo.Embarcacion e ON m.IdEmbarcacion = e.IdEmbarcacion WHERE m.parameter02 <> '0' AND m.parameter12 <> '0';",
        "df_mp_m_h": "SET NOCOUNT ON; SELECT m.id, e.Embarcacion AS ep, m.fecha, TRY_CAST(m.Latitude AS float) AS Latitud, TRY_CAST(m.Longitude AS float) AS Longitud, TRY_CAST(m.Speed AS float) / 10.0 AS Velocidad, m.heading AS Rumbo, TRY_CAST(m.GAL_H AS float) / 10.0 AS gal_h, TRY_CAST(m.RPM AS float) AS rpm FROM [dbo].[MP_M_H] m INNER JOIN dbo.Embarcacion e ON m.IdEmbarcacion = e.IdEmbarcacion WHERE m.RPM <> '0' AND m.GAL_H <> '0';"
    }
    try:
        df_list = []
        with pyodbc.connect(conn_str) as conn:
            for query in queries.values():
                df_list.append(pd.read_sql(query, conn))
        df_total = pd.concat(df_list, ignore_index=True)
        del df_list

        # Asegurar el formato datetime de la fecha
        df_total['fecha']       = pd.to_datetime(df_total['fecha'], errors='coerce')
        
        df_total['Velocidad']   = pd.to_numeric(df_total['Velocidad'], errors='coerce')
        df_total['rpm']         = pd.to_numeric(df_total['rpm'], errors='coerce')
        df_total['gal_h']       = pd.to_numeric(df_total['gal_h'], errors='coerce')
        df_total.rename(columns={'rpm': 'RPM', 'gal_h': 'Galon/hora'}, inplace=True)
        df_total.dropna(subset=['ep', 'fecha', 'RPM', 'Velocidad', 'Galon/hora'], inplace=True)

        df_total['ep'] = df_total['ep'].astype(str).str.strip().str.upper().astype('category')
        for col in ['Velocidad', 'RPM', 'Galon/hora', 'Latitud', 'Longitud', 'Rumbo']:
            if col in df_total.columns:
                df_total[col] = pd.to_numeric(df_total[col], downcast='float')
        
        df_total = df_total[df_total['RPM'] < 3000]
        return df_total
    except Exception as e:
        st.error(f"Error conectando a BD Telemetría: {e}")
        return pd.DataFrame()


@st.cache_resource(show_spinner="Cargando modelos...")
def cargar_modelos_maestros():
    ruta = 'modelos_navales/modelos_flota.joblib'
    if os.path.exists(ruta):
        return joblib.load(ruta)
    return {}


def aplicar_inferencia(df_ep_data, ep, dicc_modelos, _features, nombres_modos):
    X = df_ep_data.copy()
    if ep not in dicc_modelos:
        return X, pd.DataFrame(), pd.DataFrame(), None

    paquete_ep = dicc_modelos[ep]
    scaler = paquete_ep['scaler']
    kmeans = paquete_ep['kmeans']

    X_scaled = scaler.transform(X[_features])
    X['Cluster_Raw'] = kmeans.predict(X_scaled)

    medians  = X.groupby('Cluster_Raw')['RPM'].median().sort_values()
    mapeo_ids = {c_raw: n_id for n_id, c_raw in enumerate(medians.index)}
    X['Cluster_ID']    = X['Cluster_Raw'].map(mapeo_ids)
    X['Modo Operativo'] = X['Cluster_ID'].apply(lambda x: nombres_modos[x] if x in range(len(nombres_modos)) else "Desconocido")

    params_glm = paquete_ep['glm']
    beta0, beta1, escala = params_glm.get('beta0'), params_glm.get('beta1'), params_glm.get('escala')

    df_clean, df_anomalias = pd.DataFrame(), pd.DataFrame()
    if beta0 is not None:
        df_nav   = X[X['Cluster_ID'].isin([1, 2])].copy()
        mu_all   = np.exp(beta0 + beta1 * df_nav['RPM'])
        df_nav['Residuo_Pearson'] = ((df_nav['Galon/hora'] - mu_all) / mu_all) / np.sqrt(escala)

        mask         = np.abs(df_nav['Residuo_Pearson']) <= 2.0
        df_clean     = df_nav[mask].copy()
        df_anomalias = df_nav[~mask].copy()

    return X, df_clean, df_anomalias, (beta0, beta1)

# ===================================================================
# 3. BARRA LATERAL CORPORATIVA
# ===================================================================
with st.sidebar:
    if LOGO_B64:
        st.markdown(f'<div style="text-align:center; padding: 16px 0 8px 0;"><img src="data:image/jpeg;base64,{LOGO_B64}" style="max-width:160px; border-radius:8px; background:white; padding:8px 12px;" /></div>', unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align:center;'>🐟 Exalmar</h3>", unsafe_allow_html=True)

    st.markdown(f'<div style="height:3px; background: linear-gradient(90deg, {C_CYAN_DRK}, {C_GREEN_XLG}); border-radius:2px; margin: 8px 0 16px 0;"></div>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:0.7rem; text-transform:uppercase; letter-spacing:1.2px; color:{C_TEXT_MID}; font-weight:600; margin-bottom:6px;">Módulo de Análisis</p>', unsafe_allow_html=True)

    modulo_seleccionado = st.sidebar.radio("", ("Metas de Combustible", "Telemetría Avanzada"), label_visibility="collapsed")
    st.sidebar.markdown(f'<div style="height:1px; background:{C_BORDER}; margin: 16px 0;"></div>', unsafe_allow_html=True)

    icono = "⛽" if modulo_seleccionado == "Metas de Combustible" else "📡"
    st.markdown(f'<div style="background:{C_PANEL_MID}; border:1px solid {C_BORDER}; border-left: 3px solid {C_GREEN_XLG}; border-radius:8px; padding:10px 14px; font-size:0.82rem; color:{C_TEXT_LIGHT};">{icono} <b>{modulo_seleccionado}</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="position:fixed; bottom:20px; font-size:0.68rem; color:{C_TEXT_MID}; text-align:center; letter-spacing:0.4px;">Analytics Fleet v 1.0 · Pesquera Exalmar S.A.A.<br><span style="color:{C_CYAN_DRK};">●</span> Sistema Activo</div>', unsafe_allow_html=True)

# ===================================================================
# MÓDULO 1: METAS DE COMBUSTIBLE (BOOTSTRAPPING & SEABORN PASTEL)
# ===================================================================
if modulo_seleccionado == "Metas de Combustible":

    render_header(titulo="Eficiencia de Combustible · Análisis de Faenas", subtitulo="Bootstrapping estadístico · Detección de anomalías · Evolución por temporada")

    df_master = load_data_metas()
    if df_master.empty:
        st.stop()

    tab1_m, tab2_m, tab3_m = st.tabs(["📊  Análisis Bootstrap", "📈  Evolución por Embarcación", "🗂  Resumen de Flota"])

    with tab1_m:
        with st.container():
            st.markdown(f'<div style="background:{C_PANEL_MID}; border:1px solid {C_BORDER}; border-radius:10px; padding:16px 20px; margin-bottom:16px;">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                opciones_emb1 = ['Todas'] + sorted(df_master['Embarcacion'].cat.categories.tolist())
                emb_sel1 = st.selectbox("🚢  Embarcación", opciones_emb1, key="emb_tab1")
            with col2:
                opciones_temp1 = ['Todas'] + sorted(df_master['temporada'].cat.categories.tolist())
                temp_sel1 = st.selectbox("📅  Temporada", opciones_temp1, key="temp_tab1")
            st.markdown('</div>', unsafe_allow_html=True)

        temp_df1 = df_master.copy()
        temp_df1 = temp_df1[(temp_df1['GalonesConsumo'] < 5000)]
        temp_df1['gal_hr'] = temp_df1['GalonesConsumo'] / temp_df1['DuracionFaenaHoras']

        if emb_sel1 != 'Todas': temp_df1 = temp_df1[temp_df1['Embarcacion'] == emb_sel1]
        if temp_sel1 != 'Todas': temp_df1 = temp_df1[temp_df1['temporada'] == temp_sel1]

        if len(temp_df1) < 5:
            st.warning("⚠️  Datos insuficientes para evaluar outliers e iniciar el Bootstrapping.")
        else:
            filas_correctas, outliers_detectados = [], []

            for barco, grupo in temp_df1.groupby('Embarcacion', observed=True):
                if len(grupo) >= 5:
                    p15 = grupo['gal_hr'].quantile(0.15)
                    p90 = grupo['gal_hr'].quantile(0.90)
                    iqr_modificado = p90 - p15
                    limite_inf = p15 - 1.5 * iqr_modificado
                    limite_sup = p90 + 1.5 * iqr_modificado

                    correctos = grupo[(grupo['gal_hr'] >= limite_inf) & (grupo['gal_hr'] <= limite_sup)]
                    errores   = grupo[(grupo['gal_hr'] < limite_inf)  | (grupo['gal_hr'] > limite_sup)]

                    filas_correctas.append(correctos)
                    if not errores.empty: outliers_detectados.append(errores)
                else:
                    filas_correctas.append(grupo)

            df_datos_correctos = pd.concat(filas_correctas, ignore_index=True) if filas_correctas else pd.DataFrame()

            with st.expander("🔍  Datos Anómalos Retirados (cuantiles 15–90)"):
                if outliers_detectados:
                    df_errores_total = pd.concat(outliers_detectados, ignore_index=True)
                    st.dataframe(df_errores_total[['Embarcacion', 'temporada', 'GalonesConsumo', 'DuracionFaenaHoras', 'gal_hr']], use_container_width=True)
                else:
                    st.success("✅  No se detectaron consumos anormales según IQR modificado.")

            datos_gal = df_datos_correctos['gal_hr']

            if len(datos_gal) >= 5:
                medias_boot, media_orig, ci_inf, ci_sup, n_obs = bootstrapping_media(datos_gal)
                suma_gal   = df_datos_correctos['GalonesConsumo'].sum()
                suma_hr    = df_datos_correctos['DuracionFaenaHoras'].sum()
                ratio_global = suma_gal / suma_hr if suma_hr > 0 else 0

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Obs. Limpias", f"{int(n_obs):,}")
                k2.metric("Media Bootstrap", f"{media_orig:.2f} gal/h")
                k3.metric("IC Inf 95%", f"{ci_inf:.2f}")
                k4.metric("IC Sup 95%", f"{ci_sup:.2f}")
                st.markdown("<br>", unsafe_allow_html=True)

                fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig1.patch.set_alpha(0.0)
                
                sns.histplot(datos_gal, bins=30, ax=axes[0], color=P_TEAL, alpha=0.8, edgecolor=C_BORDER)
                axes[0].axvline(media_orig, color=P_RED, linestyle='--', label=f'μ = {media_orig:.2f}', linewidth=2)
                axes[0].set_title(f'Distribución Original gal/h (N = {n_obs:,})', color='white', pad=15)
                axes[0].set_xlabel("Galones / Hora")
                axes[0].set_ylabel("Frecuencia")
                axes[0].legend()

                sns.histplot(medias_boot, bins=30, stat="density", ax=axes[1], color=P_BLUE, alpha=0.4, kde=True, line_kws={'color': P_GREEN, 'lw': 2.5}, edgecolor=C_BORDER)
                axes[1].axvline(media_orig, color=P_RED, linestyle='--', label=f'μ = {media_orig:.2f}', linewidth=2)
                axes[1].axvline(ci_inf, color=P_ORANGE, linestyle=':', label='IC Inf', linewidth=2)
                axes[1].axvline(ci_sup, color=P_ORANGE, linestyle=':', label='IC Sup', linewidth=2)
                axes[1].set_title('Distribución de Medias Bootstrap (KDE)', color='white', pad=15)
                axes[1].set_xlabel("Media de Galones / Hora")
                axes[1].set_ylabel("Densidad")
                axes[1].legend()

                plt.tight_layout()
                st.pyplot(fig1)

                df_resumen1 = pd.DataFrame({
                    'Métrica': ['N° Observaciones Limpias', 'Media Estimada (Bootstrap)', 'Límite Inferior (IC 95%)', 'Límite Superior (IC 95%)', 'Ratio Global Puro'],
                    'Valor':   [int(n_obs), round(media_orig, 4), round(ci_inf, 4), round(ci_sup, 4), round(ratio_global, 4)]
                })
                st.dataframe(df_resumen1.style.set_table_styles(ESTILOS_TABLA), hide_index=True, use_container_width=True)

    with tab2_m:
        st.markdown(f'<p style="font-size:1.05rem; font-weight:600; color:white; margin-bottom:12px;">📈  Análisis Evolutivo por Temporada</p>', unsafe_allow_html=True)

        df_limpio2 = df_master[(df_master['GalonesConsumo'] < 5000)].copy()
        df_limpio2['gal_hr'] = df_limpio2['GalonesConsumo'] / df_limpio2['DuracionFaenaHoras']

        filas_limpias2 = []
        for barco, grupo in df_limpio2.groupby('Embarcacion', observed=True):
            if len(grupo) >= 5:
                p15, p90 = grupo['gal_hr'].quantile(0.15), grupo['gal_hr'].quantile(0.90)
                iqr       = p90 - p15
                grupo_filtrado = grupo[(grupo['gal_hr'] >= (p15 - 1.5 * iqr)) & (grupo['gal_hr'] <= (p90 + 1.5 * iqr))]
                filas_limpias2.append(grupo_filtrado)
            else:
                filas_limpias2.append(grupo)

        df_base2 = pd.concat(filas_limpias2, ignore_index=True) if filas_limpias2 else df_limpio2
        opciones_emb2 = ['Todas'] + sorted(df_base2['Embarcacion'].dropna().unique().tolist())
        emb_sel2 = st.selectbox("🚢  Embarcación", opciones_emb2, key="emb_tab2")

        temp_df2  = df_base2[df_base2['Embarcacion'] == emb_sel2] if emb_sel2 != 'Todas' else df_base2.copy()
        temporadas = sorted(temp_df2['temporada'].dropna().unique().tolist())

        if temporadas:
            lista_resultados2, lista_bootstraps, x_labels = [], [], []
            for temp in temporadas:
                df_temp    = temp_df2[temp_df2['temporada'] == temp]
                datos_gal  = df_temp['gal_hr'].dropna()

                if len(datos_gal) >= 5:
                    medias_boot, media_orig, ci_inf, ci_sup, n_obs = bootstrapping_media(datos_gal)
                    ratio_global = (df_temp['GalonesConsumo'].sum() / df_temp['DuracionFaenaHoras'].sum() if df_temp['DuracionFaenaHoras'].sum() > 0 else np.nan)
                    lista_resultados2.append({'TEMPORADA': temp, 'N° OBS': n_obs, 'MEDIA ESTIMADA': media_orig, 'LI IC 95%': ci_inf, 'LS IC 95%': ci_sup, 'RATIO GLOBAL': ratio_global})
                    lista_bootstraps.append(medias_boot)
                    x_labels.append(str(temp))

            if lista_resultados2:
                means     = [r['MEDIA ESTIMADA'] for r in lista_resultados2]
                error_inf = [m - r['LI IC 95%'] for m, r in zip(means, lista_resultados2)]
                error_sup = [r['LS IC 95%'] - m  for m, r in zip(means, lista_resultados2)]

                fig2, ax2 = plt.subplots(figsize=(12, 5.5))
                fig2.patch.set_alpha(0.0)
                
                violin_data = pd.DataFrame({
                    'Temporada': np.repeat(x_labels, [len(b) for b in lista_bootstraps]),
                    'Media Bootstrap': np.concatenate(lista_bootstraps)
                })
                
                sns.violinplot(data=violin_data, x='Temporada', y='Media Bootstrap', ax=ax2, color=P_TEAL, inner="box", linewidth=1, linecolor=C_BORDER)
                plt.setp(ax2.collections, alpha=0.4)

                ax2.errorbar(x=range(len(x_labels)), y=means, yerr=[error_inf, error_sup], fmt='-o', color=P_ORANGE, ecolor=P_RED, capsize=5, capthick=2, markersize=8, linewidth=2, label='Tendencia (Media y 95% IC)')
                
                for i, m in enumerate(means):
                    ax2.text(i, m + (error_sup[i]*1.15), f"{m:.1f}", color=P_ORANGE, ha='center', va='bottom', fontweight='bold', fontsize=10)

                ax2.set_title("Distribución Bootstrap y Evolución por Temporada", color='white', pad=15)
                ax2.set_ylabel("gal/h")
                ax2.set_xlabel("Temporada")
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig2)

                df_resumen2 = pd.DataFrame(lista_resultados2)
                st.dataframe(df_resumen2.style.format(precision=2).set_table_styles(ESTILOS_TABLA), hide_index=True, use_container_width=True)

    with tab3_m:
        st.markdown(f'<p style="font-size:1.05rem; font-weight:600; color:white; margin-bottom:12px;">🗂  Resumen General de Flota</p>', unsafe_allow_html=True)
        df_base3 = df_master.copy()
        df_base3['gal_hr'] = df_base3['GalonesConsumo'] / df_base3['DuracionFaenaHoras']
        opciones_temp3 = ['Todas'] + sorted(df_base3['temporada'].cat.categories.tolist())
        temp_sel3 = st.selectbox("📅  Temporada", opciones_temp3, key="temp_tab3")
        temp_df3 = df_base3[df_base3['temporada'] == temp_sel3] if temp_sel3 != 'Todas' else df_base3.copy()

        if not temp_df3.empty:
            lista_resultados3 = []
            for emb in sorted(temp_df3['Embarcacion'].dropna().unique().tolist()):
                df_emb = temp_df3[temp_df3['Embarcacion'] == emb]
                if len(df_emb) >= 5:
                    _, media_est, _, _, n_obs = bootstrapping_media(df_emb['gal_hr'].dropna())
                    lista_resultados3.append({'EMBARCACION': emb, 'N° OBS': n_obs, 'MEDIA ESTIMADA': media_est, 'RATIO GLOBAL': df_emb['GalonesConsumo'].sum() / df_emb['DuracionFaenaHoras'].sum()})

            if lista_resultados3:
                df_general_resumen = pd.DataFrame(lista_resultados3)
                st.dataframe(df_general_resumen.style.format(precision=2).set_table_styles(ESTILOS_TABLA), hide_index=True, use_container_width=True)
                st.download_button("⬇️  Descargar Excel", data=to_excel(df_general_resumen), file_name=f"Resumen_General_{temp_sel3}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ===================================================================
# MÓDULO 2: TELEMETRÍA AVANZADA (INFERENCIA PURA CON PLOTLY ORIGINAL)
# ===================================================================
elif modulo_seleccionado == "Telemetría Avanzada":

    # Colores corporativos originales revertidos
    COLOR_MODOS  = [C_CYAN_DRK, C_GREEN_XLG, '#3b82f6']
    NOMBRES_MODOS = ["Pesca / Espera (RPM Bajo)", "Transición (RPM Medio)", "Crucero (RPM Alto)"]

    df_flota = load_data_telemetria()
    diccionario_flota_modelos = cargar_modelos_maestros()

    if df_flota.empty:
        st.stop()

    lista_ep = ["— Seleccionar Embarcación —"] + sorted(df_flota['ep'].cat.categories.tolist())
    nombre_seleccionado = st.sidebar.selectbox("🚢  Flota Disponible", lista_ep)

    if nombre_seleccionado == "— Seleccionar Embarcación —":
        render_header(titulo="Telemetría Energética Naval", subtitulo="Seleccione una embarcación en el panel lateral para iniciar el análisis")
        st.markdown(f"""<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; margin-top:60px; gap:16px;">
                <div style="font-size:4.5rem;">⚓</div>
                <p style="color:{C_TEXT_MID}; font-size:1rem; text-align:center; max-width:420px;">
                    Seleccione una embarcación en el panel lateral para visualizar los indicadores operativos y el análisis energético en tiempo real.</p>
                <span class="module-badge">Telemetría · GLM Gamma · K-Means</span></div>""", unsafe_allow_html=True)
        st.stop()

    df_ep = (df_flota[df_flota['ep'] == nombre_seleccionado].copy().sort_values('fecha').reset_index(drop=True))
    df_ep = df_ep[df_ep['Galon/hora'] < 1000].reset_index(drop=True)
    features = ['RPM', 'Velocidad', 'Galon/hora']

    X, df_clean, df_anomalias, betas = aplicar_inferencia(df_ep_data=df_ep, ep=nombre_seleccionado, dicc_modelos=diccionario_flota_modelos, _features=features, nombres_modos=NOMBRES_MODOS)

    if betas is None:
        st.warning(f"⚠️  No se encontró un modelo analítico pre-entrenado para **{nombre_seleccionado}**. Ejecute el script de entrenamiento batch para generar el archivo `.joblib`.")
        st.stop()

    render_header(titulo=f"E/P  {nombre_seleccionado}", subtitulo="Dashboard Operativo · Análisis de Telemetría")

    st.markdown('<p class="kpi-section">Indicadores Clave de Operación</p>', unsafe_allow_html=True)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Registros Analizados", f"{len(X):,}")
    kpi2.metric("Mediana Velocidad",           f"{X['Velocidad'].median():.1f} Nudos")
    kpi3.metric("Mediana Consumo",             f"{X['Galon/hora'].median():.1f} Gal/H")
    kpi4.metric("RPM Máximo (Real)",           f"{X['RPM'].max():.0f}")

    st.markdown(f'<div style="height:2px; background: linear-gradient(90deg, {C_CYAN_DRK}, transparent); margin: 16px 0;"></div>', unsafe_allow_html=True)

    tab1_t, tab2_t, tab3_t, tab4_t, tab5_t = st.tabs(["🔵  Segmentación K-Means", "📐  Curva GLM Gamma", "📉  Análisis Marginal RPM", "🌐  Distribución 3D KDE", "⏱️  Telemetria"])

    with tab1_t:
        col1, col2 = st.columns([3, 2])
        with col1:
            color_map_dict = {NOMBRES_MODOS[0]: COLOR_MODOS[0], NOMBRES_MODOS[1]: COLOR_MODOS[1], NOMBRES_MODOS[2]: COLOR_MODOS[2]}
            df_visual = X.sample(n=min(6000, len(X)), random_state=42) if len(X) > 6000 else X

            fig1 = px.scatter_3d(df_visual, x='RPM', y='Velocidad', z='Galon/hora', color='Modo Operativo', color_discrete_map=color_map_dict, opacity=0.60)
            fig1.update_traces(marker=dict(size=2.5))
            fig1.update_layout(**layout_dark(
                margin=dict(l=0, r=0, b=0, t=20),
                scene=dict(xaxis=dict(title='RPM', backgroundcolor=C_PANEL_DARK, gridcolor=C_BORDER), yaxis=dict(title='Velocidad', backgroundcolor=C_PANEL_DARK, gridcolor=C_BORDER), zaxis=dict(title='Gal/H', backgroundcolor=C_PANEL_DARK, gridcolor=C_BORDER), bgcolor=C_BG_DARK),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(15,30,56,0.8)', bordercolor=C_BORDER, borderwidth=1, font=dict(size=11))
            ))
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown(f'<p style="font-weight:600; color:white; margin-bottom:8px;">Perfil de Modos Operativos</p>', unsafe_allow_html=True)
            perfiles = []
            for cid in range(3):
                subset_cluster = X[X['Cluster_ID'] == cid]
                if not subset_cluster.empty:
                    p = subset_cluster[features].describe().T
                    p['median']  = subset_cluster[features].median()
                    p['count']   = p['count'].astype(int)
                    p['% Total'] = (p['count'] / len(X)) * 100
                    p['Modo']    = NOMBRES_MODOS[cid]
                    perfiles.append(p[['count', '% Total', 'median', 'std', 'min', 'max', 'Modo']])

            if perfiles:
                df_p = pd.concat(perfiles).reset_index().rename(columns={'index': 'Var'})
                df_p = df_p[['Modo', 'Var', 'count', '% Total', 'median', 'std']]
                st.dataframe(df_p.style.format({'count': '{:,.0f}', '% Total': '{:.1f}%', 'median': '{:.1f}', 'std': '{:.2f}'}).set_table_styles(ESTILOS_TABLA), use_container_width=True, hide_index=True)

    with tab2_t:
        beta0, beta1 = betas
        if not df_clean.empty:
            df_nav = pd.concat([df_clean, df_anomalias])

            col_eq1, col_eq2 = st.columns(2)
            with col_eq1:
                st.success("Ecuación de la Curva Exponencial Aplicada")
                st.latex(rf"E[\text{{Gal/H}}] = e^{{{beta0:.4f} + {beta1:.5f} \cdot \text{{RPM}}}}")
            with col_eq2:
                st.info(f"**Filtro de Inferencia** — {len(df_clean):,} registros válidos · {len(df_anomalias):,} anomalías excluidas")

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                df_res_vis = (df_nav.sample(n=min(5000, len(df_nav)), random_state=42) if len(df_nav) > 5000 else df_nav)
                mu_all_vis = np.exp(beta0 + beta1 * df_res_vis['RPM'])

                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(
                    x=mu_all_vis, y=df_res_vis['Residuo_Pearson'], mode='markers',
                    marker=dict(
                        color=df_res_vis['Residuo_Pearson'].abs(),
                        colorscale=[[0, C_CYAN_DRK], [0.5, C_WARN_ORANGE], [1, C_DANGER_RED]],
                        size=4, opacity=0.65, showscale=False
                    ), name="Residuo"
                ))
                fig_res.add_hline(y= 2, line_dash="dash", line_color=C_DANGER_RED, line_width=2)
                fig_res.add_hline(y=-2, line_dash="dash", line_color=C_DANGER_RED, line_width=2)
                fig_res.add_hrect(y0=-2, y1=2, fillcolor=f'rgba(3,143,141,0.07)', layer="below", line_width=0)
                fig_res.update_layout(**layout_dark(title=dict(text='Distribución de Residuos Pearson', font=dict(size=13, color=C_TEXT_MID)), xaxis_title='Consumo Esperado (μ)', yaxis_title='Residuo Pearson'))
                st.plotly_chart(fig_res, use_container_width=True)

            with col_g2:
                df_anom_vis  = (df_anomalias.sample(n=min(1500, len(df_anomalias)), random_state=42) if len(df_anomalias) > 1500 else df_anomalias)
                df_clean_vis = (df_clean.sample(n=min(4500, len(df_clean)), random_state=42) if len(df_clean) > 4500 else df_clean)

                fig_reg = go.Figure()
                fig_reg.add_trace(go.Scatter(x=df_anom_vis['RPM'], y=df_anom_vis['Galon/hora'], mode='markers', marker=dict(color=C_DANGER_RED, size=3.5, opacity=0.50, symbol='x'), name="Anomalías"))
                fig_reg.add_trace(go.Scatter(x=df_clean_vis['RPM'], y=df_clean_vis['Galon/hora'], mode='markers', marker=dict(color='#3b82f6', size=3.5, opacity=0.55), name="Datos Válidos"))
                X_line = np.linspace(df_nav['RPM'].min(), df_nav['RPM'].max(), 100)
                fig_reg.add_trace(go.Scatter(x=X_line, y=np.exp(beta0 + beta1 * X_line), mode='lines', line=dict(color=C_GREEN_XLG, width=4), name="Curva GLM Gamma"))
                fig_reg.update_layout(**layout_dark(title=dict(text='Consumo vs RPM · Curva Ajustada', font=dict(size=13, color=C_TEXT_MID)), xaxis_title='RPM', yaxis_title='Gal/H', legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0)))
                st.plotly_chart(fig_reg, use_container_width=True)
        else:
            st.warning("No hay suficientes datos de navegación registrados para graficar la curva.")

    with tab3_t:
        if 'df_clean' in locals() and not df_clean.empty:
            subtab1, subtab2 = st.tabs(["🔭  Visión Macro (bloques 50 RPM)", "🔬  Zoom Detallado (bloques 25 RPM)"])

            def generar_graficos_margen(df_eval, salto_rpm):
                df_group = df_eval.copy()
                df_group['Rango_RPM'] = (df_group['RPM'] // salto_rpm) * salto_rpm
                res = df_group.groupby('Rango_RPM').agg(Vel_Mediana=('Velocidad', 'median'), Vel_Q1=('Velocidad', Q1), Vel_Q3=('Velocidad', Q3), Cons_Mediana=('Galon/hora', 'median'), Cons_Q1=('Galon/hora', Q1), Cons_Q3=('Galon/hora', Q3)).reset_index().sort_values('Rango_RPM')
                res['Dif_Vel']  = res['Vel_Mediana'].diff().round(1)
                res['Dif_Cons'] = res['Cons_Mediana'].diff().round(1)
                res = res.fillna(0)

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=("Progresión de Velocidad", "Impacto en Consumo de Combustible"), specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
                x_band = res['Rango_RPM'].tolist() + res['Rango_RPM'].tolist()[::-1]

                y_band_vel = res['Vel_Q3'].tolist() + res['Vel_Q1'].tolist()[::-1]
                fig.add_trace(go.Scatter(x=x_band, y=y_band_vel, fill='toself', fillcolor='rgba(59,130,246,0.15)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="Dispersión Q1-Q3 Vel."), row=1, col=1, secondary_y=False)
                fig.add_trace(go.Scatter(x=res['Rango_RPM'], y=res['Vel_Mediana'], mode='lines+markers+text', text=[f"{v:.1f} ({dv:+.1f})" for v, dv in zip(res['Vel_Mediana'], res['Dif_Vel'])], textposition="top center", textfont=dict(size=10), marker=dict(size=9, color='#3b82f6', line=dict(color='white', width=1.5)), line=dict(width=3, color='#3b82f6'), name="Velocidad Mediana"), row=1, col=1, secondary_y=False)
                fig.add_trace(go.Scatter(x=res['Rango_RPM'], y=res['Dif_Vel'], mode='lines+markers', marker=dict(symbol='diamond', size=6, color=C_WARN_ORANGE), line=dict(width=2, dash='dot'), name="Δ Velocidad"), row=1, col=1, secondary_y=True)

                y_band_cons = res['Cons_Q3'].tolist() + res['Cons_Q1'].tolist()[::-1]
                fig.add_trace(go.Scatter(x=x_band, y=y_band_cons, fill='toself', fillcolor='rgba(3,143,141,0.15)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name="Dispersión Q1-Q3 Cons."), row=2, col=1, secondary_y=False)
                fig.add_trace(go.Scatter(x=res['Rango_RPM'], y=res['Cons_Mediana'], mode='lines+markers+text', text=[f"{c:.0f} ({dc:+.1f})" for c, dc in zip(res['Cons_Mediana'], res['Dif_Cons'])], textposition="top center", textfont=dict(size=10), marker=dict(size=9, color=C_CYAN_DRK, line=dict(color='white', width=1.5)), line=dict(width=3, color=C_CYAN_DRK), name="Consumo Mediano"), row=2, col=1, secondary_y=False)
                fig.add_trace(go.Scatter(x=res['Rango_RPM'], y=res['Dif_Cons'], mode='lines+markers', marker=dict(symbol='diamond', size=6, color=C_WARN_ORANGE), line=dict(width=2, dash='dot'), name="Δ Consumo"), row=2, col=1, secondary_y=True)

                fig.update_layout(**layout_dark(height=700, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)')))
                fig.update_xaxes(title_text="Rango RPM", row=2, col=1, tickvals=res['Rango_RPM'], gridcolor='#1a2d4d')
                fig.update_yaxes(title_text="Nudos (Mediana)", row=1, col=1, secondary_y=False, gridcolor='#1a2d4d')
                fig.update_yaxes(title_text="Δ Neto", row=1, col=1, secondary_y=True, showgrid=False, color=C_WARN_ORANGE)
                fig.update_yaxes(title_text="Gal/H (Mediana)", row=2, col=1, secondary_y=False, gridcolor='#1a2d4d')
                fig.update_yaxes(title_text="Δ Neto", row=2, col=1, secondary_y=True, showgrid=False, color=C_WARN_ORANGE)
                for ann in fig.layout.annotations: ann.font.color = C_TEXT_MID; ann.font.size = 12

                res_tabla = res[['Rango_RPM', 'Vel_Mediana', 'Dif_Vel', 'Cons_Mediana', 'Dif_Cons']].rename(columns={'Rango_RPM': 'RPM Base', 'Vel_Mediana': 'Velocidad Mediana', 'Cons_Mediana': 'Consumo Mediano', 'Dif_Vel': 'Dif. Velocidad', 'Dif_Cons': 'Dif. Consumo'})
                return fig, res_tabla

            with subtab1:
                fig_macro, df_macro = generar_graficos_margen(df_clean, 50)
                st.plotly_chart(fig_macro, use_container_width=True)
                st.dataframe(df_macro.style.format({'RPM Base': '{:.0f}', 'Velocidad Mediana': '{:.2f}', 'Consumo Mediano': '{:.1f}', 'Dif. Velocidad': '{:+.1f}', 'Dif. Consumo': '{:+.1f}'}).set_table_styles(ESTILOS_TABLA), use_container_width=True, hide_index=True)

            with subtab2:
                df_zoom = df_clean[df_clean['RPM'] >= int(df_clean['RPM'].max() * 0.4)].copy()
                fig_micro, df_micro = generar_graficos_margen(df_zoom, 25)
                st.plotly_chart(fig_micro, use_container_width=True)
                def alertar_ineficiencia(val): return (f'background-color: rgba(239,68,68,0.18); color: {C_DANGER_RED}; font-weight:bold;' if val >= 5.0 else '')
                st.dataframe(df_micro.style.format({'RPM Base': '{:.0f}', 'Velocidad Mediana': '{:.2f}', 'Consumo Mediano': '{:.1f}', 'Dif. Velocidad': '{:+.1f}', 'Dif. Consumo': '{:+.1f}'}).map(alertar_ineficiencia, subset=['Dif. Consumo']).set_table_styles(ESTILOS_TABLA), use_container_width=True, hide_index=True)

    with tab4_t:
        if 'df_clean' in locals() and not df_clean.empty:
            st.info("📊  Distribución de concentración de la velocidad bajo condiciones de motor exigido (>40% RPM).")
            df_z = df_clean[df_clean['RPM'] >= int(df_clean['RPM'].max() * 0.4)].copy()
            df_z['Rango_RPM'] = (df_z['RPM'] // 25) * 25
            fig_kde = go.Figure()
            y_grid  = np.linspace(df_z['Velocidad'].min(), df_z['Velocidad'].max(), 100)
            rangos = sorted(df_z['Rango_RPM'].unique())
            n_rangos = len(rangos)
            
            for i, r in enumerate(rangos):
                v_data = df_z[df_z['Rango_RPM'] == r]['Velocidad']
                dens   = gaussian_kde(v_data)(y_grid) if len(v_data) > 2 else np.zeros_like(y_grid)
                t      = i / max(n_rangos - 1, 1)
                r_c = int(3   + (137 - 3)   * t)
                g_c = int(143 + (192 - 143) * t)
                b_c = int(141 + (76  - 141) * t)
                color_grad = f'rgb({r_c},{g_c},{b_c})'
                fig_kde.add_trace(go.Scatter3d(x=[r] * 100, y=y_grid, z=dens, mode='lines', line=dict(color=color_grad, width=5)))

            fig_kde.update_layout(**layout_dark(height=600, scene=dict(xaxis_title='RPM', yaxis_title='Nudos', zaxis_title='Concentración', bgcolor=C_BG_DARK, xaxis=dict(backgroundcolor=C_PANEL_DARK, gridcolor=C_BORDER), yaxis=dict(backgroundcolor=C_PANEL_DARK, gridcolor=C_BORDER), zaxis=dict(backgroundcolor=C_PANEL_DARK, gridcolor=C_BORDER)), showlegend=False, title=dict(text=f"KDE 3D · Distribución de Velocidad por Rango RPM · {nombre_seleccionado}", font=dict(size=13, color=C_TEXT_MID))))
            st.plotly_chart(fig_kde, use_container_width=True)
            
    with tab5_t:
        st.markdown(f'<p style="font-size:1.05rem; font-weight:600; color:white; margin-bottom:12px;">⏱️  Evolución Temporal de Variables Operativas</p>', unsafe_allow_html=True)

        # 1. Obtener rango de la última temporada para configurar el filtro por defecto
        df_master_t = load_data_metas()
        min_fecha_ep = df_ep['fecha'].min().date()
        max_fecha_ep = df_ep['fecha'].max().date()

        default_start = min_fecha_ep
        default_end = max_fecha_ep

        if not df_master_t.empty and 'idTemporadaRegion' in df_master_t.columns:
            max_id_temp = df_master_t['idTemporadaRegion'].max()
            df_last_season = df_master_t[df_master_t['idTemporadaRegion'] == max_id_temp]
            if not df_last_season.empty:
                s_date = df_last_season['FechaHoraZarpe'].min().date()
                e_date = df_last_season['FechaHoraArriboReal'].max().date()
                
                if pd.notnull(s_date) and pd.notnull(e_date):
                    # Validamos que las fechas extraídas tengan cruce con la telemetría disponible
                    default_start = max(min_fecha_ep, s_date)
                    default_end = min(max_fecha_ep, e_date)

        # 2. Selector de Rango de Fechas Interactivo
        col_d1, col_d2 = st.columns([1, 2])
        with col_d1:
            rango_fechas = st.date_input(
                "📅 Seleccione Rango de Fechas (Auto-ajustado a la Última Temporada)",
                value=(default_start, default_end),
                min_value=min_fecha_ep,
                max_value=max_fecha_ep,
                key="date_range_telemetria"
            )

        # 3. Filtrar DataFrame basado en la selección del usuario
        if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
            f_inicio, f_fin = rango_fechas
        elif isinstance(rango_fechas, tuple) and len(rango_fechas) == 1:
            f_inicio = f_fin = rango_fechas[0]
        else:
            f_inicio = f_fin = rango_fechas

        mask = (df_ep['fecha'].dt.date >= f_inicio) & (df_ep['fecha'].dt.date <= f_fin)
        df_ts = df_ep[mask].copy().sort_values('fecha')

        if df_ts.empty:
            st.warning("⚠️ No hay registros de telemetría para el rango de fechas seleccionado.")
        else:
            # 4. Optimización de renderizado: si la selección contiene demasiados puntos de datos, 
            # hacemos un muestreo ordenado proporcional para no congelar el navegador.
            max_puntos_visuales = 5000
            if len(df_ts) > max_puntos_visuales:
                N = len(df_ts) // max_puntos_visuales
                df_plot = df_ts.iloc[::N]
            else:
                df_plot = df_ts

            # 5. Gráfico de Serie de Tiempo (3 paneles sincronizados)
            fig_ts = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                subplot_titles=("Evolución RPM", "Evolución Velocidad (Nudos)", "Evolución Consumo (Gal/H)")
            )

            fig_ts.add_trace(go.Scatter(x=df_plot['fecha'], y=df_plot['RPM'], mode='lines', line=dict(color=C_CYAN_DRK, width=1.5), name="RPM"), row=1, col=1)
            fig_ts.add_trace(go.Scatter(x=df_plot['fecha'], y=df_plot['Velocidad'], mode='lines', line=dict(color=C_GREEN_XLG, width=1.5), name="Velocidad"), row=2, col=1)
            fig_ts.add_trace(go.Scatter(x=df_plot['fecha'], y=df_plot['Galon/hora'], mode='lines', line=dict(color=C_WARN_ORANGE, width=1.5), name="Gal/H"), row=3, col=1)

            fig_ts.update_layout(**layout_dark(
                height=700,
                showlegend=False,
                hovermode="x unified",
                title=dict(text=f"Dinámica Operativa · {nombre_seleccionado} ({f_inicio} a {f_fin})", font=dict(size=14, color=C_TEXT_MID))
            ))

            for i in range(1, 4):
                fig_ts.update_yaxes(gridcolor='#1a2d4d', row=i, col=1)
                fig_ts.update_xaxes(gridcolor='#1a2d4d', row=i, col=1)

            for ann in fig_ts.layout.annotations:
                ann.font.color = C_TEXT_MID
                ann.font.size = 12

            st.plotly_chart(fig_ts, use_container_width=True)

            # 6. Tabla Resumen del Periodo
            st.markdown(f'<p style="font-weight:600; color:white; margin-top:10px;">Resumen Estadístico del Periodo Seleccionado</p>', unsafe_allow_html=True)
            resumen_ts = df_ts[['RPM', 'Velocidad', 'Galon/hora']].describe().T
            resumen_ts = resumen_ts[['count', 'mean', '50%', 'min', 'max', 'std']].rename(columns={
                'count': 'N° Registros', 'mean': 'Media', '50%': 'Mediana', 'min': 'Mínimo', 'max': 'Máximo', 'std': 'Desviación Estándar'
            })

            st.dataframe(resumen_ts.style.format({
                'N° Registros': '{:,.0f}',
                'Media': '{:.2f}',
                'Mediana': '{:.2f}',
                'Mínimo': '{:.2f}',
                'Máximo': '{:.2f}',
                'Desviación Estándar': '{:.2f}'
            }).set_table_styles(ESTILOS_TABLA), use_container_width=True)