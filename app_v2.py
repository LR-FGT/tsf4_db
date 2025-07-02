import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from io import BytesIO
from pyproj import Transformer

# Configurar la transformación UTM -> LatLon
transformer = Transformer.from_crs("epsg:32613", "epsg:4326", always_xy=True)

def convertir_utm_a_latlon(df, x_col="coord_x", y_col="coord_y"):
    lons, lats = transformer.transform(df[x_col].values, df[y_col].values)
    df["lon"] = lons
    df["lat"] = lats
    return df

def letra_a_indice(col):
    col = col.upper()
    resultado = 0
    for c in col:
        resultado = resultado * 26 + (ord(c) - ord('A') + 1)
    return resultado - 1

# Conexión a Supabase (Postgres)
DB_URL = st.secrets["DB_URL"]

@st.cache_resource
def get_engine():
    return create_engine(DB_URL, poolclass=NullPool)

# Cachear tablas
@st.cache_data
def get_table_names():
    engine = get_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"))
            return [row[0] for row in result.fetchall()]
    except Exception as e:
        st.error(f"Error al conectar a la base de datos: {e}")
        return []

@st.cache_data
def load_table(table_name):
    engine = get_engine()
    try:
        with engine.connect() as conn:
            return pd.read_sql(f"SELECT * FROM {table_name}", conn)
    except Exception as e:
        st.error(f"Error al conectar a la base de datos: {e}")
        return []

def generar_descarga(df, formato="csv"):
    if formato == "csv":
        return df.to_csv(index=False).encode('utf-8')
    else:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        return output.getvalue()

# --- Interfaz Streamlit ---
st.title("📊 Visualización de Instrumentación TSF4")

tablas = get_table_names()

# Cargar instrumentos
df_instrumentos = load_table("instrumentos")

# Cargar registros de instalación
df_registro = load_table("registro_instalacion")

# Obtener lista de ubicaciones disponibles
ubicaciones_disponibles = df_registro["ubicacion"].unique().tolist()

# Selector de ubicación (con opción seleccionar todas)
seleccionar_todas_ubicaciones = st.checkbox(
    "✅ Seleccionar todas las ubicaciones",
    value=False
)

if seleccionar_todas_ubicaciones:
    ubicaciones_seleccionadas = ubicaciones_disponibles
else:
    ubicaciones_seleccionadas = st.multiselect(
        "📍 Selecciona una o varias ubicaciones:",
        ubicaciones_disponibles,
        default=ubicaciones_disponibles[:1]
    )

# Filtrar los instrumentos que estén instalados en esas ubicaciones
instrumentos_filtrados = df_registro[
    df_registro["ubicacion"].isin(ubicaciones_seleccionadas)
]["nombre"].unique().tolist()

# Ahora muestra solo esos instrumentos
seleccionar_todos_instrumentos = st.checkbox(
    "✅ Seleccionar todos los instrumentos en estas ubicaciones",
    value=False
)

if seleccionar_todos_instrumentos:
    instrumentos_seleccionados = instrumentos_filtrados
else:
    instrumentos_seleccionados = st.multiselect(
        "🎯 Selecciona uno o varios instrumentos en las ubicaciones elegidas:",
        instrumentos_filtrados,
        default=instrumentos_filtrados[:1]
    )

st.write("Instrumentos seleccionados:", instrumentos_seleccionados)
st.write("Ubicaciones seleccionadas:", ubicaciones_seleccionadas)

st.sidebar.header("⏱️ Filtro de fechas")

df_lecturas = load_table("lecturas")
df_lecturas["fecha"] = pd.to_datetime(df_lecturas["fecha"], errors="coerce")
min_fecha = df_lecturas["fecha"].min()
max_fecha = df_lecturas["fecha"].max()

rango_fechas = st.sidebar.date_input(
    "Selecciona rango de fechas",
    value=(min_fecha, max_fecha),
    min_value=min_fecha,
    max_value=max_fecha
)

# Contenido de tablas
st.header("📋 Contenido de las tablas")
for tabla in tablas:
    with st.expander(f"▶️ {tabla}", expanded=False):
        df = load_table(tabla)
        if "nombre" in df.columns:
            df = df[df["nombre"].isin(instrumentos_seleccionados)]
        columnas = df.columns.tolist()
        columnas_mostrar = st.multiselect(
            f"Columnas a mostrar en {tabla}",
            columnas,
            default=columnas
        )
        df_filtrado = df[columnas_mostrar]
        st.dataframe(df_filtrado, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Descargar CSV",
                data=generar_descarga(df_filtrado, formato="csv"),
                file_name=f"{tabla}.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                "⬇️ Descargar Excel",
                data=generar_descarga(df_filtrado, formato="excel"),
                file_name=f"{tabla}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Mapa
px.set_mapbox_access_token(st.secrets["MAPBOX_TOKEN"])

st.header("🗺️ Mapa de instrumentos")
df_map = df_instrumentos[df_instrumentos["nombre"].isin(instrumentos_seleccionados)]
df_map = convertir_utm_a_latlon(df_map)

if not df_map.empty:
    fig_map = px.scatter_mapbox(
        df_map,
        lat="lat",
        lon="lon",
        hover_name="nombre",
        zoom=14,
        height=600
    )
    fig_map.update_layout(mapbox_style="satellite")
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No hay coordenadas para los instrumentos seleccionados.")

# Gráfica de elevación
st.header("📈 Gráfica de elevación vs fecha")

df_lecturas_filtradas = df_lecturas[
    (df_lecturas["nombre"].isin(instrumentos_seleccionados)) &
    (df_lecturas["fecha"].between(pd.to_datetime(rango_fechas[0]), pd.to_datetime(rango_fechas[1])))
]

if not df_lecturas_filtradas.empty:
    fig_elev = px.scatter(
        df_lecturas_filtradas,
        x="fecha",
        y="elevacion",
        color="nombre",
        title="Elevación vs Fecha"
    )
    st.plotly_chart(fig_elev, use_container_width=True)
else:
    st.warning("No hay lecturas en el rango de fechas seleccionado.")

# Subida de archivos
uploaded_files = st.file_uploader(
    "📁 Sube archivos .dat para actualizar lecturas",
    type=["dat"],
    accept_multiple_files=True
)

def actualizar_lecturas_desde_hub(uploaded_files):
    engine = get_engine()
    with engine.begin() as conn:
        df_hub = pd.read_sql("SELECT * FROM coordenadas_hub", conn)

        for _, row in df_hub.iterrows():
            nombre = row["nombre"]
            pattern = row["archivo"]
            matching_files = [f for f in uploaded_files if pattern in f.name]
            if not matching_files:
                st.warning(f"No se encontró archivo que contenga: {pattern}")
                continue
            archivo = matching_files[0]

            try:
                df = pd.read_csv(archivo, sep=",", header=None, skiprows=4)
                fecha_idx = letra_a_indice(row['columna_fecha'])
                unidades_b_idx = letra_a_indice(row['columna_unidades_b'])
                temp_idx = letra_a_indice(row['columna_temperatura'])

                fechas = pd.to_datetime(df.iloc[:, fecha_idx], errors="coerce")
                unidades_b = pd.to_numeric(df.iloc[:, unidades_b_idx], errors="coerce")
                temperaturas = pd.to_numeric(df.iloc[:, temp_idx], errors="coerce")

                df_proc = pd.DataFrame({
                    "fecha": fechas,
                    "unidades_b": unidades_b,
                    "temperatura": temperaturas
                }).dropna(subset=["fecha"])

                op = pd.read_sql(
                    "SELECT * FROM variables_funcionamiento WHERE nombre = %s",
                    conn, params=(nombre,)
                )
                inst = pd.read_sql(
                    "SELECT * FROM registro_instalacion WHERE nombre = %s ORDER BY fecha_instalacion DESC LIMIT 1",
                    conn, params=(nombre,)
                )

                if op.empty or inst.empty:
                    st.warning(f"No se encontró configuración para {nombre}")
                    continue

                op_vars = op.iloc[0]
                inst_vars = inst.iloc[0]

                df_proc["presion_lineal_kpa"] = (
                    op_vars['cf'] * (inst_vars['li'] - df_proc["unidades_b"])
                    - op_vars['tk'] * (df_proc["temperatura"] - inst_vars['ti'])
                )
                df_proc["presion_lineal_m"] = df_proc["presion_lineal_kpa"].clip(lower=0) / 9.81

                df_proc["presion_polinomica_kpa"] = (
                    op_vars['a'] * df_proc["unidades_b"]**2 +
                    op_vars['b'] * df_proc["unidades_b"] +
                    op_vars['c'] - op_vars["tk"] * (inst_vars['ti'] - df_proc["temperatura"])
                )
                df_proc["presion_polinomica_m"] = df_proc["presion_polinomica_kpa"].clip(lower=0) / 9.81

                df_proc["elevacion"] = inst_vars["sensor_elev"] + df_proc["presion_polinomica_m"]
                df_proc["nombre"] = nombre

                fechas_existentes = pd.read_sql(
                    "SELECT fecha FROM lecturas WHERE nombre = %s",
                    conn, params=(nombre,)
                )["fecha"].dropna().unique()

                df_nuevas = df_proc[~df_proc["fecha"].isin(pd.to_datetime(fechas_existentes, errors="coerce"))]

                if not df_nuevas.empty:
                    df_nuevas.to_sql("lecturas", conn, if_exists="append", index=False)
                    st.success(f"{len(df_nuevas)} nuevas lecturas agregadas para {nombre}")
                else:
                    st.info(f"No había nuevas lecturas para {nombre}")

            except Exception as e:
                st.error(f"Error procesando {nombre}: {e}")

if st.button("🔄 Actualizar lecturas desde archivos"):
    if uploaded_files:
        actualizar_lecturas_desde_hub(uploaded_files)
        st.experimental_rerun()
    else:
        st.warning("Primero selecciona archivos .dat")

