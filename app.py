"""Ecommerce Insights: sube el CSV de ventas de tu tienda online y obtén un
dashboard de ventas, productos y clientes.

Ejecutar en local:  streamlit run app.py
"""
import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import loader, metrics, segmentation

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_online_retail.csv"
SOURCE_SAMPLE = "Datos de ejemplo"
SOURCE_UPLOAD = "Subir mi CSV"
NO_COLUMN = "(ninguna)"
BLUE, GREY = "#1F77B4", "#B0B7C3"

st.set_page_config(page_title="Ecommerce Insights", page_icon="📊", layout="wide")


# ---------------------------------------------------------------- utilidades
def money(value: float, symbol: str) -> str:
    return f"{value:,.0f}".replace(",", ".") + f" {symbol}"


def integer(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def percent(fraction: float) -> str:
    return f"{fraction * 100:.1f} %".replace(".", ",")


# El caché vive solo en la memoria del servidor y caduca a los 15 minutos:
# los archivos subidos no se escriben en disco.
@st.cache_data(ttl=900, max_entries=3, show_spinner="Leyendo el archivo...")
def read_csv_bytes(content: bytes) -> pd.DataFrame:
    return loader.load_csv(io.BytesIO(content))


@st.cache_data(ttl=900, max_entries=3, show_spinner="Limpiando los datos...")
def prepare(raw: pd.DataFrame, mapping: dict, dayfirst: bool, retail_codes: bool):
    df = loader.standardize(raw, mapping, dayfirst=dayfirst)
    codes = loader.RETAIL_NON_PRODUCT_CODES if retail_codes else None
    return loader.clean(df, exclude_product_codes=codes)


@st.cache_data(ttl=900, max_entries=10, show_spinner="Agrupando clientes...")
def cached_groups(rfm: pd.DataFrame, k: int):
    return segmentation.kmeans_groups(rfm, k)


# ------------------------------------------------------------------ gráficos
def chart_monthly(monthly: pd.DataFrame, symbol: str) -> go.Figure:
    colors = [BLUE if ok else GREY for ok in monthly["complete"]]
    fig = go.Figure(
        go.Bar(
            x=monthly["month"],
            y=monthly["sales"],
            marker_color=colors,
            customdata=monthly[["returns", "orders"]],
            hovertemplate=(
                "<b>%{x|%Y-%m}</b><br>Ventas netas: %{y:,.0f}"
                "<br>Devoluciones: %{customdata[0]:,.0f}"
                "<br>Pedidos: %{customdata[1]:,}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Ventas netas por mes",
        yaxis_title=f"Ventas netas ({symbol})",
        showlegend=False,
        margin=dict(t=50, b=10),
    )
    return fig


def chart_top_products(top: pd.DataFrame, by: str, symbol: str) -> go.Figure:
    top = top.iloc[::-1]  # el mayor arriba
    label = f"Ventas netas ({symbol})" if by == "sales" else "Unidades netas"
    fig = go.Figure(go.Bar(x=top[by], y=top["product"], orientation="h", marker_color=BLUE))
    fig.update_layout(
        title="Top de productos",
        xaxis_title=label,
        margin=dict(t=50, b=10, l=10),
        height=430,
    )
    return fig


def chart_customers(nr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        [
            go.Bar(x=nr["month"], y=nr["returning_customers"], name="Recurrentes", marker_color=BLUE),
            go.Bar(x=nr["month"], y=nr["new_customers"], name="Nuevos", marker_color="#FF7F0E"),
        ]
    )
    fig.update_layout(
        title="Clientes activos por mes: nuevos y recurrentes",
        barmode="stack",
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=70, b=10),
    )
    return fig


def chart_countries(countries: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=countries["country"], y=countries["sales"], marker_color=BLUE))
    fig.update_layout(
        title="Ventas netas por país",
        yaxis_title=f"Ventas netas ({symbol})",
        margin=dict(t=50, b=10),
    )
    return fig


def chart_segments(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        [
            go.Bar(x=summary["segment"], y=summary["customers_share"] * 100, name="% de clientes", marker_color=GREY),
            go.Bar(x=summary["segment"], y=summary["sales_share"] * 100, name="% de las ventas", marker_color=BLUE),
        ]
    )
    fig.update_layout(
        title="Peso de cada segmento: clientes frente a ventas",
        barmode="group",
        yaxis_title="%",
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=70, b=10),
    )
    return fig


def chart_groups(result: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    for group, part in result.groupby("group"):
        fig.add_trace(
            go.Scatter(
                x=part["frequency"],
                y=part["monetary"],
                mode="markers",
                name=f"Grupo {group}",
                marker=dict(size=5, opacity=0.6),
            )
        )
    fig.update_xaxes(type="log", title="Pedidos por cliente (escala logarítmica)")
    fig.update_yaxes(type="log", title=f"Valor neto ({symbol}, escala logarítmica)")
    fig.update_layout(title="Clientes por grupo", margin=dict(t=50, b=10))
    return fig


# ------------------------------------------------------------- entrada de datos
st.title("📊 Ecommerce Insights")
st.write(
    "Sube el CSV de ventas de tu tienda online y obtén un dashboard de ventas, "
    "productos y clientes."
)

with st.sidebar:
    st.header("Datos")
    source = st.radio("Origen", [SOURCE_SAMPLE, SOURCE_UPLOAD], label_visibility="collapsed")
    uploaded, dayfirst = None, False
    if source == SOURCE_UPLOAD:
        uploaded = st.file_uploader("Archivo CSV", type="csv")
        dayfirst = st.checkbox("Las fechas tienen el día primero (31/01/2025)", value=True)
    symbol = st.text_input("Moneda", value="£" if source == SOURCE_SAMPLE else "€", key=f"currency_{source}")
    st.caption("Los archivos se procesan en memoria y no se guardan en disco.")

if source == SOURCE_SAMPLE:
    raw = read_csv_bytes(SAMPLE_PATH.read_bytes())
    mapping = loader.suggest_mapping(raw.columns)
    st.caption(
        "Estás viendo una muestra de **Online Retail II**, un dataset público de una "
        "tienda online británica. Elige *Subir mi CSV* en la barra lateral para usar el tuyo."
    )
else:
    if uploaded is None:
        st.info("Sube un archivo CSV en la barra lateral para empezar.")
        st.stop()
    raw = read_csv_bytes(uploaded.getvalue())
    suggested = loader.suggest_mapping(raw.columns)
    options = [NO_COLUMN] + list(raw.columns)
    mapping = {}
    with st.expander("Columnas del archivo", expanded=True):
        st.caption("Comprueba qué columna corresponde a cada dato. Las obligatorias son las cinco primeras.")
        grid = st.columns(3)
        for i, field in enumerate(loader.REQUIRED_FIELDS + loader.OPTIONAL_FIELDS):
            required = field in loader.REQUIRED_FIELDS
            default = suggested.get(field)
            index = options.index(default) if default in options else 0
            label = loader.FIELD_LABELS[field] + ("" if required else " (opcional)")
            choice = grid[i % 3].selectbox(label, options, index=index, key=f"map_{field}")
            if choice != NO_COLUMN:
                mapping[field] = choice

try:
    df, report = prepare(raw, mapping, dayfirst, source == SOURCE_SAMPLE)
    kpis = metrics.kpi_summary(df)
    monthly = metrics.monthly_sales(df)
except ValueError as error:
    st.error(str(error))
    st.stop()

# ------------------------------------------------------------- informe y KPIs
invalid = report.removed.get("Fecha, cantidad o precio no válidos", 0)
if invalid > 0.05 * report.rows_in:
    st.warning(
        f"{percent(invalid / report.rows_in)} de las líneas tienen una fecha, cantidad o precio que no se "
        "ha podido leer. Revisa el mapeo de columnas y el formato de las fechas."
    )

with st.expander(f"Informe de limpieza: {integer(report.rows_in)} → {integer(report.rows_out)} líneas"):
    st.write(
        "Estas líneas se han descartado porque falsean el análisis. Las devoluciones "
        f"no se descartan: se conservan ({integer(report.returns_kept)} líneas) y se restan de las ventas."
    )
    st.dataframe(report.to_frame(), hide_index=True, width="stretch")

st.caption(
    f"Datos del {kpis['start']:%d/%m/%Y} al {kpis['end']:%d/%m/%Y} · "
    f"{integer(report.rows_out)} líneas analizadas"
)

columns = st.columns(5)
columns[0].metric("Ventas netas", money(kpis["total_sales"], symbol), help="Compras menos devoluciones.")
if kpis["last_month"] is not None:
    delta = None if kpis["mom_change"] is None else f"{kpis['mom_change'] * 100:+.1f} % vs mes anterior".replace(".", ",")
    columns[1].metric(
        f"Último mes completo ({kpis['last_month']:%Y-%m})",
        money(kpis["last_month_sales"], symbol),
        delta=delta,
    )
columns[2].metric("Pedidos", integer(kpis["orders"]))
columns[3].metric("Clientes", integer(kpis["customers"]))
columns[4].metric("Ticket medio", money(kpis["avg_order_value"], symbol), help="Valor medio de un pedido, antes de devoluciones.")
st.caption(f"Devoluciones: {money(kpis['returns'], symbol)}, un {percent(kpis['return_rate'])} de las ventas.")

incomplete = monthly.loc[~monthly["complete"], "month"]
if not incomplete.empty:
    months = ", ".join(incomplete.dt.strftime("%Y-%m"))
    st.info(f"Mes incompleto ({months}): aparece en gris y no se usa para comparar con el mes anterior.")

# ------------------------------------------------------------------- pestañas
tab_sales, tab_products, tab_customers, tab_segments, tab_countries = st.tabs(
    ["Ventas", "Productos", "Clientes", "Segmentos", "Países"]
)

with tab_sales:
    st.plotly_chart(chart_monthly(monthly, symbol), width="stretch")

with tab_products:
    try:
        by_label = st.radio("Ordenar por", ["Importe", "Unidades"], horizontal=True)
        by = "sales" if by_label == "Importe" else "units"
        top = metrics.top_products(df, n=10, by=by)
        st.plotly_chart(chart_top_products(top, by, symbol), width="stretch")
        st.dataframe(top, hide_index=True, width="stretch")
    except ValueError as error:
        st.info(str(error))

with tab_customers:
    try:
        nr = metrics.new_vs_returning(df)
        st.plotly_chart(chart_customers(nr), width="stretch")
        st.caption("El primer mes de los datos no se muestra: todos los clientes parecerían nuevos.")
    except ValueError as error:
        st.info(str(error))

with tab_segments:
    try:
        rfm = segmentation.rfm_segments(df)
    except ValueError as error:
        st.info(str(error))
    else:
        summary = segmentation.segment_summary(rfm)
        best = summary.loc[summary["sales_share"].idxmax()]
        st.markdown(
            f"**{best['segment']}**: el {percent(best['customers_share'])} de los clientes "
            f"genera el {percent(best['sales_share'])} de las ventas."
        )
        excluded = kpis["customers"] - len(rfm)
        st.caption(
            "Segmentación RFM: cada cliente puntúa de 1 a 5 según lo reciente de su última compra (R) "
            "y su frecuencia (F); 5 es el 20 % mejor."
            + (f" Se excluyen {integer(excluded)} clientes que devolvieron más de lo que compraron." if excluded else "")
        )
        st.plotly_chart(chart_segments(summary), width="stretch")

        shown = pd.DataFrame(
            {
                "Segmento": summary["segment"],
                "Clientes": summary["customers"],
                "% clientes": (summary["customers_share"] * 100).round(1),
                f"Ventas netas ({symbol})": summary["sales"].round(0),
                "% ventas": (summary["sales_share"] * 100).round(1),
                "Días desde la última compra": summary["recency"].round(0),
                "Pedidos por cliente": summary["frequency"].round(1),
                "Qué hacer": summary["action"],
            }
        )
        st.dataframe(shown, hide_index=True, width="stretch")

        export = rfm.reset_index()[
            ["customer_id", "segment", "recency", "frequency", "monetary", "r_score", "f_score", "m_score"]
        ]
        export["segment"] = export["segment"].astype(str)
        export.columns = [
            "cliente", "segmento", "dias_desde_ultima_compra", "pedidos",
            "valor_neto", "puntuacion_r", "puntuacion_f", "puntuacion_m",
        ]
        st.download_button(
            "Descargar clientes con su segmento (CSV)",
            export.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
            file_name="clientes_segmentados.csv",
            mime="text/csv",
        )

        with st.expander("Agrupación automática con K-Means (avanzado)"):
            st.write(
                "Agrupa a los clientes por su comportamiento (recencia, frecuencia y valor) sin reglas "
                "previas. El grupo 1 es siempre el que más gasta."
            )
            k = st.slider("Número de grupos", 2, 8, 4)
            try:
                result, profile, silhouette = cached_groups(rfm, k)
            except ValueError as error:
                st.info(str(error))
            else:
                st.caption(
                    f"Coeficiente de silueta: {silhouette:.2f}".replace(".", ",")
                    + " (de -1 a 1; cuanto más alto, más separados están los grupos)."
                )
                st.plotly_chart(chart_groups(result, symbol), width="stretch")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Grupo": profile["group"],
                            "Clientes": profile["customers"],
                            f"Ventas netas ({symbol})": profile["sales"].round(0),
                            "% ventas": (profile["sales_share"] * 100).round(1),
                            "Días desde la última compra (mediana)": profile["recency"].round(0),
                            "Pedidos (mediana)": profile["frequency"],
                            f"Valor neto ({symbol}, mediana)": profile["monetary"].round(0),
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )

with tab_countries:
    try:
        countries = metrics.sales_by_country(df, n=10)
        st.plotly_chart(chart_countries(countries, symbol), width="stretch")
    except ValueError as error:
        st.info(str(error))