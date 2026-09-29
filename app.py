"""Ecommerce Insights: sube el CSV de ventas de tu tienda online y obtén un
dashboard de ventas, productos y clientes.

Ejecutar en local:  streamlit run app.py
"""
import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import forecast, loader, metrics, segmentation

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_online_retail.csv"
SOURCE_SAMPLE = "Datos de ejemplo"
SOURCE_UPLOAD = "Subir mi CSV"
NO_COLUMN = "(ninguna)"
APP_NAME = "Ecommerce Insights"
TAGLINE = "Convierte el CSV de tu tienda online en decisiones."

# Paleta del proyecto (los mismos colores están en .streamlit/config.toml).
VIOLET, CORAL, TEAL, AMBER, GREY = "#6C4DF6", "#E8482A", "#0B8B7C", "#C77D00", "#8F89AD"
ICON_PATH = Path(__file__).parent / "assets" / "icon.png"

# Monedas del desplegable: símbolo o código -> nombre. Para añadir una, una línea más.
CURRENCIES = {
    "€": "€ · Euro",
    "£": "£ · Libra esterlina",
    "$": "$ · Dólar estadounidense",
    "CHF": "CHF · Franco suizo",
    "MXN": "MXN · Peso mexicano",
    "COP": "COP · Peso colombiano",
    "ARS": "ARS · Peso argentino",
    "CLP": "CLP · Peso chileno",
    "PEN": "PEN · Sol peruano",
}

st.set_page_config(
    page_title=APP_NAME,
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "📊",
    layout="wide",
)


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


@st.cache_data(ttl=900, max_entries=10, show_spinner="Calculando la previsión...")
def cached_forecast(monthly: pd.DataFrame, horizon: int):
    return forecast.forecast_sales(monthly, horizon)


# ------------------------------------------------------------------ gráficos
def chart_monthly(monthly: pd.DataFrame, symbol: str) -> go.Figure:
    colors = [VIOLET if ok else GREY for ok in monthly["complete"]]
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
        yaxis_title=f"Ventas netas ({symbol})",
        showlegend=False,
        margin=dict(t=10, b=10),
    )
    return fig


def chart_top_products(top: pd.DataFrame, by: str, symbol: str) -> go.Figure:
    top = top.iloc[::-1]  # el mayor arriba
    label = f"Ventas netas ({symbol})" if by == "sales" else "Unidades netas"
    fig = go.Figure(go.Bar(x=top[by], y=top["product"], orientation="h", marker_color=VIOLET))
    fig.update_layout(xaxis_title=label, margin=dict(t=10, b=10, l=10), height=430)
    return fig


def chart_customers(nr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        [
            go.Bar(x=nr["month"], y=nr["returning_customers"], name="Recurrentes", marker_color=VIOLET),
            go.Bar(x=nr["month"], y=nr["new_customers"], name="Nuevos", marker_color=CORAL),
        ]
    )
    fig.update_layout(barmode="stack", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), margin=dict(t=40, b=10))
    return fig


def chart_countries(countries: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=countries["country"], y=countries["sales"], marker_color=VIOLET))
    fig.update_layout(yaxis_title=f"Ventas netas ({symbol})", margin=dict(t=10, b=10))
    return fig


def chart_segments(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        [
            go.Bar(x=summary["segment"], y=summary["customers_share"] * 100, name="% de clientes", marker_color=TEAL),
            go.Bar(x=summary["segment"], y=summary["sales_share"] * 100, name="% de las ventas", marker_color=VIOLET),
        ]
    )
    fig.update_layout(barmode="group", yaxis_title="%", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), margin=dict(t=40, b=10))
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
    fig.update_xaxes(
        type="log",
        title="Pedidos por cliente (escala logarítmica)",
        tickvals=[1, 2, 5, 10, 20, 50, 100],
        ticktext=["1", "2", "5", "10", "20", "50", "100"],
    )
    fig.update_yaxes(
        type="log",
        title=f"Valor neto ({symbol}, escala logarítmica)",
        tickvals=[10, 100, 1000, 10000, 100000],
        ticktext=["10", "100", "1.000", "10.000", "100.000"],
    )
    fig.update_layout(margin=dict(t=10, b=10))
    return fig


def chart_forecast(result: "forecast.ForecastResult", symbol: str) -> go.Figure:
    history, fc, test = result.history, result.forecast, result.last_test
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=history.index, y=history.values, name="Ventas reales", mode="lines+markers", line=dict(color=VIOLET))
    )
    fig.add_trace(
        go.Scatter(
            x=list(fc["month"]) + list(fc["month"][::-1]),
            y=list(fc["upper"]) + list(fc["lower"][::-1]),
            fill="toself",
            fillcolor="rgba(232,72,42,0.16)",
            line=dict(width=0),
            name="Rango probable (80 %)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[history.index[-1]] + list(fc["month"]),
            y=[history.iloc[-1]] + list(fc["forecast"]),
            name="Previsión",
            mode="lines+markers",
            line=dict(color=CORAL, dash="dash"),
        )
    )
    if not test.empty:
        fig.add_trace(
            go.Scatter(
                x=test["month"],
                y=test["predicted"],
                name="Lo que habría previsto el modelo",
                mode="markers",
                marker=dict(symbol="x", size=11, color=TEAL),
            )
        )
    fig.update_layout(yaxis_title=f"Ventas netas ({symbol})", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), margin=dict(t=40, b=10))
    return fig


def render_header() -> None:
    logo = (
        '<svg width="46" height="46" viewBox="0 0 46 46" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        '<rect width="46" height="46" rx="13" fill="rgba(255,255,255,0.18)"/>'
        '<rect x="9" y="26" width="7" height="11" rx="2.5" fill="#fff"/>'
        '<rect x="19.5" y="18" width="7" height="19" rx="2.5" fill="#fff"/>'
        '<rect x="30" y="9" width="7" height="28" rx="2.5" fill="#FFB627"/></svg>'
    )
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:1rem;padding:1.1rem 1.4rem;border-radius:1.1rem;
                    margin-bottom:1rem;color:#fff;
                    background:linear-gradient(115deg,#4B2FD1 0%,#7A45F0 58%,#E8482A 130%);">
            {logo}
            <div>
                <div style="font-family:inherit;font-weight:700;font-size:1.75rem;line-height:1.15;">{APP_NAME}</div>
                <div style="opacity:.92;font-size:1rem;">{TAGLINE}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------- entrada de datos
render_header()

with st.sidebar:
    st.header("Datos")
    source = st.radio("Origen", [SOURCE_SAMPLE, SOURCE_UPLOAD], label_visibility="collapsed")
    uploaded, dayfirst = None, False
    if source == SOURCE_UPLOAD:
        uploaded = st.file_uploader("Archivo CSV", type="csv")
        dayfirst = st.checkbox("Las fechas tienen el día primero (31/01/2025)", value=True)
    symbol = st.selectbox(
        "Moneda",
        list(CURRENCIES),
        index=list(CURRENCIES).index("£" if source == SOURCE_SAMPLE else "€"),
        format_func=CURRENCIES.get,
        key=f"currency_{source}",
    )
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
columns[0].metric("Ventas netas", money(kpis["total_sales"], symbol), help="Compras menos devoluciones.", border=True)
if kpis["last_month"] is not None:
    delta = None if kpis["mom_change"] is None else f"{kpis['mom_change'] * 100:+.1f} % vs mes anterior".replace(".", ",")
    columns[1].metric(
        f"Último mes completo ({kpis['last_month']:%Y-%m})",
        money(kpis["last_month_sales"], symbol),
        delta=delta,
        border=True,
    )
columns[2].metric("Pedidos", integer(kpis["orders"]), border=True)
columns[3].metric("Clientes", integer(kpis["customers"]), border=True)
columns[4].metric("Ticket medio", money(kpis["avg_order_value"], symbol), help="Valor medio de un pedido, antes de devoluciones.", border=True)
st.caption(f"Devoluciones: {money(kpis['returns'], symbol)}, un {percent(kpis['return_rate'])} de las ventas.")

incomplete = monthly.loc[~monthly["complete"], "month"]
if not incomplete.empty:
    months = ", ".join(incomplete.dt.strftime("%Y-%m"))
    st.info(f"Mes incompleto ({months}): aparece en gris y no se usa para comparar con el mes anterior.")

# ------------------------------------------------------------------- pestañas
tab_sales, tab_forecast, tab_products, tab_customers, tab_segments, tab_countries = st.tabs(
    ["Ventas", "Previsión", "Productos", "Clientes", "Segmentos", "Países"]
)

with tab_sales:
    st.markdown("##### Ventas netas por mes")
    st.plotly_chart(chart_monthly(monthly, symbol), width="stretch")

with tab_forecast:
    horizon = st.slider("Meses a prever", 1, 6, 3)
    try:
        result = cached_forecast(monthly, horizon)
    except ValueError as error:
        st.info(str(error))
    else:
        total = result.forecast["forecast"].sum()
        period = "el próximo mes" if horizon == 1 else f"los próximos {horizon} meses"
        st.markdown(f"**Ventas previstas para {period}: {money(total, symbol)}**")
        st.caption(f"Método elegido: {result.model_label}.")

        if result.best_wape is not None:
            cols = st.columns(3)
            cols[0].metric(
                "Error medio de esta previsión",
                percent(result.best_wape),
                help="Se calcula previendo los últimos meses como si no se conocieran y comparando con lo que ocurrió.",
                border=True,
            )
            if result.baseline_wape is not None:
                cols[1].metric("Error de la media de los últimos 3 meses", percent(result.baseline_wape), border=True)
            if result.improvement is not None:
                cols[2].metric("Reducción del error", percent(max(result.improvement, 0.0)), border=True)

        st.markdown("##### Ventas netas mensuales y previsión")
        st.plotly_chart(chart_forecast(result, symbol), width="stretch")

        if not incomplete.empty and result.forecast["month"].iloc[0] == incomplete.iloc[-1]:
            st.caption(
                f"El mes en curso ({incomplete.iloc[-1]:%Y-%m}) tiene datos solo hasta el {kpis['end']:%d/%m/%Y} "
                "y no se usa para entrenar; la previsión lo incluye como primer mes."
            )
        for note in result.notes:
            st.info(note)

        st.dataframe(
            pd.DataFrame(
                {
                    "Mes": result.forecast["month"].dt.strftime("%Y-%m"),
                    f"Previsión ({symbol})": result.forecast["forecast"].round(0),
                    "Mínimo probable": result.forecast["lower"].round(0),
                    "Máximo probable": result.forecast["upper"].round(0),
                }
            ),
            hide_index=True,
            width="stretch",
        )

        with st.expander("Cómo se ha evaluado"):
            st.write(
                "Se prueban varios métodos, del más simple al más complejo. A cada uno se le esconden los "
                "últimos meses: se entrena con lo anterior, se prevé y se compara con lo que pasó de verdad. "
                "Gana el que menos se equivoca. El rango probable se calcula con los errores de esas pruebas "
                "(el 80 % de las veces el error fue menor que el que indica el rango)."
            )
            evaluation = result.backtest
            st.dataframe(
                pd.DataFrame(
                    {
                        "Método": evaluation["label"],
                        "Error medio": [percent(w) if ok else "No evaluable (poco historial)" for w, ok in zip(evaluation["wape"], evaluation["evaluated"])],
                        "Sesgo": [f"{b * 100:+.1f} %".replace(".", ",") if ok else "" for b, ok in zip(evaluation["bias"], evaluation["evaluated"])],
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Sesgo: si es negativo, el método tiende a quedarse corto; si es positivo, a pasarse. "
                f"Las pruebas cubren los últimos {forecast.MAX_FOLDS + forecast.BACKTEST_HORIZON - 1} meses de datos."
            )

with tab_products:
    try:
        by_label = st.radio("Ordenar por", ["Importe", "Unidades"], horizontal=True)
        by = "sales" if by_label == "Importe" else "units"
        top = metrics.top_products(df, n=10, by=by)
        st.markdown("##### Top de productos")
        st.plotly_chart(chart_top_products(top, by, symbol), width="stretch")
        st.dataframe(top, hide_index=True, width="stretch")
    except ValueError as error:
        st.info(str(error))

with tab_customers:
    try:
        nr = metrics.new_vs_returning(df)
        nr = nr[nr["month"].isin(monthly.loc[monthly["complete"], "month"])]
        st.markdown("##### Clientes activos por mes: nuevos y recurrentes")
        st.plotly_chart(chart_customers(nr), width="stretch")
        st.caption("No se muestran el primer mes (todos los clientes parecerían nuevos) ni los meses incompletos.")
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
        st.markdown("##### Peso de cada segmento: clientes frente a ventas")
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
                st.markdown("##### Clientes por grupo")
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
        st.markdown("##### Ventas netas por país")
        st.plotly_chart(chart_countries(countries, symbol), width="stretch")
    except ValueError as error:
        st.info(str(error))