"""Segmentación de clientes: RFM (recencia, frecuencia, valor) y K-Means.

Recibe el DataFrame que devuelve loader.clean(). Las devoluciones se restan del
valor del cliente, pero solo las compras cuentan para la recencia y la frecuencia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MIN_CUSTOMERS = 5

SEGMENT_ORDER = [
    "Campeones",
    "Fieles",
    "Recientes",
    "Necesitan atención",
    "En riesgo",
    "Inactivos",
]

SEGMENT_ACTIONS = {
    "Campeones": "Recompénsalos y pídeles reseñas: son tus mejores clientes.",
    "Fieles": "Ofréceles venta cruzada o un programa de fidelidad.",
    "Recientes": "Consigue su segunda compra con un mensaje de bienvenida.",
    "Necesitan atención": "Recuérdales tu tienda con ofertas personalizadas.",
    "En riesgo": "Compraban a menudo y ya no: lanza una campaña de reactivación.",
    "Inactivos": "Campaña de bajo coste o déjalos fuera de las acciones de pago.",
}


def rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por cliente con recencia (días), frecuencia (pedidos) y valor neto.

    La recencia se mide hasta el día siguiente a la última fecha de los datos.
    Se descartan los clientes cuyo valor neto es 0 o negativo (devolvieron todo).
    """
    if df.empty:
        raise ValueError("No hay datos que segmentar.")
    purchases = df[df["quantity"] > 0]
    if purchases.empty:
        raise ValueError("No hay compras que segmentar.")

    snapshot = df["date"].max().normalize() + pd.Timedelta(days=1)
    table = pd.DataFrame(
        {
            "last_purchase": purchases.groupby("customer_id")["date"].max(),
            "frequency": purchases.groupby("customer_id")["order_id"].nunique(),
            "monetary": df.groupby("customer_id")["amount"].sum(),
        }
    )
    table = table.dropna(subset=["last_purchase"])  # clientes que solo tienen devoluciones
    table = table[table["monetary"] > 0].copy()
    table["frequency"] = table["frequency"].astype(int)
    table["recency"] = (snapshot - table["last_purchase"].dt.normalize()).dt.days
    return table[["last_purchase", "recency", "frequency", "monetary"]].rename_axis("customer_id")


def _quintile(values: pd.Series, tiebreak: pd.Series, ascending: bool = True) -> pd.Series:
    """Puntuación de 1 a 5 por quintiles (5 = el 20 % mejor).

    Como hay muchos empates (p. ej. clientes con un solo pedido), estos se
    desempatan por 'tiebreak' para que los cinco grupos salgan del mismo tamaño.
    """
    order = pd.DataFrame({"v": values, "t": tiebreak}).sort_values(
        ["v", "t"], ascending=[ascending, True], kind="stable"
    )
    position = pd.Series(np.arange(len(order)), index=order.index)
    score = pd.qcut(position, 5, labels=False) + 1
    return score.reindex(values.index).astype(int)


def _segment(recency_score: int, frequency_score: int) -> str:
    if recency_score >= 4 and frequency_score >= 4:
        return "Campeones"
    if recency_score >= 3 and frequency_score >= 3:
        return "Fieles"
    if recency_score >= 4:
        return "Recientes"
    if recency_score == 3:
        return "Necesitan atención"
    if frequency_score >= 3:
        return "En riesgo"
    return "Inactivos"


def rfm_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla RFM con puntuaciones de 1 a 5 (r_score, f_score, m_score) y el segmento."""
    table = rfm_table(df)
    if len(table) < MIN_CUSTOMERS:
        raise ValueError(f"Se necesitan al menos {MIN_CUSTOMERS} clientes para segmentar.")
    table["r_score"] = _quintile(table["recency"], table["monetary"], ascending=False)
    table["f_score"] = _quintile(table["frequency"], table["monetary"])
    table["m_score"] = _quintile(table["monetary"], table["frequency"])
    segments = [_segment(r, f) for r, f in zip(table["r_score"], table["f_score"])]
    table["segment"] = pd.Categorical(segments, categories=SEGMENT_ORDER, ordered=True)
    return table


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Resumen por segmento: clientes, ventas, su peso y la acción recomendada."""
    grouped = rfm.groupby("segment", observed=False)
    out = grouped.agg(
        customers=("monetary", "size"),
        sales=("monetary", "sum"),
        recency=("recency", "mean"),
        frequency=("frequency", "mean"),
    ).reset_index()
    out["customers_share"] = out["customers"] / out["customers"].sum()
    out["sales_share"] = out["sales"] / out["sales"].sum()
    out["segment"] = out["segment"].astype(str)
    out["action"] = out["segment"].map(SEGMENT_ACTIONS)
    return out


def kmeans_groups(
    rfm: pd.DataFrame, k: int = 4, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Agrupa a los clientes con K-Means sobre recencia, frecuencia y valor.

    Las tres variables se pasan por logaritmo (tienen colas muy largas) y se
    estandarizan. Los grupos se numeran de mayor a menor valor mediano, así que
    el grupo 1 es siempre el que más gasta. Devuelve (clientes con su grupo,
    perfil de cada grupo, coeficiente de silueta).
    """
    if not 2 <= k <= 10:
        raise ValueError("k debe estar entre 2 y 10.")
    if len(rfm) < max(MIN_CUSTOMERS, k):
        raise ValueError("Hay muy pocos clientes para formar tantos grupos.")

    features = np.log1p(rfm[["recency", "frequency", "monetary"]])
    X = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)

    labels = pd.Series(model.labels_, index=rfm.index)
    by_value = rfm["monetary"].groupby(labels).median().sort_values(ascending=False).index
    renumber = {old: new for new, old in enumerate(by_value, start=1)}
    groups = labels.map(renumber)

    sample = len(rfm) if len(rfm) <= 5000 else 5000
    silhouette = float(silhouette_score(X, groups, sample_size=sample, random_state=random_state))

    result = rfm.assign(group=groups)
    profile = (
        result.groupby("group")
        .agg(
            customers=("monetary", "size"),
            sales=("monetary", "sum"),
            recency=("recency", "median"),
            frequency=("frequency", "median"),
            monetary=("monetary", "median"),
        )
        .reset_index()
    )
    profile["sales_share"] = profile["sales"] / profile["sales"].sum()
    return result, profile, silhouette