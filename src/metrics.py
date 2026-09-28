"""Métricas para el dashboard de ventas.

Todas las funciones reciben el DataFrame que devuelve loader.clean(), con las
columnas: order_id, date, customer_id, quantity, unit_price, amount y,
opcionalmente, product_id, product_name y country. Devuelven DataFrames o
diccionarios sencillos que la interfaz de Streamlit puede pintar directamente.
"""
from __future__ import annotations

import pandas as pd


def _require_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("No hay datos que analizar después de la limpieza.")


def _month_start(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("M").dt.to_timestamp()


def monthly_sales(df: pd.DataFrame, tolerance_days: int = 3) -> pd.DataFrame:
    """Ventas, pedidos y clientes activos por mes.

    Devuelve una fila por mes (los meses sin ventas aparecen con ceros) y una
    columna 'complete' que marca si el mes está completo. Solo el primer y el
    último mes pueden estar incompletos: se consideran incompletos si los datos
    empiezan o terminan a más de 'tolerance_days' días del límite del mes. Un
    último mes incompleto no debe usarse para previsión ni para comparar.
    """
    _require_data(df)
    data = df.assign(month=_month_start(df["date"]))
    out = data.groupby("month").agg(
        sales=("amount", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique"),
    )
    full_range = pd.date_range(out.index.min(), out.index.max(), freq="MS")
    out = out.reindex(full_range).fillna(0)
    out[["orders", "customers"]] = out[["orders", "customers"]].astype(int)

    tolerance = pd.Timedelta(days=tolerance_days)
    first_day, last_day = df["date"].min().normalize(), df["date"].max().normalize()
    first_month, last_month = out.index[0], out.index[-1]
    last_month_end = last_month + pd.offsets.MonthEnd(0)

    complete = pd.Series(True, index=out.index)
    complete.iloc[0] = complete.iloc[0] and first_day <= first_month + tolerance
    complete.iloc[-1] = complete.iloc[-1] and last_day >= last_month_end - tolerance
    out["complete"] = complete

    return out.rename_axis("month").reset_index()


def kpi_summary(df: pd.DataFrame) -> dict:
    """Cifras principales para las tarjetas del dashboard.

    'mom_change' compara el último mes completo con el anterior (None si no hay
    dos meses completos consecutivos).
    """
    _require_data(df)
    total = float(df["amount"].sum())
    orders = int(df["order_id"].nunique())

    summary = {
        "total_sales": total,
        "orders": orders,
        "customers": int(df["customer_id"].nunique()),
        "avg_order_value": total / orders if orders else 0.0,
        "start": df["date"].min(),
        "end": df["date"].max(),
        "last_month": None,
        "last_month_sales": None,
        "mom_change": None,
    }

    complete = monthly_sales(df).query("complete")
    if len(complete) >= 1:
        last = complete.iloc[-1]
        summary["last_month"] = last["month"]
        summary["last_month_sales"] = float(last["sales"])
    if len(complete) >= 2:
        prev = complete.iloc[-2]
        consecutive = prev["month"] + pd.offsets.MonthBegin(1) == last["month"]
        if consecutive and prev["sales"] > 0:
            summary["mom_change"] = float(last["sales"] / prev["sales"] - 1)
    return summary


def top_products(df: pd.DataFrame, n: int = 10, by: str = "sales") -> pd.DataFrame:
    """Los n productos que más venden, por importe ('sales') o unidades ('units').

    Agrupa por código de producto si existe (así los cambios de descripción no
    duplican filas) y muestra el nombre más frecuente de cada código.
    """
    _require_data(df)
    if by not in ("sales", "units"):
        raise ValueError("'by' debe ser 'sales' o 'units'.")

    if "product_id" in df.columns:
        key = "product_id"
    elif "product_name" in df.columns:
        key = "product_name"
    else:
        raise ValueError("El archivo no tiene columna de producto.")

    grouped = df.groupby(key).agg(
        sales=("amount", "sum"),
        units=("quantity", "sum"),
        orders=("order_id", "nunique"),
    )

    if key == "product_id" and "product_name" in df.columns:
        names = (
            df.dropna(subset=["product_name"])
            .groupby(["product_id", "product_name"])
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
            .drop_duplicates("product_id")
            .set_index("product_id")["product_name"]
        )
        grouped = grouped.join(names)
        grouped["product"] = grouped["product_name"].fillna(grouped.index.to_series())
        grouped = grouped.drop(columns="product_name").reset_index()
        columns = ["product_id", "product", "sales", "units", "orders"]
    else:
        grouped = grouped.reset_index().rename(columns={key: "product"})
        columns = ["product", "sales", "units", "orders"]

    return grouped.sort_values(by, ascending=False).head(n)[columns].reset_index(drop=True)


def new_vs_returning(df: pd.DataFrame, skip_first_month: bool = True) -> pd.DataFrame:
    """Clientes nuevos y recurrentes en cada mes.

    Un cliente es nuevo en el mes de su primera compra y recurrente en los
    siguientes. En el primer mes de los datos todos parecerían nuevos porque no
    hay historial anterior, así que por defecto ese mes se omite.
    """
    _require_data(df)
    data = pd.DataFrame({"customer_id": df["customer_id"], "month": _month_start(df["date"])})
    first_month = data.groupby("customer_id")["month"].min().rename("first_month")
    active = data.drop_duplicates(["customer_id", "month"]).join(first_month, on="customer_id")
    active["is_new"] = active["month"] == active["first_month"]

    out = active.groupby("month")["is_new"].agg(new_customers="sum", active_customers="count")
    out["returning_customers"] = out["active_customers"] - out["new_customers"]
    out = out[["new_customers", "returning_customers"]].astype(int).reset_index()

    if skip_first_month and len(out) > 1:
        out = out.iloc[1:].reset_index(drop=True)
    return out


def sales_by_country(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Ventas y clientes por país; los países fuera del top n se agrupan en 'Otros'."""
    _require_data(df)
    if "country" not in df.columns:
        raise ValueError("El archivo no tiene columna de país.")

    by_country = (
        df.groupby("country")
        .agg(sales=("amount", "sum"), customers=("customer_id", "nunique"))
        .sort_values("sales", ascending=False)
    )
    top, rest = by_country.head(n), by_country.iloc[n:]
    if not rest.empty:
        others = pd.DataFrame(
            {
                "sales": [rest["sales"].sum()],
                "customers": [df.loc[df["country"].isin(rest.index), "customer_id"].nunique()],
            },
            index=["Otros"],
        )
        top = pd.concat([top, others])
    return top.rename_axis("country").reset_index()