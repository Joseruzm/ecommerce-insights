"""Métricas para el dashboard de ventas.

Todas las funciones reciben el DataFrame que devuelve loader.clean(), con las
columnas: order_id, date, customer_id, quantity, unit_price, amount y,
opcionalmente, product_id, product_name y country. Devuelven DataFrames o
diccionarios sencillos que la interfaz de Streamlit puede pintar directamente.

Criterio con las devoluciones (líneas con cantidad negativa):
- Las ventas son NETAS: las devoluciones se restan en el mes en que ocurren.
- Los pedidos, los clientes y los nuevos/recurrentes cuentan solo compras.
"""
from __future__ import annotations

import pandas as pd


def _require_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("No hay datos que analizar después de la limpieza.")


def _month_start(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("M").dt.to_timestamp()


def _is_return(df: pd.DataFrame) -> pd.Series:
    return df["quantity"] < 0


def monthly_sales(df: pd.DataFrame, tolerance_days: int = 3) -> pd.DataFrame:
    """Ventas netas, devoluciones, pedidos y clientes activos por mes.

    Columnas: month, sales (netas), returns (importe devuelto, en positivo),
    orders y customers (solo compras) y complete.

    Devuelve una fila por mes (los meses sin ventas aparecen con ceros). La
    columna 'complete' marca si el mes está completo: solo el primer y el último
    mes pueden estar incompletos, si los datos empiezan o terminan a más de
    'tolerance_days' días del límite del mes. Un último mes incompleto no debe
    usarse para previsión ni para comparar.
    """
    _require_data(df)
    data = df.assign(month=_month_start(df["date"]))
    is_return = _is_return(data)
    purchases, returns = data[~is_return], data[is_return]

    out = pd.DataFrame(
        {
            "sales": data.groupby("month")["amount"].sum(),
            "returns": -returns.groupby("month")["amount"].sum(),
            "orders": purchases.groupby("month")["order_id"].nunique(),
            "customers": purchases.groupby("month")["customer_id"].nunique(),
        }
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

    - total_sales: ventas netas (compras menos devoluciones).
    - gross_sales / returns / return_rate: ventas antes de devoluciones, importe
      devuelto y porcentaje que suponen.
    - avg_order_value: ticket medio, calculado sobre las compras (antes de
      devoluciones).
    - mom_change: variación del último mes completo frente al anterior
      (None si no hay dos meses completos consecutivos).
    """
    _require_data(df)
    is_return = _is_return(df)
    purchases = df[~is_return]

    gross = float(purchases["amount"].sum())
    returned = float(-df.loc[is_return, "amount"].sum())
    orders = int(purchases["order_id"].nunique())

    summary = {
        "total_sales": float(df["amount"].sum()),
        "gross_sales": gross,
        "returns": returned,
        "return_rate": returned / gross if gross else 0.0,
        "orders": orders,
        "customers": int(purchases["customer_id"].nunique()),
        "avg_order_value": gross / orders if orders else 0.0,
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
    """Los n productos que más venden, por importe neto ('sales') o unidades netas ('units').

    Las devoluciones se restan, así que un producto vendido y devuelto no aparece
    arriba. Agrupa por código de producto si existe (así los cambios de
    descripción no duplican filas) y muestra el nombre más frecuente de cada
    código. 'orders' cuenta pedidos de compra.
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

    grouped = df.groupby(key).agg(sales=("amount", "sum"), units=("quantity", "sum"))
    grouped["orders"] = df[~_is_return(df)].groupby(key)["order_id"].nunique()
    grouped["orders"] = grouped["orders"].fillna(0).astype(int)

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
    """Clientes nuevos y recurrentes en cada mes (solo cuentan compras).

    Un cliente es nuevo en el mes de su primera compra y recurrente en los
    siguientes. En el primer mes de los datos todos parecerían nuevos porque no
    hay historial anterior, así que por defecto ese mes se omite.
    """
    _require_data(df)
    purchases = df[~_is_return(df)]
    if purchases.empty:
        raise ValueError("No hay compras que analizar.")

    data = pd.DataFrame(
        {"customer_id": purchases["customer_id"], "month": _month_start(purchases["date"])}
    )
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
    """Ventas netas y clientes por país; los países fuera del top n se agrupan en 'Otros'."""
    _require_data(df)
    if "country" not in df.columns:
        raise ValueError("El archivo no tiene columna de país.")

    purchases = df[~_is_return(df)]
    by_country = df.groupby("country").agg(sales=("amount", "sum"))
    by_country["customers"] = purchases.groupby("country")["customer_id"].nunique()
    by_country["customers"] = by_country["customers"].fillna(0).astype(int)
    by_country = by_country.sort_values("sales", ascending=False)

    top, rest = by_country.head(n), by_country.iloc[n:]
    if not rest.empty:
        others = pd.DataFrame(
            {
                "sales": [rest["sales"].sum()],
                "customers": [
                    purchases.loc[purchases["country"].isin(rest.index), "customer_id"].nunique()
                ],
            },
            index=["Otros"],
        )
        top = pd.concat([top, others])
    return top.rename_axis("country").reset_index()