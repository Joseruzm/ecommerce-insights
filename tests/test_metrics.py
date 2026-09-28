import pandas as pd
import pytest

from src.metrics import (
    kpi_summary,
    monthly_sales,
    new_vs_returning,
    sales_by_country,
    top_products,
)


@pytest.fixture
def df() -> pd.DataFrame:
    rows = [
        # order_id, date, customer, qty, price, product_id, product_name, country
        ("o1", "2025-01-02", "A", 2, 10.0, "SKU1", "Taza", "ES"),
        ("o2", "2025-01-30", "B", 1, 25.0, "SKU2", "Vaso", "ES"),
        ("o3", "2025-02-05", "A", 1, 10.0, "SKU1", "Taza", "ES"),
        ("o4", "2025-02-27", "C", 3, 5.0, "SKU3", "Plato", "FR"),
        ("o5", "2025-03-03", "C", 1, 5.0, "SKU3", "Plato", "FR"),
        ("o6", "2025-03-08", "D", 2, 10.0, "SKU1", "Taza", "ES"),
    ]
    out = pd.DataFrame(
        rows,
        columns=["order_id", "date", "customer_id", "quantity", "unit_price",
                 "product_id", "product_name", "country"],
    )
    out["date"] = pd.to_datetime(out["date"])
    out["amount"] = out["quantity"] * out["unit_price"]
    return out


def test_monthly_sales_values_and_incomplete_last_month(df):
    monthly = monthly_sales(df)
    assert monthly["sales"].tolist() == [45.0, 25.0, 25.0]
    assert monthly["orders"].tolist() == [2, 2, 2]
    assert monthly["customers"].tolist() == [2, 2, 2]
    # Los datos acaban el 8 de marzo: marzo está incompleto.
    assert monthly["complete"].tolist() == [True, True, False]


def test_monthly_sales_fills_gaps_with_zeros():
    gap = pd.DataFrame(
        {
            "order_id": ["1", "2"],
            "date": pd.to_datetime(["2025-01-15", "2025-03-15"]),
            "customer_id": ["A", "A"],
            "quantity": [1, 1],
            "unit_price": [10.0, 10.0],
            "amount": [10.0, 10.0],
        }
    )
    monthly = monthly_sales(gap)
    assert len(monthly) == 3
    assert monthly.loc[1, "sales"] == 0
    assert monthly.loc[1, "orders"] == 0


def test_kpi_summary(df):
    kpis = kpi_summary(df)
    assert kpis["total_sales"] == 95.0
    assert kpis["orders"] == 6
    assert kpis["customers"] == 4
    assert kpis["avg_order_value"] == pytest.approx(95 / 6)
    # Último mes completo: febrero (25) frente a enero (45).
    assert kpis["last_month"] == pd.Timestamp("2025-02-01")
    assert kpis["last_month_sales"] == 25.0
    assert kpis["mom_change"] == pytest.approx(25 / 45 - 1)


def test_top_products_by_sales_and_units(df):
    by_sales = top_products(df, n=2)
    assert by_sales["product_id"].tolist() == ["SKU1", "SKU2"]
    assert by_sales.loc[0, "product"] == "Taza"
    assert by_sales.loc[0, "sales"] == 50.0

    by_units = top_products(df, n=3, by="units")
    assert by_units["product_id"].tolist() == ["SKU1", "SKU3", "SKU2"]


def test_top_products_requires_product_column(df):
    with pytest.raises(ValueError, match="columna de producto"):
        top_products(df.drop(columns=["product_id", "product_name"]))


def test_new_vs_returning_skips_first_month(df):
    result = new_vs_returning(df)
    assert result["month"].tolist() == [pd.Timestamp("2025-02-01"), pd.Timestamp("2025-03-01")]
    assert result["new_customers"].tolist() == [1, 1]
    assert result["returning_customers"].tolist() == [1, 1]

    with_first = new_vs_returning(df, skip_first_month=False)
    assert with_first.loc[0, "new_customers"] == 2  # todos nuevos en el primer mes
    assert with_first.loc[0, "returning_customers"] == 0


def test_sales_by_country_groups_the_rest_as_others(df):
    result = sales_by_country(df, n=1)
    assert result["country"].tolist() == ["ES", "Otros"]
    assert result["sales"].tolist() == [75.0, 20.0]
    assert result["customers"].tolist() == [3, 1]


def test_empty_data_raises(df):
    with pytest.raises(ValueError, match="No hay datos"):
        kpi_summary(df.iloc[0:0])