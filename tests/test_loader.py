import io

import pandas as pd
import pytest

from src.loader import (
    RETAIL_NON_PRODUCT_CODES,
    clean,
    load_csv,
    standardize,
    suggest_mapping,
)

RETAIL_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"]


def make_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Invoice": ["1001", "1001", "C1002", "1003", "1004", "1005", "1005"],
            "StockCode": ["85048", "79323P", "85048", "POST", "22041", "22041", "22041"],
            "Description": ["A", "B", "A", "Postage", "C", "C", "C"],
            "Quantity": [12, 6, -12, 1, 3, 2, 2],
            "InvoiceDate": ["2010-01-04 10:00"] * 7,
            "Price": [2.5, 1.0, 2.5, 18.0, 0.0, 4.0, 4.0],
            "CustomerID": [13085.0, 13085.0, 13085.0, 13085.0, 12583.0, None, None],
            "Country": ["United Kingdom"] * 7,
        }
    )


def test_suggest_mapping_detects_online_retail_columns():
    mapping = suggest_mapping(RETAIL_COLUMNS)
    assert mapping["order_id"] == "Invoice"
    assert mapping["date"] == "InvoiceDate"
    assert mapping["customer_id"] == "CustomerID"
    assert mapping["quantity"] == "Quantity"
    assert mapping["unit_price"] == "Price"
    assert mapping["product_id"] == "StockCode"


def test_suggest_mapping_understands_spanish_names():
    mapping = suggest_mapping(["Nº Pedido", "Fecha", "Cliente", "Cantidad", "Precio Unitario"])
    assert mapping["date"] == "Fecha"
    assert mapping["customer_id"] == "Cliente"
    assert mapping["quantity"] == "Cantidad"
    assert mapping["unit_price"] == "Precio Unitario"


def test_standardize_requires_mandatory_columns():
    raw = make_raw()
    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        standardize(raw, {"order_id": "Invoice", "date": "InvoiceDate"})


def test_standardize_normalizes_customer_id():
    raw = make_raw()
    df = standardize(raw, suggest_mapping(raw.columns))
    assert df["customer_id"].iloc[0] == "13085"  # no "13085.0"
    assert df["customer_id"].isna().sum() == 2


def test_clean_removes_bad_rows_and_reports_reasons():
    raw = make_raw()
    df = standardize(raw, suggest_mapping(raw.columns))
    out, report = clean(df, exclude_product_codes=RETAIL_NON_PRODUCT_CODES)

    # Solo sobreviven las dos primeras líneas (factura 1001).
    assert len(out) == 2
    assert report.rows_in == 7
    assert report.rows_out == 2
    assert report.removed_total == 5
    assert report.removed["Sin identificador de cliente"] == 2
    assert report.removed["Devoluciones o cancelaciones (cantidad <= 0)"] == 1
    assert report.removed["Precio <= 0 (regalos, ajustes)"] == 1
    assert report.removed["Conceptos que no son productos (envío, comisiones...)"] == 1
    assert out["amount"].tolist() == [30.0, 6.0]


def test_load_csv_handles_semicolons_and_decimal_comma():
    content = "Fecha;Cliente;Cantidad;Precio\n04/01/2010;A1;2;6,95\n05/01/2010;A2;1;12,50\n"
    df = load_csv(io.BytesIO(content.encode("latin-1")))
    assert list(df.columns) == ["Fecha", "Cliente", "Cantidad", "Precio"]
    assert df["Precio"].tolist() == [6.95, 12.5]


def test_dayfirst_dates():
    raw = pd.DataFrame(
        {"pedido": ["1"], "fecha": ["31/01/2025"], "cliente": ["A"], "cantidad": [1], "precio": [5.0]}
    )
    df = standardize(raw, suggest_mapping(raw.columns), dayfirst=True)
    assert df["date"].iloc[0] == pd.Timestamp("2025-01-31")