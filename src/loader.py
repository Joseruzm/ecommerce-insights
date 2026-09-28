"""Carga, mapeo de columnas y limpieza de ficheros CSV de ventas.

Flujo previsto (la interfaz de Streamlit se apoya en estas funciones):

    raw = load_csv(archivo)
    mapping = suggest_mapping(raw.columns)      # el usuario puede corregirlo
    df = standardize(raw, mapping)
    df_clean, report = clean(df)
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pandas as pd

# Esquema interno: el resto de la app solo conoce estos nombres.
REQUIRED_FIELDS = ["order_id", "date", "customer_id", "quantity", "unit_price"]
OPTIONAL_FIELDS = ["product_id", "product_name", "country"]

FIELD_LABELS = {
    "order_id": "Pedido / factura",
    "date": "Fecha",
    "customer_id": "Cliente",
    "quantity": "Cantidad",
    "unit_price": "Precio unitario",
    "product_id": "Código de producto",
    "product_name": "Nombre del producto",
    "country": "País",
}

# Alias para autodetectar columnas. Se comparan en minúsculas y sin símbolos
# ("Invoice Date", "invoice_date" e "InvoiceDate" se reducen a "invoicedate").
ALIASES = {
    "order_id": ["invoice", "invoiceno", "orderid", "order", "pedido", "factura", "ticket"],
    "date": ["invoicedate", "date", "orderdate", "fecha", "fechapedido"],
    "customer_id": ["customerid", "customer", "customeruniqueid", "cliente", "idcliente"],
    "quantity": ["quantity", "qty", "cantidad", "unidades"],
    "unit_price": ["price", "unitprice", "precio", "preciounitario"],
    "product_id": ["stockcode", "sku", "productid", "codigo", "codigoproducto"],
    "product_name": ["description", "productname", "producto", "descripcion"],
    "country": ["country", "pais"],
}

# Códigos de Online Retail II que no son productos (envío, comisiones, ajustes...).
# Es específico de ese dataset, por eso clean() no los aplica salvo que se le pasen.
RETAIL_NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "C2", "D", "S", "BANK CHARGES", "ADJUST", "AMAZONFEE",
}


@dataclass
class CleaningReport:
    """Cuántas filas se eliminaron y por qué (se muestra al usuario en la app)."""

    rows_in: int
    removed: dict[str, int] = field(default_factory=dict)
    rows_out: int = 0
    returns_kept: int = 0  # líneas de devolución que se conservan (cantidad negativa)

    @property
    def removed_total(self) -> int:
        return self.rows_in - self.rows_out

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"Motivo": list(self.removed), "Filas eliminadas": list(self.removed.values())})


def load_csv(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Lee un CSV detectando separador, decimal y codificación.

    Acepta una ruta o un objeto de archivo (como el que devuelve st.file_uploader).
    """
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    else:
        source.seek(0)
        raw = source.read()

    sample = raw[:20_000].decode("utf-8", errors="ignore")
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    # Los CSV exportados por Excel en español usan ";" y coma decimal.
    decimal = "," if sep == ";" else "."

    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=sep, decimal=decimal, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("No se pudo leer el archivo: codificación no soportada.")


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def suggest_mapping(columns) -> dict[str, str]:
    """Propone {campo_interno: columna_del_csv} a partir de los nombres de columna."""
    by_norm = {_normalize(c): c for c in columns}
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for internal, aliases in ALIASES.items():
        for alias in aliases:
            column = by_norm.get(alias)
            if column is not None and column not in used:
                mapping[internal] = column
                used.add(column)
                break
    return mapping


def _clean_id(series: pd.Series) -> pd.Series:
    """Identificadores como texto: '13085.0' -> '13085', vacíos -> NA."""
    s = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    empty = (s.eq("") | s.str.lower().isin(["nan", "none", "null"])).fillna(False).astype(bool)
    return s.mask(empty)


def _parse_dates(series: pd.Series, dayfirst: bool) -> pd.Series:
    """Convierte a fecha sin equivocarse con el formato.

    Las fechas ISO (2025-03-04 o 2025-03-04 10:30:00) se leen siempre como año-mes-día:
    'dayfirst' solo se aplica al resto de formatos (31/01/2025). Las que no se
    puedan leer quedan como NaT y clean() las descarta.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    text = series.astype("string").str.strip()
    present = text.dropna()
    if len(present) and present.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9:
        return pd.to_datetime(text, format="ISO8601", errors="coerce")
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)
    if parsed.isna().mean() > text.isna().mean() + 0.05:  # formatos mezclados
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst, format="mixed")
    return parsed


def standardize(raw: pd.DataFrame, mapping: dict[str, str], dayfirst: bool = False) -> pd.DataFrame:
    """Renombra las columnas al esquema interno y convierte los tipos.

    dayfirst=True para fechas tipo 31/01/2025 (día antes que mes). Las fechas en
    formato ISO (2025-01-31) se leen bien con cualquier valor.
    """
    missing = [FIELD_LABELS[f] for f in REQUIRED_FIELDS if f not in mapping]
    if missing:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(missing))
    absent = [c for c in mapping.values() if c not in raw.columns]
    if absent:
        raise ValueError("Columnas que no existen en el archivo: " + ", ".join(absent))

    df = pd.DataFrame({internal: raw[column] for internal, column in mapping.items()})
    df["order_id"] = df["order_id"].astype("string").str.strip()
    df["customer_id"] = _clean_id(df["customer_id"])
    df["date"] = _parse_dates(df["date"], dayfirst)
    for column in ("quantity", "unit_price"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "product_id" in df.columns:
        df["product_id"] = _clean_id(df["product_id"])
    return df


def clean(
    df: pd.DataFrame,
    exclude_product_codes: set[str] | None = None,
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Elimina filas que falsean un análisis de ventas y devuelve un informe.

    Las reglas se aplican en orden y cada fila se cuenta en la primera regla que
    la elimina.

    Las devoluciones y cancelaciones (cantidad negativa) NO se eliminan: se
    conservan marcadas en la columna 'is_return' y con importe negativo en
    'amount'. Así las métricas pueden restarlas de las ventas. Descartarlas
    inflaba las ventas (en Online Retail II, un 6 %) y dejaba pedidos que se
    anularon minutos después, como si se hubieran vendido.

    Añade las columnas 'amount' (cantidad * precio unitario) e 'is_return'.
    """
    report = CleaningReport(rows_in=len(df))
    rules = [
        ("Fecha, cantidad o precio no válidos",
         lambda x: x["date"].isna() | x["quantity"].isna() | x["unit_price"].isna()),
        ("Sin identificador de cliente", lambda x: x["customer_id"].isna()),
        ("Cantidad igual a 0", lambda x: x["quantity"] == 0),
        ("Precio <= 0 (regalos, ajustes)", lambda x: x["unit_price"] <= 0),
    ]
    if exclude_product_codes and "product_id" in df.columns:
        codes = {c.upper() for c in exclude_product_codes}
        rules.append((
            "Conceptos que no son productos (envío, comisiones...)",
            lambda x: x["product_id"].str.upper().isin(codes).fillna(False).astype(bool),
        ))
    if drop_duplicates:
        rules.append(("Filas duplicadas", lambda x: x.duplicated()))

    out = df
    for reason, rule in rules:
        mask = rule(out)
        report.removed[reason] = int(mask.sum())
        out = out.loc[~mask]

    out = out.assign(
        amount=out["quantity"] * out["unit_price"],
        is_return=out["quantity"] < 0,
    ).reset_index(drop=True)
    report.rows_out = len(out)
    report.returns_kept = int(out["is_return"].sum())
    return out, report