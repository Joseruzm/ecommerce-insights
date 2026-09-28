"""Convierte onlineretail2.rda (R) en una muestra CSV pequeña para la demo.

Uso (desde la raíz del repositorio):
    pip install pyreadr
    python scripts/prepare_sample.py

Entrada : data/raw/onlineretail2.rda   (no se sube a GitHub, está en .gitignore)
Salida  : data/sample_online_retail.csv (sí se sube)

La muestra se hace por clientes, no por filas, para que cada cliente conserve
todo su historial (lo necesita el análisis RFM). Se añade una proporción parecida
de filas sin cliente para que la demo enseñe también la limpieza.
"""
from pathlib import Path

import pandas as pd
import pyreadr

RAW = Path("data/raw/onlineretail2.rda")
OUT = Path("data/sample_online_retail.csv")
FRACTION = 0.05
SEED = 42


def main() -> None:
    df = pyreadr.read_r(str(RAW))["onlineretail2"]

    customers = df["CustomerID"].dropna().unique()
    rng = pd.Series(customers).sample(frac=FRACTION, random_state=SEED)
    with_customer = df[df["CustomerID"].isin(rng)]

    without_customer = df[df["CustomerID"].isna()].sample(frac=FRACTION, random_state=SEED)

    sample = pd.concat([with_customer, without_customer]).sort_values("InvoiceDate")
    sample["CustomerID"] = sample["CustomerID"].astype("Int64")  # 13085.0 -> 13085
    sample.to_csv(OUT, index=False)
    print(f"{len(sample):,} filas -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()