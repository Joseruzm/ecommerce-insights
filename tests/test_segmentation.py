import pandas as pd
import pytest

from src.segmentation import (
    SEGMENT_ORDER,
    kmeans_groups,
    rfm_segments,
    rfm_table,
    segment_summary,
)


def make_orders(rows) -> pd.DataFrame:
    """rows: (order_id, fecha, cliente, cantidad, precio)."""
    df = pd.DataFrame(rows, columns=["order_id", "date", "customer_id", "quantity", "unit_price"])
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["quantity"] * df["unit_price"]
    return df


@pytest.fixture
def crowd() -> pd.DataFrame:
    """100 clientes: el 99 es el más reciente, frecuente y gastador; el 0, lo contrario."""
    base = pd.Timestamp("2025-06-30")
    rows = []
    for i in range(100):
        last = base - pd.Timedelta(days=99 - i)
        for k in range(1 + i // 10):  # de 1 a 10 pedidos
            rows.append((f"c{i}-{k}", last - pd.Timedelta(days=7 * k), f"C{i:03d}", 1, 10.0 + i))
    return make_orders(rows)


def test_rfm_table_values_and_returns():
    df = make_orders(
        [
            ("o1", "2025-01-01", "A", 1, 10.0),
            ("o2", "2025-01-20", "A", 1, 20.0),
            ("r1", "2025-01-25", "A", -1, 5.0),  # devolución: resta valor, no cuenta como compra
            ("o3", "2025-01-10", "B", 1, 50.0),
        ]
    )
    table = rfm_table(df)
    # Referencia: 26/01 (día siguiente a la última fecha de los datos)
    assert table.loc["A", "recency"] == 6
    assert table.loc["A", "frequency"] == 2
    assert table.loc["A", "monetary"] == 25.0
    assert table.loc["B", "recency"] == 16
    assert table.loc["B", "frequency"] == 1


def test_customers_who_returned_everything_are_excluded():
    df = make_orders(
        [
            ("o1", "2025-01-01", "A", 1, 10.0),
            ("o2", "2025-01-02", "B", 1, 10.0),
            ("r1", "2025-01-03", "B", -1, 10.0),  # B devuelve todo: valor neto 0
            ("r2", "2025-01-03", "C", -1, 10.0),  # C solo tiene una devolución
        ]
    )
    assert rfm_table(df).index.tolist() == ["A"]


def test_scores_and_segments(crowd):
    rfm = rfm_segments(crowd)
    assert len(rfm) == 100
    for column in ("r_score", "f_score", "m_score"):
        assert rfm[column].between(1, 5).all()
        assert rfm[column].value_counts().eq(20).all()  # quintiles: 20 clientes por puntuación
    assert set(rfm["segment"].astype(str)) <= set(SEGMENT_ORDER)
    assert rfm.loc["C099", "segment"] == "Campeones"
    assert rfm.loc["C000", "segment"] == "Inactivos"


def test_segment_summary_adds_up(crowd):
    summary = segment_summary(rfm_segments(crowd))
    assert summary["segment"].tolist() == SEGMENT_ORDER
    assert summary["customers"].sum() == 100
    assert summary["customers_share"].sum() == pytest.approx(1.0)
    assert summary["sales_share"].sum() == pytest.approx(1.0)
    assert summary["action"].notna().all()


def test_too_few_customers_raises():
    df = make_orders([("o1", "2025-01-01", "A", 1, 10.0), ("o2", "2025-01-02", "B", 1, 10.0)])
    with pytest.raises(ValueError, match="al menos"):
        rfm_segments(df)


def test_kmeans_groups_are_ordered_by_value_and_reproducible(crowd):
    rfm = rfm_segments(crowd)
    result, profile, silhouette = kmeans_groups(rfm, k=3)

    assert sorted(result["group"].unique()) == [1, 2, 3]
    assert profile["customers"].sum() == 100
    assert profile["monetary"].is_monotonic_decreasing  # el grupo 1 es el que más gasta
    assert -1 <= silhouette <= 1

    again, _, _ = kmeans_groups(rfm, k=3)
    assert again["group"].tolist() == result["group"].tolist()


def test_kmeans_validates_k(crowd):
    rfm = rfm_segments(crowd)
    with pytest.raises(ValueError, match="entre 2 y 10"):
        kmeans_groups(rfm, k=1)