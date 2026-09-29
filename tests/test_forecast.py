import numpy as np
import pandas as pd
import pytest

from src.forecast import MIN_MONTHS, forecast_sales


def make_monthly(values, start="2022-01-01", incomplete_last=False) -> pd.DataFrame:
    months = pd.date_range(start, periods=len(values), freq="MS")
    complete = [True] * len(values)
    if incomplete_last:
        complete[-1] = False
    return pd.DataFrame({"month": months, "sales": values, "complete": complete})


def seasonal(n: int, trend: float = 0.0) -> list[float]:
    t = np.arange(n)
    return (100 + 30 * np.sin(2 * np.pi * t / 12) + trend * t).tolist()


def test_seasonal_series_picks_a_seasonal_model_and_beats_the_baseline():
    result = forecast_sales(make_monthly(seasonal(30, trend=0.5)), horizon=3)
    assert result.model in {"snaive", "sgrowth", "holt_winters"}
    assert result.best_wape < 0.05
    assert result.baseline_wape > 0.10           # la media de 3 meses no ve la estacionalidad
    assert result.improvement > 0.5


def test_forecast_frame_is_consistent():
    monthly = make_monthly(seasonal(24))
    result = forecast_sales(monthly, horizon=4)
    fc = result.forecast
    assert len(fc) == 4
    assert fc["month"].iloc[0] == monthly["month"].iloc[-1] + pd.offsets.MonthBegin(1)
    assert (fc["lower"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["upper"]).all()
    assert (fc["forecast"] >= 0).all()
    assert len(result.last_test) == 3


def test_incomplete_last_month_is_not_used():
    values = seasonal(24)
    with_partial = make_monthly(values + [5.0], incomplete_last=True)  # mes a medias, casi vacío
    result = forecast_sales(with_partial, horizon=3)
    assert len(result.history) == 24
    assert result.history.iloc[-1] == pytest.approx(values[-1])
    # La previsión empieza justo después del último mes COMPLETO.
    assert result.forecast["month"].iloc[0] == with_partial["month"].iloc[-2] + pd.offsets.MonthBegin(1)


def test_short_history_falls_back_to_simple_methods_and_says_so():
    result = forecast_sales(make_monthly([100, 110, 105, 120, 115, 125, 130, 128, 135, 140]), horizon=2)
    assert result.model in {"naive3", "ets"}
    evaluated = result.backtest.set_index("model")["evaluated"]
    assert not evaluated["snaive"] and not evaluated["holt_winters"]
    assert any("24 meses" in note for note in result.notes)


def test_without_enough_months_for_a_backtest_uses_baseline():
    result = forecast_sales(make_monthly([100, 110, 105, 120, 115, 125]), horizon=2)
    assert result.model == "naive3"
    assert result.best_wape is None and result.improvement is None
    assert result.interval_error == pytest.approx(0.30)


def test_holt_winters_is_only_evaluated_with_enough_history():
    short = forecast_sales(make_monthly(seasonal(24)))
    long = forecast_sales(make_monthly(seasonal(40, trend=0.3)))
    assert not short.backtest.set_index("model").loc["holt_winters", "evaluated"]
    assert long.backtest.set_index("model").loc["holt_winters", "evaluated"]


def test_too_few_months_raises():
    with pytest.raises(ValueError, match="al menos"):
        forecast_sales(make_monthly([1.0] * (MIN_MONTHS - 1)))


def test_horizon_is_validated():
    with pytest.raises(ValueError, match="entre 1 y 12"):
        forecast_sales(make_monthly(seasonal(24)), horizon=0)


def test_results_are_reproducible():
    monthly = make_monthly(seasonal(30, trend=0.5))
    a, b = forecast_sales(monthly), forecast_sales(monthly)
    assert a.model == b.model
    assert a.forecast["forecast"].tolist() == b.forecast["forecast"].tolist()