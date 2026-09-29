"""Previsión de ventas mensuales con backtesting.

Se prueban varios métodos, desde el más simple (la media de los últimos meses)
hasta Holt-Winters. Cada uno se evalúa "escondiendo" los últimos meses: se
entrena con lo anterior, se prevé y se compara con lo que pasó de verdad. Gana
el que menos se equivoca, y esa cifra de error es la que se enseña al usuario.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

MIN_MONTHS = 6          # mínimo de meses completos para prever algo
BACKTEST_HORIZON = 3    # cuántos meses se prevén en cada prueba
MAX_FOLDS = 4           # cuántas pruebas se hacen (moviendo el punto de corte)
MIN_TRAIN = 6           # meses mínimos con los que se entrena en una prueba
BASELINE = "naive3"
DEFAULT_INTERVAL_ERROR = 0.30  # rango orientativo si no se puede hacer backtest


# ------------------------------------------------------------------- métodos
def _naive3(train: pd.Series, h: int) -> np.ndarray:
    return np.repeat(train.iloc[-3:].mean(), h)


def _seasonal_naive(train: pd.Series, h: int) -> np.ndarray:
    start = len(train) - 12
    return train.iloc[start : start + h].to_numpy()


def _seasonal_growth(train: pd.Series, h: int, k: int = 3) -> np.ndarray:
    """Mismo mes del año anterior, multiplicado por el crecimiento de los últimos k meses."""
    last_year = train.iloc[-12 - k : -12].sum()
    growth = train.iloc[-k:].sum() / last_year if last_year > 0 else 1.0
    return _seasonal_naive(train, h) * float(np.clip(growth, 0.5, 1.5))


def _ets_damped(train: pd.Series, h: int) -> np.ndarray:
    model = ExponentialSmoothing(
        train.asfreq("MS"), trend="add", damped_trend=True, initialization_method="estimated"
    )
    return model.fit().forecast(h).to_numpy()


def _holt_winters(train: pd.Series, h: int) -> np.ndarray:
    model = ExponentialSmoothing(
        train.asfreq("MS"), seasonal="add", seasonal_periods=12, initialization_method="estimated"
    )
    return model.fit().forecast(h).to_numpy()


# clave: (nombre para el usuario, función, meses mínimos de entrenamiento)
# Van de más simple a más compleja: en caso de empate gana la primera.
MODELS = {
    "naive3": ("Media de los últimos 3 meses", _naive3, 3),
    "snaive": ("Mismo mes del año anterior", _seasonal_naive, 12),
    "ets": ("Suavizado exponencial con tendencia amortiguada", _ets_damped, MIN_TRAIN),
    "sgrowth": ("Mismo mes del año anterior con ajuste de crecimiento", _seasonal_growth, 15),
    "holt_winters": ("Holt-Winters (estacionalidad aditiva)", _holt_winters, 24),
}


def _predict(key: str, train: pd.Series, h: int) -> np.ndarray:
    _, function, _ = MODELS[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = np.asarray(function(train, h), dtype=float)
    if prediction.shape != (h,) or not np.isfinite(prediction).all():
        raise ValueError(f"El modelo {key} no ha devuelto una previsión válida.")
    return np.clip(prediction, 0, None)


# ------------------------------------------------------------------- resultado
@dataclass
class ForecastResult:
    history: pd.Series            # ventas netas de los meses completos
    forecast: pd.DataFrame        # month, forecast, lower, upper
    model: str
    model_label: str
    backtest: pd.DataFrame        # una fila por método: wape, bias, evaluated
    baseline_wape: float | None   # error de la media de los últimos 3 meses
    best_wape: float | None       # error del método elegido
    improvement: float | None     # reducción del error frente a la línea base
    interval_error: float         # error relativo usado para el rango probable
    last_test: pd.DataFrame       # month, actual, predicted: la última prueba
    notes: list[str] = field(default_factory=list)


def _wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.abs(actual).sum()
    return float(np.abs(actual - predicted).sum() / denominator) if denominator > 0 else float("nan")


def _backtest(y: pd.Series):
    """Devuelve (tabla de resultados, errores relativos por método, última prueba por método)."""
    n = len(y)
    last_origin = n - BACKTEST_HORIZON
    first_origin = max(MIN_TRAIN, last_origin - (MAX_FOLDS - 1))
    origins = list(range(first_origin, last_origin + 1))

    rows, relative_errors, last_tests = [], {}, {}
    for key, (label, _, min_train) in MODELS.items():
        row = {"model": key, "label": label, "wape": np.nan, "bias": np.nan, "evaluated": False}
        if origins and first_origin >= min_train:
            actual, predicted, months = [], [], []
            try:
                for origin in origins:
                    train = y.iloc[:origin]
                    test = y.iloc[origin : origin + BACKTEST_HORIZON]
                    prediction = _predict(key, train, BACKTEST_HORIZON)
                    actual += test.tolist()
                    predicted += prediction.tolist()
                    months += test.index.tolist()
                    if origin == last_origin:
                        last_tests[key] = pd.DataFrame(
                            {"month": test.index, "actual": test.to_numpy(), "predicted": prediction}
                        )
            except Exception:  # un método que falla simplemente no compite
                rows.append(row)
                continue
            a, p = np.array(actual), np.array(predicted)
            row.update(wape=_wape(a, p), bias=float(p.sum() / a.sum() - 1) if a.sum() else np.nan, evaluated=True)
            positive = a > 0
            relative_errors[key] = np.abs(a[positive] - p[positive]) / a[positive]
        rows.append(row)
    return pd.DataFrame(rows), relative_errors, last_tests


def forecast_sales(monthly: pd.DataFrame, horizon: int = 3) -> ForecastResult:
    """Prevé las ventas netas de los próximos 'horizon' meses.

    'monthly' es la tabla que devuelve metrics.monthly_sales(). Solo se usan los
    meses completos: un mes a medias haría creer que las ventas se desploman.
    """
    if not 1 <= horizon <= 12:
        raise ValueError("El horizonte debe estar entre 1 y 12 meses.")
    complete = monthly.loc[monthly["complete"]]
    y = complete.set_index("month")["sales"].astype(float).asfreq("MS").fillna(0.0)
    if len(y) < MIN_MONTHS:
        raise ValueError(
            f"Se necesitan al menos {MIN_MONTHS} meses completos para prever "
            f"(hay {len(y)})."
        )

    notes = []
    backtest, relative_errors, last_tests = _backtest(y)
    evaluated = backtest[backtest["evaluated"]]

    if evaluated.empty:
        chosen = BASELINE
        best_wape = baseline_wape = None
        notes.append("Hay pocos meses para evaluar el modelo: se usa la media de los últimos 3 meses.")
    else:
        best = evaluated.sort_values("wape", kind="stable").iloc[0]  # empate: el más simple
        chosen, best_wape = best["model"], float(best["wape"])
        base = evaluated[evaluated["model"] == BASELINE]["wape"]
        baseline_wape = float(base.iloc[0]) if len(base) else None

    skipped = backtest[~backtest["evaluated"] & (backtest["model"] != BASELINE)]
    if len(y) < 24:
        notes.append(
            "Con menos de 24 meses completos la estacionalidad (la forma de cada año) "
            "se ha visto solo una o dos veces: la previsión es orientativa."
        )
    if not skipped.empty:
        names = ", ".join(skipped["label"])
        notes.append(f"No se han podido evaluar por falta de historial: {names}.")

    try:
        values = _predict(chosen, y, horizon)
    except Exception:
        chosen = BASELINE
        values = _predict(BASELINE, y, horizon)
        notes.append("El método elegido falló con todos los datos y se usó la media de los últimos 3 meses.")

    errors = relative_errors.get(chosen)
    if errors is not None and len(errors):
        interval_error = float(np.quantile(errors, 0.8))
    else:
        interval_error = DEFAULT_INTERVAL_ERROR
        notes.append(f"Sin backtest, el rango probable es un ±{DEFAULT_INTERVAL_ERROR:.0%} orientativo.")

    months = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    forecast = pd.DataFrame(
        {
            "month": months,
            "forecast": values,
            "lower": np.clip(values * (1 - interval_error), 0, None),
            "upper": values * (1 + interval_error),
        }
    )

    improvement = None
    if best_wape is not None and baseline_wape:
        improvement = (baseline_wape - best_wape) / baseline_wape

    last_test = last_tests.get(chosen, pd.DataFrame(columns=["month", "actual", "predicted"]))
    return ForecastResult(
        history=y,
        forecast=forecast,
        model=chosen,
        model_label=MODELS[chosen][0],
        backtest=backtest,
        baseline_wape=baseline_wape,
        best_wape=best_wape,
        improvement=improvement,
        interval_error=interval_error,
        last_test=last_test,
        notes=notes,
    )