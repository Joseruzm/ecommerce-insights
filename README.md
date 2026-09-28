# Ecommerce Insights

Sube el fichero CSV de ventas de tu tienda online y obtén un dashboard, una segmentación de clientes
y una previsión de la demanda futura

**Demo en vivo:** (futuro enlace)

![Captura del dashboard](docs/screenshot.png)

## Problema resuelto
(Una o dos frases en lenguaje de negocio: qué decisiones ayuda a tomar.)

## Funcionalidades
- Dashboard de ventas, ticket medio y top productos
- Segmentación de clientes (RFM + K-Means)
- Previsión de ventas con backtesting frente a una línea base

## Stack
Python, pandas, Streamlit, Plotly, scikit-learn, statsmodels

## Ejecución en local
```bash
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py
```

## Datos y privacidad
Los archivos subidos se procesan en memoria y no se almacenan.


## Licencia 
MIT 