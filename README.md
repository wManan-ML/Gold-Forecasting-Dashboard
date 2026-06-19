# Gold Forecasting Dashboard

A Streamlit-based dashboard for forecasting gold prices using Monte Carlo simulation and ARIMA benchmarking on historical GLD market data.

## Overview

This project explores the application of probabilistic forecasting techniques to gold price prediction. Historical GLD ETF data is analyzed to estimate market returns and volatility, which are then used to generate future price scenarios through Monte Carlo simulation. Forecast performance is evaluated through multi-year backtesting and compared against a traditional ARIMA time-series model.

## Features

* Interactive Streamlit dashboard
* Historical gold price visualization
* Monte Carlo simulation-based forecasting
* Confidence interval estimation
* Multi-year backtesting framework
* ARIMA benchmark comparison
* Forecast uncertainty analysis

## Technologies Used

* Python
* Pandas
* NumPy
* Streamlit
* Plotly
* yfinance
* SciPy
* Statsmodels

## Methodology

The forecasting engine uses historical GLD market data to calculate average returns and volatility. These statistics are used to simulate thousands of potential future price paths through Monte Carlo methods. Rather than producing a single deterministic prediction, the model evaluates a distribution of possible outcomes, providing insight into both expected returns and forecast uncertainty.

## Results

* Average forecasting error: 10.84%
* Median forecasting error: 8.32%
* Multi-year backtesting performed on historical data
* Forecast uncertainty quantified through probabilistic simulations
* ARIMA model used as a benchmark for comparison

## Run Locally

```bash
pip install pandas numpy yfinance matplotlib plotly streamlit scipy statsmodels

streamlit run app.py
```

## Future Improvements

* Additional forecasting models (LSTM, Prophet, XGBoost)
* Automated model selection
* Portfolio and risk analysis features
* Enhanced visualization and scenario analysis tools
