# Machine Learning Models for Stock Price Predictions
Applying Machine Learning models (Linear Regression, Random Forest, SVM, and Neural Networks) to predict stock returns using historical data from Yahoo Finance

# ML-Stock-Price-Prediction

Applying machine learning models (Linear Regression, Random Forest, SVM, and Neural Networks) to predict stock returns using historical data from Yahoo Finance.

## Overview

This project compares four machine learning models for predicting daily stock returns across five major tech stocks (AAPL, MSFT, GOOGL, AMZN, TSLA) using historical data from January 2019 to December 2023. The S&P 500 (^GSPC) is used as a market benchmark.

## Models

- **Linear Regression (OLS)** — baseline model
- **Random Forest (RF)** — ensemble tree-based model
- **Support Vector Machine (SVM)** — with RBF and linear kernel tuning
- **Neural Network (NN)** — two hidden layers with dropout, built with Keras

## Features

The following features were engineered from raw price data:

- Daily and log returns
- Moving averages (10-day, 50-day, 200-day) and exponential moving averages
- Relative Strength Index (RSI)
- Rolling volatility (5-day, 10-day, 20-day)
- S&P 500 market index returns

## Results

Linear Regression outperformed all other models across all five stocks based on MSE, MAE, and R² metrics. Neural Network and SVM models produced negative R² values, indicating poor generalisation on this dataset.

| Model             | MSE (range)           | R² (range)          |
|-------------------|-----------------------|---------------------|
| Linear Regression | 0.000004 – 0.000029   | 0.954 – 0.981       |
| Random Forest     | 0.000057 – 0.000380   | 0.438 – 0.515       |
| Neural Network    | 0.001135 – 0.003269   | -16.12 – -1.57      |
| SVM               | 0.000116 – 0.000722   | -0.003 – -0.0000002 |

5-fold cross-validation was used to assess model stability and prevent overfitting.

## Project Structure

ml-stock-price-prediction/
├── 2400724_Assignment_2.ipynb   # Main Jupyter notebook
├── 2400724_Assignment_2.py      # Python script version
├── report.pdf                   # Full written report
└── README.md

## Requirements

pandas
numpy
scikit-learn
tensorflow / keras
yfinance
matplotlib

Install dependencies:

```bash
pip install pandas numpy scikit-learn tensorflow yfinance matplotlib

## Usage

Run the Jupyter notebook:

```bash
jupyter notebook .ipynb
```

Or run the Python script directly:

```bash
python .py

## Data

Historical daily OHLCV data was fetched from Yahoo Finance using `yfinance` for the period January 2019 – December 2023. Outliers were removed using the IQR method and features were standardised using Z-score normalisation.

## References

- Kumar et al. (2020) — Linear regression for high-frequency stock return prediction
- Livieris et al. (2020) — Random forest for outlier detection in financial time series
- Zhang & Wang (2022) — Adaptive kernel selection in SVM for financial forecasting
- Soni & Bhatia (2023) — Deep neural networks for stock market prediction
- Zhang et al. (2020) — Technical analysis and sentiment for stock prediction
