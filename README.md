# Modular Retail Demand Forecasting Pipeline

A professional time-series forecasting pipeline built in Python. This project compares a classical statistical approach (**SARIMA**) against a machine learning regressor (**Random Forest**) using the real-world **Kaggle Store Item Demand Forecasting** dataset. 

The pipeline is built with a modular software engineering structure, separating data ingestion, feature engineering, and model execution.

---
## Project Structure
This project is organized into self-contained modules to ensure maintainability, testability, and clean code separation:

```text
demand-forecasting-project/
│
├── .gitignore
├── requirements.txt
├── README.md
├── main.py                  # Project orchestrator
│
├── data/                    # Local storage for datasets (Git-ignored)
│   ├── train.csv
│   └── test.csv
│
└── src/                     # Core pipeline modules
    ├── __init__.py          # Marks src as a package
    ├── data_prep.py         # Data loading and feature engineering
    └── models.py            # Model definition and metrics calculation
```
## Key Features

Modular Architecture: Business logic is decoupled from execution scripts. Adding new data sources or modeling techniques requires no changes to the main runner.

Feature Engineering: Transforms the sequential time-series into a supervised learning problem for the Random Forest model using:
- Lag Features: 1, 7, and 30-day past sales values.
- Rolling Statistics: 7-day rolling mean and standard deviation to capture local trends.
- Temporal Features: Extraction of day of the week, month, year, and a weekend binary indicator.

Rigid Temporal Validation: Splitting is based strictly on a temporal cutoff (holding out the last 30 days) to prevent lookahead bias/data leakage during testing.

Statistical Diagnostics: Integrates the Augmented Dickey-Fuller (ADF) test to evaluate series stationarity before fitting statistical models.

##Prerequisites & Environment Setup

**Create and activate environment**

conda create --name demand_env python=3.10 -y

conda activate demand_env

**Install dependencies***

pip install -r requirements.txt

**Execute code**

python main.py



## Key Takeaways & Visual Analysis
Evaluating on granular, item-level retail data revealed critical differences in model adaptability:

- SARIMA’s Trend-Projection Bias: The classical SARIMA model consistently overpredicted demand throughout December 2017. Because SARIMA relies heavily on historical mathematical averages, it projected a long-term upward trend into the future and was unable to adapt to the lower actual volume of late 2017.
- Random Forest’s Adaptability: The Random Forest model closely tracked both the scale and the weekly rhythm of the actual sales. By leveraging localized features like the 1-day lag and the 7-day rolling average, it adapted immediately to recent sales volumes, keeping its predictions anchored to the real demand curve.
