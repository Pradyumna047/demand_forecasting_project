import pandas as pd
from statsmodels.tsa.stattools import adfuller

def load_kaggle_data(filepath="data/train.csv", store_id=1, item_id=1):
    """
    Loads real Kaggle data, filters for a single store/item combination,
    and formats the index as a datetime series.
    """
    # Load the training dataset
    df = pd.read_csv(filepath)
    
    # Filter for a specific store and item to isolate a single time series
    filtered_df = df[(df["store"] == store_id) & (df["item"] == item_id)].copy()
    
    # Format dates
    filtered_df["date"] = pd.to_datetime(filtered_df["date"])
    filtered_df.set_index("date", inplace=True)
    
    # Rename 'sales' to 'Demand' to maintain compatibility with downstream models
    filtered_df.rename(columns={"sales": "Demand"}, inplace=True)
    
    # Keep only the target column
    final_df = filtered_df[["Demand"]]
    
    return final_df

def run_stationarity_test(series):
    """Performs Augmented Dickey-Fuller test on a series."""
    result = adfuller(series)
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Stationary": result[1] < 0.05
    }

def engineer_ml_features(df, test_days=30):
    """Creates time series lag and rolling features for ML models."""
    ml_df = df.copy()
    
    # Calendar features
    ml_df["dayofweek"] = ml_df.index.dayofweek
    ml_df["month"] = ml_df.index.month
    ml_df["year"] = ml_df.index.year
    ml_df["is_weekend"] = ml_df["dayofweek"].isin([5, 6]).astype(int)
    
    # Lag and rolling features
    ml_df["lag_1"] = ml_df["Demand"].shift(1)
    ml_df["lag_7"] = ml_df["Demand"].shift(7)
    ml_df["lag_30"] = ml_df["Demand"].shift(30)
    ml_df["rolling_mean_7"] = ml_df["Demand"].shift(1).rolling(window=7).mean()
    ml_df["rolling_std_7"] = ml_df["Demand"].shift(1).rolling(window=7).std()
    
    ml_df.dropna(inplace=True)
    
    # Split into train and test
    train_data = ml_df.iloc[:-test_days]
    test_data = ml_df.iloc[-test_days:]
    
    features = ["dayofweek", "month", "year", "is_weekend", "lag_1", "lag_7", "lag_30", "rolling_mean_7", "rolling_std_7"]
    target = "Demand"
    
    return train_data[features], train_data[target], test_data[features], test_data[target]