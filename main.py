import matplotlib.pyplot as plt
from src.data_prep import load_kaggle_data, run_stationarity_test, engineer_ml_features
from src.models import train_sarima, train_random_forest, calculate_metrics

def main():
    test_days = 30
    
    # 1. Load Real Kaggle Data for Store 1, Item 1
    print("Loading real Kaggle sales data...")
    try:
        df = load_kaggle_data("data/train.csv", store_id=1, item_id=1)
    except FileNotFoundError:
        print("Error: Could not find 'data/train.csv'. Please ensure your directory structure is correct.")
        return
    
    # 2. Check Stationarity
    adf_results = run_stationarity_test(df["Demand"])
    print(f"Stationarity Check -> ADF Statistic: {adf_results['ADF Statistic']:.4f}, p-value: {adf_results['p-value']:.4f}")
    if adf_results['Stationary']:
        print("The series is stationary.")
    else:
        print("The series is non-stationary. Models will adjust internally via differencing.")
    
    # 3. Fit SARIMA
    print("Fitting SARIMA model on real data...")
    sarima_train = df["Demand"].iloc[:-test_days]
    sarima_predictions = train_sarima(sarima_train, steps=test_days)
    
    # 4. Fit Random Forest
    print("Engineering ML features and training Random Forest on real data...")
    X_train, y_train, X_test, y_test = engineer_ml_features(df, test_days)
    rf_predictions = train_random_forest(X_train, y_train, X_test)
    
    # 5. Evaluate Performance
    actual_test_values = df["Demand"].iloc[-test_days:]
    sarima_metrics = calculate_metrics(actual_test_values, sarima_predictions)
    rf_metrics = calculate_metrics(actual_test_values, rf_predictions)
    
    print("\n--- Model Evaluation (Real Kaggle Data) ---")
    print(f"SARIMA Metrics:        MAE: {sarima_metrics['MAE']:.2f} | RMSE: {sarima_metrics['RMSE']:.2f} | MAPE: {sarima_metrics['MAPE (%)']:.2f}%")
    print(f"Random Forest Metrics: MAE: {rf_metrics['MAE']:.2f} | RMSE: {rf_metrics['RMSE']:.2f} | MAPE: {rf_metrics['MAPE (%)']:.2f}%")
    
    # 6. Plot Results
    plt.figure(figsize=(12, 6))
    plt.plot(actual_test_values.index, actual_test_values.values, label="Actual Demand", color="black", linewidth=2)
    plt.plot(actual_test_values.index, sarima_predictions, label="SARIMA Forecast", color="blue", linestyle="--")
    plt.plot(actual_test_values.index, rf_predictions, label="Random Forest Forecast", color="green", linestyle="-.")
    plt.title("Real Kaggle Demand Forecast Comparison (Store 1, Item 1)")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.grid(True)
    plt.savefig("forecast_comparison_real.png")
    print("\nPlot saved successfully as 'forecast_comparison_real.png'")
    plt.show()

if __name__ == "__main__":
    main()