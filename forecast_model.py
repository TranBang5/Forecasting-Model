import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from scipy.stats import shapiro
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import statsmodels
import warnings
warnings.filterwarnings("ignore")

# Loading data from all sheets in the Excel file
file_path = "vinamilk.xlsx"  # Adjust this path to your local file location
try:
    xl = pd.ExcelFile(file_path, engine='openpyxl')
    sheets = xl.sheet_names
    print("Sheet names:", sheets)
except Exception as e:
    print(f"Error loading vinamilk.xlsx: {e}")
    raise

# Translating Vietnamese metric names to English
translation_map = {
    'TỔNG CỘNG TÀI SẢN': 'Total Assets',
    '3. Doanh thu thuần về bán hàng và cung cấp dịch vụ (10 = 01 - 02)': 'Net Revenue',
    '18. Lợi nhuận sau thuế thu nhập doanh nghiệp(60=50-51-52)': 'Profit After Tax',
}

metrics_vn = list(translation_map.keys())
metrics_en = ['Total Assets', 'Net Revenue', 'Profit After Tax']

# Initialize combined DataFrame
combined_data = pd.DataFrame()

# Process each sheet
for sheet in sheets:
    try:
        df = pd.read_excel(file_path, sheet_name=sheet, engine='openpyxl')
        print(f"\nProcessing sheet: {sheet}")
        print("Original columns:", df.columns.tolist())
        print("First few rows:\n", df.head())

        if 'Unnamed: 0' in df.columns:
            df = df[df['Unnamed: 0'].isin(metrics_vn)].copy()
            print(f"Filtered rows in {sheet}:\n", df)

            df['Unnamed: 0'] = df['Unnamed: 0'].map(lambda x: translation_map.get(str(x).strip(), x))
            df.set_index('Unnamed: 0', inplace=True)
            df = df.T
            df.index = pd.to_datetime(df.index.astype(str), format='%Y')
            df = df.apply(pd.to_numeric, errors='coerce')

            if not combined_data.empty:
                combined_data = combined_data.join(df, how='outer')
            else:
                combined_data = df
    except Exception as e:
        print(f"Error processing sheet {sheet}: {e}")
        continue

print("\nCombined DataFrame:\n", combined_data)
print("Combined DataFrame columns:", combined_data.columns.tolist())
print("Any NaN in Combined DataFrame:\n", combined_data.isna().any())

# Fill NaN and scale to billions VND
combined_data = combined_data.fillna(0)
combined_data[metrics_en] = combined_data[metrics_en] / 1_000_000_000

# Verify metrics
available_metrics = [m for m in metrics_en if m in combined_data.columns]
if not available_metrics:
    print("Warning: No metrics found in columns:", combined_data.columns.tolist())
    raise KeyError("Required metrics not found")
print("Available metrics:", available_metrics)

# Initialize results
accuracy_results = {}
forecast_results = {}

# Anchor tests (only 2022)
anchor_tests = [
    {'anchor': 2022, 'train_years': [2020, 2021, 2022], 'test_years': [2023, 2024]}
]

for test in anchor_tests:
    anchor = test['anchor']
    train_years = [pd.to_datetime(str(year)) for year in test['train_years']]
    test_years = [pd.to_datetime(str(year)) for year in test['test_years']]
    
    print(f"\nAnchor {anchor}: Train years {train_years}, Test years {test_years}")
    
    accuracy_results[anchor] = {}
    
    for metric in ['Total Assets', 'Net Revenue', 'Profit After Tax']:
        series = combined_data[metric].dropna()
        
        valid_train_years = [y for y in train_years if y in series.index]
        valid_test_years = [y for y in test_years if y in series.index]
        if len(valid_train_years) != len(train_years):
            print(f"Skipping {metric} for anchor {anchor}: Missing train years")
            continue
        if len(valid_test_years) != len(test_years):
            print(f"Skipping {metric} for anchor {anchor}: Missing test years")
            continue
        
        train = series[valid_train_years]
        test = series[valid_test_years]
        
        print(f"{metric} - Train data:\n{train}\nAny NaN in train: {train.isna().any()}")
        print(f"{metric} - Test data:\n{test}\nAny NaN in test: {test.isna().any()}")
        
        if len(train) < 2:
            print(f"Skipping {metric} for anchor {anchor}: Insufficient data")
            continue
        if train.std() == 0:
            print(f"Skipping {metric} for anchor {anchor}: No variation")
            continue
        
        orders = [(1, 0, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]
        model_fit = None
        for order in orders:
            try:
                trend = None if order[1] > 0 else 'c'
                model = ARIMA(train, order=order, trend=trend)
                model_fit = model.fit()
                print(f"ARIMA{order} with trend='{trend}' succeeded for {metric}")
                break
            except Exception as e:
                print(f"ARIMA{order} with trend='{trend}' failed for {metric}: {e}")
                continue
        
        if model_fit is None:
            print(f"Skipping {metric} for anchor {anchor}: All ARIMA orders failed")
            continue
        
        pred = model_fit.forecast(steps=len(valid_test_years))
        
        print(f"{metric} - Predictions:\n{pred}\nAny NaN in pred: {np.isnan(pred).any()}")
        
        if np.isnan(pred).any() or test.isna().any():
            print(f"Skipping {metric} for anchor {anchor}: NaN in predictions")
            continue
        
        mae = mean_absolute_error(test, pred)
        mse = mean_squared_error(test, pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test - pred) / test)) * 100 if not np.any(test == 0) else float('inf')
        accuracy_results[anchor][metric] = {
            'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape,
            'predictions': pred, 'test_years': valid_test_years, 'actuals': test
        }

# Forecasting 2025-2027
for metric in ['Total Assets', 'Net Revenue', 'Profit After Tax']:
    series = combined_data[metric].dropna()
    
    if len(series) < 2:
        print(f"Skipping forecast for {metric}: Insufficient data")
        continue
    if series.std() == 0:
        print(f"Skipping forecast for {metric}: No variation")
        continue
    
    try:
        if metric == 'Profit After Tax':
            print(f"{metric} - Data:\n{series}")
            print(f"{metric} - Data variance: {series.var():,.2f}, std: {series.std():,.2f}")
            model = auto_arima(series, seasonal=False, trend='t', max_p=2, max_q=2, max_d=1, information_criterion='bic')
            forecast = model.predict(n_periods=3)
            ci = model.predict(n_periods=3, return_conf_int=True)[1]
            lower_ci = ci[:, 0]
            upper_ci = ci[:, 1]
            # Check if forecast is flat
            if np.allclose(forecast, forecast[0], rtol=1e-5):
                print(f"{metric} - auto_arima produced flat forecast, switching to ARIMA(1,0,0)")
                model = ARIMA(series, order=(1,0,0), trend='t')
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=3)
                # Bootstrap CI for ARIMA
                n_bootstraps = 1000
                forecasts_bootstrap = []
                residuals = model_fit.resid
                for _ in range(n_bootstraps):
                    noise = np.random.choice(residuals, size=len(series))
                    sim_series = series + noise
                    try:
                        model_sim = ARIMA(sim_series, order=(1,0,0), trend='t')
                        model_sim_fit = model_sim.fit()
                        sim_forecast = model_sim_fit.forecast(steps=3)
                        forecasts_bootstrap.append(sim_forecast)
                    except:
                        continue
                if len(forecasts_bootstrap) > 100:
                    forecasts_bootstrap = np.array(forecasts_bootstrap)
                    lower_ci = np.percentile(forecasts_bootstrap, 2.5, axis=0)
                    upper_ci = np.percentile(forecasts_bootstrap, 97.5, axis=0)
                else:
                    print(f"Warning: Insufficient bootstraps for {metric}, using parametric CI")
                    std_error = np.std(residuals)
                    lower_ci = forecast - 1.96 * std_error
                    upper_ci = forecast + 1.96 * std_error
        else:
            model_full = ExponentialSmoothing(series, trend='add', seasonal=None, damped=True)
            model_full_fit = model_full.fit(optimized=True)
            forecast = model_full_fit.forecast(steps=3)
            
            residuals = series - model_full_fit.fittedvalues
            residuals = residuals[np.isfinite(residuals)]
            
            p = 0.0
            if len(residuals) > 3:
                stat, p = shapiro(residuals)
                print(f"{metric} - Shapiro-Wilk: p-value = {p}")
                if p < 0.05:
                    print(f"{metric} - Residuals not normal, using bootstrap")
            
            if metric == 'Total Assets':
                print(f"{metric} - Using bootstrap CI")
                n_bootstraps = 5000
                forecasts_bootstrap = []
                for _ in range(n_bootstraps):
                    noise = np.clip(np.random.choice(residuals, size=len(series)), -0.3 * np.std(residuals), 0.3 * np.std(residuals))
                    sim_series = np.maximum(series + noise, 0)
                    try:
                        model_sim = ExponentialSmoothing(sim_series, trend='add', seasonal=None, damped=True)
                        model_sim_fit = model_sim.fit(optimized=True)
                        sim_forecast = model_sim.forecast(steps=3)
                        forecasts_bootstrap.append(sim_forecast)
                    except:
                        continue
                
                if len(forecasts_bootstrap) < 100:
                    print(f"Warning: Only {len(forecasts_bootstrap)} successful bootstraps for {metric}")
                    std_error = np.std(residuals)
                    lower_ci = forecast - 1.96 * std_error
                    upper_ci = forecast + 1.96 * std_error
                else:
                    forecasts_bootstrap = np.array(forecasts_bootstrap)
                    lower_ci = np.percentile(forecasts_bootstrap, 2.5, axis=0)
                    upper_ci = np.percentile(forecasts_bootstrap, 97.5, axis=0)
            else:
                std_error = np.std(residuals)
                lower_ci = forecast - 1.96 * std_error
                upper_ci = forecast + 1.96 * std_error
                print(f"{metric} - Using parametric CI")
        
        forecast_results[metric] = {
            'Forecast': forecast,
            'Lower CI': lower_ci,
            'Upper CI': upper_ci
        }
    except Exception as e:
        print(f"Error forecasting for {metric}: {e}")
        continue

# Plotting
for metric in ['Total Assets', 'Net Revenue', 'Profit After Tax']:
    if metric in forecast_results:
        plt.figure(figsize=(12, 6))
        series = combined_data[metric].dropna()
        plt.plot(series.index, series, 'b-', label='2020-2024 Actual')
        plt.plot(series.index, LinearRegression().fit(np.arange(len(series)).reshape(-1, 1), series).predict(np.arange(len(series)).reshape(-1, 1)), 'b--', label='Historical Trend')
        
        forecast_years = [pd.to_datetime(str(year)) for year in [2025, 2026, 2027]]
        forecast = forecast_results[metric]['Forecast']
        lower_ci = forecast_results[metric]['Lower CI']
        upper_ci = forecast_results[metric]['Upper CI']
        plt.plot(forecast_years, forecast, 'ro-', label='2025-2027 Forecast')
        plt.fill_between(forecast_years, lower_ci, upper_ci, color='r', alpha=0.3, label=f'95% CI (Width: {int(upper_ci[0]-lower_ci[0]):,} B VND)')
        for x, y in zip(forecast_years, forecast):
            plt.text(x, y, f'{y:,.0f}', ha='center', va='bottom')
        
        plt.title(f'{metric} Forecast for 2025-2027')
        plt.xlabel('Year')
        plt.ylabel(f'{metric} (Billions VND)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{metric.replace(" ", "_")}_forecast.png')
        plt.close()

# Saving results
with open('forecast_accuracy_results.txt', 'w', encoding='utf-8') as f:
    f.write('Vinamilk Financial Forecast and Accuracy\n')
    f.write('==============================================\n')
    f.write('Accuracy Metrics (Test Predictions vs Actual, in Billions VND)\n')
    f.write('----------------------------------------------------------\n')
    for anchor in accuracy_results:
        if not accuracy_results[anchor]:
            f.write(f'Anchor Year: {anchor}\nNo results due to insufficient data\n\n')
            continue
        test_config = next(t for t in anchor_tests if t['anchor'] == anchor)
        f.write(f'Anchor Year: {anchor}\n')
        f.write(f'Training Years: {", ".join(map(str, test_config["train_years"]))}\n')
        f.write(f'Test Years: {", ".join(map(str, test_config["test_years"]))}\n')
        for metric, metrics_dict in accuracy_results[anchor].items():
            f.write(f'{metric}:\n')
            f.write(f'  MAE: {metrics_dict["MAE"]:,.2f} Billions VND\n')
            f.write(f'  MSE: {metrics_dict["MSE"]:,.2f} Billions VND^2\n')
            f.write(f'  RMSE: {metrics_dict["RMSE"]:,.2f} Billions VND\n')
            f.write(f'  MAPE: {metrics_dict["MAPE"]:,.2f}%' if metrics_dict["MAPE"] != float('inf') else '  MAPE: Undefined\n')
            f.write('  Actual vs Predicted:\n')
            for year, actual, pred in zip(metrics_dict["test_years"], metrics_dict["actuals"], metrics_dict["predictions"]):
                f.write(f'    {year.year}: Actual = {actual:,.2f}, Predicted = {pred:,.2f} Billions VND\n')
            f.write('\n')
    f.write('Forecast for 2025-2027 (in Billions VND)\n')
    f.write('----------------------------------\n')
    for metric, forecast_dict in forecast_results.items():
        f.write(f'{metric}:\n')
        for year, val, lower, upper in zip(
            [2025, 2026, 2027],
            forecast_dict['Forecast'],
            forecast_dict['Lower CI'],
            forecast_dict['Upper CI']
        ):
            f.write(f'  {year}:\n')
            f.write(f'    Forecast: {val:,.2f} Billions VND\n')
            f.write(f'    95% Confidence Interval: [{lower:,.2f}, {upper:,.2f}] Billions VND\n')
