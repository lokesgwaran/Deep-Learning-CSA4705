# AIR WATCH CHENNAI
**Air Quality Forecasting and Pollution Source Analysis**

## Windows quick run
1. Extract the ZIP.
2. Install Python 3.10+ (enable Add Python to PATH).
3. Double-click `RUN_APP.bat` to install dependencies and launch the app.

## Dashboard modules
- Module 1: Air Quality Forecasting and Data Analysis (pollutant trends, monthly averages, chronological 80/20 split, rolling 7-day baseline, MAE/RMSE).
- Module 2: Pollution Source Analysis and Visualization (synthetic traffic/industry/dust proxies and pollutant correlation heatmap).
- Upload CSVs with Date and PM2.5; pollutant columns PM10, NO2, SO2, CO and O3 optional. Proxy columns are optional.

## Data warning
Included dataset is SYNTHETIC, for demonstration/testing only. Source proxy values are simulated and are not actual emissions apportionment. Replace with valid, cited real observations (e.g., CPCB/TNPCB data where access/usage terms permit) before claiming empirical findings.

## Results warning
MAE and RMSE are calculated from the selected dataset and the chronological holdout split in this application. They are not precomputed claims. The baseline is a simple 7-day rolling mean, not an N-BEATS model.
