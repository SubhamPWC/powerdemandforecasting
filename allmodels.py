import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prophet import Prophet
import torch
import torch.nn as nn
from datetime import timedelta
import io

st.set_page_config(layout="wide")
st.title("🔮 Power Demand Forecast with Weather & Calendar Features")

# 📤 Upload Files
demand_file = st.file_uploader("Upload Demand File", type=["xlsx"])
weather_file = st.file_uploader("Upload Weather File", type=["xlsx"])
calendar_file = st.file_uploader("Upload Calendar File", type=["xlsx"])

# 🔧 Model Selection
model_list = ["RandomForest", "XGBoost", "SVR", "RNN", "LSTM", "GRU", "TCN", "Transformer", "Prophet"]
selected_model = st.sidebar.selectbox("Choose Forecasting Model", model_list)

if demand_file and weather_file and calendar_file:
    # 📊 Load Data
    demand_df = pd.read_excel(demand_file)
    weather_df = pd.read_excel(weather_file)
    calendar_df = pd.read_excel(calendar_file)

    # 🧩 Preprocess & Merge
    demand_df['Date'] = pd.to_datetime(demand_df['Date'])
    demand_df['Hour'] = pd.to_datetime(demand_df['Hour'], format='%H:%M:%S').dt.hour
    demand_df['Datetime'] = demand_df['Date'] + pd.to_timedelta(demand_df['Hour'], unit='h')
    weather_df['Datetime'] = pd.to_datetime(weather_df['Date'].astype(str) + ' ' + weather_df['Time'].astype(str))
    calendar_df['Date'] = pd.to_datetime(calendar_df['Date'], errors='coerce')

    full_df = demand_df.merge(weather_df, on=['State', 'Datetime'], how='left')
    full_df['Date'] = full_df['Datetime'].dt.date
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    full_df = full_df.merge(calendar_df, on=['State', 'Date'], how='left')

    # 🧠 Feature Engineering
    def generate_features(df):
        df = df.sort_values('Datetime')
        df['Hour'] = df['Datetime'].dt.hour
        df['DayOfWeek'] = df['Datetime'].dt.dayofweek
        df['Month'] = df['Datetime'].dt.month
        df['Lag_1'] = df['Demand'].shift(1)
        df['Lag_24'] = df['Demand'].shift(24)
        df['RollingMean_3'] = df['Demand'].rolling(3).mean()
        df['RollingStd_3'] = df['Demand'].rolling(3).std()

        for col in ['Temperature_2m', 'DNI', 'Relative_humidity_2m', 'Dew_point_2m',
                    'Apparent_temperature', 'Rain', 'Cloud_cover']:
            if col in df.columns:
                df[f'{col}_Lag1'] = df[col].shift(1)
                df[f'{col}_Diff'] = df[col].diff()

        df = df.fillna(method='ffill').fillna(method='bfill')
        return df.dropna(subset=['Demand'])

    feature_df = generate_features(full_df)

    # 📍 Select State
    state = st.selectbox("Select State", feature_df['State'].unique())
    state_df = feature_df[feature_df['State'] == state]

    # 📈 Forecasting Setup
    exclude_cols = ['State', 'Datetime', 'Demand']
    X = state_df.drop(columns=exclude_cols, errors='ignore').select_dtypes(include=[np.number])
    y = state_df['Demand']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]

    # 🔮 Model Training
    if selected_model == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        forecast_test = model.predict(X_test)
    elif selected_model == "XGBoost":
        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        forecast_test = model.predict(X_test)
    elif selected_model == "SVR":
        model = SVR(kernel='rbf')
        model.fit(X_train, y_train)
        forecast_test = model.predict(X_test)
    elif selected_model == "Prophet":
        prophet_df = state_df[['Datetime', 'Demand']].rename(columns={'Datetime': 'ds', 'Demand': 'y'})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=245 * 24, freq='H')
        forecast_df = model.predict(future)
        forecast_test = forecast_df['yhat'][-len(y_test):].values
    elif selected_model in ["RNN", "LSTM", "GRU", "TCN", "Transformer"]:
        class TimeSeriesModel(nn.Module):
            def __init__(self, model_type):
                super().__init__()
                if model_type == "LSTM":
                    self.rnn = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
                elif model_type == "GRU":
                    self.rnn = nn.GRU(input_size=1, hidden_size=50, batch_first=True)
                elif model_type == "RNN":
                    self.rnn = nn.RNN(input_size=1, hidden_size=50, batch_first=True)
                else:
                    self.rnn = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
                self.fc = nn.Linear(50, 1)

            def forward(self, x):
                out, _ = self.rnn(x)
                out = out[:, -1, :]
                return self.fc(out)

        def create_sequences(data, window=5):
            X, y = [], []
            for i in range(window, len(data)):
                X.append(data[i-window:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        window = 5
        scaled_series = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        X_seq, y_seq = create_sequences(scaled_series, window)
        split = int(len(X_seq) * 0.8)
        X_train_seq = torch.tensor(X_seq[:split], dtype=torch.float32).unsqueeze(-1)
        y_train_seq = torch.tensor(y_seq[:split], dtype=torch.float32).unsqueeze(-1)
        X_test_seq = torch.tensor(X_seq[split:], dtype=torch.float32).unsqueeze(-1)

        model = TimeSeriesModel(selected_model)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_seq)
            loss = criterion(output, y_train_seq)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            forecast_scaled = model(X_test_seq).squeeze().numpy()
        forecast_test = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

    # 📅 Future Forecast
    last_datetime = state_df['Datetime'].max()
    future_dates = [last_datetime + timedelta(hours=i) for i in range(1, 245 * 24 + 1)]
    future_df = pd.DataFrame({'Datetime': future_dates})
    future_df['State'] = state
    future_df['Hour'] = future_df['Datetime'].dt.hour
    future_df['DayOfWeek'] = future_df['Datetime'].dt.dayofweek
    future_df['Month'] = future_df['Datetime'].dt.month

    for col in X.columns:
        future_df[col] = X[col].mean()

    if selected_model in ["RandomForest", "XGBoost", "SVR"]:
        future_scaled = scaler.transform(future_df[X.columns])
        future_forecast = model.predict(future_scaled)
    elif selected_model == "Prophet":
        future_forecast = forecast_df['yhat'][-len(future_df):].values
    elif selected_model in ["RNN", "LSTM", "GRU", "TCN", "Transformer"]:
        future_scaled = scaler.transform(future_df[X.columns])
        future_seq = []
        for i in range(len(future_scaled) - window):
            future_seq.append(future_scaled[i:i+window])
        future_seq_tensor = torch.tensor(future_seq, dtype=torch.float32).unsqueeze(-1)
        with torch.no_grad():
            future_scaled_pred = model(future_seq_tensor).squeeze().numpy()
        future_forecast = scaler.inverse_transform(future_scaled_pred.reshape(-1, 1)).flatten()
    else:
        future_forecast = np.full(len(future_df), y.mean())

    future_df['Forecasted_Demand'] = future_forecast[:len(future_df)]

    # 🕰️ Historical Forecast DataFrame
    historical_df = state_df.iloc[split:].copy()
    historical_df = historical_df[['Datetime', 'State']].copy()
    historical_df['Forecasted_Demand'] = forecast_test[:len(historical_df)]

    # 📦 Combine Historical + Future
    combined_df = pd.concat([historical_df, future_df[['Datetime', 'State', 'Forecasted_Demand']]], ignore_index=True)

    # 📉 Full Forecast Chart
    st.subheader("📈 Full Forecast (Historical + Future)")
    st.line_chart(combined_df.set_index('Datetime')['Forecasted_Demand'])

    # 📁 Export Full CSV
    st.subheader("📁 Export Full Forecast CSV")
    csv_buffer = io.StringIO()
    combined_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Full Forecast CSV",
        data=csv_buffer.getvalue(),
        file_name=f"{state}_full_forecast.csv",
        mime="text/csv"
    )

    # 📊 Metrics
    st.subheader("📊 Model Performance on Historical Test Set")
    min_len = min(len(y_test), len(forecast_test))
    y_test = y_test[:min_len]
    forecast_test = forecast_test[:min_len]
    rmse = np.sqrt(mean_squared_error(y_test, forecast_test))
    mae = mean_absolute_error(y_test, forecast_test)
    r2 = r2_score(y_test, forecast_test)

    st.write(f"**RMSE**: {rmse:.2f}")
    st.write(f"**MAE**: {mae:.2f}")
    st.write(f"**R² Score**: {r2:.2f}")
