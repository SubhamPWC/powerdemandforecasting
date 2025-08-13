import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import io

st.set_page_config(layout="wide")
st.title("🔮 Transformer-Based Power Demand Forecast")

# 📤 Upload Excel
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # 🕒 Parse datetime
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = pd.to_datetime(df['time'],format='%H:%M:%S').dt.hour
    df['datetime'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')

    # 🧠 Time features
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if 7 <= x <= 10 or 17 <= x <= 20 else 0)
    df['weekend tag'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['holiday tag'] = 0

    # 🔠 Encode state
    le = LabelEncoder()
    df['state_encoded'] = le.fit_transform(df['state'])

    # ⏳ Lag and rolling features
    df['demand_lag1'] = df.groupby('state')['demand'].shift(1)
    df['demand_lag24'] = df.groupby('state')['demand'].shift(24)
    df['demand_rolling_mean_24'] = df.groupby('state')['demand'].rolling(window=24).mean().reset_index(level=0, drop=True)
    df['demand_rolling_std_24'] = df.groupby('state')['demand'].rolling(window=24).std().reset_index(level=0, drop=True)
    df['demand_ema_12'] = df.groupby('state')['demand'].transform(lambda x: x.ewm(span=12).mean())

    # 🧮 Interaction features
    df['temp_rain_interaction'] = df['temperature'] * df['rain']
    df['dni_peak_hour_interaction'] = df['DNI'] * df['is_peak_hour']

    df = df.dropna()

    # 🎯 Features and target
    features = ['temperature', 'rain', 'DNI', 'weekend tag', 'holiday tag',
                'hour', 'dayofweek', 'month', 'is_peak_hour',
                'state_encoded', 'demand_lag1', 'demand_lag24',
                'demand_rolling_mean_24', 'demand_rolling_std_24', 'demand_ema_12',
                'temp_rain_interaction', 'dni_peak_hour_interaction']
    target = 'demand'

    # 📊 Normalize
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    df[target] = scaler.fit_transform(df[[target]])

    # 🧱 Sequence creation
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, -1])
        return np.array(X), np.array(y)

    seq_length = 24
    data = df[features + [target]].values
    X, y = create_sequences(data, seq_length)

    # 🧠 Transformer model
    def build_transformer_model(seq_len, num_features):
        inputs = Input(shape=(seq_len, num_features))
        x = LayerNormalization()(inputs)
        x = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        x = Dropout(0.1)(x)
        x = LayerNormalization()(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        return Model(inputs, outputs)

    model = build_transformer_model(seq_length, len(features)+1)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1, verbose=0)

    # 📈 Historical predictions
    df['forecasted_demand'] = np.nan
    for i in range(seq_length, len(df)):
        seq = df[features + [target]].iloc[i-seq_length:i].values
        seq = np.expand_dims(seq, axis=0)
        df.loc[df.index[i], 'forecasted_demand'] = model.predict(seq, verbose=0)[0][0]

    df['date'] = df['datetime'].dt.strftime('%d-%m-%Y')
    df['time'] = df['datetime'].dt.strftime('%H:%M:%S')
    historical_df = df[['state', 'date', 'time', 'demand', 'forecasted_demand'] + features].copy()
    historical_df = historical_df.rename(columns={
        'demand': 'Actual Demand',
        'forecasted_demand': 'Forecasted Demand'
    })
    historical_df['Type'] = 'Historical'

    # 🔮 Future forecast
    future_forecasts = []
    for state in df['state'].unique():
        state_df = df[df['state'] == state].copy()
        last_sequence = state_df[features + ['forecasted_demand']].tail(seq_length).values

        for i in range(1200):  # ~50 days hourly
            input_seq = np.expand_dims(last_sequence, axis=0)
            next_pred = model.predict(input_seq, verbose=0)[0][0]

            next_dt = state_df.iloc[-1]['datetime'] + pd.Timedelta(hours=1)
            hour = next_dt.hour
            dayofweek = next_dt.dayofweek
            month = next_dt.month
            is_peak_hour = 1 if 7 <= hour <= 10 or 17 <= hour <= 20 else 0
            weekend_tag = 1 if dayofweek >= 5 else 0
            holiday_tag = 0

            temp = state_df.iloc[-1]['temperature']
            rain = state_df.iloc[-1]['rain']
            dni = state_df.iloc[-1]['DNI']
            state_encoded = state_df.iloc[-1]['state_encoded']

            lag1 = last_sequence[-1][-1]
            lag24 = last_sequence[-24][-1] if len(last_sequence) >= 24 else lag1
            rolling_mean_24 = last_sequence[-24:][:, -1].mean()
            rolling_std_24 = last_sequence[-24:][:, -1].std()
            ema_12 = np.mean(last_sequence[-12:][:, -1])
            temp_rain_interaction = temp * rain
            dni_peak_hour_interaction = dni * is_peak_hour

            next_row = np.array([
                temp, rain, dni, weekend_tag, holiday_tag,
                hour, dayofweek, month, is_peak_hour,
                state_encoded, lag1, lag24, rolling_mean_24, rolling_std_24, ema_12,
                temp_rain_interaction, dni_peak_hour_interaction, next_pred
            ])
            last_sequence = np.vstack([last_sequence[1:], next_row])

            future_forecasts.append({
                'state': state,
                'date': next_dt.strftime('%d-%m-%Y'),
                'time': next_dt.strftime('%H:%M:%S'),
                'Actual Demand': np.nan,
                'Forecasted Demand': next_pred,
                'Type': 'Forecast',
                'temperature': temp,
                'rain': rain,
                'DNI': dni,
                'weekend tag': weekend_tag,
                'holiday tag': holiday_tag,
                'hour': hour,
                'dayofweek': dayofweek,
                'month': month,
                'is_peak_hour': is_peak_hour,
                'state_encoded': state_encoded,
                'demand_lag1': lag1,
                'demand_lag24': lag24,
                'demand_rolling_mean_24': rolling_mean_24,
                'demand_rolling_std_24': rolling_std_24,
                'demand_ema_12': ema_12,
                'temp_rain_interaction': temp_rain_interaction,
                'dni_peak_hour_interaction': dni_peak_hour_interaction
            })

    # 🧾 Combine and export
    future_df = pd.DataFrame(future_forecasts)
    combined_df = pd.concat([historical_df, future_df], ignore_index=True)

    st.subheader("📈 Forecast Preview: Actual vs Forecasted Demand")

    # 🕒 Create datetime index
    combined_df['datetime'] = pd.to_datetime(combined_df['date'] + ' ' + combined_df['time'])
    combined_df = combined_df.sort_values('datetime')

    # 📊 Select relevant columns
    plot_df = combined_df.set_index('datetime')[['Actual Demand', 'Forecasted Demand']]

    # 📈 Plot both series
    st.line_chart(plot_df)


    # 📁 Export Combined Forecast CSV
    csv_buffer = io.StringIO()
    combined_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="📥 Download Combined Forecast CSV",
        data=csv_buffer.getvalue(),
        file_name="combined_actual_and_forecasted_demand_transformer.csv",
        mime="text/csv"
    )

    st.success("✅ Forecasting complete and CSV ready for download!")
