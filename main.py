import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Генерація або завантаження даних (штучні для прикладу)
def generate_data():
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    df = pd.DataFrame({"date": dates})
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = df["date"].dt.weekday >= 5
    df["energy_consumption"] = (
        50
        + 30 * np.sin(2 * np.pi * df["day_of_year"] / 365)  # сезонність
        - 10 * df["is_weekend"].astype(int)  # менше у вихідні
        + np.random.normal(0, 5, len(df))  # шум
    )
    return df 

df = generate_data()

# 2. Підготовка даних
X = df[["day_of_year", "is_weekend"]].values
y = df["energy_consumption"].values

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 3. Побудова простої нейромережі
model = Sequential([
    Dense(16, activation='relu', input_shape=(2,)),
    Dense(8, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, verbose=0)

# 4. Оцінка
loss = model.evaluate(X_test, y_test)
print(f"MSE на тестових даних: {loss:.4f}")

# 5. Прогнозування для конкретного дня
def predict_energy(day_of_year, is_weekend):
    X_input = scaler_X.transform([[day_of_year, is_weekend]])
    prediction_scaled = model.predict(X_input)
    prediction = scaler_y.inverse_transform(prediction_scaled)
    return prediction[0][0]

examples = [
    ("15 січня", 15), 
    ("15 липня", 196), 
    ("25 грудня", 360)
]

for label, day in examples:
    val = predict_energy(day, int(datetime(2024, 1, 1).replace(day=day).weekday() >= 5))
    print(f"Прогноз на {label}: {val:.2f} кВт⋅год")

# 6. Графік
y_pred = model.predict(X_scaled)
y_pred_inv = scaler_y.inverse_transform(y_pred)

plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["energy_consumption"], label="Реальне")
plt.plot(df["date"], y_pred_inv, label="Прогноз", alpha=0.7)
plt.legend()
plt.xlabel("Дата")
plt.ylabel("Споживання енергії (кВт⋅год)")
plt.title("Прогноз споживання енергії нейронною мережею")
plt.tight_layout()
plt.show()
