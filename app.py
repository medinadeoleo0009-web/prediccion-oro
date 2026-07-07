
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression

# Descargar datos del oro desde 2015 hasta 2026
gold = yf.download("GC=F", start="2015-01-01", end="2026-01-01")

# Paso 3: Crear indicadores técnicos básicos
gold['MA50'] = gold['Close'].rolling(50).mean()

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

gold['RSI'] = compute_rsi(gold['Close'])

# Paso 4: Preparar dataset para predicción
gold['Target'] = gold['Close'].shift(-1)  # precio del día siguiente
gold = gold.dropna()  # eliminar filas con valores nulos

X = gold[['Close', 'MA50', 'RSI']]
y = gold['Target']

# Paso 5: Entrenar modelo sencillo (Regresión Lineal)
model = LinearRegression()
model.fit(X, y)

# Predicción para el último día disponible
last_row = X.iloc[-1].values.reshape(1, -1)
prediction = model.predict(last_row)[0]

print("Precio actual:", gold['Close'].iloc[-1])
print("Predicción para mañana:", prediction)

