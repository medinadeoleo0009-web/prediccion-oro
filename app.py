import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

st.title("Predicción del precio del oro")
st.write("Este dashboard muestra el precio actual del oro y una predicción para mañana usando regresión lineal.")

gold = yf.download("GC=F", start="2015-01-01", end="2026-01-01")
gold['MA50'] = gold['Close'].rolling(50).mean()

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

gold['RSI'] = compute_rsi(gold['Close'])
gold['Target'] = gold['Close'].shift(-1)
gold = gold.dropna()

X = gold[['Close', 'MA50', 'RSI']]
y = gold['Target']

model = LinearRegression()
model.fit(X, y)

last_row = X.iloc[-1].values.reshape(1, -1)
prediction = model.predict(last_row)[0]

st.metric("Precio actual", f"${gold['Close'].iloc[-1]:.2f}")
st.metric("Predicción para mañana", f"${prediction:.2f}")
