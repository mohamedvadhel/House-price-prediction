import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# Load data
csv_file_path = "filtered_dataa.csv"
df = pd.read_csv(csv_file_path).dropna()

# Preprocessing
categorical_features = ['salon', 'room', 'sittingArea', 'gatheringRoom', 'storageRoom', 'kitchen', 'publicBathroom', 'workersHouse']
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
df_encoded[df_encoded != 0] = 1
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp_model.fit(X_train_scaled, y_train)
joblib.dump(mlp_model, 'mlp_model.pkl')  # Save MLP model to a file

dl_model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
dl_model.compile(optimizer='adam', loss='mean_absolute_error')
dl_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
dl_model.save('dl_model.h5')  # Save Deep Learning model to a file

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
joblib.dump(lr_model, 'lr_model.pkl')  # Save Linear Regression model to a file

# Streamlit app
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS to style the app
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Arial', sans-serif;
    }
    .styled-button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        cursor: pointer;
        display: inline-block;
    }
    .styled-button:hover {
        background-color: #0056b3;
    }
    .title {
        font-size: 2.5rem;
        color: #333;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('House Price Prediction')

# Sidebar for feature inputs and model selection
st.sidebar.title('Enter Feature Values')
features = [st.sidebar.number_input(feature, value=0, step=1, key=feature) for feature in X.columns]

selected_models = st.sidebar.multiselect('Select models for prediction:', ['MLP Regressor', 'Linear Regression'])

if st.sidebar.button('Predict', key='predict_button'):
    st.subheader('Predicted Prices:')
    for model_name, model in [('MLP Regressor', mlp_model), ('Linear Regression', lr_model)]:
        if model_name in selected_models:
            X_input = scaler.transform([features])
            y_pred = model.predict(X_input)
            st.write(f'{model_name}: {y_pred[0]:.2f}')
