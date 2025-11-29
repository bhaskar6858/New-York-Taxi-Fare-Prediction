import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from geopy.distance import geodesic
import joblib
import os

# --------------------------- Helper Functions ---------------------------
def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return r2, mae, mse, rmse

def is_valid_location(lat, lon):
    return -90 <= lat <= 90 and -180 <= lon <= 180

def preprocess_data(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['pickup_year'] = df['pickup_datetime'].dt.year
    df['pickup_month'] = df['pickup_datetime'].dt.month
    df['pickup_day'] = df['pickup_datetime'].dt.day
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
    
    df = df[(df['pickup_longitude'] != 0) & (df['pickup_latitude'] != 0) &
            (df['dropoff_longitude'] != 0) & (df['dropoff_latitude'] != 0)]
    
    df['pickup_coords'] = list(zip(df['pickup_latitude'], df['pickup_longitude']))
    df['dropoff_coords'] = list(zip(df['dropoff_latitude'], df['dropoff_longitude']))
    
    df['distance'] = df.apply(
        lambda row: geodesic(row['pickup_coords'], row['dropoff_coords']).km
        if is_valid_location(row['pickup_latitude'], row['pickup_longitude']) and
           is_valid_location(row['dropoff_latitude'], row['dropoff_longitude'])
        else np.nan, axis=1
    )
    
    df = df.dropna(subset=['distance'])
    X = df.drop(columns=[col for col in ['fare_amount', 'pickup_datetime'] if col in df.columns])
    X = X.select_dtypes(include=[np.number])
    y = df['fare_amount']
    
    return X, y

def train_models(X_train, y_train):
    st.info("Training models... Please wait!")
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42)
    rf_model.fit(X_train, y_train)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    st.success("Models trained successfully!")
    return rf_model, lr_model, dt_model

def predict_fare(model, input_data):
    return model.predict(input_data)[0]

# --------------------------- Load Dataset ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", low_memory=False)
    return df

df = load_data()
X, y = preprocess_data(df)

# --------------------------- Sidebar ---------------------------
st.sidebar.image("background.jpg", use_container_width=True)

st.sidebar.markdown("Predict your taxi fare in seconds!")

with st.sidebar:
        st.write("")  # Creates some space

        st.sidebar.markdown("<br>", unsafe_allow_html=True) 
        
        # Use columns to push content down
        _, col = st.columns([1, 10])
        with col:
            st.markdown("""
            <style>
            .credits {
                font-size: 0.8em;
                color: #666;
                position: relative;
                top: 100px;
            }
            </style>
            <div class='credits'>
                Engineered by Bhaskar Singh<br>
                ©2025 All rights reserved
            </div>
            """, unsafe_allow_html=True)

# Model file paths
rf_path = "rf_model.joblib"
lr_path = "lr_model.joblib"
dt_path = "dt_model.joblib"

# --------------------------- Train or Load Models ---------------------------
retrain = st.sidebar.button("Retrain Models")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if retrain or not (os.path.exists(rf_path) and os.path.exists(lr_path) and os.path.exists(dt_path)):
    rf_model, lr_model, dt_model = train_models(X_train, y_train)
    joblib.dump(rf_model, rf_path)
    joblib.dump(lr_model, lr_path)
    joblib.dump(dt_model, dt_path)
else:
    rf_model = joblib.load(rf_path)
    lr_model = joblib.load(lr_path)
    dt_model = joblib.load(dt_path)
    st.success("Models loaded successfully!")

# --------------------------- Main Page ---------------------------
st.title("🚖 NYC Taxi Fare Prediction")

st.subheader("Enter Pickup & Dropoff Details")
with st.form(key="fare_form"):
    pickup_lat = st.number_input("Pickup Latitude", value=40.7128, step=0.0001)
    pickup_lon = st.number_input("Pickup Longitude", value=-74.0060, step=0.0001)
    dropoff_lat = st.number_input("Dropoff Latitude", value=40.7812, step=0.0001)
    dropoff_lon = st.number_input("Dropoff Longitude", value=-73.9665, step=0.0001)
    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)
    pickup_hour = st.number_input("Pickup Hour (0-23)", min_value=0, max_value=23, value=14)
    pickup_day = st.number_input("Pickup Day (1-31)", min_value=1, max_value=31, value=15)
    pickup_month = st.number_input("Pickup Month (1-12)", min_value=1, max_value=12, value=6)
    pickup_weekday = st.number_input("Pickup Weekday (0=Mon, 6=Sun)", min_value=0, max_value=6, value=4)
    pickup_year = st.number_input("Pickup Year", min_value=2009, max_value=2025, value=2020)
    submit_button = st.form_submit_button("Predict Fare")

if submit_button:
    input_df = pd.DataFrame({
        'pickup_latitude': [pickup_lat],
        'pickup_longitude': [pickup_lon],
        'dropoff_latitude': [dropoff_lat],
        'dropoff_longitude': [dropoff_lon],
        'passenger_count': [passenger_count],
        'pickup_hour': [pickup_hour],
        'pickup_day': [pickup_day],
        'pickup_month': [pickup_month],
        'pickup_weekday': [pickup_weekday],
        'pickup_year': [pickup_year],
        'distance': [geodesic((pickup_lat, pickup_lon), (dropoff_lat, dropoff_lon)).km]
    })
    input_df = input_df[X.columns]
    fare = predict_fare(rf_model, input_df)
    st.success(f"Predicted Fare: ${fare:.2f}")

# --------------------------- Evaluation Metrics ---------------------------
st.header("📊 Model Evaluation Metrics")
models = {"Random Forest": rf_model, "Linear Regression": lr_model, "Decision Tree": dt_model}
metrics_data = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    r2, mae, mse, rmse = evaluate_model(y_test, y_pred)
    metrics_data.append([name, r2, mae, mse, rmse])
metrics_df = pd.DataFrame(metrics_data, columns=["Model", "R2 Score", "MAE", "MSE", "RMSE"])
st.dataframe(metrics_df)

# --------------------------- Sample Dataset ---------------------------
st.header("📄 Sample Data")
st.dataframe(df.head(10))

# --------------------------- Feature Importance ---------------------------
st.header("🌟 Feature Importance (Random Forest)")
feature_importances = rf_model.feature_importances_
features = X.columns
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x=feature_importances, y=features, ax=ax)
ax.set_title("Feature Importance")
ax.set_xlabel("Importance")
ax.set_ylabel("Features")
st.pyplot(fig)

# --------------------------- Charts ---------------------------
st.header("📈 Fare Distribution")
fig, ax = plt.subplots(figsize=(10,6))
sns.histplot(df['fare_amount'], kde=True, color='blue', ax=ax)
ax.set_xlabel("Fare Amount")
ax.set_ylabel("Frequency")
ax.set_title("Fare Amount Distribution")
st.pyplot(fig)

st.header("📊 Correlation Heatmap")
numeric_df = df.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
st.pyplot(fig)
