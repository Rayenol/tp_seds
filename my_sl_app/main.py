import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
# Set Streamlit page configuration
st.set_page_config(page_title="Bitcoin Price Dashboard", layout="wide")


# Fetch Bitcoin price data
@st.cache_data
def fetch_bitcoin_data(start_date, end_date):
    btc_data = yf.download("BTC-USD", start=start_date, end=end_date)
    return btc_data

# Sidebar for date selection
st.sidebar.header("Date Range Selection")
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=30))
end_date = st.sidebar.date_input("End Date", datetime.today())

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
else:
    btc_data = fetch_bitcoin_data(start_date, end_date)


# Descriptive statistics
if not btc_data.empty:
    st.header("Descriptive Statistics")
    st.dataframe(btc_data.describe())
else:
    st.warning("No data available for the selected date range.")


if not btc_data.empty:
    # Price trend visualization
    st.header("Bitcoin Price Trend")
    # Ensure data is structured correctly
btc_data.reset_index(inplace=True)  # Make the index a column for plotly
if "Close" in btc_data.columns:
    fig = px.line(btc_data, x="Date", y=btc_data["Close"].squeeze(), title="Bitcoin Closing Price Over Time")
    st.write(btc_data.head())
    st.write(btc_data["Close"].shape)

    st.plotly_chart(fig)
else:
    st.warning("The 'Close' column is missing in the data. Unable to plot price trend.")
    fig = px.line(btc_data, x=btc_data.index, y="Close", title="Bitcoin Closing Price Over Time")
    st.plotly_chart(fig)

    # Correlation heatmap
    st.header("Correlation Analysis")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(btc_data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Histogram of daily returns
    st.header("Daily Returns Distribution")
    btc_data['Daily Returns'] = btc_data['Close'].pct_change()
    fig = px.histogram(btc_data, x='Daily Returns', nbins=50, title="Distribution of Daily Returns")
    st.plotly_chart(fig)

# Function to fetch Bitcoin data
def fetch_bitcoin_data(start_date, end_date):
    # Simulate fetching data
    # Replace this with your API or data source logic
    date_range = pd.date_range(start=start_date, end=end_date)
    data = pd.DataFrame({
        "Date": date_range,
        "Open": np.random.uniform(20000, 30000, len(date_range)),
        "High": np.random.uniform(20000, 30000, len(date_range)),
        "Low": np.random.uniform(20000, 30000, len(date_range)),
        "Close": np.random.uniform(20000, 30000, len(date_range)),
        "Volume": np.random.uniform(1000, 10000, len(date_range)),
    })
    return data

# Preprocessing function
def preprocess_data(data):
    if "Close" not in data.columns:
        raise KeyError("The 'Close' column is missing from the data.")
    
    # Calculate daily returns
    data["Daily Returns"] = data["Close"].pct_change()
    data.dropna(inplace=True)
    
    # Create the target column for ML
    data["Target"] = data["Close"].shift(-1)
    data.dropna(subset=["Target"], inplace=True)
    return data

# Plot price trend
def plot_price_trend(data):
    fig = px.line(data, x="Date", y="Close", title="Bitcoin Closing Price Trend")
    st.plotly_chart(fig)

# Plot correlation heatmap
def plot_correlation_heatmap(data):
    corr_data = data[["Open", "High", "Low", "Close", "Volume"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_data, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Build and train the ML model
def build_and_train_model(data):
    features = data[["Open", "High", "Low", "Volume"]]
    target = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    return model, predictions, y_test

# Streamlit App Layout
st.title("Bitcoin Analysis and Prediction")

# Sidebar
st.sidebar.header("Date Range Selection")
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=30), key="start_date_1")
end_date = st.sidebar.date_input("End Date", datetime.today(), key="end_date_1")

#st.sidebar.header("ML Configuration")
#start_date_ml = st.sidebar.date_input("ML Start Date", datetime.today() - timedelta(days=365), key="start_date_2")
#end_date_ml = st.sidebar.date_input("ML End Date", datetime.today(), key="end_date_2")

# Tabbed Interface
tab1, tab2 = st.tabs(["Data Analysis", "Machine Learning"])

# Data Retrieval
if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
else:
    btc_data = fetch_bitcoin_data(start_date, end_date)


# Data Analysis Tab
with tab1:
    st.header("Data Analysis")
    if not btc_data.empty:
        plot_price_trend(btc_data)
        plot_correlation_heatmap(btc_data)
    else:
        st.warning("No data available for analysis.")

# Machine Learning Tab
with tab2:
    st.header("Machine Learning Prediction Model")
    if not btc_data.empty:
        try:
            preprocessed_data = preprocess_data(btc_data)
            if preprocessed_data.empty:
                st.error("Not enough data available for training. Adjust the date range.")
            else:
                model, predictions, y_test = build_and_train_model(preprocessed_data)
                st.write("### Model Performance")
                st.write(f"Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}")
                st.write(f"R² Score: {r2_score(y_test, predictions):.2f}")
                
                # Comparison of Predictions
                comparison = pd.DataFrame({"Actual": y_test, "Predicted": predictions}).reset_index(drop=True)
                st.write("### Prediction vs Actual")
                st.write(comparison)
        except KeyError as e:
            st.error(f"Error in data preprocessing: {e}")
    else:
        st.warning("No data available for training.")

