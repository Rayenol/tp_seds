import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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


### part 2
def preprocess_data(data):
    # Ensure the required columns exist
    if "Close" not in data.columns:
        raise KeyError("The 'Close' column is missing from the data.")
    
    # Calculate daily returns
    data["Daily Returns"] = data["Close"].pct_change()
    
    # Remove rows with NaN caused by Daily Returns
    data.dropna(inplace=True)
    
    # Create the Target column (next day's closing price)
    data["Target"] = data["Close"].shift(-1)
    
    # Remove rows with NaN in Target
    data.dropna(subset=["Target"], inplace=True)
    
    return data







# Sidebar for date selection (first set)
st.sidebar.header("Date Range Selection")
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=30), key="start_date_1")
end_date = st.sidebar.date_input("End Date", datetime.today(), key="end_date_1")

# Sidebar for configuration (second set)
st.sidebar.header("Configuration")
start_date_ml = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=365), key="start_date_2")
end_date_ml = st.sidebar.date_input("End Date", datetime.today(), key="end_date_2")
if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
else:
    btc_data = fetch_bitcoin_data(start_date, end_date)

if start_date_ml > end_date_ml:
    st.sidebar.error("Start date for ML must be before end date.")
else:
    btc_data_ml = fetch_bitcoin_data(start_date_ml, end_date_ml)

tab1, tab2 = st.tabs(["Data Analysis", "Machine Learning"])

def plot_price_trend(data):
    fig = px.line(data, x="Date", y="Close", title="Bitcoin Closing Price Trend")
    st.plotly_chart(fig)

def plot_correlation_heatmap(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


def build_and_train_model(data):
    features = data[["Open", "High", "Low", "Volume"]]
    target = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    return model, predictions, y_test


with tab2:
    st.header("Machine Learning Prediction Model")
    
    if not btc_data.empty:
        # Flatten columns if necessary
        btc_data.columns = [col[0] if isinstance(col, tuple) else col for col in btc_data.columns]
        
        try:
            preprocessed_data = preprocess_data(btc_data)
            
            if preprocessed_data.empty:
                st.error("Not enough data available for training. Please adjust the date range.")
            else:
                model, predictions, y_test = build_and_train_model(preprocessed_data)
                st.write("Model Performance")
                st.write(f"Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}")
                st.write(f"R² Score: {r2_score(y_test, predictions):.2f}")
                
                # Display prediction vs actual
                comparison = pd.DataFrame({"Actual": y_test, "Predicted": predictions}).reset_index(drop=True)
                st.write(comparison)
        except KeyError as e:
            st.error(f"Error in data preprocessing: {e}")
    else:
        st.warning("No data available for training.")


# Flatten multi-index columns
btc_data.columns = [col[0] if isinstance(col, tuple) else col for col in btc_data.columns]

# Ensure Date is a column
if btc_data.index.name == "Date":
    btc_data.reset_index(inplace=True)

# Plot price trend
with tab1:
    st.header("Data Analysis")
    if not btc_data.empty:
        plot_price_trend(btc_data)
        plot_correlation_heatmap(btc_data)
    else:
        st.warning("No data available for analysis.")
