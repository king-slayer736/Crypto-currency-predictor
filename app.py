import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64
from utils.data_processing import preprocess_data, calculate_metrics, calculate_returns
from utils.visualization import plot_returns, plot_correlation_matrix, plot_volatility, plot_cumulative_returns
from utils.ml_models import train_model, evaluate_model, predict_future

# Set page config
st.set_page_config(
    page_title="Crypto Analysis App",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for zombie theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    .stApp {
        background-color: #0A1929;
        background-image: linear-gradient(45deg, rgba(0, 0, 0, 0.95) 0%, rgba(10, 25, 41, 0.95) 100%);
        font-family: 'Inter', sans-serif;
        color: #FFFFFF;
    }

    /* Trading chart colors */
    .js-plotly-plot .plotly .candlestick .increases {
        fill: #26a69a;
        stroke: #26a69a;
    }

    .js-plotly-plot .plotly .candlestick .decreases {
        fill: #ef5350;
        stroke: #ef5350;
    }

    /* Trading specific styles */
    .element-container div[data-testid="stVerticalBlock"] div[data-testid="stDataFrame"] {
        background: #132F4C;
        border-radius: 8px;
        border: 1px solid rgba(0, 255, 187, 0.1);
    }

    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    .stApp {
        background-image: 
            linear-gradient(45deg, rgba(0, 0, 0, 0.80) 0%, rgba(15, 15, 20, 0.80) 100%),
            url("./attached_assets/images.jpeg");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
        font-family: 'Roboto', sans-serif;
        background-color: #0a0a0f;
    }

    .zombie-title {
        font-family: 'Creepster', cursive;
        color: #8B0000;
        text-shadow: 2px 2px 4px #000;
        animation: drip 2s infinite;
    }

    @keyframes drip {
        0% { text-shadow: 2px 2px 4px #8B0000; }
        50% { text-shadow: 2px 4px 8px #FF0000; }
        100% { text-shadow: 2px 2px 4px #8B0000; }
    }

    .card {
        border: 2px solid #2b2b2b;
        background-color: rgba(10, 10, 15, 0.9);
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .card:hover {
        border-color: #ff1744;
        box-shadow: 0 0 20px rgba(255, 23, 68, 0.3);
    }

    div.stButton > button {
        background-color: #00FFBB;
        color: #fff;
        border: 2px solid #7366ff;
        text-shadow: 1px 1px 2px #000;
        font-family: 'Courier New', monospace;
    }

    div.stButton > button:hover {
        background-color: #7366ff;
        border-color: #00FFBB;
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(0, 255, 187, 0.5);
    }

    .title-text {
        background: linear-gradient(45deg, #00FFBB, #7366ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Courier New', monospace;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }

    .card {
        background-color: rgba(31, 45, 45, 0.8);
        border: 1px solid #00FFBB;
        box-shadow: 0 0 10px rgba(0, 255, 187, 0.3);
    }

    div.stButton > button {
        background-color: #00FFBB;
        color: #fff;
        border: 2px solid #7366ff;
    }

    div.stButton > button:hover {
        background-color: #7366ff;
        border-color: #00FFBB;
    }
    div.stButton > button {
        border-radius: 10px;
        font-weight: bold;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 4px 4px 10px rgba(0,0,0,0.3);
    }
    .css-1v3fvcr {
        background-color: transparent;
    }
    .title-text {
        background: linear-gradient(45deg, #7366ff, #3a86ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .animate-fade-in {
        animation: fadeIn 1.5s ease-in-out;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
        background-color: rgba(255, 255, 255, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(115, 102, 255, 0.2) !important;
        border-bottom: 2px solid #7366ff !important;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    .icon-title {
        display: inline-flex;
        align-items: center;
        font-size: 1.5em;
        margin-bottom: 1em;
    }
    .icon-title svg {
        margin-right: 0.5em;
    }
    .card {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        margin-bottom: 1em;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: rgba(255, 255, 255, 1.0);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .metric-label {
        font-size: 0.9em;
        color: rgba(255, 255, 255, 1.0);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        font-weight: 500;
    }
    /* Typography enhancements */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    p, li {
        color: rgba(255, 255, 255, 0.9) !important;
        text-shadow: 0px 0px 2px rgba(0,0,0,0.3);
        font-family: 'Poppins', sans-serif;
        line-height: 1.6;
        letter-spacing: 0.3px;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(45deg, #00FFBB, #7366ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: 0.8px;
        margin-bottom: 1em;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        padding: 10px 0;
        border-bottom: 2px solid rgba(115, 102, 255, 0.2);
    }

    h1 {
        font-size: 2.5em;
        text-align: center;
        margin-top: 20px;
    }

    h2 {
        font-size: 2em;
        color: #00FFBB;
    }

    h3 {
        font-size: 1.75em;
        color: #7366ff;
    }

    .title-text {
        background: linear-gradient(45deg, #00FFBB, #7366ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Poppins', sans-serif;
    }

    .metric-value {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }

    .metric-label {
        font-family: 'Poppins', sans-serif;
        font-weight: 400;
    }

    .card {
        border-radius: 15px;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'crypto_list' not in st.session_state:
    st.session_state.crypto_list = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'returns' not in st.session_state:
    st.session_state.returns = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'selected_crypto' not in st.session_state:
    st.session_state.selected_crypto = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'Linear Regression'
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None

# Function to create a download link for dataframe
def get_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Main application
def main():
    # Initialize page in session state if not exists
    if 'page' not in st.session_state:
        st.session_state.page = "Welcome"

    # Sidebar for navigation and data loading
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", ["Welcome", "Data Loading", "Data Analysis", "Risk Metrics", "Predictive Modeling", "Results"], index=["Welcome", "Data Loading", "Data Analysis", "Risk Metrics", "Predictive Modeling", "Results"].index(st.session_state.page))

        st.markdown("---")
        st.subheader("Data Source Options")

        # Upload data option
        uploaded_file = st.file_uploader("Upload cryptocurrency dataset", type=["csv", "xlsx"])

        # Yahoo Finance option
        st.subheader("Or fetch from Yahoo Finance")
        crypto_symbols = st.multiselect(
            "Select cryptocurrencies",
            ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD", "DOT-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "MATIC-USD"],
            default=[]
        )

        # Date range options
        time_period = st.radio(
            "Select time period",
            ["Last Month", "Last 3 Months", "Last 6 Months", "Last Year", "Last 2 Years", "Last 5 Years", "Custom"],
            index=3,
            horizontal=True
        )

        # Set dates based on selection
        end_date = datetime.now()
        if time_period == "Last Month":
            start_date = end_date - timedelta(days=30)
        elif time_period == "Last 3 Months":
            start_date = end_date - timedelta(days=90)
        elif time_period == "Last 6 Months":
            start_date = end_date - timedelta(days=180)
        elif time_period == "Last Year":
            start_date = end_date - timedelta(days=365)
        elif time_period == "Last 2 Years":
            start_date = end_date - timedelta(days=730)
        elif time_period == "Last 5 Years":
            start_date = end_date - timedelta(days=1825)
        else:  # Custom
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("Start date", end_date - timedelta(days=365))
            with date_col2:
                end_date = st.date_input("End date", end_date)

        # Display selected date range
        st.caption(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        if st.button("Fetch Data"):
            if crypto_symbols:
                try:
                    # Fetch data from Yahoo Finance
                    raw_data = yf.download(crypto_symbols, start=start_date, end=end_date)

                    # Check if 'Adj Close' column exists, otherwise use 'Close'
                    if 'Adj Close' in raw_data.columns:
                        data = raw_data['Adj Close']
                    else:
                        data = raw_data['Close']

                    if isinstance(data, pd.Series):
                        data = pd.DataFrame(data)
                        data.columns = [crypto_symbols[0]]

                    # Reset index to make date a column
                    data = data.reset_index()

                    st.session_state.data = data
                    st.session_state.crypto_list = crypto_symbols
                    st.session_state.selected_crypto = crypto_symbols[0] if crypto_symbols else None
                    st.success(f"Successfully fetched data for {', '.join(crypto_symbols)}")
                except Exception as e:
                    st.error(f"Error fetching data: {e}")

        # Process uploaded data if provided
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)

                st.session_state.data = data
                date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    data[date_cols[0]] = pd.to_datetime(data[date_cols[0]])

                # Try to determine crypto columns (excluding date columns)
                crypto_cols = [col for col in data.columns if col not in date_cols]
                st.session_state.crypto_list = crypto_cols
                st.session_state.selected_crypto = crypto_cols[0] if crypto_cols else None

                st.success(f"Successfully loaded data with {len(data)} rows and {len(data.columns)} columns")
            except Exception as e:
                st.error(f"Error loading data: {e}")

        # Model selection for predictive modeling
        st.markdown("---")
        st.subheader("Model Selection")
        model_type = st.selectbox(
            "Select Machine Learning Model",
            ["Linear Regression", "Logistic Regression", "K-Means Clustering"],
            index=0
        )
        st.session_state.model_type = model_type

    # Update session state with selected page
    st.session_state.page = page

    # Main content based on selected page
    if page == "Welcome":
        display_welcome_page()
    elif page == "Data Loading":
        display_data_loading_page()
    elif page == "Data Analysis":
        display_data_analysis_page()
    elif page == "Risk Metrics":
        display_risk_metrics_page()
    elif page == "Predictive Modeling":
        display_predictive_modeling_page()
    elif page == "Results":
        display_results_page()

# Page functions
def display_welcome_page():
    # Animated welcome title with gradient
    st.markdown('<h1 style="color: #FF0000;" class="animate-fade-in">Crypto Analysis Dashboard</h1>', unsafe_allow_html=True)

    # Add crypto-themed GIF
    st.markdown("""
    <div style="display: flex; justify-content: center; margin: 20px 0;">
        <img src="https://media.giphy.com/media/trN9ht5RlE3Dcwavg2/giphy.gif" width="300px" style="border-radius: 10px; box-shadow: 0 0 15px rgba(115, 102, 255, 0.5);">
    </div>
    """, unsafe_allow_html=True)

    # Main content in a card-like container
    st.markdown('<div class="animate-fade-in" style="animation-delay: 0.3s;">', unsafe_allow_html=True)

    # Display cryptocurrency-themed image
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="card">
        <h2 style="color: #FF0000;">Analyze and Predict Cryptocurrency Trends</h2>

        <p style="font-size: 1.1em; margin-bottom: 1.5em;">
        This application helps you analyze cryptocurrency data and make predictions using advanced machine learning algorithms. 
        You can upload your own data or fetch real-time data from Yahoo Finance for in-depth analysis.
        </p>

        <h3 style="color: #FF0000; margin-top: 1em;">✨ Key Features:</h3>
        <ul style="list-style-type: none; padding-left: 0;">
            <li style="margin: 0.5em 0;">🔍 Comprehensive return calculations</li>
            <li style="margin: 0.5em 0;">📊 Advanced risk metrics</li>
            <li style="margin: 0.5em 0;">📈 Interactive data visualization</li>
            <li style="margin: 0.5em 0;">🧠 Predictive modeling for price forecasting</li>
            <li style="margin: 0.5em 0;">📅 Historical performance analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="background-color: rgba(115, 102, 255, 0.1);">
        <h3 style="color: #FF0000;">🚀 Getting Started:</h3>
        <ol style="padding-left: 1.5em;">
            <li style="margin: 0.5em 0;">Use the sidebar to navigate between pages</li>
            <li style="margin: 0.5em 0;">Upload your data or fetch from Yahoo Finance</li>
            <li style="margin: 0.5em 0;">Analyze returns and risk metrics</li>
            <li style="margin: 0.5em 0;">Build and evaluate predictive models</li>
            <li style="margin: 0.5em 0;">Download your results</li>
        </ol>
        <p style="font-style: italic; margin-top: 1em;">Let's get started with your cryptocurrency analysis journey!</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image("https://images.unsplash.com/photo-1644361566696-3d442b5b482a", caption="Cryptocurrency Trading")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top: 1em;">', unsafe_allow_html=True)
        st.image("https://images.unsplash.com/photo-1523961131990-5ea7c61b2107", caption="Data Visualization")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Feature buttons for navigation with animated cards
    st.markdown('<h3 class="animate-fade-in" style="animation-delay: 0.6s; color: #7366ff; margin-top: 1em;">Explore App Features</h3>', unsafe_allow_html=True)

    # Creating feature cards with animations and hover effects
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="animate-fade-in" style="animation-delay: 0.8s;">
            <div class="card" style="cursor: pointer;" onclick="document.getElementById('btn_data_loading').click();">
                <h3 style="color: #FF0000;">📊 Load & Process Data</h3>
                <p>Import cryptocurrency data and prepare it for analysis</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Load Data", key="btn_data_loading"):
            st.session_state.page = "Data Loading"
            st.rerun()

        st.markdown("""
        <div class="animate-fade-in" style="animation-delay: 1.0s;">
            <div class="card" style="cursor: pointer;" onclick="document.getElementById('btn_data_analysis').click();">
                <h3 style="color: #FF0000;">📈 Data Analysis</h3>
                <p>Visualize price trends and analyze returns</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Analyze Data", key="btn_data_analysis"):
            st.session_state.page = "Data Analysis"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="animate-fade-in" style="animation-delay: 1.2s;">
            <div class="card" style="cursor: pointer;" onclick="document.getElementById('btn_risk_metrics').click();">
                <h3 style="color: #FF0000;">⚖️ Risk Metrics</h3>
                <p>Calculate volatility, Sharpe ratio, and drawdowns</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Risk Metrics", key="btn_risk_metrics"):
            st.session_state.page = "Risk Metrics"
            st.rerun()

        st.markdown("""
        <div class="animate-fade-in" style="animation-delay: 1.4s;">
            <div class="card" style="cursor: pointer;" onclick="document.getElementById('btn_predictive').click();">
                <h3 style="color: #FF0000;">🧠 Predictive Modeling</h3>
                <p>Build machine learning models to forecast prices</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Build Models", key="btn_predictive"):
            st.session_state.page = "Predictive Modeling"
            st.rerun()

def display_data_loading_page():
    # Animated title with gradient
    st.markdown('<h1 style="color: #FF0000;" class="animate-fade-in">📊 Data Loading and Preview</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.session_state.data is not None:
            # Data Preview Card
            st.markdown('<div class="card animate-fade-in" style="animation-delay: 0.3s;">', unsafe_allow_html=True)
            st.subheader("📋 Data Preview")
            st.dataframe(st.session_state.data.head(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Data Information Card
            st.markdown('<div class="card animate-fade-in" style="animation-delay: 0.5s;">', unsafe_allow_html=True)
            st.subheader("ℹ️ Data Information")
            buffer = io.StringIO()
            st.session_state.data.info(buf=buffer)
            st.text(buffer.getvalue())
            st.markdown('</div>', unsafe_allow_html=True)

            # Statistics Card
            st.markdown('<div class="card animate-fade-in" style="animation-delay: 0.7s;">', unsafe_allow_html=True)
            st.subheader("📊 Descriptive Statistics")
            st.dataframe(st.session_state.data.describe(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Process Data Button with enhanced styling
            st.markdown('<div class="animate-fade-in" style="animation-delay: 0.9s; text-align: center; margin-top: 20px;">', unsafe_allow_html=True)
            if st.button("🔄 Process Data", key="process_data", use_container_width=True):
                # Process data
                try:
                    with st.spinner("Processing data... Please wait."):
                        st.session_state.data = preprocess_data(st.session_state.data)
                        st.session_state.returns = calculate_returns(st.session_state.data, st.session_state.crypto_list)
                        st.session_state.metrics = calculate_metrics(st.session_state.returns)
                    st.success("✅ Data processed successfully!")
                    # Navigate to Data Analysis page
                    st.session_state.page = "Data Analysis"
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error processing data: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # No data yet - show info card
            st.markdown('<div class="card animate-fade-in" style="animation-delay: 0.3s; text-align: center; padding: 40px;">', unsafe_allow_html=True)
            st.info("👈 Please upload a dataset or fetch data from Yahoo Finance using the sidebar.")
            st.image("https://images.unsplash.com/photo-1639762681057-408e52192e55", caption="Waiting for data...")
            st.markdown("""
            <div style="margin-top: 20px; font-style: italic; color: rgba(255,255,255,0.7);">
                Select cryptocurrency symbols and a time period in the sidebar, then click "Fetch Data" to begin your analysis.
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Image Card
        st.markdown('<div class="card animate-fade-in" style="animation-delay: 0.4s;">', unsafe_allow_html=True)
        st.image("https://images.unsplash.com/photo-1634097538301-5d5f8b09eb84", caption="Data Analysis")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.data is not None:
            # Dataset Summary Card
            st.markdown('<div class="card animate-fade-in" style="animation-delay: 0.6s; margin-top: 20px;">', unsafe_allow_html=True)
            st.subheader("📈 Dataset Summary")

            # Show metrics in an attractive format
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div class="metric-value">{st.session_state.data.shape[0]:,}</div>
                    <div class="metric-label">Rows</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div class="metric-value">{st.session_state.data.shape[1]}</div>
                    <div class="metric-label">Columns</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Available Cryptocurrencies Card
            if st.session_state.crypto_list:
                st.markdown('<div class="card animate-fade-in" style="animation-delay: 0.8s; margin-top: 20px;">', unsafe_allow_html=True)
                st.subheader("💰 Available Cryptocurrencies")

                for i, crypto in enumerate(st.session_state.crypto_list):
                    st.markdown(f"""
                    <div style="display: inline-block; padding: 8px 15px; margin: 5px; 
                         background-color: rgba(115, 102, 255, 0.15); border-radius: 20px; 
                         font-weight: bold; color: #7366ff;">
                        {crypto}
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

def display_data_analysis_page():
    st.title("📈 Data Analysis")

    if st.session_state.data is None:
        st.warning("No data available. Please load data first.")
        return

    if st.session_state.returns is None:
        st.warning("Data not processed. Please process data on the Data Loading page.")
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Price Trends")
        if len(st.session_state.crypto_list) > 0:
            selected_cryptos = st.multiselect(
                "Select cryptocurrencies to display",
                st.session_state.crypto_list,
                default=[st.session_state.crypto_list[0]]
            )

            if selected_cryptos:
                # Plot price trends
                fig = go.Figure()

                for crypto in selected_cryptos:
                    # Check if crypto exists in the dataframe
                    if crypto in st.session_state.data.columns:
                        fig.add_trace(go.Scatter(
                            x=st.session_state.data['Date'] if 'Date' in st.session_state.data.columns else st.session_state.data.index,
                            y=st.session_state.data[crypto],
                            mode='lines',
                            name=crypto
                        ))

                fig.update_layout(
                    title="Price Trends",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    legend_title="Cryptocurrency",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Return analysis
                st.subheader("Return Analysis")
                # Check which cryptos have valid return data
                valid_return_cryptos = [crypto for crypto in selected_cryptos if f'{crypto}_Daily_Return' in st.session_state.returns.columns]

                if valid_return_cryptos:
                    st.plotly_chart(plot_returns(st.session_state.returns, valid_return_cryptos), use_container_width=True)

                    # Cumulative returns
                    st.subheader("Cumulative Returns")
                    st.plotly_chart(plot_cumulative_returns(st.session_state.returns, valid_return_cryptos), use_container_width=True)
                else:
                    st.info("No valid return data available for the selected cryptocurrencies.")

                if st.button("Calculate Risk Metrics", key="calc_risk_metrics"):
                    st.session_state.metrics = calculate_metrics(st.session_state.returns)
                    st.success("Risk metrics calculated!")
                    # Navigate to Risk Metrics page
                    st.session_state.page = "Risk Metrics"
                    st.rerun()
        else:
            st.info("No cryptocurrency data available for analysis.")

    with col2:
        st.image("https://images.unsplash.com/photo-1639762681485-074b7f938ba0", caption="Crypto Data Visualization")

        if st.session_state.returns is not None:
            st.subheader("Return Statistics")
            st.dataframe(st.session_state.returns.describe())

            if len(st.session_state.crypto_list) > 1:
                st.subheader("Correlation Analysis")
                # Get list of columns that actually exist in the returns dataframe
                # Look for daily return columns for each crypto
                daily_return_cols = [col for col in st.session_state.returns.columns if '_Daily_Return' in col]
                valid_cryptos = [col.split('_Daily_Return')[0] for col in daily_return_cols]

                if len(valid_cryptos) > 1:
                    # Create correlation matrix using the return columns
                    return_cols = [f'{crypto}_Daily_Return' for crypto in valid_cryptos]
                    corr_matrix = st.session_state.returns[return_cols].corr()

                    # Rename the matrix for better display
                    corr_matrix.columns = valid_cryptos
                    corr_matrix.index = valid_cryptos

                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 valid cryptocurrencies to show correlation analysis.")

def display_risk_metrics_page():
    st.title("⚖️ Risk Metrics Analysis")

    if st.session_state.metrics is None:
        st.warning("Risk metrics not calculated. Please process data and calculate metrics first.")
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Key Risk & Return Metrics")
        st.dataframe(st.session_state.metrics)

        # Risk vs. Return scatterplot
        st.subheader("Risk vs. Return Analysis")
        fig = px.scatter(
            st.session_state.metrics,
            x='Volatility (Annualized)',
            y='Return (Annualized)',
            text=st.session_state.metrics.index,
            size='Volatility (Annualized)',
            color='Sharpe Ratio',
            color_continuous_scale='Viridis',
            hover_data=['Maximum Drawdown', 'Sortino Ratio']
        )

        fig.update_traces(textposition='top center')
        fig.update_layout(
            height=600,
            title_text='Risk-Return Profile of Cryptocurrencies',
            xaxis_title='Risk (Annualized Volatility)',
            yaxis_title='Return (Annualized)',
            coloraxis_colorbar=dict(title='Sharpe Ratio')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Volatility over time
        st.subheader("Volatility Over Time")
        if st.session_state.returns is not None and len(st.session_state.crypto_list) > 0:
            selected_cryptos = st.multiselect(
                "Select cryptocurrencies for volatility analysis",
                st.session_state.crypto_list,
                default=[st.session_state.crypto_list[0]]
            )

            if selected_cryptos:
                # Get list of valid cryptos for volatility calculation
                valid_volatility_cryptos = [crypto for crypto in selected_cryptos if f'{crypto}_Daily_Return' in st.session_state.returns.columns]

                if valid_volatility_cryptos:
                    st.plotly_chart(plot_volatility(st.session_state.returns, valid_volatility_cryptos), use_container_width=True)
                else:
                    st.info("No valid cryptocurrencies available for volatility analysis.")

        if st.button("Proceed to Predictive Modeling", key="proceed_to_modeling"):
            st.success("Ready for predictive modeling!")
            # Navigate to Predictive Modeling page
            st.session_state.page = "Predictive Modeling"
            st.rerun()

    with col2:
        st.image("https://images.unsplash.com/photo-1554260570-e9689a3418b8", caption="Blockchain Financial Analysis")

        st.subheader("Understanding Risk Metrics")
        st.markdown("""
        **Volatility (Annualized)** - Measures price fluctuation; higher values indicate greater risk.

        **Sharpe Ratio** - Return per unit of risk; higher values indicate better risk-adjusted returns.

        **Sortino Ratio** - Similar to Sharpe but only considers downside risk; higher values are better.

        **Maximum Drawdown** - Largest peak-to-trough decline; smaller (less negative) values are better.

        **Value at Risk (VaR)** - Maximum potential loss at a 95% confidence level.

        **Expected Shortfall** - Average loss in the worst 5% of cases.
        """)

        if st.session_state.metrics is not None:
            st.subheader("Best Performing")
            best = st.session_state.metrics.sort_values('Return (Annualized)', ascending=False).head(1)
            st.markdown(f"**Highest Return:** {best.index[0]} ({best['Return (Annualized)'].values[0]:.2%})")

            best = st.session_state.metrics.sort_values('Sharpe Ratio', ascending=False).head(1)
            st.markdown(f"**Best Risk-Adjusted:** {best.index[0]} (Sharpe: {best['Sharpe Ratio'].values[0]:.2f})")

            st.subheader("Riskiest")
            worst = st.session_state.metrics.sort_values('Maximum Drawdown').head(1)
            st.markdown(f"**Largest Drawdown:** {worst.index[0]} ({worst['Maximum Drawdown'].values[0]:.2%})")

def display_predictive_modeling_page():
    st.title("🔮 Predictive Modeling")

    if st.session_state.data is None or st.session_state.returns is None:
        st.warning("No processed data available. Please load and process data first.")
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Model Configuration")

        # Select cryptocurrency for prediction
        selected_crypto = st.selectbox(
            "Select cryptocurrency for prediction",
            st.session_state.crypto_list,
            index=0
        )
        st.session_state.selected_crypto = selected_crypto

        # Model parameters
        st.write(f"Selected model: **{st.session_state.model_type}**")

        if st.session_state.model_type == "Linear Regression":
            test_size = st.slider("Test data size (%)", 10, 40, 20)
            forecast_days = st.slider("Number of days to forecast", 7, 90, 30)

            features = st.multiselect(
                "Select features to include",
                ["Price", "Volume", "MA_7", "MA_14", "MA_30", "RSI", "MACD", "Upper_Band", "Lower_Band"],
                default=["Price", "MA_7", "MA_30", "RSI"]
            )

        elif st.session_state.model_type == "Logistic Regression":
            test_size = st.slider("Test data size (%)", 10, 40, 20)
            forecast_days = st.slider("Number of days to forecast", 7, 90, 30)
            threshold = st.slider("Price increase threshold (%)", 0.5, 5.0, 1.0)

            features = st.multiselect(
                "Select features to include",
                ["Price", "Volume", "MA_7", "MA_14", "MA_30", "RSI", "MACD", "Upper_Band", "Lower_Band"],
                default=["Price", "MA_7", "MA_30", "RSI"]
            )

        elif st.session_state.model_type == "K-Means Clustering":
            n_clusters = st.slider("Number of clusters", 2, 10, 5)
            features = st.multiselect(
                "Select features for clustering",
                ["Return", "Volatility", "Volume_Change", "MA_Crossover", "RSI"],
                default=["Return", "Volatility", "RSI"]
            )

        # Train model button
        if st.button("Train Model", key="train_model_button"):
            try:
                with st.spinner("Training model..."):
                    if st.session_state.model_type == "Linear Regression":
                        model, X_test, y_test, predictions, feature_importance = train_model(
                            st.session_state.data,
                            selected_crypto,
                            model_type="linear",
                            test_size=test_size/100,
                            features=features
                        )

                        st.session_state.trained_model = model
                        st.session_state.model_results = evaluate_model(model, X_test, y_test, predictions)
                        st.session_state.feature_importance = feature_importance

                        # Generate future predictions
                        future_pred = predict_future(
                            model, 
                            st.session_state.data, 
                            selected_crypto, 
                            days=forecast_days,
                            features=features
                        )
                        st.session_state.predictions = future_pred

                    elif st.session_state.model_type == "Logistic Regression":
                        model, X_test, y_test, predictions, feature_importance = train_model(
                            st.session_state.data,
                            selected_crypto,
                            model_type="logistic",
                            test_size=test_size/100,
                            threshold=threshold/100,
                            features=features
                        )

                        st.session_state.trained_model = model
                        st.session_state.model_results = evaluate_model(model, X_test, y_test, predictions, is_classification=True)
                        st.session_state.feature_importance = feature_importance

                        # Generate future predictions
                        future_pred = predict_future(
                            model, 
                            st.session_state.data, 
                            selected_crypto, 
                            days=forecast_days,
                            features=features,
                            is_classification=True
                        )
                        st.session_state.predictions = future_pred

                    elif st.session_state.model_type == "K-Means Clustering":
                        model, cluster_data, centroids = train_model(
                            st.session_state.data,
                            selected_crypto,
                            model_type="kmeans",
                            n_clusters=n_clusters,
                            features=features
                        )

                        st.session_state.trained_model = model
                        st.session_state.model_results = cluster_data
                        st.session_state.predictions = centroids

                st.success(f"Model trained successfully!")
                # Navigate to Results page
                st.session_state.page = "Results"
                st.rerun()

            except Exception as e:
                st.error(f"Error training model: {e}")

    with col2:
        st.image("https://images.unsplash.com/photo-1556155092-490a1ba16284", caption="Blockchain Financial Analysis")

        st.subheader("Model Information")

        if st.session_state.model_type == "Linear Regression":
            st.markdown("""
            **Linear Regression** predicts continuous price values based on historical data and technical indicators.

            **Use for:**
            - Price forecasting
            - Trend analysis
            - Understanding feature relationships

            **Key metrics:**
            - RMSE (lower is better)
            - R² (higher is better)
            - MAE (lower is better)
            """)

        elif st.session_state.model_type == "Logistic Regression":
            st.markdown("""
            **Logistic Regression** predicts whether the price will increase or decrease in the future.

            **Use for:**
            - Directional prediction (up/down)
            - Trading signal generation
            - Risk assessment

            **Key metrics:**
            - Accuracy (higher is better)
            - Precision & Recall
            - F1 Score
            """)

        elif st.session_state.model_type == "K-Means Clustering":
            st.markdown("""
            **K-Means Clustering** groups similar trading periods based on selected features.

            **Use for:**
            - Market regime identification
            - Volatility clustering
            - Pattern recognition

            **Key metrics:**
            - Silhouette score
            - Inertia
            - Cluster distribution
            """)

def display_results_page():
    st.title("📊 Results and Predictions")

    if st.session_state.trained_model is None or st.session_state.model_results is None:
        st.warning("No model has been trained yet. Please train a model on the Predictive Modeling page.")
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(f"Model Results for {st.session_state.selected_crypto}")

        if st.session_state.model_type == "Linear Regression":
            # Display regression metrics
            metrics_df = pd.DataFrame({
                'Metric': ['R² Score', 'Mean Absolute Error', 'Root Mean Squared Error', 'Mean Absolute Percentage Error'],
                'Value': [
                    st.session_state.model_results['r2'],
                    st.session_state.model_results['mae'],
                    st.session_state.model_results['rmse'],
                    st.session_state.model_results['mape']
                ]
            })

            st.dataframe(metrics_df)

            # Plot actual vs predicted
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=np.arange(len(st.session_state.model_results['y_test'])),
                y=st.session_state.model_results['y_test'],
                mode='lines',
                name='Actual'
            ))

            fig.add_trace(go.Scatter(
                x=np.arange(len(st.session_state.model_results['predictions'])),
                y=st.session_state.model_results['predictions'],
                mode='lines',
                name='Predicted'
            ))

            fig.update_layout(
                title="Actual vs Predicted Prices",
                xaxis_title="Time Period",
                yaxis_title="Price",
                legend_title="Data",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Future predictions
            if st.session_state.predictions is not None:
                st.subheader("Price Forecasts")

                fig = go.Figure()

                # Historical data
                fig.add_trace(go.Scatter(
                    x=st.session_state.data['Date'].tail(30) if 'Date' in st.session_state.data.columns else np.arange(30),
                    y=st.session_state.data[st.session_state.selected_crypto].tail(30),
                    mode='lines',
                    name='Historical'
                ))

                # Predicted data
                future_dates = pd.date_range(
                    start=st.session_state.data['Date'].max() if 'Date' in st.session_state.data.columns else pd.Timestamp.today(),
                    periods=len(st.session_state.predictions)+1
                )[1:]

                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=st.session_state.predictions,
                    mode='lines',
                    name='Forecast',
                    line=dict(dash='dash')
                ))

                fig.update_layout(
                    title=f"Price Forecast for {st.session_state.selected_crypto}",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend_title="Data",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Create downloadable dataframe
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted_Price': st.session_state.predictions
                })

                st.markdown(get_download_link(forecast_df, "price_forecast.csv"), unsafe_allow_html=True)

        elif st.session_state.model_type == "Logistic Regression":
            # Display classification metrics
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                'Value': [
                    st.session_state.model_results['accuracy'],
                    st.session_state.model_results['precision'],
                    st.session_state.model_results['recall'],
                    st.session_state.model_results['f1']
                ]
            })

            st.dataframe(metrics_df)

            # Confusion matrix
            cm = st.session_state.model_results['confusion_matrix']
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Decrease', 'Increase'],
                y=['Decrease', 'Increase'],
                text_auto=True,
                color_continuous_scale='Blues'
            )

            fig.update_layout(
                title="Confusion Matrix",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Future predictions
            if st.session_state.predictions is not None:
                st.subheader("Directional Forecasts")

                future_dates = pd.date_range(
                    start=st.session_state.data['Date'].max() if 'Date' in st.session_state.data.columns else pd.Timestamp.today(),
                    periods=len(st.session_state.predictions)+1
                )[1:]

                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Prediction': ['Increase' if p == 1 else 'Decrease' for p in st.session_state.predictions],
                    'Probability': np.random.uniform(0.6, 0.9, len(st.session_state.predictions))  # Example probabilities
                })

                st.dataframe(forecast_df)

                # Create bar chart of predictions
                fig = px.bar(
                    forecast_df,
                    x='Date',
                    y='Probability',
                    color='Prediction',
                    color_discrete_map={'Increase': 'green', 'Decrease': 'red'},
                    title=f"Directional Forecast for {st.session_state.selected_crypto}"
                )

                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(get_download_link(forecast_df, "direction_forecast.csv"), unsafe_allow_html=True)

        elif st.session_state.model_type == "K-Means Clustering":
            # Display clustering results
            if isinstance(st.session_state.model_results, pd.DataFrame):
                st.subheader("Cluster Analysis")

                # Show cluster counts
                cluster_counts = st.session_state.model_results['Cluster'].value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']

                fig = px.bar(
                    cluster_counts,
                    x='Cluster',
                    y='Count',
                    color='Cluster',
                    title="Distribution of Clusters"
                )

                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Scatter plot of clusters (first two features)
                features = st.session_state.model_results.columns[:-1]  # Exclude 'Cluster' column

                if len(features) >= 2:
                    fig = px.scatter(
                        st.session_state.model_results,
                        x=features[0],
                        y=features[1],
                        color='Cluster',
                        title=f"Cluster Visualization ({features[0]} vs {features[1]})",
                        hover_data=features
                    )

                    # Add centroids
                    if st.session_state.predictions is not None:
                        centroids = st.session_state.predictions
                        for i in range(len(centroids)):
                            fig.add_trace(go.Scatter(
                                x=[centroids[i][0]],
                                y=[centroids[i][1]],
                                mode='markers',
                                marker=dict(
                                    symbol='star',
                                    size=15,
                                    color=i,
                                    line=dict(width=2, color='DarkSlateGrey')
                                ),
                                name=f'Centroid {i}'
                            ))

                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown(get_download_link(st.session_state.model_results, "cluster_analysis.csv"), unsafe_allow_html=True)

    with col2:
        st.image("https://images.unsplash.com/photo-1640661089711-708d6043d0c7", caption="Cryptocurrency Trading")

        # Feature importance for regression and classification models
        if st.session_state.feature_importance is not None and st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
            st.subheader("Feature Importance")

            feature_imp_df = pd.DataFrame({
                'Feature': st.session_state.feature_importance.index,
                'Importance': st.session_state.feature_importance.values
            }).sort_values('Importance', ascending=False)

            fig = px.bar(
                feature_imp_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance",
                color='Importance'
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Download all results
        st.subheader("Download All Results")

        if st.session_state.metrics is not None:
            st.markdown(get_download_link(st.session_state.metrics, "risk_metrics.csv"), unsafe_allow_html=True)

        if st.session_state.returns is not None:
            st.markdown(get_download_link(st.session_state.returns, "returns_data.csv"), unsafe_allow_html=True)

        if st.session_state.model_type == "Linear Regression" and st.session_state.predictions is not None:
            future_dates = pd.date_range(
                start=st.session_state.data['Date'].max() if 'Date' in st.session_state.data.columns else pd.Timestamp.today(),
                periods=len(st.session_state.predictions)+1
            )[1:]

            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Price': st.session_state.predictions
            })

            st.markdown(get_download_link(forecast_df, "price_forecast.csv"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()