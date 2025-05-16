# Cryptocurrency Analysis Dashboard

A comprehensive cryptocurrency analysis application built with Streamlit that provides advanced financial insights, predictive modeling, and interactive data visualization for crypto investors.

## Features

- **Multi-cryptocurrency return and risk analysis**
- **Advanced technical indicators**
- **Machine learning predictive modeling**
  - Linear Regression
  - Logistic Regression
  - K-Means Clustering
- **Interactive data visualization**
- **Real-time crypto market insights**

## Getting Started

### Local Development

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements-github.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

### Deploying to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app" and select this repository
5. Set the main file path to `app.py`
6. Deploy!

## Required Packages

- streamlit
- pandas
- pandas-ta
- numpy
- yfinance
- plotly
- scikit-learn

## App Structure

- `app.py`: Main application file
- `utils/`: Utility modules
  - `data_processing.py`: Data preprocessing and analysis functions
  - `ml_models.py`: Machine learning model implementations
  - `visualization.py`: Data visualization functions
- `streamlit/config.toml`: Streamlit configuration

## License

This project is licensed under the MIT License - see the LICENSE file for details.