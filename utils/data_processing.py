import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def preprocess_data(data):
    """
    Preprocess the cryptocurrency data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The raw cryptocurrency data.
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data.
    """
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Check if 'Date' column exists, if not try to find a date column
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break
    
    # Ensure the date column is properly formatted
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        if date_col != 'Date':
            df.rename(columns={date_col: 'Date'}, inplace=True)
        df.sort_values('Date', inplace=True)
    else:
        # If no date column is found, create one
        if not isinstance(df.index, pd.DatetimeIndex):
            df['Date'] = pd.date_range(end=datetime.now(), periods=len(df))
        else:
            df['Date'] = df.index
            df.reset_index(drop=True, inplace=True)
    
    # Check for missing values and handle them
    if df.isnull().sum().sum() > 0:
        df.fillna(method='ffill', inplace=True)  # Forward fill
        df.fillna(method='bfill', inplace=True)  # Backward fill for any remaining NAs
    
    # Add technical indicators if price columns are available
    for column in df.columns:
        if column not in ['Date', 'Volume']:
            # Calculate moving averages
            df[f'{column}_MA7'] = df[column].rolling(window=7).mean()
            df[f'{column}_MA14'] = df[column].rolling(window=14).mean()
            df[f'{column}_MA30'] = df[column].rolling(window=30).mean()
            
            # RSI (Relative Strength Index)
            # Calculate a simple version
            delta = df[column].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df[f'{column}_RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            std = df[column].rolling(window=20).std()
            middle_band = df[column].rolling(window=20).mean()
            df[f'{column}_Upper_Band'] = middle_band + (std * 2)
            df[f'{column}_Lower_Band'] = middle_band - (std * 2)
            
            # MACD (Moving Average Convergence Divergence)
            ema12 = df[column].ewm(span=12, adjust=False).mean()
            ema26 = df[column].ewm(span=26, adjust=False).mean()
            df[f'{column}_MACD'] = ema12 - ema26
            df[f'{column}_MACD_Signal'] = df[f'{column}_MACD'].ewm(span=9, adjust=False).mean()
    
    # Drop rows with NaN values that might have been introduced
    df.dropna(inplace=True)
    
    return df

def calculate_returns(data, crypto_list):
    """
    Calculate returns for cryptocurrency data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The preprocessed cryptocurrency data.
    crypto_list : list
        List of cryptocurrency columns to calculate returns for.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with calculated returns.
    """
    # Make a copy of the data
    df = data.copy()
    
    # Initialize returns DataFrame with dates
    if 'Date' in df.columns:
        returns = pd.DataFrame(df['Date'])
    else:
        returns = pd.DataFrame(index=df.index)
    
    # Calculate daily returns for each cryptocurrency
    for crypto in crypto_list:
        if crypto in df.columns:
            # Daily returns
            returns[f'{crypto}_Daily_Return'] = df[crypto].pct_change()
            
            # Log returns (useful for normalization and statistical analysis)
            returns[f'{crypto}_Log_Return'] = np.log(df[crypto] / df[crypto].shift(1))
            
            # Weekly returns (5-day rolling)
            returns[f'{crypto}_Weekly_Return'] = df[crypto].pct_change(5)
            
            # Monthly returns (21-day rolling)
            returns[f'{crypto}_Monthly_Return'] = df[crypto].pct_change(21)
    
    # Drop NaN values
    returns.dropna(inplace=True)
    
    return returns

def calculate_metrics(returns_data):
    """
    Calculate risk metrics for cryptocurrency returns.
    
    Parameters:
    -----------
    returns_data : pandas.DataFrame
        DataFrame containing cryptocurrency returns.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with calculated risk metrics.
    """
    # Get list of cryptocurrencies from column names
    crypto_list = []
    for col in returns_data.columns:
        if 'Daily_Return' in col:
            crypto = col.split('_Daily_Return')[0]
            crypto_list.append(crypto)
    
    # Initialize metrics DataFrame
    metrics = pd.DataFrame(index=crypto_list)
    
    # Calculate risk metrics for each cryptocurrency
    for crypto in crypto_list:
        daily_returns = returns_data[f'{crypto}_Daily_Return']
        log_returns = returns_data[f'{crypto}_Log_Return']
        
        # Annualized return and volatility (assuming 252 trading days per year)
        trading_days = 252
        metrics.loc[crypto, 'Return (Daily)'] = daily_returns.mean()
        metrics.loc[crypto, 'Return (Annualized)'] = daily_returns.mean() * trading_days
        metrics.loc[crypto, 'Volatility (Daily)'] = daily_returns.std()
        metrics.loc[crypto, 'Volatility (Annualized)'] = daily_returns.std() * np.sqrt(trading_days)
        
        # Sharpe Ratio (assuming risk-free rate of 0)
        sharpe_ratio = (daily_returns.mean() * trading_days) / (daily_returns.std() * np.sqrt(trading_days))
        metrics.loc[crypto, 'Sharpe Ratio'] = sharpe_ratio
        
        # Sortino Ratio (considering only downside risk)
        downside_returns = daily_returns[daily_returns < 0]
        sortino_ratio = (daily_returns.mean() * trading_days) / (downside_returns.std() * np.sqrt(trading_days)) if len(downside_returns) > 0 else np.nan
        metrics.loc[crypto, 'Sortino Ratio'] = sortino_ratio
        
        # Maximum Drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        metrics.loc[crypto, 'Maximum Drawdown'] = max_drawdown
        
        # Value at Risk (VaR) at 95% confidence level
        var_95 = daily_returns.quantile(0.05)
        metrics.loc[crypto, 'Value at Risk (95%)'] = var_95
        
        # Expected Shortfall (Conditional VaR) at 95% confidence level
        es_95 = daily_returns[daily_returns <= var_95].mean()
        metrics.loc[crypto, 'Expected Shortfall (95%)'] = es_95
        
        # Skewness and Kurtosis (from log returns)
        metrics.loc[crypto, 'Skewness'] = log_returns.skew()
        metrics.loc[crypto, 'Kurtosis'] = log_returns.kurt()
    
    return metrics
