import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_returns(returns_data, selected_cryptos):
    """
    Create a plot of daily returns for selected cryptocurrencies.
    
    Parameters:
    -----------
    returns_data : pandas.DataFrame
        DataFrame containing cryptocurrency returns.
    selected_cryptos : list
        List of cryptocurrencies to plot.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Returns plot figure.
    """
    fig = go.Figure()
    
    for crypto in selected_cryptos:
        fig.add_trace(go.Scatter(
            x=returns_data['Date'] if 'Date' in returns_data.columns else returns_data.index,
            y=returns_data[f'{crypto}_Daily_Return'],
            mode='lines',
            name=f'{crypto} Daily Return'
        ))
    
    fig.update_layout(
        title="Daily Returns",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        legend_title="Cryptocurrency",
        height=500
    )
    
    return fig

def plot_cumulative_returns(returns_data, selected_cryptos):
    """
    Create a plot of cumulative returns for selected cryptocurrencies.
    
    Parameters:
    -----------
    returns_data : pandas.DataFrame
        DataFrame containing cryptocurrency returns.
    selected_cryptos : list
        List of cryptocurrencies to plot.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Cumulative returns plot figure.
    """
    fig = go.Figure()
    
    for crypto in selected_cryptos:
        daily_returns = returns_data[f'{crypto}_Daily_Return']
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        
        fig.add_trace(go.Scatter(
            x=returns_data['Date'] if 'Date' in returns_data.columns else returns_data.index,
            y=cumulative_returns,
            mode='lines',
            name=f'{crypto} Cumulative Return'
        ))
    
    fig.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        legend_title="Cryptocurrency",
        height=500
    )
    
    return fig

def plot_correlation_matrix(returns_data, crypto_list):
    """
    Create a correlation matrix heatmap for cryptocurrency returns.
    
    Parameters:
    -----------
    returns_data : pandas.DataFrame
        DataFrame containing cryptocurrency returns.
    crypto_list : list
        List of cryptocurrencies to include in the correlation matrix.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Correlation matrix figure.
    """
    # Get daily returns for each cryptocurrency
    daily_returns_cols = [f'{crypto}_Daily_Return' for crypto in crypto_list]
    
    # Calculate correlation matrix
    corr_matrix = returns_data[daily_returns_cols].corr()
    
    # Rename columns and index for better display
    corr_matrix.columns = crypto_list
    corr_matrix.index = crypto_list
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="Correlation Matrix of Cryptocurrency Returns"
    )
    
    fig.update_layout(height=600)
    
    return fig

def plot_volatility(returns_data, selected_cryptos, window=21):
    """
    Create a plot of rolling volatility for selected cryptocurrencies.
    
    Parameters:
    -----------
    returns_data : pandas.DataFrame
        DataFrame containing cryptocurrency returns.
    selected_cryptos : list
        List of cryptocurrencies to plot.
    window : int, optional
        Rolling window size for volatility calculation.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Volatility plot figure.
    """
    fig = go.Figure()
    
    for crypto in selected_cryptos:
        daily_returns = returns_data[f'{crypto}_Daily_Return']
        
        # Calculate rolling volatility (annualized)
        rolling_vol = daily_returns.rolling(window).std() * np.sqrt(252)
        
        fig.add_trace(go.Scatter(
            x=returns_data['Date'] if 'Date' in returns_data.columns else returns_data.index,
            y=rolling_vol,
            mode='lines',
            name=f'{crypto} {window}-day Volatility'
        ))
    
    fig.update_layout(
        title=f"{window}-day Rolling Volatility (Annualized)",
        xaxis_title="Date",
        yaxis_title="Volatility",
        legend_title="Cryptocurrency",
        height=500
    )
    
    return fig

def plot_drawdown(returns_data, selected_cryptos):
    """
    Create a plot of drawdowns for selected cryptocurrencies.
    
    Parameters:
    -----------
    returns_data : pandas.DataFrame
        DataFrame containing cryptocurrency returns.
    selected_cryptos : list
        List of cryptocurrencies to plot.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Drawdown plot figure.
    """
    fig = go.Figure()
    
    for crypto in selected_cryptos:
        daily_returns = returns_data[f'{crypto}_Daily_Return']
        
        # Calculate drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        
        fig.add_trace(go.Scatter(
            x=returns_data['Date'] if 'Date' in returns_data.columns else returns_data.index,
            y=drawdown,
            mode='lines',
            name=f'{crypto} Drawdown',
            fill='tozeroy'
        ))
    
    fig.update_layout(
        title="Drawdown Analysis",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        legend_title="Cryptocurrency",
        height=500
    )
    
    return fig

def plot_risk_return_profile(metrics, crypto_list):
    """
    Create a risk-return scatter plot for cryptocurrencies.
    
    Parameters:
    -----------
    metrics : pandas.DataFrame
        DataFrame containing risk metrics for cryptocurrencies.
    crypto_list : list
        List of cryptocurrencies to include in the plot.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Risk-return profile figure.
    """
    fig = px.scatter(
        metrics,
        x='Volatility (Annualized)',
        y='Return (Annualized)',
        text=metrics.index,
        size='Sharpe Ratio',
        hover_data=['Maximum Drawdown', 'Sortino Ratio'],
        title="Risk-Return Profile of Cryptocurrencies"
    )
    
    fig.update_traces(textposition='top center')
    
    fig.update_layout(
        xaxis_title="Risk (Annualized Volatility)",
        yaxis_title="Return (Annualized)",
        height=600
    )
    
    # Add a reference point at (0,0)
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        marker=dict(
            color='black',
            size=10,
            symbol='x'
        ),
        name='Risk-Free'
    ))
    
    return fig

def plot_feature_importance(feature_importance, title="Feature Importance"):
    """
    Create a bar plot of feature importance.
    
    Parameters:
    -----------
    feature_importance : pandas.Series
        Series containing feature importance values.
    title : str, optional
        Plot title.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Feature importance figure.
    """
    # Sort feature importance
    feature_importance = feature_importance.sort_values(ascending=False)
    
    fig = px.bar(
        x=feature_importance.values,
        y=feature_importance.index,
        orientation='h',
        title=title
    )
    
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=500
    )
    
    return fig
