import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score
)
import datetime

def prepare_features(data, crypto, features=None, is_classification=False, threshold=0.01):
    """
    Prepare features for machine learning models.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The preprocessed cryptocurrency data.
    crypto : str
        The cryptocurrency to analyze.
    features : list, optional
        List of features to include.
    is_classification : bool, optional
        Whether to prepare for classification (direction prediction).
    threshold : float, optional
        Threshold for price increase/decrease classification.
        
    Returns:
    --------
    tuple
        (X, y) features and target variable.
    """
    df = data.copy()
    
    # Default features if none specified
    if features is None:
        features = ['Price', 'MA_7', 'MA_30', 'RSI']
    
    # Extract features
    X_cols = []
    for feature in features:
        if feature == 'Price':
            X_cols.append(crypto)
        elif feature == 'Volume' and f'{crypto}_Volume' in df.columns:
            X_cols.append(f'{crypto}_Volume')
        else:
            feature_col = f'{crypto}_{feature}'
            if feature_col in df.columns:
                X_cols.append(feature_col)
    
    # Check if we have enough features
    if len(X_cols) == 0:
        # Use default columns if no features match
        X_cols = [col for col in df.columns if col != 'Date' and col.startswith(crypto)]
        if not X_cols:
            X_cols = [crypto]
    
    X = df[X_cols].values
    
    # Create target variable
    if is_classification:
        # For classification: predict direction (up/down)
        y = np.where(df[crypto].pct_change().shift(-1) > threshold, 1, 0)
    else:
        # For regression: predict next price
        y = df[crypto].shift(-1).values
    
    # Remove last row (we don't have y for it)
    X = X[:-1]
    y = y[:-1]
    
    return X, y

def train_model(data, crypto, model_type='linear', test_size=0.2, features=None, 
               n_clusters=5, threshold=0.01, **kwargs):
    """
    Train a machine learning model on cryptocurrency data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The preprocessed cryptocurrency data.
    crypto : str
        The cryptocurrency to analyze.
    model_type : str, optional
        Type of model to train ('linear', 'logistic', or 'kmeans').
    test_size : float, optional
        Proportion of data to use for testing.
    features : list, optional
        List of features to include.
    n_clusters : int, optional
        Number of clusters for K-means clustering.
    threshold : float, optional
        Threshold for price increase/decrease classification.
    **kwargs : dict
        Additional parameters for model training.
        
    Returns:
    --------
    tuple
        Model and results (varies by model type).
    """
    if model_type == 'linear':
        # Linear Regression for price prediction
        X, y = prepare_features(data, crypto, features, is_classification=False)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        predictions = model.predict(X_test_scaled)
        
        # Feature importance (coefficients)
        if features is None:
            features = ['Price', 'MA_7', 'MA_30', 'RSI']
            
        feature_importance = pd.Series(
            model.coef_,
            index=[f for f in features if f in data.columns or f == 'Price'][:len(model.coef_)]
        )
        
        return model, X_test_scaled, y_test, predictions, feature_importance
    
    elif model_type == 'logistic':
        # Logistic Regression for direction prediction
        X, y = prepare_features(data, crypto, features, is_classification=True, threshold=threshold)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(max_iter=1000, **kwargs)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        predictions = model.predict(X_test_scaled)
        
        # Feature importance (coefficients)
        if features is None:
            features = ['Price', 'MA_7', 'MA_30', 'RSI']
            
        feature_importance = pd.Series(
            model.coef_[0],
            index=[f for f in features if f in data.columns or f == 'Price'][:len(model.coef_[0])]
        )
        
        return model, X_test_scaled, y_test, predictions, feature_importance
    
    elif model_type == 'kmeans':
        # K-means clustering for market regimes
        
        # Prepare data for clustering
        if features is None:
            features = ['Return', 'Volatility', 'RSI']
        
        cluster_data = pd.DataFrame()
        
        # Daily returns
        if 'Return' in features:
            cluster_data['Return'] = data[crypto].pct_change()
        
        # Volatility (20-day rolling)
        if 'Volatility' in features:
            cluster_data['Volatility'] = data[crypto].pct_change().rolling(window=20).std()
        
        # RSI
        if 'RSI' in features and f'{crypto}_RSI' in data.columns:
            cluster_data['RSI'] = data[f'{crypto}_RSI']
        
        # Volume change
        if 'Volume_Change' in features and f'{crypto}_Volume' in data.columns:
            cluster_data['Volume_Change'] = data[f'{crypto}_Volume'].pct_change()
        
        # MA Crossover
        if 'MA_Crossover' in features and f'{crypto}_MA7' in data.columns and f'{crypto}_MA30' in data.columns:
            cluster_data['MA_Crossover'] = data[f'{crypto}_MA7'] - data[f'{crypto}_MA30']
        
        # Drop NaN values
        cluster_data.dropna(inplace=True)
        
        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Train K-means model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to data
        cluster_data['Cluster'] = clusters
        
        # Get centroids
        centroids = kmeans.cluster_centers_
        
        return kmeans, cluster_data, centroids

def evaluate_model(model, X_test, y_test, predictions, is_classification=False):
    """
    Evaluate machine learning model performance.
    
    Parameters:
    -----------
    model : object
        Trained model.
    X_test : numpy.ndarray
        Test features.
    y_test : numpy.ndarray
        Test target values.
    predictions : numpy.ndarray
        Model predictions.
    is_classification : bool, optional
        Whether the model is a classifier.
        
    Returns:
    --------
    dict
        Performance metrics.
    """
    results = {}
    
    if is_classification:
        # Classification metrics
        results['accuracy'] = accuracy_score(y_test, predictions)
        results['precision'] = precision_score(y_test, predictions, zero_division=0)
        results['recall'] = recall_score(y_test, predictions, zero_division=0)
        results['f1'] = f1_score(y_test, predictions, zero_division=0)
        results['confusion_matrix'] = confusion_matrix(y_test, predictions)
        results['y_test'] = y_test
        results['predictions'] = predictions
    else:
        # Regression metrics
        results['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
        results['mae'] = mean_absolute_error(y_test, predictions)
        results['r2'] = r2_score(y_test, predictions)
        # Mean absolute percentage error
        results['mape'] = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        results['y_test'] = y_test
        results['predictions'] = predictions
    
    return results

def predict_future(model, data, crypto, days=30, features=None, is_classification=False):
    """
    Generate future predictions using the trained model.
    
    Parameters:
    -----------
    model : object
        Trained model.
    data : pandas.DataFrame
        The preprocessed cryptocurrency data.
    crypto : str
        The cryptocurrency to analyze.
    days : int, optional
        Number of days to forecast.
    features : list, optional
        List of features to include.
    is_classification : bool, optional
        Whether the model is a classifier.
        
    Returns:
    --------
    numpy.ndarray
        Future predictions.
    """
    # Make a copy of the most recent data point to predict from
    df = data.copy().tail(1)
    
    # Get historical price info to determine trend
    historical_prices = data[crypto].tail(30).values
    if len(historical_prices) >= 2:
        # Calculate average daily change over recent history (last 30 days)
        avg_daily_change = (historical_prices[-1] - historical_prices[0]) / len(historical_prices)
    else:
        avg_daily_change = 0
    
    # Initialize array to store predictions
    predictions = []
    
    # Initial price (last known price)
    last_price = data[crypto].iloc[-1]
    
    # Calculate volatility from historical data
    if len(historical_prices) > 5:
        historical_returns = np.diff(historical_prices) / historical_prices[:-1]
        volatility = np.std(historical_returns)
    else:
        volatility = 0.01  # Default volatility if not enough data
    
    # Generate predictions for each future day
    for i in range(days):
        # Extract features (from the last row)
        X_cols = []
        
        if features is None:
            features = ['Price', 'MA_7', 'MA_30', 'RSI']
        
        for feature in features:
            if feature == 'Price':
                X_cols.append(crypto)
            elif feature == 'Volume' and f'{crypto}_Volume' in df.columns:
                X_cols.append(f'{crypto}_Volume')
            else:
                feature_col = f'{crypto}_{feature}'
                if feature_col in df.columns:
                    X_cols.append(feature_col)
        
        # If no features match, use the first few columns
        if len(X_cols) == 0:
            X_cols = [col for col in df.columns if col != 'Date' and col.startswith(crypto)]
            if not X_cols:
                X_cols = [crypto]
        
        X = df[X_cols].values.reshape(1, -1)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Make prediction
        if is_classification:
            # Classification (direction)
            pred = model.predict(X_scaled)[0]
        else:
            # Regression (price)
            base_pred = model.predict(X_scaled)[0]
            
            # Add some randomness based on historical volatility
            noise = np.random.normal(0, volatility * last_price)
            
            # Incorporate trend (positive or negative) based on historical data
            # and add a slight upward bias for more realistic projections
            trend_component = avg_daily_change * (i + 1) * 0.5
            
            # Final prediction combines model output, trend, and noise
            pred = base_pred + trend_component + noise
            
            # Ensure prediction is positive
            pred = max(pred, last_price * 0.95)
        
        # Store prediction
        predictions.append(pred)
        
        # Update data with prediction for next iteration
        if not is_classification:
            last_price = pred
            df[crypto] = pred
            
            # Update technical indicators
            if f'{crypto}_MA7' in df.columns:
                # Simple approach: assume new value affects MAs
                df[f'{crypto}_MA7'] = (df[f'{crypto}_MA7'] * 6 + pred) / 7
            
            if f'{crypto}_MA14' in df.columns:
                df[f'{crypto}_MA14'] = (df[f'{crypto}_MA14'] * 13 + pred) / 14
            
            if f'{crypto}_MA30' in df.columns:
                df[f'{crypto}_MA30'] = (df[f'{crypto}_MA30'] * 29 + pred) / 30
            
            # Update RSI (simplified)
            if f'{crypto}_RSI' in df.columns:
                # Keep RSI in a reasonable range with slight changes
                current_rsi = df[f'{crypto}_RSI'].values[0]
                new_rsi = current_rsi + np.random.uniform(-3, 3)  # Random walk
                # Keep within bounds
                new_rsi = max(min(new_rsi, 80), 20)
                df[f'{crypto}_RSI'] = new_rsi
    
    return np.array(predictions)
