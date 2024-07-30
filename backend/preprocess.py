import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    # Load your stock data
    df = pd.read_csv(file_path)
    
    # Example preprocessing steps
    df = df[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Scale features
    scaler = MinMaxScaler()
    df['Close'] = scaler.fit_transform(df[['Close']])
    
    return df, scaler

def create_dataset(df, look_back=60):
    X, y = [], []
    for i in range(len(df) - look_back):
        X.append(df.iloc[i:i + look_back].values)
        y.append(df.iloc[i + look_back].values)
    return np.array(X), np.array(y)


