import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from lstm_model import FinancialLSTM

WINDOW_SIZE = 60
BATCH_SIZE = 32
EPOCHS = 60

def create_sequences_by_ticker(df, window):
    X, y = [], []
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker]
        feat = ticker_data[['return', 'volatility_7d', 'sentiment_score']].values
        targ = ticker_data['target'].values
        
        for i in range(len(feat) - window):
            X.append(feat[i:i+window])
            y.append(targ[i+window])
    return np.array(X), np.array(y)

def train():
    df = pd.read_csv('data/processed_multi_ticker.csv')
    
    scaler = MinMaxScaler()
    df[['return', 'volatility_7d', 'sentiment_score']] = scaler.fit_transform(
        df[['return', 'volatility_7d', 'sentiment_score']]
    )
    joblib.dump(scaler, 'models/scaler.pkl')

    X_np, y_np = create_sequences_by_ticker(df, WINDOW_SIZE)
    
    split = int(len(X_np) * 0.8)
    X_train = torch.tensor(X_np[:split], dtype=torch.float32)
    y_train = torch.tensor(y_np[:split], dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    model = FinancialLSTM(input_size=3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("--- Đang huấn luyện ---")
    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x).squeeze(), batch_y)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), 'models/stock_lstm_model.pth')
    print("--- Đã lưu mô hình và Scaler ---")

if __name__ == "__main__":
    train()