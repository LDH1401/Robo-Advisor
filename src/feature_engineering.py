import pandas as pd
import json
import os

def run_feature_engineering():
    print("--- Đang xử lý dữ liệu với logic Ticker-Sentiment ---")
    
    with open('data/stock_prices.json', 'r') as f:
        df_stock = pd.DataFrame(json.load(f))
    df_stock['date'] = pd.to_datetime(df_stock['date']).dt.normalize()
    df_stock = df_stock.sort_values(['ticker', 'date'])

    df_stock['return'] = df_stock.groupby('ticker')['close'].pct_change()
    df_stock['volatility_7d'] = df_stock.groupby('ticker')['return'].transform(lambda x: x.rolling(7).std())

    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df_stock['rsi'] = df_stock.groupby('ticker')['close'].transform(lambda x: calculate_rsi(x))

    with open('data/articles.json', 'r') as f:
        df_art = pd.DataFrame(json.load(f))
    with open('data/article_sentiments.json', 'r') as f:
        df_sent = pd.DataFrame(json.load(f))

    df_news = pd.merge(df_art, df_sent, left_on='_id', right_on='article_id')
    df_news['date'] = pd.to_datetime(df_news['published_at']).dt.normalize()

    df_news = df_news.explode('mentioned_tickers')
    df_news = df_news.rename(columns={'mentioned_tickers': 'ticker'})

    daily_ticker_sentiment = df_news.groupby(['date', 'ticker']).agg({
        'sentiment_score': 'mean'
    }).reset_index()

    final_df = pd.merge(df_stock, daily_ticker_sentiment, on=['date', 'ticker'], how='left')
    final_df['sentiment_score'] = final_df['sentiment_score'].fillna(0)

    final_df['target'] = final_df.groupby('ticker')['return'].shift(-1)
    final_df = final_df.dropna()

    os.makedirs('data', exist_ok=True)
    final_df.to_csv('data/processed_multi_ticker.csv', index=False)
    print(f"--- Đã lưu dữ liệu sạch: {len(final_df)} dòng ---")

if __name__ == "__main__":
    run_feature_engineering()