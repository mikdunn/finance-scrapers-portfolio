import pandas as pd
from utils.scrapers import fetch_dynamic
from utils.sentiment import score_text

tickers = ['AAPL', 'TSLA', 'SPY']  # Your list
news_data = []

for ticker in tickers:
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    soup = fetch_dynamic(url)  # News often JS-loaded
    headlines = [h.text for h in soup.select('.svelte-3az9ch')]  # Inspect selectors
    for hl in headlines[:10]:  # Limit
        news_data.append({'ticker': ticker, 'headline': hl, 'sentiment': score_text(hl)})

df = pd.DataFrame(news_data)
pivot = df.pivot_table(values='sentiment', index='ticker', columns='headline', aggfunc='mean').fillna(0)

import plotly.express as px
fig = px.imshow(pivot, title='Stock Sentiment Heatmap', aspect='auto')
fig.write_html('sentiment_heatmap.html')
fig.show()
df.to_csv('data/sentiments.csv')
