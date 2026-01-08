from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def score_text(text):
    return analyzer.polarity_scores(text)['compound']
