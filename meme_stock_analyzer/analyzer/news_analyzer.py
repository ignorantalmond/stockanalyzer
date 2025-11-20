"""
News data fetching and sentiment analysis module
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from textblob import TextBlob
from typing import Dict, List, Optional
import time

from .config import Config

class NewsAnalyzer:
    """Handles news data fetching and sentiment analysis"""
    
    def __init__(self, alpha_vantage_key: Optional[str] = None, finnhub_key: Optional[str] = None):
        """Initialize news analyzer with API keys"""
        self.alpha_vantage_key = alpha_vantage_key
        self.finnhub_key = finnhub_key
        self.config = Config()
    
    def get_stock_news(self, symbol: str, days_back: int = None) -> pd.DataFrame:
        """
        Fetch recent news for a stock symbol from multiple sources
        
        Args:
            symbol: Stock ticker symbol
            days_back: Number of days to look back
            
        Returns:
            DataFrame with news articles
        """
        if days_back is None:
            days_back = self.config.NEWS_HISTORY_DAYS
            
        news_data = []
        
        # Try Alpha Vantage News API
        if self.alpha_vantage_key:
            news_data.extend(self._get_alpha_vantage_news(symbol, days_back))
        
        # Try Finnhub News API  
        if self.finnhub_key:
            news_data.extend(self._get_finnhub_news(symbol, days_back))
        
        # Fallback: Yahoo Finance news
        if not news_data:
            news_data.extend(self._get_yahoo_news(symbol))
        
        return pd.DataFrame(news_data)
    
    def _get_alpha_vantage_news(self, symbol: str, days_back: int) -> List[Dict]:
        """Fetch news from Alpha Vantage API"""
        news_data = []
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'time_from': (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%dT%H%M'),
                'limit': 20,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'feed' in data:
                    for article in data['feed']:
                        news_data.append({
                            'date': datetime.strptime(article['time_published'], '%Y%m%dT%H%M%S'),
                            'title': article['title'],
                            'summary': article.get('summary', ''),
                            'sentiment': float(article.get('overall_sentiment_score', 0)),
                            'source': article.get('source', 'Alpha Vantage'),
                            'url': article.get('url', ''),
                            'api_source': 'alpha_vantage'
                        })
        except Exception as e:
            print(f"Alpha Vantage news error for {symbol}: {e}")
        
        return news_data
    
    def _get_finnhub_news(self, symbol: str, days_back: int) -> List[Dict]:
        """Fetch news from Finnhub API"""
        news_data = []
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.finnhub_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for article in data:
                    news_data.append({
                        'date': datetime.fromtimestamp(article['datetime']),
                        'title': article['headline'],
                        'summary': article.get('summary', ''),
                        'sentiment': 0,  # Finnhub doesn't provide sentiment directly
                        'source': article.get('source', 'Finnhub'),
                        'url': article.get('url', ''),
                        'api_source': 'finnhub'
                    })
        except Exception as e:
            print(f"Finnhub news error for {symbol}: {e}")
        
        return news_data
    
    def _get_yahoo_news(self, symbol: str) -> List[Dict]:
        """Fetch news from Yahoo Finance (fallback)"""
        news_data = []
        
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            for article in news[:10]:  # Limit to 10 articles
                news_data.append({
                    'date': datetime.fromtimestamp(article.get('providerPublishTime', time.time())),
                    'title': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'sentiment': 0,
                    'source': article.get('publisher', 'Yahoo Finance'),
                    'url': article.get('link', ''),
                    'api_source': 'yahoo'
                })
        except Exception as e:
            print(f"Yahoo Finance news error for {symbol}: {e}")
        
        return news_data
    
    def analyze_news_sentiment(self, news_df: pd.DataFrame) -> Dict:
        """
        Analyze sentiment of news articles
        
        Args:
            news_df: DataFrame with news articles
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if news_df.empty:
            return {
                'avg_sentiment': 0,
                'news_count': 0,
                'bullish_news': 0,
                'bearish_news': 0,
                'neutral_news': 0,
                'recent_news': [],
                'sentiment_trend': 'neutral',
                'confidence': 0,
                'has_major_catalyst': False,
                'catalyst_score': 0
            }
        
        sentiments = []
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for _, article in news_df.iterrows():
            # Use provided sentiment or calculate with TextBlob
            if article['sentiment'] != 0:
                sentiment = article['sentiment']
            else:
                text = f"{article['title']} {article['summary']}"
                sentiment = TextBlob(text).sentiment.polarity
            
            sentiments.append(sentiment)
            
            # Classify sentiment
            if sentiment > self.config.BULLISH_SENTIMENT_THRESHOLD:
                bullish_count += 1
            elif sentiment < self.config.BEARISH_SENTIMENT_THRESHOLD:
                bearish_count += 1
            else:
                neutral_count += 1
        
        # Calculate sentiment trend (recent vs older articles)
        if len(news_df) > 3:
            recent_articles = news_df.head(len(news_df)//2)
            older_articles = news_df.tail(len(news_df)//2)
            
            recent_sentiment = np.mean([
                self._get_article_sentiment(article) for _, article in recent_articles.iterrows()
            ])
            older_sentiment = np.mean([
                self._get_article_sentiment(article) for _, article in older_articles.iterrows()
            ])
            
            sentiment_change = recent_sentiment - older_sentiment
            if sentiment_change > 0.1:
                sentiment_trend = 'improving'
            elif sentiment_change < -0.1:
                sentiment_trend = 'deteriorating'
            else:
                sentiment_trend = 'stable'
        else:
            sentiment_trend = 'insufficient_data'
        
        # Calculate confidence based on number of articles and sentiment consistency
        confidence = min(1.0, len(news_df) / 10)  # More articles = higher confidence
        sentiment_std = np.std(sentiments) if len(sentiments) > 1 else 1.0
        confidence *= (1.0 - min(sentiment_std, 1.0))  # Less variance = higher confidence
        
        # Detect major catalysts
        catalyst_detection = self.detect_news_catalysts(news_df)
        
        return {
            'avg_sentiment': np.mean(sentiments) if sentiments else 0,
            'news_count': len(news_df),
            'bullish_news': bullish_count,
            'bearish_news': bearish_count,
            'neutral_news': neutral_count,
            'recent_news': news_df.head(3).to_dict('records'),
            'sentiment_trend': sentiment_trend,
            'confidence': confidence,
            'sentiment_distribution': {
                'bullish_pct': bullish_count / len(news_df) * 100,
                'bearish_pct': bearish_count / len(news_df) * 100,
                'neutral_pct': neutral_count / len(news_df) * 100
            },
            'has_major_catalyst': catalyst_detection['has_major_catalyst'],
            'catalyst_score': catalyst_detection['catalyst_score']
        }
    
    def _get_article_sentiment(self, article: pd.Series) -> float:
        """Get sentiment for a single article"""
        if article['sentiment'] != 0:
            return article['sentiment']
        else:
            text = f"{article['title']} {article['summary']}"
            return TextBlob(text).sentiment.polarity
    
    def detect_news_catalysts(self, news_df: pd.DataFrame) -> Dict:
        """
        Detect potential news catalysts that could affect stock price
        
        Args:
            news_df: DataFrame with news articles
            
        Returns:
            Dictionary with catalyst analysis
        """
        if news_df.empty:
            return {'catalysts': [], 'catalyst_score': 0, 'has_major_catalyst': False}
        
        # Keywords that often indicate important catalysts
        catalyst_keywords = {
            'earnings': ['earnings', 'quarterly results', 'eps', 'revenue', 'guidance'],
            'corporate_action': ['merger', 'acquisition', 'buyout', 'dividend', 'split'],
            'regulatory': ['fda approval', 'clinical trial', 'investigation', 'lawsuit'],
            'partnership': ['partnership', 'deal', 'contract', 'agreement'],
            'management': ['ceo', 'resignation', 'hiring', 'management change'],
            'financial': ['bankruptcy', 'debt', 'funding', 'loan', 'restructuring']
        }
        
        catalysts = []
        total_catalyst_score = 0
        
        for _, article in news_df.iterrows():
            text = f"{article['title']} {article['summary']}".lower()
            article_catalysts = []
            
            for category, keywords in catalyst_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        article_catalysts.append(category)
                        break
            
            if article_catalysts:
                sentiment = self._get_article_sentiment(article)
                catalyst_score = abs(sentiment) * len(article_catalysts)
                total_catalyst_score += catalyst_score
                
                catalysts.append({
                    'title': article['title'],
                    'date': article['date'],
                    'categories': article_catalysts,
                    'sentiment': sentiment,
                    'score': catalyst_score
                })
        
        # Sort catalysts by score (most important first)
        catalysts.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'catalysts': catalysts[:5],  # Top 5 catalysts
            'catalyst_score': total_catalyst_score,
            'has_major_catalyst': total_catalyst_score > 2.0
        }
    
    def get_sentiment_summary(self, news_sentiment: Dict) -> str:
        """
        Get human-readable news sentiment summary
        
        Args:
            news_sentiment: News sentiment analysis results
            
        Returns:
            Summary string
        """
        avg_sentiment = news_sentiment.get('avg_sentiment', 0)
        trend = news_sentiment.get('sentiment_trend', 'neutral')
        confidence = news_sentiment.get('confidence', 0)
        news_count = news_sentiment.get('news_count', 0)
        
        # Determine sentiment level
        if avg_sentiment > 0.3:
            sentiment_level = "Very Bullish"
        elif avg_sentiment > 0.1:
            sentiment_level = "Bullish"
        elif avg_sentiment < -0.3:
            sentiment_level = "Very Bearish"
        elif avg_sentiment < -0.1:
            sentiment_level = "Bearish"
        else:
            sentiment_level = "Neutral"
        
        # Add trend information
        trend_desc = {
            'improving': 'Improving',
            'deteriorating': 'Deteriorating',
            'stable': 'Stable',
            'insufficient_data': 'Limited Data'
        }.get(trend, 'Unknown')
        
        # Add confidence level
        if confidence > 0.7:
            conf_level = "High"
        elif confidence > 0.4:
            conf_level = "Medium"
        else:
            conf_level = "Low"
        
        return f"{sentiment_level} ({trend_desc}, {conf_level} confidence, {news_count} articles)"