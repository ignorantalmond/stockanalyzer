"""
Reddit data fetching and analysis module
"""

import praw
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from collections import Counter
from textblob import TextBlob
from typing import List, Dict, Optional
import yfinance as yf

from .config import Config
from .enums import PUMP_KEYWORDS, DUMP_KEYWORDS, OPTIONS_CALLS_KEYWORDS, OPTIONS_PUTS_KEYWORDS, COMMON_WORDS

class RedditAnalyzer:
    """Handles Reddit data fetching and sentiment analysis"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """Initialize Reddit API connection"""
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.config = Config()
    
    def get_trending_stocks(self, subreddit: str = 'wallstreetbets', limit: int = None) -> List[str]:
        """
        Identify currently trending stocks from Reddit discussions
        
        Args:
            subreddit: Subreddit name to analyze
            limit: Number of posts to analyze
            
        Returns:
            List of trending stock symbols
        """
        if limit is None:
            limit = self.config.REDDIT_SEARCH_LIMIT
            
        try:
            subreddit_obj = self.reddit.subreddit(subreddit)
            stock_mentions = Counter()
            
            # Get hot posts from the last 24 hours
            for post in subreddit_obj.hot(limit=limit):
                # Extract stock symbols from title and text
                text = f"{post.title} {post.selftext}".upper()
                
                # Find potential stock symbols (1-5 letters, often prefixed with $)
                symbols = re.findall(r'\$?([A-Z]{1,5})(?=\s|$|[^A-Z])', text)
                
                for symbol in symbols:
                    # Filter out common words that aren't stocks
                    if symbol not in COMMON_WORDS:
                        # Weight by post score and comments
                        weight = max(1, post.score) * max(1, post.num_comments / 10)
                        stock_mentions[symbol] += weight
            
            # Get top mentioned stocks
            trending = [symbol for symbol, count in stock_mentions.most_common(20)]
            
            # Filter for known stocks and validate they exist
            validated_stocks = []
            for symbol in trending:
                try:
                    # Quick validation - try to get basic info
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    if info and 'symbol' in info and info.get('regularMarketPrice', 0) > 0:
                        validated_stocks.append(symbol)
                        if len(validated_stocks) >= self.config.TRENDING_STOCKS_LIMIT:
                            break
                except:
                    continue
                    
            return validated_stocks
            
        except Exception as e:
            print(f"Error getting trending stocks: {e}")
            return self.config.KNOWN_MEME_STOCKS[:self.config.TRENDING_STOCKS_LIMIT]
    
    def get_posts_for_symbol(self, symbol: str, subreddit: str = 'wallstreetbets', 
                           days_back: int = None, limit: int = None) -> pd.DataFrame:
        """
        Fetch recent Reddit posts mentioning a stock symbol
        
        Args:
            symbol: Stock ticker symbol
            subreddit: Subreddit to search
            days_back: Number of days to look back
            limit: Maximum number of posts to fetch
            
        Returns:
            DataFrame with post data
        """
        if days_back is None:
            days_back = 30
        if limit is None:
            limit = self.config.REDDIT_POST_LIMIT
            
        posts_data = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            subreddit_obj = self.reddit.subreddit(subreddit)
            search_query = f"${symbol} OR {symbol}"
            
            for post in subreddit_obj.search(search_query, time_filter='month', limit=limit):
                post_date = datetime.fromtimestamp(post.created_utc)
                if post_date >= cutoff_date:
                    posts_data.append({
                        'date': post_date,
                        'title': post.title,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'upvote_ratio': post.upvote_ratio,
                        'selftext': post.selftext,
                        'created_utc': post.created_utc
                    })
                    
            return pd.DataFrame(posts_data)
            
        except Exception as e:
            print(f"Error fetching Reddit data for {symbol}: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment_and_keywords(self, text: str) -> Dict:
        """
        Analyze sentiment and detect keywords in text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores and keyword counts
        """
        text_str = str(text).lower()
        
        # Sentiment analysis using TextBlob
        sentiment = TextBlob(text_str).sentiment.polarity
        
        # Count keywords
        pump_count = sum(1 for keyword in PUMP_KEYWORDS if keyword in text_str)
        dump_count = sum(1 for keyword in DUMP_KEYWORDS if keyword in text_str)
        calls_mentions = sum(1 for keyword in OPTIONS_CALLS_KEYWORDS if keyword in text_str)
        puts_mentions = sum(1 for keyword in OPTIONS_PUTS_KEYWORDS if keyword in text_str)
        
        return {
            'sentiment': sentiment,
            'pump_keywords': pump_count,
            'dump_keywords': dump_count,
            'calls_mentions': calls_mentions,
            'puts_mentions': puts_mentions
        }
    
    def calculate_reddit_metrics(self, reddit_df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive Reddit metrics
        
        Args:
            reddit_df: DataFrame with Reddit posts
            
        Returns:
            Dictionary with calculated metrics
        """
        if reddit_df.empty:
            return {
                'avg_sentiment': 0,
                'post_count': 0,
                'total_score': 0,
                'avg_score': 0,
                'pump_keyword_ratio': 0,
                'dump_keyword_ratio': 0,
                'calls_put_ratio': 1,
                'activity_trend': 'neutral'
            }
        
        # Add sentiment analysis to each post
        sentiment_data = reddit_df.apply(
            lambda x: self.analyze_sentiment_and_keywords(x['title'] + ' ' + str(x['selftext'])), 
            axis=1
        )
        
        # Extract sentiment metrics
        sentiments = [data['sentiment'] for data in sentiment_data]
        pump_keywords = [data['pump_keywords'] for data in sentiment_data]
        dump_keywords = [data['dump_keywords'] for data in sentiment_data]
        calls_mentions = [data['calls_mentions'] for data in sentiment_data]
        puts_mentions = [data['puts_mentions'] for data in sentiment_data]
        
        # Calculate metrics
        total_keywords = sum(pump_keywords) + sum(dump_keywords)
        pump_ratio = sum(pump_keywords) / max(total_keywords, 1)
        dump_ratio = sum(dump_keywords) / max(total_keywords, 1)
        
        total_options = sum(calls_mentions) + sum(puts_mentions)
        calls_put_ratio = sum(calls_mentions) / max(sum(puts_mentions), 1)
        
        # Recent activity trend (last 7 days vs previous 7 days)
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_posts = reddit_df[reddit_df['date'] >= recent_cutoff]
        older_posts = reddit_df[reddit_df['date'] < recent_cutoff]
        
        recent_activity = len(recent_posts)
        older_activity = len(older_posts)
        
        if older_activity > 0:
            activity_ratio = recent_activity / older_activity
            if activity_ratio > 1.5:
                activity_trend = 'increasing'
            elif activity_ratio < 0.5:
                activity_trend = 'decreasing'
            else:
                activity_trend = 'stable'
        else:
            activity_trend = 'new' if recent_activity > 0 else 'none'
        
        return {
            'avg_sentiment': np.mean(sentiments) if sentiments else 0,
            'post_count': len(reddit_df),
            'total_score': reddit_df['score'].sum(),
            'avg_score': reddit_df['score'].mean(),
            'pump_keyword_ratio': pump_ratio,
            'dump_keyword_ratio': dump_ratio,
            'calls_put_ratio': calls_put_ratio,
            'activity_trend': activity_trend,
            'recent_posts': recent_activity,
            'total_pump_keywords': sum(pump_keywords),
            'total_dump_keywords': sum(dump_keywords),
            'options_mentions': total_options
        }
    
    def get_sentiment_summary(self, reddit_metrics: Dict) -> str:
        """
        Get human-readable sentiment summary
        
        Args:
            reddit_metrics: Reddit metrics dictionary
            
        Returns:
            Sentiment summary string
        """
        avg_sentiment = reddit_metrics.get('avg_sentiment', 0)
        pump_ratio = reddit_metrics.get('pump_keyword_ratio', 0)
        dump_ratio = reddit_metrics.get('dump_keyword_ratio', 0)
        activity_trend = reddit_metrics.get('activity_trend', 'neutral')
        
        # Determine overall sentiment
        if avg_sentiment > 0.2 and pump_ratio > 0.6:
            sentiment_desc = "Very Bullish"
        elif avg_sentiment > 0.1 or pump_ratio > 0.4:
            sentiment_desc = "Bullish"
        elif avg_sentiment < -0.2 and dump_ratio > 0.6:
            sentiment_desc = "Very Bearish"
        elif avg_sentiment < -0.1 or dump_ratio > 0.4:
            sentiment_desc = "Bearish"
        else:
            sentiment_desc = "Neutral"
        
        # Add activity trend
        trend_desc = {
            'increasing': 'Growing Interest',
            'decreasing': 'Declining Interest',
            'stable': 'Stable Interest',
            'new': 'New Interest',
            'none': 'No Interest'
        }.get(activity_trend, 'Unknown')
        
        return f"{sentiment_desc} ({trend_desc})"