"""
Reddit data fetching and analysis module
Enhanced with penny stock and pump/dump subreddits
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
import time

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
        
        # Comprehensive list of subreddits for pump/dump and meme stock analysis
        self.target_subreddits = [
            # Primary meme stock subreddits
            'wallstreetbets',
            'Superstonk',
            'amcstock', 
            'GME',
            
            # Penny stock subreddits (major pump/dump activity)
            'pennystocks',
            'RobinHoodPennyStocks',
            'penny_stocks',
            'PennyStocksDD',
            'pennystockwatch',
            'OTCMarkets',
            
            # Small cap and speculative trading
            'SmallCapStocks',
            'stocks',
            'SecurityAnalysis',
            'ValueInvesting',
            'investing',
            'StockMarket',
            
            # Options and derivatives
            'options',
            'thetagang',
            'wallstreetbetsnew',
            'wallstreetbetsOGs',
            
            # Crypto-related (often pump/dump activity)
            'CryptoPumping',
            'SatoshiStreetBets',
            'CryptoMoonShots',
            
            # Other trading communities
            'daytrading',
            'SwingTrading',
            'trading',
            'RobinHood',
            'SecurityAnalysis',
            'financialindependence',
            
            # Smaller/newer communities
            'Shortsqueeze',
            'squeezeplays',
            'BBIG',
            'SPRT',
            'PROG'
        ]
        
        print(f"🎯 Monitoring {len(self.target_subreddits)} subreddits for pump/dump activity")
    
    def get_trending_stocks(self, limit: int = None) -> List[str]:
        """
        Identify currently trending stocks from multiple Reddit communities
        
        Args:
            limit: Number of posts to analyze per subreddit
            
        Returns:
            List of trending stock symbols
        """
        if limit is None:
            limit = self.config.REDDIT_SEARCH_LIMIT // len(self.target_subreddits)  # Distribute across subreddits
            
        stock_mentions = Counter()
        total_posts_analyzed = 0
        
        print(f"🔍 Scanning {len(self.target_subreddits)} subreddits...")
        
        for subreddit_name in self.target_subreddits:
            try:
                subreddit_mentions = self._analyze_subreddit_for_stocks(subreddit_name, limit)
                
                # Weight mentions by subreddit relevance
                weight = self._get_subreddit_weight(subreddit_name)
                
                for symbol, count in subreddit_mentions.items():
                    stock_mentions[symbol] += count * weight
                
                total_posts_analyzed += min(limit, len(subreddit_mentions))
                
                # Rate limiting to avoid API issues
                time.sleep(0.5)
                
            except Exception as e:
                print(f"⚠️  Error analyzing r/{subreddit_name}: {e}")
                continue
        
        print(f"📊 Analyzed {total_posts_analyzed} posts across {len(self.target_subreddits)} subreddits")
        
        # Get top mentioned stocks
        trending = [symbol for symbol, count in stock_mentions.most_common(30)]
        
        # Filter and validate stocks
        validated_stocks = self._validate_stock_symbols(trending)
        
        # Limit to configured amount
        final_stocks = validated_stocks[:self.config.TRENDING_STOCKS_LIMIT]
        
        print(f"🎯 Found {len(final_stocks)} trending stocks: {', '.join(final_stocks)}")
        return final_stocks
    
    def _analyze_subreddit_for_stocks(self, subreddit_name: str, limit: int) -> Counter:
        """Analyze a single subreddit for stock mentions"""
        
        stock_mentions = Counter()
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get hot posts from the last 24 hours
            posts_analyzed = 0
            
            for post in subreddit.hot(limit=limit):
                # Skip very old posts (older than 3 days)
                post_age = datetime.now() - datetime.fromtimestamp(post.created_utc)
                if post_age.days > 3:
                    continue
                
                # Extract stock symbols from title and text
                text = f"{post.title} {post.selftext}".upper()
                
                # Find potential stock symbols
                symbols = self._extract_stock_symbols(text)
                
                for symbol in symbols:
                    # Weight by post engagement
                    engagement_weight = self._calculate_engagement_weight(post)
                    stock_mentions[symbol] += engagement_weight
                
                posts_analyzed += 1
            
            print(f"  📈 r/{subreddit_name}: {posts_analyzed} posts, {len(stock_mentions)} unique symbols")
            
        except Exception as e:
            print(f"  ❌ Error with r/{subreddit_name}: {e}")
        
        return stock_mentions
    
    def _extract_stock_symbols(self, text: str) -> List[str]:
        """Extract potential stock symbols from text"""
        
        # Find potential stock symbols (1-5 letters, often prefixed with $)
        symbols = re.findall(r'\$?([A-Z]{1,5})(?=\s|$|[^A-Z])', text)
        
        # Filter out common words and obvious non-stocks
        filtered_symbols = []
        for symbol in symbols:
            if (symbol not in COMMON_WORDS and 
                len(symbol) >= 2 and  # At least 2 characters
                symbol not in ['TO', 'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 
                              'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS',
                              'DD', 'TA', 'PT', 'ATH', 'ATL', 'CEO', 'CFO', 'IPO', 'NYSE',
                              'NASDAQ', 'ETF', 'SPY', 'QQQ', 'IWM', 'VIX'] and
                not symbol.isdigit()):
                filtered_symbols.append(symbol)
        
        return filtered_symbols
    
    def _calculate_engagement_weight(self, post) -> float:
        """Calculate engagement weight for a post"""
        
        # Base weight
        weight = 1.0
        
        # Score weight (upvotes - downvotes)
        if post.score > 100:
            weight *= 3.0
        elif post.score > 50:
            weight *= 2.0
        elif post.score > 10:
            weight *= 1.5
        elif post.score < 0:
            weight *= 0.5
        
        # Comments weight
        if post.num_comments > 100:
            weight *= 2.0
        elif post.num_comments > 50:
            weight *= 1.5
        elif post.num_comments > 10:
            weight *= 1.2
        
        # Award weight (Reddit awards indicate high engagement)
        if hasattr(post, 'total_awards_received') and post.total_awards_received > 0:
            weight *= (1 + post.total_awards_received * 0.1)
        
        # Time decay (newer posts get slight boost)
        post_age_hours = (datetime.now() - datetime.fromtimestamp(post.created_utc)).total_seconds() / 3600
        if post_age_hours < 6:
            weight *= 1.3
        elif post_age_hours < 24:
            weight *= 1.1
        
        return weight
    
    def _get_subreddit_weight(self, subreddit_name: str) -> float:
        """Get relevance weight for different subreddits"""
        
        weights = {
            # Primary meme stock communities (highest weight)
            'wallstreetbets': 2.0,
            'Superstonk': 1.8,
            'amcstock': 1.5,
            'GME': 1.5,
            
            # Penny stock communities (high weight for pump/dump detection)
            'pennystocks': 1.8,
            'RobinHoodPennyStocks': 1.6,
            'penny_stocks': 1.5,
            'PennyStocksDD': 1.7,
            'pennystockwatch': 1.4,
            'OTCMarkets': 1.3,
            
            # Options communities
            'options': 1.4,
            'thetagang': 1.2,
            'wallstreetbetsOGs': 1.6,
            'wallstreetbetsnew': 1.4,
            
            # General trading communities
            'SmallCapStocks': 1.3,
            'stocks': 1.1,
            'StockMarket': 1.1,
            'daytrading': 1.2,
            'SwingTrading': 1.2,
            'trading': 1.1,
            
            # Squeeze communities
            'Shortsqueeze': 1.6,
            'squeezeplays': 1.5,
            
            # Crypto (often has pump/dump patterns)
            'CryptoPumping': 1.4,
            'SatoshiStreetBets': 1.3,
            'CryptoMoonShots': 1.2,
        }
        
        return weights.get(subreddit_name, 1.0)  # Default weight of 1.0
    
    def _validate_stock_symbols(self, symbols: List[str]) -> List[str]:
        """Validate that symbols are real stocks"""
        
        validated_stocks = []
        
        for symbol in symbols:
            try:
                # Quick validation - try to get basic info
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if info and 'symbol' in info:
                    # Additional checks for penny stocks and small caps
                    current_price = info.get('regularMarketPrice') or info.get('currentPrice', 0)
                    market_cap = info.get('marketCap', 0)
                    
                    # Include penny stocks and small caps (often pump targets)
                    if current_price > 0:  # Any price above 0
                        validated_stocks.append(symbol)
                        
                        # Log penny stocks for special attention
                        if current_price < 5.0:
                            print(f"  🪙 Penny stock detected: {symbol} @ ${current_price:.3f}")
                        elif market_cap and market_cap < 1e9:  # Less than $1B market cap
                            print(f"  🏢 Small cap detected: {symbol} (${market_cap/1e6:.0f}M market cap)")
                
                if len(validated_stocks) >= self.config.TRENDING_STOCKS_LIMIT * 2:
                    break  # Don't validate too many
                    
            except:
                continue
        
        return validated_stocks
    
    def get_posts_for_symbol(self, symbol: str, subreddits: List[str] = None, 
                           days_back: int = None, limit: int = None) -> pd.DataFrame:
        """
        Fetch recent Reddit posts mentioning a stock symbol from multiple subreddits
        
        Args:
            symbol: Stock ticker symbol
            subreddits: List of subreddits to search (uses all if None)
            days_back: Number of days to look back
            limit: Maximum number of posts to fetch per subreddit
            
        Returns:
            DataFrame with post data
        """
        if days_back is None:
            days_back = 30
        if limit is None:
            limit = self.config.REDDIT_POST_LIMIT // len(self.target_subreddits)
        if subreddits is None:
            subreddits = self.target_subreddits
            
        posts_data = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        print(f"🔍 Searching for ${symbol} across {len(subreddits)} subreddits...")
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                search_query = f"${symbol} OR {symbol}"
                
                subreddit_posts = 0
                for post in subreddit.search(search_query, time_filter='month', limit=limit):
                    post_date = datetime.fromtimestamp(post.created_utc)
                    if post_date >= cutoff_date:
                        posts_data.append({
                            'date': post_date,
                            'title': post.title,
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'upvote_ratio': post.upvote_ratio,
                            'selftext': post.selftext,
                            'created_utc': post.created_utc,
                            'subreddit': subreddit_name,
                            'url': f"https://reddit.com{post.permalink}",
                            'awards': getattr(post, 'total_awards_received', 0)
                        })
                        subreddit_posts += 1
                
                if subreddit_posts > 0:
                    print(f"  📍 r/{subreddit_name}: {subreddit_posts} posts")
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f"  ❌ Error with r/{subreddit_name}: {e}")
                continue
        
        total_posts = len(posts_data)
        print(f"📊 Total posts found for {symbol}: {total_posts}")
        
        return pd.DataFrame(posts_data)
    
    def analyze_sentiment_and_keywords(self, text: str) -> Dict:
        """
        Analyze sentiment and detect keywords in text with detailed positive/negative tracking
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores and detailed keyword counts
        """
        text_str = str(text).lower()
        
        # Sentiment analysis using TextBlob
        sentiment = TextBlob(text_str).sentiment.polarity
        
        # Define comprehensive positive (bullish) keywords
        positive_keywords = [
            'call', 'calls', 'bullish', 'moon', 'rocket', 'tendies', 'squeeze', 
            'breakout', 'buy', 'long', 'hold', 'hodl', 'diamond hands', 'ape',
            'rally', 'gain', 'gains', 'profit', 'winner', 'winner', 'catalyst',
            'potential', 'undervalued', 'gem', 'golden', 'opportunity', 'massive',
            'huge', 'strong', 'bullrun', 'mooning', 'lambo', 'yolo', 'btfd',
            'buy the dip', 'accumulate', 'loading', 'position', 'conviction',
            '🚀', '💎', '🦍', '📈', '🌙', 'to the moon', 'short squeeze',
            'gamma squeeze', 'momentum', 'breakout play', 'mega', 'explosive'
        ]
        
        # Define comprehensive negative (bearish) keywords
        negative_keywords = [
            'put', 'puts', 'bearish', 'short', 'dump', 'crash', 'falling',
            'bagholder', 'bagholding', 'bags', 'dead', 'scam', 'fraud',
            'overvalued', 'bubble', 'worthless', 'sell', 'exit', 'loss',
            'losses', 'rip', 'rekt', 'pump and dump', 'rug pull', 'avoid',
            'stay away', 'trap', 'dumpster fire', 'disaster', 'tanking',
            'plummeting', 'bleeding', 'underwater', 'negative', 'declining',
            'weak', 'failing', 'bankruptcy', 'delisting', 'investigation',
            '📉', '🔴', '💀', 'fud', 'fear', 'panic', 'selling', 'dump it'
        ]
        
        # Count positive mentions
        positive_count = 0
        positive_matches = []
        for keyword in positive_keywords:
            count = text_str.count(keyword)
            if count > 0:
                positive_count += count
                positive_matches.append((keyword, count))
        
        # Count negative mentions
        negative_count = 0
        negative_matches = []
        for keyword in negative_keywords:
            count = text_str.count(keyword)
            if count > 0:
                negative_count += count
                negative_matches.append((keyword, count))
        
        # Original keyword tracking for backward compatibility
        pump_count = sum(1 for keyword in PUMP_KEYWORDS if keyword in text_str)
        dump_count = sum(1 for keyword in DUMP_KEYWORDS if keyword in text_str)
        calls_mentions = sum(1 for keyword in OPTIONS_CALLS_KEYWORDS if keyword in text_str)
        puts_mentions = sum(1 for keyword in OPTIONS_PUTS_KEYWORDS if keyword in text_str)
        
        # Calculate sentiment ratio
        total_keywords = positive_count + negative_count
        positive_ratio = positive_count / max(total_keywords, 1)
        negative_ratio = negative_count / max(total_keywords, 1)
        
        return {
            'sentiment': sentiment,
            'pump_keywords': pump_count,
            'dump_keywords': dump_count,
            'calls_mentions': calls_mentions,
            'puts_mentions': puts_mentions,
            'positive_mentions': positive_count,
            'negative_mentions': negative_count,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'positive_matches': positive_matches,
            'negative_matches': negative_matches,
            'net_sentiment_keywords': positive_count - negative_count
        }
    
    def calculate_reddit_metrics(self, reddit_df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive Reddit metrics including detailed positive/negative mention counts
        
        Args:
            reddit_df: DataFrame with Reddit posts
            
        Returns:
            Dictionary with calculated metrics including detailed sentiment breakdowns
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
                'activity_trend': 'neutral',
                'subreddit_distribution': {},
                'cross_subreddit_mentions': 0,
                'positive_mentions': 0,
                'negative_mentions': 0,
                'net_sentiment': 0,
                'positive_ratio': 0,
                'negative_ratio': 0,
                'top_positive_keywords': [],
                'top_negative_keywords': []
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
        
        # Extract detailed positive/negative counts
        positive_mentions = [data['positive_mentions'] for data in sentiment_data]
        negative_mentions = [data['negative_mentions'] for data in sentiment_data]
        positive_ratios = [data['positive_ratio'] for data in sentiment_data]
        negative_ratios = [data['negative_ratio'] for data in sentiment_data]
        
        # Aggregate all positive and negative keyword matches
        all_positive_matches = {}
        all_negative_matches = {}
        
        for data in sentiment_data:
            for keyword, count in data['positive_matches']:
                all_positive_matches[keyword] = all_positive_matches.get(keyword, 0) + count
            for keyword, count in data['negative_matches']:
                all_negative_matches[keyword] = all_negative_matches.get(keyword, 0) + count
        
        # Get top keywords
        top_positive = sorted(all_positive_matches.items(), key=lambda x: x[1], reverse=True)[:10]
        top_negative = sorted(all_negative_matches.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Calculate totals
        total_positive = sum(positive_mentions)
        total_negative = sum(negative_mentions)
        net_sentiment = total_positive - total_negative
        
        # Calculate ratios
        total_mentions = total_positive + total_negative
        overall_positive_ratio = total_positive / max(total_mentions, 1)
        overall_negative_ratio = total_negative / max(total_mentions, 1)
        
        # Original metrics
        total_keywords = sum(pump_keywords) + sum(dump_keywords)
        pump_ratio = sum(pump_keywords) / max(total_keywords, 1)
        dump_ratio = sum(dump_keywords) / max(total_keywords, 1)
        
        total_options = sum(calls_mentions) + sum(puts_mentions)
        calls_put_ratio = sum(calls_mentions) / max(sum(puts_mentions), 1)
        
        # Subreddit distribution analysis
        subreddit_distribution = {}
        if 'subreddit' in reddit_df.columns:
            subreddit_counts = reddit_df['subreddit'].value_counts()
            subreddit_distribution = subreddit_counts.to_dict()
            cross_subreddit_mentions = len(subreddit_counts)
        else:
            cross_subreddit_mentions = 1
        
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
            'options_mentions': total_options,
            'subreddit_distribution': subreddit_distribution,
            'cross_subreddit_mentions': cross_subreddit_mentions,
            'avg_awards': reddit_df['awards'].mean() if 'awards' in reddit_df.columns else 0,
            'high_engagement_posts': len(reddit_df[reddit_df['score'] > 100]),
            'penny_stock_indicators': self._detect_penny_stock_patterns(reddit_df),
            
            # New detailed sentiment metrics
            'positive_mentions': total_positive,
            'negative_mentions': total_negative,
            'net_sentiment': net_sentiment,
            'positive_ratio': overall_positive_ratio,
            'negative_ratio': overall_negative_ratio,
            'top_positive_keywords': top_positive,
            'top_negative_keywords': top_negative,
            'bullish_posts': sum(1 for ratio in positive_ratios if ratio > 0.6),
            'bearish_posts': sum(1 for ratio in negative_ratios if ratio > 0.6),
            'neutral_posts': sum(1 for pos, neg in zip(positive_ratios, negative_ratios) if pos <= 0.6 and neg <= 0.6)
        }
    
    def _detect_penny_stock_patterns(self, reddit_df: pd.DataFrame) -> Dict:
        """Detect patterns specific to penny stock pump and dumps"""
        
        if reddit_df.empty:
            return {'pump_pattern_detected': False, 'coordination_score': 0}
        
        # Look for coordination patterns
        coordination_indicators = 0
        
        # Multiple posts in short time frame
        if len(reddit_df) > 5:
            time_spans = []
            for _, post in reddit_df.iterrows():
                post_time = post['date']
                similar_time_posts = reddit_df[
                    (reddit_df['date'] >= post_time - timedelta(hours=2)) & 
                    (reddit_df['date'] <= post_time + timedelta(hours=2))
                ]
                time_spans.append(len(similar_time_posts))
            
            if max(time_spans) > 3:  # More than 3 posts in 4-hour window
                coordination_indicators += 1
        
        # Similar language across posts
        if len(reddit_df) > 2:
            titles = reddit_df['title'].str.lower()
            common_phrases = ['to the moon', 'huge potential', 'massive gains', 'next big thing']
            
            phrase_matches = 0
            for phrase in common_phrases:
                if sum(titles.str.contains(phrase, na=False)) > 1:
                    phrase_matches += 1
            
            if phrase_matches > 1:
                coordination_indicators += 1
        
        # Cross-subreddit posting pattern
        if 'subreddit' in reddit_df.columns:
            unique_subreddits = reddit_df['subreddit'].nunique()
            if unique_subreddits > 3:  # Posted in many subreddits
                coordination_indicators += 1
        
        return {
            'pump_pattern_detected': coordination_indicators >= 2,
            'coordination_score': coordination_indicators,
            'rapid_posting': max(time_spans) if 'time_spans' in locals() else 0,
            'cross_subreddit_spread': unique_subreddits if 'unique_subreddits' in locals() else 1
        }
    
    def get_sentiment_summary(self, reddit_metrics: Dict) -> str:
        """
        Get human-readable sentiment summary including multi-subreddit analysis
        
        Args:
            reddit_metrics: Reddit metrics dictionary
            
        Returns:
            Sentiment summary string
        """
        avg_sentiment = reddit_metrics.get('avg_sentiment', 0)
        pump_ratio = reddit_metrics.get('pump_keyword_ratio', 0)
        dump_ratio = reddit_metrics.get('dump_keyword_ratio', 0)
        activity_trend = reddit_metrics.get('activity_trend', 'neutral')
        cross_subreddit = reddit_metrics.get('cross_subreddit_mentions', 0)
        
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
        
        # Add multi-subreddit context
        if cross_subreddit > 3:
            multi_context = f" (Spread across {cross_subreddit} subreddits)"
        else:
            multi_context = ""
        
        # Check for pump patterns
        penny_indicators = reddit_metrics.get('penny_stock_indicators', {})
        if penny_indicators.get('pump_pattern_detected'):
            pattern_warning = " ⚠️ COORDINATED PUMP PATTERN DETECTED"
        else:
            pattern_warning = ""
        
        return f"{sentiment_desc} ({trend_desc}{multi_context}){pattern_warning}"