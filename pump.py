import praw
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
from textblob import TextBlob
from collections import Counter
import warnings
import time
import requests
from enum import Enum
import json
warnings.filterwarnings('ignore')

class PumpDumpPhase(Enum):
    """Enum for different phases of pump and dump cycle"""
    ACCUMULATION = "Accumulation"
    EARLY_PUMP = "Early Pump"
    MAIN_PUMP = "Main Pump"
    PEAK_FRENZY = "Peak Frenzy"
    EARLY_DUMP = "Early Dump"
    MAIN_DUMP = "Main Dump"
    BAGHOLDERS = "Bagholders"
    RECOVERY_ATTEMPT = "Recovery Attempt"
    DEAD = "Dead/Delisted Risk"

class OptionsStrategy(Enum):
    """Options trading strategies"""
    BUY_CALLS = "Buy Calls"
    BUY_PUTS = "Buy Puts"
    SELL_CALLS = "Sell Calls (Covered)"
    SELL_PUTS = "Sell Puts (Cash Secured)"
    CALL_SPREADS = "Call Spreads"
    PUT_SPREADS = "Put Spreads"
    STRADDLE = "Long Straddle"
    STRANGLE = "Long Strangle"
    IRON_CONDOR = "Iron Condor"
    AVOID_OPTIONS = "Avoid All Options"

class MemeStockAnalyzer:
    def __init__(self, reddit_client_id, reddit_client_secret, reddit_user_agent, 
                 alpha_vantage_key=None, finnhub_key=None):
        """
        Initialize the analyzer with API credentials
        """
        self.reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )
        
        # News API keys (optional but recommended)
        self.alpha_vantage_key = alpha_vantage_key
        self.finnhub_key = finnhub_key
        
        # Known meme stock symbols for monitoring
        self.known_meme_stocks = [
            'GME', 'AMC', 'CLOV', 'BB', 'WISH', 'SNDL', 'SPRT', 'IRNT',
            'BBBY', 'NOK', 'NAKD', 'EXPR', 'KOSS', 'RKT', 'UWMC', 'CLOV',
            'WKHS', 'RIDE', 'GOEV', 'PLTR', 'NIO', 'TSLA', 'MVIS', 'TLRY',
            'PROG', 'ATER', 'DWAC', 'PHUN', 'BKKT', 'MARK', 'GREE', 'PTON'
        ]
        
    def get_stock_news(self, symbol, days_back=7):
        """
        Fetch recent news for a stock symbol from multiple sources
        """
        news_data = []
        
        # Try Alpha Vantage News API
        if self.alpha_vantage_key:
            try:
                url = f"https://www.alphavantage.co/query"
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
                                'url': article.get('url', '')
                            })
            except Exception as e:
                print(f"Alpha Vantage news error for {symbol}: {e}")
        
        # Try Finnhub News API
        if self.finnhub_key:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                
                url = f"https://finnhub.io/api/v1/company-news"
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
                            'url': article.get('url', '')
                        })
            except Exception as e:
                print(f"Finnhub news error for {symbol}: {e}")
        
        # Fallback: Try to get news from Yahoo Finance (basic)
        if not news_data:
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
                        'url': article.get('link', '')
                    })
            except Exception as e:
                print(f"Yahoo Finance news error for {symbol}: {e}")
        
        return pd.DataFrame(news_data)
    
    def analyze_news_sentiment(self, news_df):
        """
        Analyze sentiment of news articles
        """
        if news_df.empty:
            return {'avg_sentiment': 0, 'news_count': 0, 'bullish_news': 0, 'bearish_news': 0}
        
        sentiments = []
        bullish_count = 0
        bearish_count = 0
        
        for _, article in news_df.iterrows():
            # Use provided sentiment or calculate with TextBlob
            if article['sentiment'] != 0:
                sentiment = article['sentiment']
            else:
                text = f"{article['title']} {article['summary']}"
                sentiment = TextBlob(text).sentiment.polarity
            
            sentiments.append(sentiment)
            
            if sentiment > 0.1:
                bullish_count += 1
            elif sentiment < -0.1:
                bearish_count += 1
        
        return {
            'avg_sentiment': np.mean(sentiments) if sentiments else 0,
            'news_count': len(news_df),
            'bullish_news': bullish_count,
            'bearish_news': bearish_count,
            'recent_news': news_df.head(3).to_dict('records')
        }
    
    def get_options_data(self, symbol):
        """
        Get basic options data for the symbol
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current stock info
            info = ticker.info
            current_price = info.get('currentPrice', 0)
            
            # Get options expiration dates
            options_dates = ticker.options
            
            if not options_dates:
                return None
            
            # Get options chain for nearest expiration
            nearest_expiry = options_dates[0]
            options_chain = ticker.option_chain(nearest_expiry)
            
            calls = options_chain.calls
            puts = options_chain.puts
            
            # Find at-the-money options
            atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
            atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]
            
            return {
                'current_price': current_price,
                'nearest_expiry': nearest_expiry,
                'atm_call_iv': atm_call['impliedVolatility'].iloc[0] if not atm_call.empty else 0,
                'atm_put_iv': atm_put['impliedVolatility'].iloc[0] if not atm_put.empty else 0,
                'atm_call_volume': atm_call['volume'].iloc[0] if not atm_call.empty else 0,
                'atm_put_volume': atm_put['volume'].iloc[0] if not atm_put.empty else 0,
                'call_put_ratio': (atm_call['volume'].iloc[0] / atm_put['volume'].iloc[0]) if not atm_call.empty and not atm_put.empty and atm_put['volume'].iloc[0] > 0 else 1
            }
            
        except Exception as e:
            print(f"Options data error for {symbol}: {e}")
            return None
    
    def get_trending_stocks_from_reddit(self, subreddit='wallstreetbets', limit=100):
        """
        Identify currently trending stocks from Reddit discussions
        """
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
                    if symbol not in ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE', 'WSB', 'GME', 'APE', 'MOON', 'YOLO']:
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
                        if len(validated_stocks) >= 10:  # Limit to top 10
                            break
                except:
                    continue
                    
            return validated_stocks
            
        except Exception as e:
            print(f"Error getting trending stocks: {e}")
            return self.known_meme_stocks[:10]  # Fallback to known meme stocks
    
    def get_reddit_posts(self, symbol, subreddit='wallstreetbets', days_back=30, limit=200):
        """
        Fetch recent Reddit posts mentioning a stock symbol
        """
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
    
    def get_stock_data(self, symbol, days_back=90):
        """
        Fetch recent stock price data with enhanced metrics
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            df.reset_index(inplace=True)
            
            # Get additional info
            info = stock.info
            
            # Calculate additional metrics
            if not df.empty:
                df['daily_return'] = df['Close'].pct_change()
                df['volatility'] = df['daily_return'].rolling(20).std()
                df['volume_ma'] = df['Volume'].rolling(20).mean()
                df['price_ma_20'] = df['Close'].rolling(20).mean()
                df['price_ma_50'] = df['Close'].rolling(50).mean()
                
                # RSI calculation
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
            
            return df, info
            
        except Exception as e:
            print(f"Error fetching stock data for {symbol}: {e}")
            return pd.DataFrame(), {}
    
    def analyze_sentiment_and_keywords(self, text):
        """
        Enhanced sentiment and keyword analysis
        """
        text_str = str(text).lower()
        
        # Sentiment analysis
        sentiment = TextBlob(text_str).sentiment.polarity
        
        # Pump keywords (bullish)
        pump_keywords = [
            'moon', 'rocket', 'diamond hands', 'hold', 'hodl', 'squeeze', 
            'short squeeze', 'gamma squeeze', 'to the moon', 'buy the dip', 
            'yolo', 'ape', 'tendies', 'lambo', 'bullish', 'breakout',
            '🚀', '💎', '🦍', '📈', 'rally', 'momentum', 'calls', 'long'
        ]
        
        # Dump keywords (bearish)
        dump_keywords = [
            'sell', 'dump', 'crash', 'falling', 'bearish', 'short', 
            'puts', 'bags', 'bagholders', 'dead', 'scam', 'pump and dump',
            'fraud', 'overvalued', 'bubble', '📉', 'rip', 'loss', 'exit'
        ]
        
        # Options-specific keywords
        options_calls = ['calls', 'call option', 'buy calls', 'otm calls', 'itm calls']
        options_puts = ['puts', 'put option', 'buy puts', 'otm puts', 'itm puts']
        
        pump_count = sum(1 for keyword in pump_keywords if keyword in text_str)
        dump_count = sum(1 for keyword in dump_keywords if keyword in text_str)
        calls_mentions = sum(1 for keyword in options_calls if keyword in text_str)
        puts_mentions = sum(1 for keyword in options_puts if keyword in text_str)
        
        return sentiment, pump_count, dump_count, calls_mentions, puts_mentions
    
    def determine_pump_dump_phase(self, symbol, stock_df, reddit_df, stock_info, news_sentiment):
        """
        Enhanced phase determination with news sentiment
        """
        if stock_df.empty:
            return PumpDumpPhase.DEAD, "No stock data available", 0.0
        
        # Get recent data (last 30 days)
        recent_days = 30
        recent_stock = stock_df.tail(recent_days).copy()
        
        if len(recent_stock) < 10:
            return PumpDumpPhase.DEAD, "Insufficient data", 0.0
        
        # Calculate key metrics
        current_price = recent_stock['Close'].iloc[-1]
        price_30d_ago = recent_stock['Close'].iloc[0] if len(recent_stock) >= 30 else recent_stock['Close'].iloc[0]
        price_change_30d = (current_price - price_30d_ago) / price_30d_ago
        
        recent_high = recent_stock['High'].max()
        recent_low = recent_stock['Low'].min()
        price_from_high = (current_price - recent_high) / recent_high
        price_from_low = (current_price - recent_low) / recent_low
        
        avg_volume_recent = recent_stock['Volume'].mean()
        current_volume = recent_stock['Volume'].iloc[-1]
        volume_trend = current_volume / avg_volume_recent if avg_volume_recent > 0 else 1
        
        current_rsi = recent_stock['rsi'].iloc[-1] if not pd.isna(recent_stock['rsi'].iloc[-1]) else 50
        volatility = recent_stock['volatility'].iloc[-1] if not pd.isna(recent_stock['volatility'].iloc[-1]) else 0.02
        
        # Reddit sentiment analysis
        reddit_score = 0.5  # Neutral default
        if not reddit_df.empty:
            reddit_recent = reddit_df[reddit_df['date'] >= (datetime.now() - timedelta(days=7))]
            if not reddit_recent.empty:
                sentiments = []
                for _, post in reddit_recent.iterrows():
                    sentiment, _, _, _, _ = self.analyze_sentiment_and_keywords(post['title'] + ' ' + str(post['selftext']))
                    sentiments.append(sentiment)
                reddit_score = np.mean(sentiments) if sentiments else 0.5
        
        # Combine Reddit and news sentiment
        combined_sentiment = (reddit_score + news_sentiment.get('avg_sentiment', 0)) / 2
        
        # Confidence calculation
        confidence = min(1.0, len(recent_stock) / 30) * min(1.0, len(reddit_df) / 10)
        if news_sentiment.get('news_count', 0) > 0:
            confidence *= 1.2  # Boost confidence if we have news data
        
        # Enhanced phase determination with news sentiment
        if current_price < 1.0 and price_change_30d < -0.5:
            return PumpDumpPhase.DEAD, f"Price ${current_price:.2f}, down {price_change_30d:.1%}", confidence
        
        elif price_change_30d > 2.0 and current_rsi > 80 and volume_trend > 3:
            return PumpDumpPhase.PEAK_FRENZY, f"Extreme pump: +{price_change_30d:.1%}, RSI {current_rsi:.0f}", confidence
        
        elif price_change_30d > 0.5 and current_rsi > 70 and combined_sentiment > 0.2:
            return PumpDumpPhase.MAIN_PUMP, f"Strong pump: +{price_change_30d:.1%}, bullish sentiment", confidence
        
        elif price_change_30d > 0.2 and (volume_trend > 1.5 or combined_sentiment > 0.3):
            return PumpDumpPhase.EARLY_PUMP, f"Early pump: +{price_change_30d:.1%}, positive signals", confidence
        
        elif price_from_high < -0.3 and (price_change_30d < -0.2 or combined_sentiment < -0.2):
            return PumpDumpPhase.MAIN_DUMP, f"Dumping: {price_from_high:.1%} from high", confidence
        
        elif price_from_high < -0.15 and (volume_trend < 0.8 or combined_sentiment < -0.1):
            return PumpDumpPhase.EARLY_DUMP, f"Early dump: {price_from_high:.1%} from high", confidence
        
        elif price_change_30d < -0.4 and combined_sentiment < -0.1:
            return PumpDumpPhase.BAGHOLDERS, f"Bagholders: {price_change_30d:.1%}, negative sentiment", confidence
        
        elif price_from_low > 0.2 and price_from_high < -0.2:
            return PumpDumpPhase.RECOVERY_ATTEMPT, f"Recovery attempt from lows", confidence
        
        else:
            return PumpDumpPhase.ACCUMULATION, f"Quiet: {price_change_30d:.1%} change", confidence
    
    def suggest_options_strategy(self, phase, reddit_sentiment, news_sentiment, options_data, stock_info):
        """
        Suggest options trading strategy based on phase and sentiment
        """
        combined_sentiment = (reddit_sentiment + news_sentiment.get('avg_sentiment', 0)) / 2
        current_price = options_data.get('current_price', 0) if options_data else stock_info.get('currentPrice', 0)
        iv = options_data.get('atm_call_iv', 0.5) if options_data else 0.5  # Implied volatility
        
        strategy_recommendations = []
        risk_level = "UNKNOWN"
        
        if phase == PumpDumpPhase.ACCUMULATION:
            if combined_sentiment > 0.1:
                strategy_recommendations.append({
                    'strategy': OptionsStrategy.BUY_CALLS,
                    'reasoning': 'Accumulation phase with positive sentiment - potential for upward movement',
                    'risk': 'MEDIUM',
                    'timeframe': '30-60 days',
                    'strike_guidance': 'Slightly OTM calls (5-10% above current price)'
                })
            elif iv < 0.3:  # Low implied volatility
                strategy_recommendations.append({
                    'strategy': OptionsStrategy.LONG_STRADDLE,
                    'reasoning': 'Low volatility in accumulation - expect big move in either direction',
                    'risk': 'MEDIUM',
                    'timeframe': '30-45 days',
                    'strike_guidance': 'ATM straddle'
                })
            risk_level = "MEDIUM"
        
        elif phase == PumpDumpPhase.EARLY_PUMP:
            if combined_sentiment > 0.2:
                strategy_recommendations.append({
                    'strategy': OptionsStrategy.BUY_CALLS,
                    'reasoning': 'Early pump with strong sentiment - ride the momentum',
                    'risk': 'HIGH',
                    'timeframe': '14-30 days',
                    'strike_guidance': 'ATM or slightly ITM calls'
                })
            else:
                strategy_recommendations.append({
                    'strategy': OptionsStrategy.CALL_SPREADS,
                    'reasoning': 'Early pump but mixed sentiment - limit risk with spreads',
                    'risk': 'MEDIUM-HIGH',
                    'timeframe': '14-30 days',
                    'strike_guidance': 'Bull call spread'
                })
            risk_level = "HIGH"
        
        elif phase == PumpDumpPhase.MAIN_PUMP:
            strategy_recommendations.append({
                'strategy': OptionsStrategy.SELL_CALLS,
                'reasoning': 'Main pump phase - high probability of reversal, sell premium',
                'risk': 'HIGH',
                'timeframe': '7-14 days',
                'strike_guidance': 'OTM covered calls if you own stock'
            })
            if iv > 0.6:  # High implied volatility
                strategy_recommendations.append({
                    'strategy': OptionsStrategy.IRON_CONDOR,
                    'reasoning': 'Extremely high IV - sell premium with defined risk',
                    'risk': 'MEDIUM',
                    'timeframe': '7-21 days',
                    'strike_guidance': 'Wide iron condor around current price'
                })
            risk_level = "EXTREME"
        
        elif phase == PumpDumpPhase.PEAK_FRENZY:
            strategy_recommendations.append({
                'strategy': OptionsStrategy.BUY_PUTS,
                'reasoning': 'Peak frenzy - imminent dump likely, profit from decline',
                'risk': 'HIGH',
                'timeframe': '7-21 days',
                'strike_guidance': 'ATM or slightly OTM puts'
            })
            strategy_recommendations.append({
                'strategy': OptionsStrategy.AVOID_OPTIONS,
                'reasoning': 'Extreme volatility and unpredictability - too dangerous',
                'risk': 'EXTREME',
                'timeframe': 'N/A',
                'strike_guidance': 'Stay away completely'
            })
            risk_level = "EXTREME"
        
        elif phase == PumpDumpPhase.EARLY_DUMP:
            strategy_recommendations.append({
                'strategy': OptionsStrategy.BUY_PUTS,
                'reasoning': 'Early dump phase - more downside likely',
                'risk': 'MEDIUM-HIGH',
                'timeframe': '14-30 days',
                'strike_guidance': 'ATM puts'
            })
            if combined_sentiment < -0.2:
                strategy_recommendations.append({
                    'strategy': OptionsStrategy.PUT_SPREADS,
                    'reasoning': 'Strong negative sentiment - bear put spreads for defined risk',
                    'risk': 'MEDIUM',
                    'timeframe': '14-30 days',
                    'strike_guidance': 'Bear put spread'
                })
            risk_level = "HIGH"
        
        elif phase == PumpDumpPhase.MAIN_DUMP:
            strategy_recommendations.append({
                'strategy': OptionsStrategy.SELL_PUTS,
                'reasoning': 'Main dump - sell puts if willing to own stock at lower prices',
                'risk': 'HIGH',
                'timeframe': '30-45 days',
                'strike_guidance': 'OTM cash-secured puts'
            })
            strategy_recommendations.append({
                'strategy': OptionsStrategy.AVOID_OPTIONS,
                'reasoning': 'Falling knife - wait for stabilization',
                'risk': 'EXTREME',
                'timeframe': 'N/A',
                'strike_guidance': 'Avoid until trend reverses'
            })
            risk_level = "EXTREME"
        
        elif phase == PumpDumpPhase.BAGHOLDERS:
            strategy_recommendations.append({
                'strategy': OptionsStrategy.SELL_CALLS,
                'reasoning': 'If you own shares, sell covered calls to generate income',
                'risk': 'MEDIUM',
                'timeframe': '30-60 days',
                'strike_guidance': 'OTM covered calls'
            })
            strategy_recommendations.append({
                'strategy': OptionsStrategy.AVOID_OPTIONS,
                'reasoning': 'Low probability of significant movement',
                'risk': 'LOW',
                'timeframe': 'N/A',
                'strike_guidance': 'Wait for better opportunities'
            })
            risk_level = "LOW"
        
        elif phase == PumpDumpPhase.RECOVERY_ATTEMPT:
            if combined_sentiment > 0:
                strategy_recommendations.append({
                    'strategy': OptionsStrategy.LONG_STRANGLE,
                    'reasoning': 'Recovery attempt - could break either way',
                    'risk': 'MEDIUM',
                    'timeframe': '30-45 days',
                    'strike_guidance': 'OTM calls and puts'
                })
            strategy_recommendations.append({
                'strategy': OptionsStrategy.CALL_SPREADS,
                'reasoning': 'Limited upside potential with defined risk',
                'risk': 'MEDIUM',
                'timeframe': '30-60 days',
                'strike_guidance': 'Bull call spread'
            })
            risk_level = "MEDIUM"
        
        elif phase == PumpDumpPhase.DEAD:
            strategy_recommendations.append({
                'strategy': OptionsStrategy.AVOID_OPTIONS,
                'reasoning': 'Dead/delisted risk - no options activity recommended',
                'risk': 'EXTREME',
                'timeframe': 'N/A',
                'strike_guidance': 'Completely avoid'
            })
            risk_level = "EXTREME"
        
        return {
            'recommendations': strategy_recommendations,
            'risk_level': risk_level,
            'combined_sentiment': combined_sentiment,
            'iv_rank': 'HIGH' if iv > 0.6 else 'MEDIUM' if iv > 0.3 else 'LOW'
        }
    
    def analyze_stock(self, symbol):
        """
        Complete analysis of a single stock with options strategy
        """
        print(f"Analyzing {symbol}...")
        
        # Get all data
        stock_df, stock_info = self.get_stock_data(symbol)
        reddit_df = self.get_reddit_posts(symbol)
        news_df = self.get_stock_news(symbol)
        options_data = self.get_options_data(symbol)
        
        if stock_df.empty:
            return None
        
        # Analyze sentiments
        news_sentiment = self.analyze_news_sentiment(news_df)
        
        # Calculate Reddit sentiment
        reddit_sentiment = 0
        if not reddit_df.empty:
            sentiments = []
            for _, post in reddit_df.iterrows():
                sentiment, _, _, _, _ = self.analyze_sentiment_and_keywords(post['title'] + ' ' + str(post['selftext']))
                sentiments.append(sentiment)
            reddit_sentiment = np.mean(sentiments) if sentiments else 0
        
        # Determine phase
        phase, description, confidence = self.determine_pump_dump_phase(symbol, stock_df, reddit_df, stock_info, news_sentiment)
        
        # Get options strategy recommendations
        options_strategy = self.suggest_options_strategy(phase, reddit_sentiment, news_sentiment, options_data, stock_info)
        
        # Current metrics
        current_price = stock_df['Close'].iloc[-1]
        price_change_1d = stock_df['daily_return'].iloc[-1] if len(stock_df) > 1 else 0
        volume_ratio = stock_df['Volume'].iloc[-1] / stock_df['volume_ma'].iloc[-1] if not pd.isna(stock_df['volume_ma'].iloc[-1]) else 1
        
        return {
            'symbol': symbol,
            'company_name': stock_info.get('longName', symbol),
            'current_price': current_price,
            'price_change_1d': price_change_1d,
            'volume_ratio': volume_ratio,
            'phase': phase,
            'description': description,
            'confidence': confidence,
            'reddit_sentiment': reddit_sentiment,
            'news_sentiment': news_sentiment,
            'options_strategy': options_strategy,
            'options_data': options_data,
            'stock_data': stock_df,
            'reddit_data': reddit_df,
            'news_data': news_df,
            'stock_info': stock_info
        }
    
    def scan_trending_stocks(self):
        """
        Scan all trending stocks and return analysis summary
        """
        print("🔍 Scanning for trending meme stocks...")
        trending_symbols = self.get_trending_stocks_from_reddit()
        
        print(f"Found {len(trending_symbols)} trending stocks: {', '.join(trending_symbols)}")
        
        results = []
        for symbol in trending_symbols:
            try:
                analysis = self.analyze_stock(symbol)
                if analysis:
                    results.append(analysis)
                time.sleep(1)  # Rate limiting for APIs
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        return results
    
    def display_stock_summary(self, results):
        """
        Display enhanced summary table with options strategies
        """
        if not results:
            print("No stocks to display")
            return
        
        print(f"\n{'='*140}")
        print("🎯 MEME STOCK ANALYSIS WITH OPTIONS STRATEGIES")
        print(f"{'='*140}")
        print(f"{'#':<3} {'Symbol':<6} {'Company':<20} {'Price':<8} {'1D %':<7} {'Vol':<6} {'Phase':<15} {'Sentiment':<10} {'Options':<15} {'Risk':<8}")
        print(f"{'-'*140}")
        
        for i, result in enumerate(results, 1):
            phase_emoji = {
                PumpDumpPhase.ACCUMULATION: "🔄",
                PumpDumpPhase.EARLY_PUMP: "📈",
                PumpDumpPhase.MAIN_PUMP: "🚀",
                PumpDumpPhase.PEAK_FRENZY: "🌋",
                PumpDumpPhase.EARLY_DUMP: "📉",
                PumpDumpPhase.MAIN_DUMP: "🔻",
                PumpDumpPhase.BAGHOLDERS: "💼",
                PumpDumpPhase.RECOVERY_ATTEMPT: "🔄",
                PumpDumpPhase.DEAD: "☠️"
            }
            
            emoji = phase_emoji.get(result['phase'], "❓")
            company_short = result['company_name'][:18] + ".." if len(result['company_name']) > 20 else result['company_name']
            
            # Get primary options strategy
            strategies = result['options_strategy']['recommendations']
            primary_strategy = strategies[0]['strategy'].value[:13] if strategies else "No Strategy"
            
            # Combined sentiment
            combined_sentiment = result['options_strategy']['combined_sentiment']
            sentiment_emoji = "🟢" if combined_sentiment > 0.1 else "🔴" if combined_sentiment < -0.1 else "🟡"
            
            risk_level = result['options_strategy']['risk_level']
            risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "EXTREME": "🔴"}.get(risk_level, "❓")
            
            print(f"{i:<3} {result['symbol']:<6} {company_short:<20} "
                  f"${result['current_price']:<7.2f} {result['price_change_1d']:<6.1%} "
                  f"{result['volume_ratio']:<5.1f}x {emoji} {result['phase'].value:<13} "
                  f"{sentiment_emoji} {combined_sentiment:<8.2f} {primary_strategy:<15} {risk_emoji} {risk_level:<6}")
    
    def interactive_stock_selector(self, results):
        """
        Enhanced interactive interface with options focus
        """
        while True:
            print(f"\n{'='*100}")
            print("📊 SELECT A STOCK FOR DETAILED OPTIONS ANALYSIS")
            print(f"{'='*100}")
            
            for i, result in enumerate(results, 1):
                strategies = result['options_strategy']['recommendations']
                primary_strategy = strategies[0]['strategy'].value if strategies else "No Strategy"
                risk_level = result['options_strategy']['risk_level']
                
                risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "EXTREME": "🔴"}.get(risk_level, "❓")
                
                print(f"{i:2d}. {result['symbol']:<6} | {result['phase'].value:<15} | {primary_strategy:<15} | {risk_color} {risk_level}")
            
            print(f"{len(results)+1:2d}. 🔄 Refresh/Rescan stocks")
            print(f"{len(results)+2:2d}. ❌ Exit")
            
            try:
                choice = input(f"\nEnter your choice (1-{len(results)+2}): ").strip()
                
                if choice == str(len(results)+1):
                    return "refresh"
                elif choice == str(len(results)+2):
                    return "exit"
                elif choice.isdigit() and 1 <= int(choice) <= len(results):
                    selected_stock = results[int(choice)-1]
                    self.display_detailed_analysis(selected_stock)
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                return "exit"
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def display_detailed_analysis(self, stock_analysis):
        """
        Enhanced detailed analysis with comprehensive options strategies
        """
        symbol = stock_analysis['symbol']
        phase = stock_analysis['phase']
        
        print(f"\n{'='*120}")
        print(f"📈 COMPREHENSIVE ANALYSIS: {symbol} - {stock_analysis['company_name']}")
        print(f"{'='*120}")
        
        # Current status
        print(f"💰 Current Price: ${stock_analysis['current_price']:.2f}")
        print(f"📊 24h Change: {stock_analysis['price_change_1d']:.2%}")
        print(f"🔊 Volume Ratio: {stock_analysis['volume_ratio']:.1f}x normal")
        print(f"🎯 Current Phase: {phase.value}")
        print(f"📝 Description: {stock_analysis['description']}")
        print(f"🎚️ Confidence: {stock_analysis['confidence']:.0%}")
        
        # Sentiment Analysis
        print(f"\n{'📊 SENTIMENT ANALYSIS':<60}")
        print("-" * 60)
        reddit_sentiment = stock_analysis['reddit_sentiment']
        news_sentiment = stock_analysis['news_sentiment']
        combined_sentiment = stock_analysis['options_strategy']['combined_sentiment']
        
        print(f"Reddit Sentiment: {reddit_sentiment:+.3f} {'🟢 Bullish' if reddit_sentiment > 0.1 else '🔴 Bearish' if reddit_sentiment < -0.1 else '🟡 Neutral'}")
        print(f"News Sentiment: {news_sentiment.get('avg_sentiment', 0):+.3f} ({'🟢 Bullish' if news_sentiment.get('avg_sentiment', 0) > 0.1 else '🔴 Bearish' if news_sentiment.get('avg_sentiment', 0) < -0.1 else '🟡 Neutral'})")
        print(f"Combined Sentiment: {combined_sentiment:+.3f} ({'🟢 Bullish' if combined_sentiment > 0.1 else '🔴 Bearish' if combined_sentiment < -0.1 else '🟡 Neutral'})")
        
        if news_sentiment.get('news_count', 0) > 0:
            print(f"Recent News: {news_sentiment['bullish_news']} bullish, {news_sentiment['bearish_news']} bearish")
        
        # Options Analysis
        options_data = stock_analysis['options_data']
        if options_data:
            print(f"\n{'📈 OPTIONS MARKET DATA':<60}")
            print("-" * 60)
            print(f"ATM Call IV: {options_data['atm_call_iv']:.1%}")
            print(f"ATM Put IV: {options_data['atm_put_iv']:.1%}")
            print(f"Call/Put Volume Ratio: {options_data['call_put_ratio']:.2f}")
            print(f"IV Rank: {stock_analysis['options_strategy']['iv_rank']}")
            print(f"Next Expiry: {options_data['nearest_expiry']}")
        
        # Options Strategy Recommendations
        print(f"\n{'🎯 OPTIONS STRATEGY RECOMMENDATIONS':<60}")
        print("-" * 60)
        
        strategies = stock_analysis['options_strategy']['recommendations']
        risk_level = stock_analysis['options_strategy']['risk_level']
        
        print(f"Overall Risk Level: {risk_level}")
        print()
        
        for i, strategy in enumerate(strategies, 1):
            strategy_emoji = {
                OptionsStrategy.BUY_CALLS: "📈",
                OptionsStrategy.BUY_PUTS: "📉",
                OptionsStrategy.SELL_CALLS: "💰",
                OptionsStrategy.SELL_PUTS: "💰",
                OptionsStrategy.CALL_SPREADS: "📊",
                OptionsStrategy.PUT_SPREADS: "📊",
                OptionsStrategy.STRADDLE: "🎯",
                OptionsStrategy.STRANGLE: "🎯",
                OptionsStrategy.IRON_CONDOR: "🦅",
                OptionsStrategy.AVOID_OPTIONS: "🚫"
            }.get(strategy['strategy'], "📋")
            
            risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "MEDIUM-HIGH": "🟠", "HIGH": "🔴", "EXTREME": "☠️"}.get(strategy['risk'], "❓")
            
            print(f"{strategy_emoji} Strategy #{i}: {strategy['strategy'].value}")
            print(f"   Risk Level: {risk_color} {strategy['risk']}")
            print(f"   Timeframe: {strategy['timeframe']}")
            print(f"   Reasoning: {strategy['reasoning']}")
            print(f"   Strike Guidance: {strategy['strike_guidance']}")
            print()
        
        # Phase-specific warnings
        print(f"\n{'🚨 PHASE-SPECIFIC WARNINGS & GUIDANCE':<60}")
        print("-" * 60)
        
        if phase == PumpDumpPhase.PEAK_FRENZY:
            print("🔴 EXTREME DANGER: This stock is likely at peak pump!")
            print("⚠️  DO NOT BUY CALLS - Imminent dump highly probable")
            print("💡 Consider puts if you must trade, but beware of high IV")
            print("🛑 Best strategy: STAY AWAY until dust settles")
            
        elif phase == PumpDumpPhase.MAIN_PUMP:
            print("🟠 HIGH RISK: Stock is in active pump phase")
            print("⚠️  Buying calls is extremely dangerous here")
            print("💡 If holding calls, consider taking profits")
            print("📊 Selling premium might be profitable due to high IV")
            
        elif phase == PumpDumpPhase.EARLY_PUMP:
            print("🟡 CAUTION: Early pump phase detected")
            print("⚠️  High risk/reward for call options")
            print("💡 Consider tight stop losses on any positions")
            print("📈 Monitor volume and momentum closely")
            
        elif phase == PumpDumpPhase.EARLY_DUMP:
            print("🟠 DUMP BEGINNING: Price declining from highs")
            print("⚠️  Avoid buying the dip too early")
            print("💡 Put options may be profitable")
            print("📉 Wait for trend confirmation before reversing")
            
        elif phase == PumpDumpPhase.MAIN_DUMP:
            print("🔴 MAJOR DUMP: Significant price decline")
            print("⚠️  Falling knife - extremely dangerous")
            print("💡 Puts may work but watch for dead cat bounces")
            print("🚫 Avoid calls completely")
            
        elif phase == PumpDumpPhase.BAGHOLDERS:
            print("🔴 BAGHOLDERS PHASE: Late investors trapped")
            print("💼 If you own shares, consider covered calls for income")
            print("💡 Long-term recovery uncertain")
            print("⏰ Patience required - could take months/years")
            
        elif phase == PumpDumpPhase.ACCUMULATION:
            print("🟢 ACCUMULATION: Relatively safer phase")
            print("💡 Best time for new positions if bullish")
            print("📊 Monitor for early pump signals")
            print("🎯 Consider long-term options strategies")
            
        elif phase == PumpDumpPhase.RECOVERY_ATTEMPT:
            print("🟡 RECOVERY: Attempting to bounce back")
            print("⚠️  Could be dead cat bounce - use caution")
            print("💡 Wide strangles might work for volatility")
            print("📈 Wait for sustained momentum confirmation")
            
        elif phase == PumpDumpPhase.DEAD:
            print("☠️  DEAD STOCK: Extreme delisting risk")
            print("🚫 AVOID ALL OPTIONS - Total loss likely")
            print("⚠️  Company may be in severe financial distress")
            print("🛑 Do not attempt any trades")
        
        # Recent News Summary
        if not stock_analysis['news_data'].empty:
            print(f"\n{'📰 RECENT NEWS HEADLINES':<60}")
            print("-" * 60)
            recent_news = news_sentiment.get('recent_news', [])
            for i, article in enumerate(recent_news[:3], 1):
                sentiment_emoji = "🟢" if article.get('sentiment', 0) > 0.1 else "🔴" if article.get('sentiment', 0) < -0.1 else "🟡"
                print(f"{i}. {sentiment_emoji} {article.get('title', 'No title')[:80]}...")
                if article.get('date'):
                    print(f"   📅 {article['date'].strftime('%Y-%m-%d %H:%M')}")
                print()
        
        # Technical Indicators
        if not stock_analysis['stock_data'].empty:
            recent_data = stock_analysis['stock_data'].tail(5)
            print(f"\n{'📊 TECHNICAL INDICATORS':<60}")
            print("-" * 60)
            
            current_rsi = recent_data['rsi'].iloc[-1]
            if not pd.isna(current_rsi):
                rsi_signal = "Overbought 🔴" if current_rsi > 70 else "Oversold 🟢" if current_rsi < 30 else "Neutral 🟡"
                print(f"RSI (14): {current_rsi:.1f} ({rsi_signal})")
            
            current_vol = recent_data['volatility'].iloc[-1]
            if not pd.isna(current_vol):
                print(f"20-day Volatility: {current_vol:.2%}")
            
            # Moving averages
            if 'price_ma_20' in recent_data.columns and 'price_ma_50' in recent_data.columns:
                ma20 = recent_data['price_ma_20'].iloc[-1]
                ma50 = recent_data['price_ma_50'].iloc[-1]
                current_price = recent_data['Close'].iloc[-1]
                
                if not pd.isna(ma20) and not pd.isna(ma50):
                    ma_trend = "Bullish 🟢" if ma20 > ma50 else "Bearish 🔴"
                    price_vs_ma = "Above 🟢" if current_price > ma20 else "Below 🔴"
                    print(f"MA Trend (20 vs 50): {ma_trend}")
                    print(f"Price vs MA20: {price_vs_ma}")
        
        print(f"\n{'⚠️  DISCLAIMER':<60}")
        print("-" * 60)
        print("This analysis is for educational purposes only and not financial advice.")
        print("Options trading involves significant risk and may result in total loss.")
        print("Meme stocks are highly volatile and unpredictable.")
        print("Always do your own research and consider consulting a financial advisor.")
        
        input("\nPress Enter to continue...")
    
    def run_interactive_analyzer(self):
        """
        Main interactive loop with enhanced options focus
        """
        print("🚀 ADVANCED MEME STOCK & OPTIONS ANALYZER")
        print("=" * 60)
        print("Features: Pump/Dump Detection | Options Strategies | News Sentiment")
        print("=" * 60)
        
        while True:
            results = self.scan_trending_stocks()
            
            if not results:
                print("❌ No stocks found. Please check your API credentials.")
                break
            
            self.display_stock_summary(results)
            
            action = self.interactive_stock_selector(results)
            
            if action == "exit":
                print("👋 Goodbye! Trade safely and remember: This is not financial advice!")
                break
            elif action == "refresh":
                print("🔄 Refreshing data...")
                continue

# Example usage with your credentials
def main():
    """
    Main function to run the interactive analyzer with your Reddit credentials
    """
    # Your Reddit API credentials
    REDDIT_CLIENT_ID = "HQS2vBLyvf7KC1wu5XCYaQ"
    REDDIT_CLIENT_SECRET = "C_cfUlj-jhJzictGVvXaYTmX2gu2Dg"
    REDDIT_USER_AGENT = "MemeStockAnalyzer/2.0 by AdvancedTrader"
    
    # Optional: Add these for enhanced news analysis
    # Get free API keys from:
    # Alpha Vantage: https://www.alphavantage.co/support/#api-key
    # Finnhub: https://finnhub.io/register
    ALPHA_VANTAGE_KEY = None  # "your_alpha_vantage_key_here"
    FINNHUB_KEY = None        # "your_finnhub_key_here"
    
    # Initialize and run analyzer
    analyzer = MemeStockAnalyzer(
        REDDIT_CLIENT_ID, 
        REDDIT_CLIENT_SECRET, 
        REDDIT_USER_AGENT,
        ALPHA_VANTAGE_KEY,
        FINNHUB_KEY
    )
    
    print("🔥 Starting Advanced Meme Stock Analysis...")
    print("⚠️  Remember: This tool helps identify risks, not guarantee profits!")
    print()
    
    analyzer.run_interactive_analyzer()

if __name__ == "__main__":
    main()