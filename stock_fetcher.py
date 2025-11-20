import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import warnings
import re
import requests
from collections import Counter, defaultdict
import math
warnings.filterwarnings('ignore')

class OptionsAnalyzer:
    """Analyze options data for stocks"""
    
    def __init__(self):
        pass
    
    def black_scholes_call(self, S, K, T, r, sigma):
        """Calculate Black-Scholes call option price"""
        try:
            if T <= 0 or sigma <= 0:
                return 0
            
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            # Approximation of cumulative normal distribution
            def norm_cdf(x):
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))
            
            call_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
            return max(call_price, 0)
        except:
            return 0
    
    def black_scholes_put(self, S, K, T, r, sigma):
        """Calculate Black-Scholes put option price"""
        try:
            if T <= 0 or sigma <= 0:
                return 0
            
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            # Approximation of cumulative normal distribution
            def norm_cdf(x):
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))
            
            put_price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
            return max(put_price, 0)
        except:
            return 0
    
    def calculate_implied_volatility(self, option_price, S, K, T, r, option_type='call'):
        """Calculate implied volatility using Newton-Raphson method"""
        try:
            if T <= 0 or option_price <= 0:
                return 0
            
            # Initial guess
            sigma = 0.3
            tolerance = 1e-6
            max_iterations = 100
            
            for i in range(max_iterations):
                if option_type == 'call':
                    price = self.black_scholes_call(S, K, T, r, sigma)
                else:
                    price = self.black_scholes_put(S, K, T, r, sigma)
                
                diff = price - option_price
                if abs(diff) < tolerance:
                    return sigma
                
                # Calculate vega (derivative of price with respect to volatility)
                vega = S * math.sqrt(T) * math.exp(-0.5 * ((math.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*math.sqrt(T)))**2) / math.sqrt(2*math.pi)
                
                if vega == 0:
                    break
                
                sigma = sigma - diff / vega
                sigma = max(0.01, min(5.0, sigma))  # Keep sigma in reasonable bounds
            
            return sigma
        except:
            return 0.3  # Default volatility if calculation fails
    
    def analyze_options_chain(self, symbol, current_price, prediction_data=None):
        """Analyze options chain for a given symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            try:
                expiration_dates = ticker.options
            except:
                return {"error": "No options data available"}
            
            if not expiration_dates:
                return {"error": "No options available for this symbol"}
            
            # Filter for dates within 3 months
            today = datetime.now().date()
            three_months_out = today + timedelta(days=90)
            
            valid_expirations = []
            for exp_date_str in expiration_dates:
                try:
                    exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                    if today < exp_date <= three_months_out:
                        valid_expirations.append(exp_date_str)
                except:
                    continue
            
            if not valid_expirations:
                return {"error": "No options expiring within 3 months"}
            
            # Analyze up to 3 expiration dates
            options_analysis = {}
            risk_free_rate = 0.05  # Approximate risk-free rate
            
            for exp_date in valid_expirations[:3]:
                try:
                    # Get options chain for this expiration
                    opt_chain = ticker.option_chain(exp_date)
                    calls = opt_chain.calls
                    puts = opt_chain.puts
                    
                    # Calculate time to expiration
                    exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                    time_to_exp = (exp_datetime.date() - today).days / 365.0
                    
                    # Analyze calls and puts
                    call_analysis = self.analyze_option_type(calls, current_price, time_to_exp, risk_free_rate, 'call', prediction_data)
                    put_analysis = self.analyze_option_type(puts, current_price, time_to_exp, risk_free_rate, 'put', prediction_data)
                    
                    options_analysis[exp_date] = {
                        'expiration': exp_date,
                        'days_to_expiration': (exp_datetime.date() - today).days,
                        'calls': call_analysis,
                        'puts': put_analysis,
                        'recommendations': self.generate_options_recommendations(call_analysis, put_analysis, prediction_data, time_to_exp)
                    }
                
                except Exception as e:
                    print(f"Error analyzing options for {exp_date}: {str(e)}")
                    continue
            
            return options_analysis
            
        except Exception as e:
            return {"error": f"Failed to analyze options: {str(e)}"}
    
    def analyze_option_type(self, options_df, current_price, time_to_exp, risk_free_rate, option_type, prediction_data):
        """Analyze calls or puts for a specific expiration"""
        if options_df.empty:
            return {"error": "No options data available"}
        
        analysis = {
            'total_volume': options_df['volume'].sum(),
            'total_open_interest': options_df['openInterest'].sum(),
            'options': []
        }
        
        # Focus on options around current price
        price_range = current_price * 0.2  # +/- 20% from current price
        relevant_options = options_df[
            (options_df['strike'] >= current_price - price_range) & 
            (options_df['strike'] <= current_price + price_range)
        ].copy()
        
        for _, option in relevant_options.iterrows():
            strike = option['strike']
            last_price = option['lastPrice']
            volume = option['volume']
            open_interest = option['openInterest']
            bid = option['bid']
            ask = option['ask']
            
            # Calculate metrics
            moneyness = strike / current_price
            spread = ask - bid
            spread_pct = (spread / last_price * 100) if last_price > 0 else 0
            
            # Calculate implied volatility
            if last_price > 0:
                implied_vol = self.calculate_implied_volatility(
                    last_price, current_price, strike, time_to_exp, risk_free_rate, option_type
                )
            else:
                implied_vol = 0
            
            # Calculate theoretical price
            if option_type == 'call':
                theoretical_price = self.black_scholes_call(current_price, strike, time_to_exp, risk_free_rate, implied_vol)
                intrinsic_value = max(0, current_price - strike)
            else:
                theoretical_price = self.black_scholes_put(current_price, strike, time_to_exp, risk_free_rate, implied_vol)
                intrinsic_value = max(0, strike - current_price)
            
            time_value = last_price - intrinsic_value
            
            # Determine if option is over/undervalued
            price_diff = last_price - theoretical_price
            overvalued = price_diff > 0
            
            option_analysis = {
                'strike': strike,
                'last_price': last_price,
                'bid': bid,
                'ask': ask,
                'volume': volume,
                'open_interest': open_interest,
                'implied_volatility': implied_vol,
                'moneyness': moneyness,
                'intrinsic_value': intrinsic_value,
                'time_value': time_value,
                'theoretical_price': theoretical_price,
                'overvalued': overvalued,
                'price_difference': price_diff,
                'spread_pct': spread_pct,
                'liquidity_score': min(10, (volume + open_interest/10) / 10)  # Simple liquidity metric
            }
            
            analysis['options'].append(option_analysis)
        
        # Sort by volume and open interest for best options
        analysis['options'] = sorted(analysis['options'], 
                                   key=lambda x: x['volume'] + x['open_interest']/10, 
                                   reverse=True)
        
        return analysis
    
    def generate_options_recommendations(self, call_analysis, put_analysis, prediction_data, time_to_exp):
        """Generate trading recommendations based on analysis"""
        recommendations = []
        
        if not prediction_data or 'technical_analysis' not in prediction_data:
            return ["No prediction data available for recommendations"]
        
        ta = prediction_data['technical_analysis']
        if 'error' in ta:
            return ["Technical analysis unavailable - limited recommendations possible"]
        
        prediction = ta.get('overall_prediction', 'Sideways/Uncertain')
        confidence = ta.get('confidence', 50)
        
        # Get best call and put options by liquidity
        best_calls = call_analysis.get('options', [])[:3] if 'error' not in call_analysis else []
        best_puts = put_analysis.get('options', [])[:3] if 'error' not in put_analysis else []
        
        if prediction == 'Likely Increase' and confidence > 60:
            recommendations.append(f"🟢 BULLISH STRATEGY (Confidence: {confidence}%)")
            
            # Recommend calls
            for call in best_calls:
                if call['moneyness'] > 0.95 and call['moneyness'] < 1.05:  # Near the money
                    action = "BUY" if not call['overvalued'] else "MONITOR"
                    recommendations.append(
                        f"  {action} Call ${call['strike']:.0f} - "
                        f"Last: ${call['last_price']:.2f}, "
                        f"IV: {call['implied_volatility']*100:.1f}%, "
                        f"Liquidity: {call['liquidity_score']:.1f}/10"
                    )
            
            # Recommend selling puts
            for put in best_puts:
                if put['moneyness'] > 0.90 and put['moneyness'] < 0.98:  # Out of the money
                    if put['overvalued']:
                        recommendations.append(
                            f"  SELL Put ${put['strike']:.0f} - "
                            f"Collect: ${put['last_price']:.2f}, "
                            f"IV: {put['implied_volatility']*100:.1f}% (High)"
                        )
        
        elif prediction == 'Likely Decrease' and confidence > 60:
            recommendations.append(f"🔴 BEARISH STRATEGY (Confidence: {confidence}%)")
            
            # Recommend puts
            for put in best_puts:
                if put['moneyness'] > 0.95 and put['moneyness'] < 1.05:  # Near the money
                    action = "BUY" if not put['overvalued'] else "MONITOR"
                    recommendations.append(
                        f"  {action} Put ${put['strike']:.0f} - "
                        f"Last: ${put['last_price']:.2f}, "
                        f"IV: {put['implied_volatility']*100:.1f}%, "
                        f"Liquidity: {put['liquidity_score']:.1f}/10"
                    )
            
            # Recommend selling calls
            for call in best_calls:
                if call['moneyness'] > 1.02 and call['moneyness'] < 1.10:  # Out of the money
                    if call['overvalued']:
                        recommendations.append(
                            f"  SELL Call ${call['strike']:.0f} - "
                            f"Collect: ${call['last_price']:.2f}, "
                            f"IV: {call['implied_volatility']*100:.1f}% (High)"
                        )
        
        else:
            recommendations.append(f"🟡 NEUTRAL STRATEGY (Confidence: {confidence}%)")
            recommendations.append("  Consider iron condors or straddles")
            recommendations.append("  Look for high IV options to sell")
            
            # Find high IV options to sell
            high_iv_calls = [c for c in best_calls if c['implied_volatility'] > 0.4 and c['overvalued']]
            high_iv_puts = [p for p in best_puts if p['implied_volatility'] > 0.4 and p['overvalued']]
            
            for call in high_iv_calls[:2]:
                recommendations.append(
                    f"  SELL Call ${call['strike']:.0f} - "
                    f"High IV: {call['implied_volatility']*100:.1f}%"
                )
            
            for put in high_iv_puts[:2]:
                recommendations.append(
                    f"  SELL Put ${put['strike']:.0f} - "
                    f"High IV: {put['implied_volatility']*100:.1f}%"
                )
        
        # Add general notes
        recommendations.append("\n📋 TRADING NOTES:")
        recommendations.append(f"  • Time to expiration: {time_to_exp*365:.0f} days")
        recommendations.append("  • Consider liquidity (volume + open interest)")
        recommendations.append("  • Monitor implied volatility levels")
        recommendations.append("  • Set stop-losses for long positions")
        
        return recommendations if recommendations else ["No specific recommendations available"]
    
    def get_options_summary(self, options_data):
        """Generate a summary of options analysis"""
        if 'error' in options_data:
            return f"Options analysis failed: {options_data['error']}"
        
        summary = []
        summary.append("📊 OPTIONS ANALYSIS SUMMARY")
        summary.append("=" * 50)
        
        for exp_date, data in options_data.items():
            summary.append(f"\nExpiration: {exp_date} ({data['days_to_expiration']} days)")
            
            # Call metrics
            if 'error' not in data['calls']:
                call_volume = data['calls']['total_volume']
                call_oi = data['calls']['total_open_interest']
                summary.append(f"  Calls: Volume {call_volume:,}, OI {call_oi:,}")
            
            # Put metrics  
            if 'error' not in data['puts']:
                put_volume = data['puts']['total_volume']
                put_oi = data['puts']['total_open_interest']
                summary.append(f"  Puts: Volume {put_volume:,}, OI {put_oi:,}")
                
                # Put/Call ratio
                if call_volume > 0:
                    pc_ratio = put_volume / call_volume
                    summary.append(f"  Put/Call Ratio: {pc_ratio:.2f}")
            
            # Add top recommendation
            if data['recommendations']:
                summary.append(f"  Top Rec: {data['recommendations'][0]}")
        
        return "\n".join(summary)

class RedditSentimentAnalyzer:
    """Analyze Reddit sentiment from r/wallstreetbets"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'StockAnalyzer/1.0 by YourUsername'
        }
        
        # Common stock symbols for filtering
        self.common_symbols = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD',
            'INTC', 'ORCL', 'CRM', 'ADBE', 'PYPL', 'QCOM', 'TXN', 'AVGO', 'MU', 'AMAT',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'AXP', 'BLK',
            'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY',
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND', 'VTEB', 'VNQ',
            'GME', 'AMC', 'PLTR', 'BB', 'NOK', 'WISH', 'CLOV', 'SPCE', 'SOFI', 'HOOD'
        }
        
        # Sentiment keywords
        self.positive_words = {
            'moon', 'rocket', 'bullish', 'buy', 'calls', 'pump', 'squeeze', 'diamond', 'hands',
            'hold', 'hodl', 'up', 'green', 'gains', 'profit', 'winner', 'strong', 'breakout',
            'rally', 'surge', 'boom', 'explosion', 'massive', 'huge', 'epic', 'legendary',
            'tendies', 'lambo', 'yolo', 'long', 'bull', 'support', 'resistance', 'breakthrough'
        }
        
        self.negative_words = {
            'bear', 'bearish', 'crash', 'dump', 'sell', 'puts', 'short', 'red', 'loss', 'losses',
            'down', 'drop', 'fall', 'decline', 'tank', 'plummet', 'disaster', 'terrible',
            'awful', 'bad', 'worse', 'worst', 'fail', 'failure', 'dead', 'rip', 'broke',
            'broke', 'bankruptcy', 'overvalued', 'bubble', 'correction', 'recession'
        }
    
    def get_reddit_posts(self, limit=100, time_filter='day'):
        """Fetch posts from r/wallstreetbets"""
        try:
            url = f"https://www.reddit.com/r/wallstreetbets/hot.json?limit={limit}&t={time_filter}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                posts = []
                
                for post in data['data']['children']:
                    post_data = post['data']
                    posts.append({
                        'title': post_data.get('title', ''),
                        'selftext': post_data.get('selftext', ''),
                        'score': post_data.get('score', 0),
                        'num_comments': post_data.get('num_comments', 0),
                        'created_utc': post_data.get('created_utc', 0),
                        'url': post_data.get('url', ''),
                        'author': post_data.get('author', ''),
                        'upvote_ratio': post_data.get('upvote_ratio', 0)
                    })
                
                print(f"Successfully fetched {len(posts)} posts from r/wallstreetbets")
                return posts
            else:
                print(f"Failed to fetch Reddit data. Status code: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching Reddit data: {str(e)}")
            return []
    
    def extract_stock_symbols(self, text):
        """Extract stock symbols from text"""
        # Look for $SYMBOL or just SYMBOL patterns
        pattern = r'\$?([A-Z]{1,5})\b'
        matches = re.findall(pattern, text.upper())
        
        # Filter to only include known stock symbols
        symbols = [match for match in matches if match in self.common_symbols]
        return symbols
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0, "Neutral"
        
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        
        if sentiment_score > 0.2:
            sentiment_label = "Bullish"
        elif sentiment_score < -0.2:
            sentiment_label = "Bearish"
        else:
            sentiment_label = "Neutral"
        
        return sentiment_score, sentiment_label
    
    def analyze_reddit_sentiment(self, limit=100, min_mentions=3):
        """Analyze sentiment for trending stocks on Reddit"""
        print("Fetching Reddit sentiment data...")
        posts = self.get_reddit_posts(limit=limit)
        
        if not posts:
            return {}
        
        symbol_data = defaultdict(lambda: {
            'mentions': 0,
            'total_sentiment': 0,
            'post_scores': [],
            'sentiment_scores': [],
            'contexts': []
        })
        
        for post in posts:
            # Combine title and text for analysis
            full_text = f"{post['title']} {post['selftext']}"
            symbols = self.extract_stock_symbols(full_text)
            
            if symbols:
                sentiment_score, sentiment_label = self.analyze_sentiment(full_text)
                
                for symbol in symbols:
                    symbol_data[symbol]['mentions'] += 1
                    symbol_data[symbol]['total_sentiment'] += sentiment_score
                    symbol_data[symbol]['post_scores'].append(post['score'])
                    symbol_data[symbol]['sentiment_scores'].append(sentiment_score)
                    symbol_data[symbol]['contexts'].append({
                        'title': post['title'][:100],
                        'sentiment': sentiment_label,
                        'score': post['score']
                    })
        
        # Filter symbols with minimum mentions and calculate averages
        trending_symbols = {}
        for symbol, data in symbol_data.items():
            if data['mentions'] >= min_mentions:
                avg_sentiment = data['total_sentiment'] / data['mentions']
                avg_post_score = sum(data['post_scores']) / len(data['post_scores'])
                
                # Determine overall sentiment
                if avg_sentiment > 0.1:
                    overall_sentiment = "Bullish"
                elif avg_sentiment < -0.1:
                    overall_sentiment = "Bearish"
                else:
                    overall_sentiment = "Neutral"
                
                trending_symbols[symbol] = {
                    'mentions': data['mentions'],
                    'avg_sentiment_score': round(avg_sentiment, 3),
                    'overall_sentiment': overall_sentiment,
                    'avg_post_score': round(avg_post_score, 1),
                    'confidence': min(data['mentions'] / 10, 1.0),  # Confidence based on mention count
                    'top_contexts': sorted(data['contexts'], 
                                         key=lambda x: x['score'], reverse=True)[:3]
                }
        
        return trending_symbols
    
    def print_reddit_summary(self, reddit_data):
        """Print summary of Reddit sentiment analysis"""
        if not reddit_data:
            print("No Reddit sentiment data available")
            return
        
        print(f"\n{'='*80}")
        print(f"REDDIT SENTIMENT ANALYSIS (r/wallstreetbets)")
        print(f"{'='*80}")
        
        # Sort by mentions
        sorted_symbols = sorted(reddit_data.items(), 
                              key=lambda x: x[1]['mentions'], reverse=True)
        
        bullish_stocks = []
        bearish_stocks = []
        neutral_stocks = []
        
        for symbol, data in sorted_symbols:
            if data['overall_sentiment'] == 'Bullish':
                bullish_stocks.append((symbol, data))
            elif data['overall_sentiment'] == 'Bearish':
                bearish_stocks.append((symbol, data))
            else:
                neutral_stocks.append((symbol, data))
        
        print(f"\nBULLISH SENTIMENT ({len(bullish_stocks)} stocks):")
        for symbol, data in bullish_stocks[:10]:
            print(f"  {symbol}: {data['mentions']} mentions, "
                  f"sentiment: {data['avg_sentiment_score']:.3f}, "
                  f"avg score: {data['avg_post_score']:.1f}")
        
        print(f"\nBEARISH SENTIMENT ({len(bearish_stocks)} stocks):")
        for symbol, data in bearish_stocks[:10]:
            print(f"  {symbol}: {data['mentions']} mentions, "
                  f"sentiment: {data['avg_sentiment_score']:.3f}, "
                  f"avg score: {data['avg_post_score']:.1f}")
        
        print(f"\nNEUTRAL SENTIMENT ({len(neutral_stocks)} stocks):")
        for symbol, data in neutral_stocks[:5]:
            print(f"  {symbol}: {data['mentions']} mentions, "
                  f"sentiment: {data['avg_sentiment_score']:.3f}, "
                  f"avg score: {data['avg_post_score']:.1f}")
        
        if sorted_symbols:
            print(f"\nMOST MENTIONED STOCKS:")
            for symbol, data in sorted_symbols[:10]:
                print(f"  {symbol}: {data['mentions']} mentions ({data['overall_sentiment']})")

class TechnicalAnalysis:
    """Technical analysis and trend prediction methods"""
    
    @staticmethod
    def calculate_sma(data, window):
        """Calculate Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(data, window):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def calculate_rsi(data, window=14):
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalAnalysis.calculate_ema(data, fast)
        ema_slow = TechnicalAnalysis.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalAnalysis.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data, window=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = TechnicalAnalysis.calculate_sma(data, window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_stochastic(high, low, close, k_window=14, d_window=3):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    @staticmethod
    def analyze_trend_strength(data, window=20):
        """Analyze trend strength and direction"""
        sma_short = TechnicalAnalysis.calculate_sma(data, window//2)
        sma_long = TechnicalAnalysis.calculate_sma(data, window)
        
        if len(sma_short) < 2 or len(sma_long) < 2:
            return "Insufficient data", 0
        
        # Current trend direction
        if sma_short.iloc[-1] > sma_long.iloc[-1]:
            trend = "Uptrend"
        else:
            trend = "Downtrend"
        
        # Trend strength (based on slope of moving average)
        recent_slope = (sma_short.iloc[-1] - sma_short.iloc[-5]) / 5
        strength = abs(recent_slope) / data.iloc[-1] * 100  # Normalize by price
        
        return trend, strength

class StockDataFetcher:
    def __init__(self):
        # Top 100 stocks by market cap (mix of different sectors)
        self.stock_symbols = [
            # Tech giants
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'ADBE', 'CRM',
            'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'INTU', 'MU', 'AMAT', 'LRCX',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'SPGI',
            'CME', 'ICE', 'COF', 'TFC', 'USB', 'PNC', 'BK', 'STT', 'NTRS', 'RF',
            
            # Healthcare
            'UNH', 'JNJ', 'PFE', 'ABT', 'TMO', 'MRK', 'ABBV', 'DHR', 'BMY', 'LLY',
            'MDT', 'GILD', 'AMGN', 'ISRG', 'SYK', 'BSX', 'REGN', 'VRTX', 'BIIB', 'ILMN',
            
            # Consumer & Retail
            'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'LOW', 'TGT', 'SBUX',
            'MCD', 'NKE', 'DIS', 'CMCSA', 'VZ', 'T', 'CL', 'KMB', 'GIS', 'K',
            
            # Industrial & Energy
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'GD',
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'OXY', 'HAL'
        ]
        
        # Initialize analyzers
        self.reddit_analyzer = RedditSentimentAnalyzer()
        self.options_analyzer = OptionsAnalyzer()
    
    def get_reddit_trending_symbols(self, limit=100, min_mentions=3):
        """Get trending symbols from Reddit with sentiment analysis"""
        return self.reddit_analyzer.analyze_reddit_sentiment(limit=limit, min_mentions=min_mentions)
    
    def combine_technical_and_sentiment(self, technical_data, reddit_data):
        """Combine technical analysis with Reddit sentiment"""
        for symbol, stock_info in technical_data.items():
            if symbol in reddit_data:
                reddit_info = reddit_data[symbol]
                
                # Add Reddit data to stock info
                stock_info['reddit_sentiment'] = {
                    'mentions': reddit_info['mentions'],
                    'sentiment': reddit_info['overall_sentiment'],
                    'sentiment_score': reddit_info['avg_sentiment_score'],
                    'confidence': reddit_info['confidence'],
                    'avg_post_score': reddit_info['avg_post_score']
                }
                
                # Adjust technical prediction based on Reddit sentiment
                if 'technical_analysis' in stock_info and 'error' not in stock_info['technical_analysis']:
                    ta = stock_info['technical_analysis']
                    current_confidence = ta.get('confidence', 50)
                    
                    # Initialize key_signals if it doesn't exist
                    if 'key_signals' not in ta:
                        ta['key_signals'] = []
                    
                    # Boost confidence if Reddit sentiment aligns with technical analysis
                    if (ta.get('overall_prediction') == 'Likely Increase' and 
                        reddit_info['overall_sentiment'] == 'Bullish'):
                        adjusted_confidence = min(95, current_confidence + reddit_info['confidence'] * 10)
                        ta['confidence'] = round(adjusted_confidence, 1)
                        ta['key_signals'].append(f"Reddit sentiment: {reddit_info['overall_sentiment']} ({reddit_info['mentions']} mentions)")
                    
                    elif (ta.get('overall_prediction') == 'Likely Decrease' and 
                          reddit_info['overall_sentiment'] == 'Bearish'):
                        adjusted_confidence = min(95, current_confidence + reddit_info['confidence'] * 10)
                        ta['confidence'] = round(adjusted_confidence, 1)
                        ta['key_signals'].append(f"Reddit sentiment: {reddit_info['overall_sentiment']} ({reddit_info['mentions']} mentions)")
                    
                    # Reduce confidence if sentiment conflicts
                    elif ((ta.get('overall_prediction') == 'Likely Increase' and 
                           reddit_info['overall_sentiment'] == 'Bearish') or
                          (ta.get('overall_prediction') == 'Likely Decrease' and 
                           reddit_info['overall_sentiment'] == 'Bullish')):
                        adjusted_confidence = max(30, current_confidence - reddit_info['confidence'] * 5)
                        ta['confidence'] = round(adjusted_confidence, 1)
                        ta['key_signals'].append(f"Reddit sentiment conflicts: {reddit_info['overall_sentiment']} ({reddit_info['mentions']} mentions)")
                    
                    else:
                        ta['key_signals'].append(f"Reddit sentiment: {reddit_info['overall_sentiment']} ({reddit_info['mentions']} mentions)")
                
                elif 'technical_analysis' not in stock_info or 'error' in stock_info['technical_analysis']:
                    # If technical analysis failed, create a basic analysis based on Reddit sentiment
                    stock_info['technical_analysis'] = {
                        'current_price': stock_info.get('current_price', 'N/A'),
                        'trend': 'Unknown',
                        'trend_strength': 0,
                        'rsi': None,
                        'rsi_signal': 'No data',
                        'macd_signal': 'No data',
                        'bollinger_position': 'No data',
                        'moving_average_signal': 'No data',
                        'overall_prediction': self.predict_from_reddit_only(reddit_info),
                        'confidence': min(60, reddit_info['confidence'] * 60),
                        'key_signals': [f"Reddit-only analysis: {reddit_info['overall_sentiment']} ({reddit_info['mentions']} mentions)",
                                      "Technical analysis unavailable - prediction based on Reddit sentiment only"],
                        'price_target': None,
                        'risk_level': 'High'
                    }
        
        return technical_data
    
    def predict_from_reddit_only(self, reddit_info):
        """Generate prediction based only on Reddit sentiment"""
        sentiment = reddit_info['overall_sentiment']
        if sentiment == 'Bullish':
            return 'Likely Increase'
        elif sentiment == 'Bearish':
            return 'Likely Decrease'
        else:
            return 'Sideways/Uncertain'
    
    def analyze_stock_trends(self, symbol, hist_data):
        """Perform comprehensive technical analysis on stock data"""
        if len(hist_data) < 30:
            return {"error": "Insufficient data for analysis"}
        
        close_prices = hist_data['Close']
        high_prices = hist_data['High']
        low_prices = hist_data['Low']
        volume = hist_data['Volume']
        
        # Moving Averages
        sma_20 = TechnicalAnalysis.calculate_sma(close_prices, 20)
        sma_50 = TechnicalAnalysis.calculate_sma(close_prices, 50)
        
        # Technical Indicators
        rsi = TechnicalAnalysis.calculate_rsi(close_prices)
        macd_line, signal_line, histogram = TechnicalAnalysis.calculate_macd(close_prices)
        upper_bb, middle_bb, lower_bb = TechnicalAnalysis.calculate_bollinger_bands(close_prices)
        
        # Current values
        current_price = close_prices.iloc[-1]
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None
        current_macd = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else None
        current_signal = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else None
        
        # Trend Analysis
        trend, strength = TechnicalAnalysis.analyze_trend_strength(close_prices)
        
        # Price position relative to Bollinger Bands
        bb_position = "Middle"
        if not pd.isna(upper_bb.iloc[-1]) and not pd.isna(lower_bb.iloc[-1]):
            if current_price > upper_bb.iloc[-1]:
                bb_position = "Above Upper Band"
            elif current_price < lower_bb.iloc[-1]:
                bb_position = "Below Lower Band"
            elif current_price > middle_bb.iloc[-1]:
                bb_position = "Upper Half"
            else:
                bb_position = "Lower Half"
        
        # Generate signals and predictions
        signals = self.generate_trading_signals(
            current_price, current_rsi, current_macd, current_signal,
            sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else None,
            sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else None,
            bb_position, trend, strength
        )
        
        analysis = {
            'current_price': current_price,
            'trend': trend,
            'trend_strength': strength,
            'rsi': current_rsi,
            'rsi_signal': self.interpret_rsi(current_rsi),
            'macd_signal': self.interpret_macd(current_macd, current_signal),
            'bollinger_position': bb_position,
            'moving_average_signal': self.interpret_moving_averages(current_price, 
                sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else None, 
                sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else None),
            'overall_prediction': signals['prediction'],
            'confidence': signals['confidence'],
            'key_signals': signals['signals'],
            'price_target': signals.get('price_target'),
            'risk_level': signals['risk_level']
        }
        
        return analysis
    
    def interpret_moving_averages(self, current_price, sma20, sma50):
        """Interpret moving average signals"""
        if sma20 is None or sma50 is None:
            return "No data"
        
        if current_price > sma20 > sma50:
            return "Strong Bullish (Price > SMA20 > SMA50)"
        elif current_price > sma20 and sma20 < sma50:
            return "Mixed (Price > SMA20 but SMA20 < SMA50)"
        elif current_price < sma20 < sma50:
            return "Strong Bearish (Price < SMA20 < SMA50)"
        elif current_price < sma20 and sma20 > sma50:
            return "Mixed (Price < SMA20 but SMA20 > SMA50)"
        elif current_price > sma20 and sma20 == sma50:
            return "Neutral with upside bias"
        elif current_price < sma20 and sma20 == sma50:
            return "Neutral with downside bias"
        else:
            return "Neutral"
    
    def generate_trading_signals(self, price, rsi, macd, signal, sma20, sma50, bb_pos, trend, strength):
        """Generate trading signals and price predictions"""
        signals = []
        bullish_count = 0
        bearish_count = 0
        total_signals = 0
        
        # RSI Analysis
        if rsi is not None:
            total_signals += 1
            if rsi < 30:
                signals.append("RSI oversold - potential upside")
                bullish_count += 1
            elif rsi > 70:
                signals.append("RSI overbought - potential downside")
                bearish_count += 1
            elif 40 <= rsi <= 60:
                signals.append("RSI neutral")
        
        # MACD Analysis
        if macd is not None and signal is not None:
            total_signals += 1
            if macd > signal:
                signals.append("MACD bullish crossover")
                bullish_count += 1
            else:
                signals.append("MACD bearish crossover")
                bearish_count += 1
        
        # Moving Average Analysis
        if sma20 is not None and sma50 is not None:
            total_signals += 1
            if price > sma20 > sma50:
                signals.append("Price above both moving averages - bullish")
                bullish_count += 1
            elif price < sma20 < sma50:
                signals.append("Price below both moving averages - bearish")
                bearish_count += 1
            else:
                signals.append("Mixed moving average signals")
        
        # Bollinger Bands Analysis
        total_signals += 1
        if bb_pos == "Below Lower Band":
            signals.append("Price below lower Bollinger Band - oversold")
            bullish_count += 1
        elif bb_pos == "Above Upper Band":
            signals.append("Price above upper Bollinger Band - overbought")
            bearish_count += 1
        
        # Trend Analysis
        if trend == "Uptrend" and strength > 0.5:
            bullish_count += 0.5
        elif trend == "Downtrend" and strength > 0.5:
            bearish_count += 0.5
        
        # Calculate prediction
        if total_signals == 0:
            prediction = "Insufficient data"
            confidence = 0
        else:
            bullish_ratio = bullish_count / total_signals
            bearish_ratio = bearish_count / total_signals
            
            if bullish_ratio > 0.6:
                prediction = "Likely Increase"
                confidence = min(90, bullish_ratio * 100)
            elif bearish_ratio > 0.6:
                prediction = "Likely Decrease"
                confidence = min(90, bearish_ratio * 100)
            else:
                prediction = "Sideways/Uncertain"
                confidence = 50
        
        # Simple price target estimation
        price_target = None
        if prediction == "Likely Increase":
            price_target = f"${price * 1.05:.2f} - ${price * 1.15:.2f}"
        elif prediction == "Likely Decrease":
            price_target = f"${price * 0.85:.2f} - ${price * 0.95:.2f}"
        
        # Risk assessment
        risk_level = "Medium"
        if rsi is not None:
            if rsi < 20 or rsi > 80:
                risk_level = "High"
            elif 35 < rsi < 65:
                risk_level = "Low"
        
        return {
            'prediction': prediction,
            'confidence': round(confidence, 1),
            'signals': signals,
            'price_target': price_target,
            'risk_level': risk_level
        }
    
    def interpret_rsi(self, rsi):
        """Interpret RSI values"""
        if rsi is None:
            return "No data"
        elif rsi < 30:
            return "Oversold (Bullish)"
        elif rsi > 70:
            return "Overbought (Bearish)"
        else:
            return "Neutral"
    
    def interpret_macd(self, macd, signal):
        """Interpret MACD signals"""
        if macd is None or signal is None:
            return "No data"
        elif macd > signal:
            return "Bullish"
        else:
            return "Bearish"
    
    def simplified_analysis(self, symbol, hist_data):
        """Simplified analysis for stocks with limited data"""
        if len(hist_data) < 5:
            return {"error": "Insufficient data even for basic analysis"}
        
        close_prices = hist_data['Close']
        
        # Basic trend analysis
        current_price = close_prices.iloc[-1]
        start_price = close_prices.iloc[0]
        price_change = ((current_price - start_price) / start_price) * 100
        
        # Simple moving average if we have enough data
        trend = "Unknown"
        if len(close_prices) >= 10:
            recent_avg = close_prices.tail(5).mean()
            older_avg = close_prices.head(5).mean()
            if recent_avg > older_avg:
                trend = "Upward"
            elif recent_avg < older_avg:
                trend = "Downward"
            else:
                trend = "Sideways"
        
        return {
            'trend': trend,
            'price_change_pct': round(price_change, 2),
            'current_price': current_price,
            'data_points': len(hist_data)
        }
    
    def print_detailed_analysis(self, stock_info, symbol):
        """Print detailed analysis for a specific stock"""
        print(f"\n{'='*60}")
        print(f"DETAILED ANALYSIS: {symbol}")
        print(f"{'='*60}")
        
        print(f"Company: {stock_info.get('company_name', 'N/A')}")
        print(f"Sector: {stock_info.get('sector', 'N/A')}")
        print(f"Current Price: ${stock_info.get('current_price', 0):.2f}")
        
        if 'market_cap' in stock_info and stock_info['market_cap'] != 'N/A':
            market_cap = stock_info['market_cap']
            if market_cap > 1e9:
                print(f"Market Cap: ${market_cap/1e9:.1f}B")
            elif market_cap > 1e6:
                print(f"Market Cap: ${market_cap/1e6:.1f}M")
            else:
                print(f"Market Cap: ${market_cap:,.0f}")
        
        if 'pct_change' in stock_info:
            print(f"Period Performance: {stock_info['pct_change']:+.2f}%")
        
        # Technical analysis details
        if 'technical_analysis' in stock_info:
            ta = stock_info['technical_analysis']
            if 'error' not in ta:
                print(f"\nTechnical Analysis:")
                print(f"  Overall Prediction: {ta.get('overall_prediction', 'N/A')}")
                print(f"  Confidence: {ta.get('confidence', 0):.1f}%")
                print(f"  Price Target: {ta.get('price_target', 'N/A')}")
                print(f"  Risk Level: {ta.get('risk_level', 'N/A')}")
                
                print(f"\nTechnical Indicators:")
                if ta.get('rsi'):
                    print(f"  RSI: {ta['rsi']:.1f} ({ta.get('rsi_signal', 'N/A')})")
                print(f"  Trend: {ta.get('trend', 'N/A')}")
                if ta.get('trend_strength'):
                    print(f"  Trend Strength: {ta['trend_strength']:.2f}")
                print(f"  MACD: {ta.get('macd_signal', 'N/A')}")
                print(f"  Moving Averages: {ta.get('moving_average_signal', 'N/A')}")
                print(f"  Bollinger Bands: {ta.get('bollinger_position', 'N/A')}")
                
                print(f"\nKey Signals:")
                signals = ta.get('key_signals', [])
                if signals:
                    for i, signal in enumerate(signals, 1):
                        print(f"  {i}. {signal}")
        
        elif 'simplified_analysis' in stock_info:
            sa = stock_info['simplified_analysis']
            print(f"\nBasic Analysis:")
            print(f"  Trend: {sa.get('trend', 'N/A')}")
            print(f"  Price Change: {sa.get('price_change_pct', 0):+.2f}%")
            print(f"  Data Points: {sa.get('data_points', 0)} days")
        
        # Reddit sentiment if available
        if 'reddit_sentiment' in stock_info:
            reddit = stock_info['reddit_sentiment']
            print(f"\nReddit Sentiment:")
            print(f"  Mentions: {reddit.get('mentions', 'N/A')}")
            print(f"  Overall Sentiment: {reddit.get('sentiment', 'N/A')}")
            print(f"  Sentiment Score: {reddit.get('sentiment_score', 'N/A')}")
        
        print(f"\n{'='*60}")

def get_user_preferences(fetcher):
    """Get user preferences for stock data fetching"""
    print("Stock Data Fetcher with Technical Analysis, Reddit Sentiment & Options")
    print("=" * 75)
    
    # Ask for mode selection
    print("\nSelect mode:")
    print("1. Fetch specific stock symbol(s)")
    print("2. Fetch from predefined list")
    print("3. Analyze Reddit trending stocks only")
    
    while True:
        try:
            mode = input("\nEnter your choice (1, 2, or 3): ").strip()
            if mode in ['1', '2', '3']:
                break
            print("Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None
    
    symbols_to_use = []
    
    if mode == '1':
        # Custom symbols mode
        print("\nEnter stock symbols (comma-separated, e.g., AAPL,MSFT,GOOGL):")
        while True:
            try:
                symbols_input = input("Symbols: ").strip().upper()
                if symbols_input:
                    symbols_to_use = [s.strip() for s in symbols_input.split(',')]
                    print(f"You entered: {', '.join(symbols_to_use)}")
                    confirm = input("Is this correct? (y/n): ").strip().lower()
                    if confirm == 'y':
                        break
                else:
                    print("Please enter at least one symbol")
            except KeyboardInterrupt:
                print("\nExiting...")
                return None
    
    elif mode == '2':
        # Predefined list mode
        print(f"\nAvailable stocks in predefined list: {len(fetcher.stock_symbols)}")
        print("How many stocks would you like to fetch data for?")
        print(f"Enter a number between 1 and {len(fetcher.stock_symbols)}")
        
        while True:
            try:
                num_stocks = input(f"Number of stocks (default: 10): ").strip()
                if not num_stocks:
                    num_stocks = 10
                    break
                
                num_stocks = int(num_stocks)
                if 1 <= num_stocks <= len(fetcher.stock_symbols):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(fetcher.stock_symbols)}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nExiting...")
                return None
        
        symbols_to_use = fetcher.stock_symbols[:num_stocks]
        print(f"Will fetch data for first {num_stocks} stocks")
    
    else:  # mode == '3'
        # Reddit trending mode
        print("\nAnalyzing Reddit trending stocks...")
        reddit_data = fetcher.get_reddit_trending_symbols()
        if reddit_data:
            symbols_to_use = list(reddit_data.keys())
            print(f"Found {len(symbols_to_use)} trending stocks from Reddit")
        else:
            print("No trending stocks found on Reddit")
            return None
    
    return {
        'symbols': symbols_to_use,
        'mode': mode
    }

def main():
    """Main execution function"""
    # Create fetcher instance
    fetcher = StockDataFetcher()
    
    # Get user preferences
    prefs = get_user_preferences(fetcher)
    if prefs is None:
        return None
    
    symbols = prefs['symbols']
    mode = prefs['mode']
    
    # Set up basic parameters
    period = '3mo'  # Use 3 months for better technical analysis
    interval = '1d'
    include_predictions = True
    include_reddit = mode == '3'  # Only get Reddit data if specifically requested
    save_to_file = False
    
    print(f"\nStarting analysis for {len(symbols)} stocks...")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {period}, Interval: {interval}")
    print(f"Technical analysis: Yes")
    print(f"Reddit sentiment: {'Yes' if include_reddit else 'No'}")
    print("-" * 50)
    
    # Fetch and analyze data
    try:
        all_data = {}
        failed_symbols = []
        reddit_data = {}
        
        # Get Reddit sentiment data ONLY if in Reddit mode or if requested
        if include_reddit:
            print("Analyzing Reddit sentiment from r/wallstreetbets...")
            reddit_data = fetcher.get_reddit_trending_symbols()
            fetcher.reddit_analyzer.print_reddit_summary(reddit_data)
        
        print(f"\nFetching and analyzing your requested stocks...")
        
        for i, symbol in enumerate(symbols, 1):
            try:
                print(f"\nAnalyzing {i}/{len(symbols)}: {symbol}")
                
                # Create ticker object
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval)
                
                if not hist.empty:
                    # Get basic info
                    info = ticker.info
                    current_price = info.get('currentPrice', hist['Close'].iloc[-1])
                    
                    # Store data
                    all_data[symbol] = {
                        'historical_data': hist,
                        'company_name': info.get('longName', 'N/A'),
                        'sector': info.get('sector', 'N/A'),
                        'current_price': current_price,
                        'market_cap': info.get('marketCap', 'N/A')
                    }
                    
                    # Calculate basic metrics
                    if len(hist) > 1:
                        price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[0]
                        pct_change = (price_change / hist['Close'].iloc[0]) * 100
                        all_data[symbol]['price_change'] = price_change
                        all_data[symbol]['pct_change'] = pct_change
                        print(f"  Current Price: ${current_price:.2f}")
                        print(f"  Price Change ({period}): {pct_change:+.2f}%")
                    
                    # Add technical analysis
                    if include_predictions:
                        analysis = fetcher.analyze_stock_trends(symbol, hist)
                        all_data[symbol]['technical_analysis'] = analysis
                        
                        if 'error' in analysis:
                            print(f"  Technical Analysis: {analysis['error']}")
                            # Try with less data requirement
                            if len(hist) >= 14:
                                print("  Attempting simplified analysis...")
                                simplified_analysis = fetcher.simplified_analysis(symbol, hist)
                                all_data[symbol]['simplified_analysis'] = simplified_analysis
                        else:
                            print(f"  Prediction: {analysis.get('overall_prediction', 'N/A')}")
                            print(f"  Confidence: {analysis.get('confidence', 0):.1f}%")
                            print(f"  Price Target: {analysis.get('price_target', 'N/A')}")
                            print(f"  Risk Level: {analysis.get('risk_level', 'N/A')}")
                            
                            # Show key technical indicators
                            if analysis.get('rsi'):
                                print(f"  RSI: {analysis['rsi']:.1f} ({analysis.get('rsi_signal', 'N/A')})")
                            print(f"  Trend: {analysis.get('trend', 'N/A')}")
                            print(f"  MACD: {analysis.get('macd_signal', 'N/A')}")
                        
                        # Check for Reddit sentiment on this specific stock
                        if symbol in reddit_data:
                            reddit_sentiment = reddit_data[symbol]['overall_sentiment']
                            reddit_mentions = reddit_data[symbol]['mentions']
                            print(f"  Reddit Sentiment: {reddit_sentiment} ({reddit_mentions} mentions)")
                        elif include_reddit:
                            print(f"  Reddit Sentiment: Not trending on WSB")
                
                else:
                    failed_symbols.append(symbol)
                    print(f"  ERROR: No price data available for {symbol}")
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                failed_symbols.append(symbol)
                print(f"  ERROR: Failed to analyze {symbol}: {str(e)}")
        
        # Combine technical and Reddit data only for the requested symbols
        if include_reddit and include_predictions and reddit_data:
            all_data = fetcher.combine_technical_and_sentiment(all_data, reddit_data)
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        if all_data:
            for symbol, info in all_data.items():
                print(f"\n{symbol} - {info.get('company_name', 'N/A')}")
                print(f"  Sector: {info.get('sector', 'N/A')}")
                print(f"  Current Price: ${info.get('current_price', 0):.2f}")
                
                if 'pct_change' in info:
                    print(f"  Performance ({period}): {info['pct_change']:+.2f}%")
                
                if 'technical_analysis' in info and 'error' not in info['technical_analysis']:
                    ta = info['technical_analysis']
                    print(f"  Prediction: {ta.get('overall_prediction', 'N/A')} ({ta.get('confidence', 0):.1f}%)")
                elif 'simplified_analysis' in info:
                    sa = info['simplified_analysis']
                    print(f"  Basic Analysis: {sa.get('trend', 'N/A')}")
                else:
                    print(f"  Analysis: Limited data available")
        
        if failed_symbols:
            print(f"\nFailed to analyze: {', '.join(failed_symbols)}")
        
        # Interactive detailed analysis
        if all_data and len(all_data) <= 5:  # Only for small requests
            print(f"\nWould you like detailed analysis for any specific stock?")
            while True:
                try:
                    choice = input("Enter symbol for details (or 'quit'): ").strip().upper()
                    if choice.lower() == 'quit' or not choice:
                        break
                    if choice in all_data:
                        fetcher.print_detailed_analysis(all_data[choice], choice)
                    else:
                        print(f"No data available for {choice}")
                except KeyboardInterrupt:
                    break
        
        return all_data
        
    except Exception as e:
        print(f"Error in main analysis: {str(e)}")
        return None

if __name__ == "__main__":
    print("=" * 75)
    print("STOCK ANALYSIS SUITE: TECHNICAL + REDDIT SENTIMENT")
    print("=" * 75)
    print("This script analyzes stocks using:")
    print("• Technical indicators (RSI, MACD, Bollinger Bands, etc.)")
    print("• Reddit sentiment analysis from r/wallstreetbets")
    print("• Combined predictions with confidence scoring")
    print("\nNote: This is for educational purposes only and should not be")
    print("considered as financial advice. Always do your own research.")
    print("=" * 75)
    
    data = main()