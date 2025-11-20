"""
Stock data fetching and technical analysis module
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from .config import Config

class StockAnalyzer:
    """Handles stock data fetching and technical analysis"""
    
    def __init__(self):
        """Initialize stock analyzer"""
        self.config = Config()
    
    def get_stock_data(self, symbol: str, days_back: int = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Fetch stock price data with technical indicators
        
        Args:
            symbol: Stock ticker symbol
            days_back: Number of days of historical data
            
        Returns:
            Tuple of (DataFrame with stock data, dict with stock info)
        """
        if days_back is None:
            days_back = self.config.STOCK_HISTORY_DAYS
            
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            df.reset_index(inplace=True)
            
            # Get additional company info
            info = stock.info
            
            # Calculate technical indicators
            if not df.empty:
                df = self._calculate_technical_indicators(df)
            
            return df, info
            
        except Exception as e:
            print(f"Error fetching stock data for {symbol}: {e}")
            return pd.DataFrame(), {}
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various technical indicators"""
        
        # Basic price metrics
        df['daily_return'] = df['Close'].pct_change()
        df['price_change'] = df['Close'].diff()
        
        # Moving averages
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['Close'].rolling(window=bb_period).mean()
        bb_std_dev = df['Close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        df['volume_change'] = df['Volume'].pct_change()
        
        # Volatility measures
        df['volatility_5d'] = df['daily_return'].rolling(window=5).std()
        df['volatility_10d'] = df['daily_return'].rolling(window=10).std()
        df['volatility_20d'] = df['daily_return'].rolling(window=20).std()
        
        # High/Low analysis
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Support and resistance levels
        df['resistance_20d'] = df['High'].rolling(window=20).max()
        df['support_20d'] = df['Low'].rolling(window=20).min()
        
        # Price relative to moving averages
        df['price_vs_sma20'] = (df['Close'] - df['sma_20']) / df['sma_20']
        df['price_vs_sma50'] = (df['Close'] - df['sma_50']) / df['sma_50']
        
        return df
    
    def get_options_data(self, symbol: str) -> Optional[Dict]:
        """
        Get basic options data for the symbol
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with options data or None if unavailable
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
            
            if calls.empty or puts.empty:
                return None
            
            # Find at-the-money options
            atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
            atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]
            
            # Calculate additional metrics
            total_call_volume = calls['volume'].sum()
            total_put_volume = puts['volume'].sum()
            total_call_oi = calls['openInterest'].sum()
            total_put_oi = puts['openInterest'].sum()
            
            return {
                'current_price': current_price,
                'nearest_expiry': nearest_expiry,
                'atm_call_iv': atm_call['impliedVolatility'].iloc[0] if not atm_call.empty else 0,
                'atm_put_iv': atm_put['impliedVolatility'].iloc[0] if not atm_put.empty else 0,
                'atm_call_volume': atm_call['volume'].iloc[0] if not atm_call.empty else 0,
                'atm_put_volume': atm_put['volume'].iloc[0] if not atm_put.empty else 0,
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'total_call_oi': total_call_oi,
                'total_put_oi': total_put_oi,
                'call_put_volume_ratio': total_call_volume / max(total_put_volume, 1),
                'call_put_oi_ratio': total_call_oi / max(total_put_oi, 1),
                'iv_rank': self._calculate_iv_rank(calls, puts),
                'options_flow': self._analyze_options_flow(calls, puts)
            }
            
        except Exception as e:
            print(f"Options data error for {symbol}: {e}")
            return None
    
    def _calculate_iv_rank(self, calls: pd.DataFrame, puts: pd.DataFrame) -> str:
        """Calculate implied volatility rank"""
        try:
            all_ivs = pd.concat([calls['impliedVolatility'], puts['impliedVolatility']])
            avg_iv = all_ivs.mean()
            
            if avg_iv > 0.8:
                return "EXTREMELY_HIGH"
            elif avg_iv > 0.6:
                return "HIGH"
            elif avg_iv > 0.4:
                return "MEDIUM"
            elif avg_iv > 0.2:
                return "LOW"
            else:
                return "VERY_LOW"
        except:
            return "UNKNOWN"
    
    def _analyze_options_flow(self, calls: pd.DataFrame, puts: pd.DataFrame) -> Dict:
        """Analyze options flow patterns"""
        try:
            # Volume analysis
            call_volume = calls['volume'].sum()
            put_volume = puts['volume'].sum()
            total_volume = call_volume + put_volume
            
            # Open interest analysis
            call_oi = calls['openInterest'].sum()
            put_oi = puts['openInterest'].sum()
            
            # Unusual volume detection
            avg_volume = (call_volume + put_volume) / len(calls) if len(calls) > 0 else 0
            unusual_volume = total_volume > avg_volume * 3
            
            return {
                'total_volume': total_volume,
                'call_volume_pct': (call_volume / max(total_volume, 1)) * 100,
                'put_volume_pct': (put_volume / max(total_volume, 1)) * 100,
                'unusual_volume': unusual_volume,
                'volume_bias': 'calls' if call_volume > put_volume else 'puts',
                'oi_bias': 'calls' if call_oi > put_oi else 'puts'
            }
        except:
            return {}
    
    def calculate_key_levels(self, df: pd.DataFrame) -> Dict:
        """
        Calculate key support and resistance levels
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Dictionary with key levels
        """
        if df.empty or len(df) < 20:
            return {}
        
        current_price = df['Close'].iloc[-1]
        
        # Recent highs and lows
        recent_20d = df.tail(20)
        recent_high = recent_20d['High'].max()
        recent_low = recent_20d['Low'].min()
        
        # Longer-term levels
        longer_60d = df.tail(60) if len(df) >= 60 else df
        support_60d = longer_60d['Low'].min()
        resistance_60d = longer_60d['High'].max()
        
        # Moving average levels
        sma_20 = df['sma_20'].iloc[-1] if not pd.isna(df['sma_20'].iloc[-1]) else current_price
        sma_50 = df['sma_50'].iloc[-1] if not pd.isna(df['sma_50'].iloc[-1]) else current_price
        
        # Calculate distances from current price
        levels = {
            'current_price': current_price,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'support_60d': support_60d,
            'resistance_60d': resistance_60d,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'distance_to_high': (recent_high - current_price) / current_price,
            'distance_to_low': (current_price - recent_low) / current_price,
            'trend': 'bullish' if sma_20 > sma_50 else 'bearish'
        }
        
        return levels
    
    def get_stock_summary(self, df: pd.DataFrame, info: Dict) -> Dict:
        """
        Get comprehensive stock summary
        
        Args:
            df: DataFrame with stock data
            info: Stock info dictionary
            
        Returns:
            Dictionary with stock summary
        """
        if df.empty:
            return {}
        
        current_data = df.iloc[-1]
        previous_data = df.iloc[-2] if len(df) > 1 else current_data
        
        # Basic metrics
        current_price = current_data['Close']
        price_change_1d = current_data['daily_return'] if not pd.isna(current_data['daily_return']) else 0
        volume_ratio = current_data['volume_ratio'] if not pd.isna(current_data['volume_ratio']) else 1
        
        # Technical indicators
        rsi = current_data['rsi'] if not pd.isna(current_data['rsi']) else 50
        macd = current_data['macd'] if not pd.isna(current_data['macd']) else 0
        bb_position = current_data['bb_position'] if not pd.isna(current_data['bb_position']) else 0.5
        
        # Volatility
        volatility = current_data['volatility_20d'] if not pd.isna(current_data['volatility_20d']) else 0
        
        # Determine technical signals
        rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        macd_signal = "Bullish" if macd > 0 else "Bearish"
        bb_signal = "Overbought" if bb_position > 0.8 else "Oversold" if bb_position < 0.2 else "Neutral"
        
        return {
            'symbol': info.get('symbol', 'N/A'),
            'company_name': info.get('longName', 'N/A'),
            'current_price': current_price,
            'price_change_1d': price_change_1d,
            'volume_ratio': volume_ratio,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('forwardPE', 0),
            'beta': info.get('beta', 1.0),
            'rsi': rsi,
            'rsi_signal': rsi_signal,
            'macd': macd,
            'macd_signal': macd_signal,
            'bb_position': bb_position,
            'bb_signal': bb_signal,
            'volatility_20d': volatility,
            'technical_score': self._calculate_technical_score(rsi, macd, bb_position)
        }
    
    def _calculate_technical_score(self, rsi: float, macd: float, bb_position: float) -> float:
        """Calculate a simple technical analysis score"""
        score = 0
        
        # RSI scoring
        if 30 <= rsi <= 70:
            score += 1  # Neutral is good
        elif rsi < 30:
            score += 0.5  # Oversold can be opportunity
        else:  # rsi > 70
            score -= 0.5  # Overbought is risky
        
        # MACD scoring
        if macd > 0:
            score += 0.5
        else:
            score -= 0.5
        
        # Bollinger Band position scoring
        if 0.2 <= bb_position <= 0.8:
            score += 0.5  # Middle range is stable
        
        return max(0, min(2, score))  # Score between 0 and 2