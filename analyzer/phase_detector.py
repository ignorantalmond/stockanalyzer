"""
Pump and dump phase detection module
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple

from .enums import PumpDumpPhase
from .config import Config

class PhaseDetector:
    """Detects the current phase of pump and dump cycle"""
    
    def __init__(self):
        """Initialize phase detector"""
        self.config = Config()
    
    def determine_phase(self, symbol: str, stock_df: pd.DataFrame, reddit_metrics: Dict, 
                       news_sentiment: Dict, options_data: Dict) -> Tuple[PumpDumpPhase, str, float]:
        """
        Determine which phase of pump/dump cycle the stock is in
        
        Args:
            symbol: Stock ticker symbol
            stock_df: DataFrame with stock data
            reddit_metrics: Reddit analysis metrics
            news_sentiment: News sentiment analysis
            options_data: Options market data
            
        Returns:
            Tuple of (phase, description, confidence)
        """
        if stock_df.empty:
            return PumpDumpPhase.DEAD, "No stock data available", 0.0
        
        # Get recent data (last 30 days)
        recent_days = 30
        recent_stock = stock_df.tail(recent_days).copy()
        
        if len(recent_stock) < 10:
            return PumpDumpPhase.DEAD, "Insufficient data", 0.0
        
        # Calculate key metrics
        metrics = self._calculate_phase_metrics(recent_stock, reddit_metrics, news_sentiment, options_data)
        
        # Determine phase based on metrics
        phase, description = self._classify_phase(metrics)
        
        # Calculate confidence
        confidence = self._calculate_confidence(metrics, len(recent_stock), reddit_metrics, news_sentiment)
        
        return phase, description, confidence
    
    def _calculate_phase_metrics(self, recent_stock: pd.DataFrame, reddit_metrics: Dict, 
                                news_sentiment: Dict, options_data: Dict) -> Dict:
        """Calculate all metrics needed for phase determination"""
        
        current_price = recent_stock['Close'].iloc[-1]
        price_30d_ago = recent_stock['Close'].iloc[0]
        
        # Price metrics
        price_change_30d = (current_price - price_30d_ago) / price_30d_ago
        recent_high = recent_stock['High'].max()
        recent_low = recent_stock['Low'].min()
        price_from_high = (current_price - recent_high) / recent_high
        price_from_low = (current_price - recent_low) / recent_low
        
        # Volume metrics
        avg_volume_recent = recent_stock['Volume'].mean()
        current_volume = recent_stock['Volume'].iloc[-1]
        volume_trend = current_volume / avg_volume_recent if avg_volume_recent > 0 else 1
        
        # Technical indicators
        current_rsi = recent_stock['rsi'].iloc[-1] if 'rsi' in recent_stock.columns and not pd.isna(recent_stock['rsi'].iloc[-1]) else 50
        volatility = recent_stock['volatility_20d'].iloc[-1] if 'volatility_20d' in recent_stock.columns and not pd.isna(recent_stock['volatility_20d'].iloc[-1]) else 0.02
        
        # Volume spike detection
        volume_spikes = (recent_stock['volume_ratio'] > self.config.HIGH_VOLUME_THRESHOLD).sum() if 'volume_ratio' in recent_stock.columns else 0
        
        # Price momentum
        price_momentum_5d = (recent_stock['Close'].iloc[-1] - recent_stock['Close'].iloc[-6]) / recent_stock['Close'].iloc[-6] if len(recent_stock) >= 6 else 0
        price_momentum_10d = (recent_stock['Close'].iloc[-1] - recent_stock['Close'].iloc[-11]) / recent_stock['Close'].iloc[-11] if len(recent_stock) >= 11 else 0
        
        # Sentiment metrics
        reddit_sentiment = reddit_metrics.get('avg_sentiment', 0)
        news_sentiment_score = news_sentiment.get('avg_sentiment', 0)
        combined_sentiment = (reddit_sentiment + news_sentiment_score) / 2
        
        # Reddit activity metrics
        reddit_activity_trend = reddit_metrics.get('activity_trend', 'neutral')
        post_count = reddit_metrics.get('post_count', 0)
        pump_keyword_ratio = reddit_metrics.get('pump_keyword_ratio', 0)
        dump_keyword_ratio = reddit_metrics.get('dump_keyword_ratio', 0)
        
        # Options metrics
        iv_level = 0.5  # Default
        call_put_ratio = 1.0  # Default
        if options_data:
            iv_level = max(options_data.get('atm_call_iv', 0), options_data.get('atm_put_iv', 0))
            call_put_ratio = options_data.get('call_put_volume_ratio', 1.0)
        
        # News catalyst detection
        has_major_catalyst = news_sentiment.get('has_major_catalyst', False)
        catalyst_score = news_sentiment.get('catalyst_score', 0)
        
        return {
            # Price metrics
            'current_price': current_price,
            'price_change_30d': price_change_30d,
            'price_from_high': price_from_high,
            'price_from_low': price_from_low,
            'price_momentum_5d': price_momentum_5d,
            'price_momentum_10d': price_momentum_10d,
            
            # Volume metrics
            'volume_trend': volume_trend,
            'volume_spikes': volume_spikes,
            
            # Technical metrics
            'current_rsi': current_rsi,
            'volatility': volatility,
            'iv_level': iv_level,
            
            # Sentiment metrics
            'reddit_sentiment': reddit_sentiment,
            'news_sentiment': news_sentiment_score,
            'combined_sentiment': combined_sentiment,
            'reddit_activity_trend': reddit_activity_trend,
            'post_count': post_count,
            'pump_keyword_ratio': pump_keyword_ratio,
            'dump_keyword_ratio': dump_keyword_ratio,
            
            # Options metrics
            'call_put_ratio': call_put_ratio,
            
            # News metrics
            'has_major_catalyst': has_major_catalyst,
            'catalyst_score': catalyst_score
        }
    
    def _classify_phase(self, metrics: Dict) -> Tuple[PumpDumpPhase, str]:
        """Classify the current phase based on calculated metrics"""
        
        # Extract key metrics
        price_change_30d = metrics['price_change_30d']
        price_from_high = metrics['price_from_high']
        price_from_low = metrics['price_from_low']
        current_rsi = metrics['current_rsi']
        volume_trend = metrics['volume_trend']
        combined_sentiment = metrics['combined_sentiment']
        reddit_activity_trend = metrics['reddit_activity_trend']
        pump_keyword_ratio = metrics['pump_keyword_ratio']
        dump_keyword_ratio = metrics['dump_keyword_ratio']
        iv_level = metrics['iv_level']
        current_price = metrics['current_price']
        
        # Phase classification logic
        
        # DEAD STOCK - Extreme low price or massive decline
        if current_price < 1.0 and price_change_30d < -0.6:
            return PumpDumpPhase.DEAD, f"Price ${current_price:.2f}, down {price_change_30d:.1%} in 30d"
        
        # PEAK FRENZY - Extreme pump conditions
        if (price_change_30d > 3.0 and current_rsi > 85 and volume_trend > 5) or \
           (price_change_30d > 2.0 and current_rsi > 80 and combined_sentiment > 0.5 and iv_level > 1.0):
            return PumpDumpPhase.PEAK_FRENZY, f"Extreme pump: +{price_change_30d:.1%}, RSI {current_rsi:.0f}, euphoric sentiment"
        
        # MAIN PUMP - Strong pump in progress
        if (price_change_30d > 1.0 and current_rsi > 70) or \
           (price_change_30d > 0.5 and current_rsi > 75 and volume_trend > 2):
            return PumpDumpPhase.MAIN_PUMP, f"Strong pump: +{price_change_30d:.1%}, RSI {current_rsi:.0f}"
        
        # EARLY PUMP - Beginning of pump
        if (price_change_30d > 0.3 and volume_trend > 1.5) or \
           (price_change_30d > 0.2 and combined_sentiment > 0.3 and reddit_activity_trend == 'increasing'):
            return PumpDumpPhase.EARLY_PUMP, f"Early pump: +{price_change_30d:.1%}, volume spike, bullish sentiment"
        
        # MAIN DUMP - Active dumping
        if (price_from_high < -0.4 and price_change_30d < -0.3) or \
           (price_from_high < -0.3 and dump_keyword_ratio > 0.5):
            return PumpDumpPhase.MAIN_DUMP, f"Major dump: {price_from_high:.1%} from high, bearish sentiment"
        
        # EARLY DUMP - Beginning of dump
        if (price_from_high < -0.2 and volume_trend < 0.8) or \
           (price_from_high < -0.15 and combined_sentiment < -0.2):
            return PumpDumpPhase.EARLY_DUMP, f"Early dump: {price_from_high:.1%} from recent high"
        
        # BAGHOLDERS - Sustained decline with negative sentiment
        if (price_change_30d < -0.5 and combined_sentiment < -0.2) or \
           (price_change_30d < -0.3 and dump_keyword_ratio > 0.6):
            return PumpDumpPhase.BAGHOLDERS, f"Bagholders phase: {price_change_30d:.1%}, negative sentiment"
        
        # RECOVERY ATTEMPT - Bouncing from lows
        if (price_from_low > 0.3 and price_from_high < -0.3) or \
           (price_from_low > 0.2 and combined_sentiment > 0.1 and reddit_activity_trend == 'increasing'):
            return PumpDumpPhase.RECOVERY_ATTEMPT, f"Recovery attempt: +{price_from_low:.1%} from lows"
        
        # ACCUMULATION - Default quiet phase
        return PumpDumpPhase.ACCUMULATION, f"Accumulation: {price_change_30d:.1%} change, low volatility"
    
    def _calculate_confidence(self, metrics: Dict, data_points: int, reddit_metrics: Dict, 
                            news_sentiment: Dict) -> float:
        """Calculate confidence level for phase determination"""
        
        confidence = 0.5  # Base confidence
        
        # Data quality factors
        confidence += min(0.3, data_points / 30)  # More data points = higher confidence
        confidence += min(0.2, reddit_metrics.get('post_count', 0) / 50)  # More Reddit posts = higher confidence
        confidence += min(0.1, news_sentiment.get('news_count', 0) / 10)  # More news = higher confidence
        
        # Signal strength factors
        price_change_magnitude = abs(metrics['price_change_30d'])
        if price_change_magnitude > 1.0:
            confidence += 0.2  # Large price moves are easier to classify
        elif price_change_magnitude > 0.5:
            confidence += 0.1
        
        # Volume confirmation
        if metrics['volume_trend'] > 2.0 or metrics['volume_trend'] < 0.5:
            confidence += 0.1  # Unusual volume confirms phase
        
        # Sentiment consistency
        reddit_sentiment = metrics['reddit_sentiment']
        news_sentiment_score = metrics['news_sentiment']
        if abs(reddit_sentiment - news_sentiment_score) < 0.2:
            confidence += 0.1  # Consistent sentiment across sources
        
        # Technical indicator alignment
        rsi = metrics['current_rsi']
        price_momentum = metrics['price_momentum_5d']
        if (rsi > 70 and price_momentum > 0) or (rsi < 30 and price_momentum < 0):
            confidence += 0.1  # Technical indicators align with price action
        
        # Cap confidence at 1.0
        return min(1.0, confidence)
    
    def get_phase_risk_assessment(self, phase: PumpDumpPhase, metrics: Dict) -> Dict:
        """
        Get risk assessment for the current phase
        
        Args:
            phase: Current pump/dump phase
            metrics: Phase metrics dictionary
            
        Returns:
            Dictionary with risk assessment
        """
        risk_factors = []
        risk_level = "MEDIUM"
        trade_recommendation = "CAUTION"
        
        if phase == PumpDumpPhase.PEAK_FRENZY:
            risk_level = "EXTREME"
            trade_recommendation = "AVOID"
            risk_factors = [
                "Extreme overvaluation likely",
                "Imminent dump highly probable", 
                "Extremely high volatility",
                "Options premiums extremely expensive"
            ]
        
        elif phase == PumpDumpPhase.MAIN_PUMP:
            risk_level = "HIGH"
            trade_recommendation = "SELL/AVOID_BUYING"
            risk_factors = [
                "Significant overvaluation",
                "High probability of reversal",
                "Elevated volatility",
                "Late entry risk"
            ]
        
        elif phase == PumpDumpPhase.EARLY_PUMP:
            risk_level = "HIGH"
            trade_recommendation = "CAUTION"
            risk_factors = [
                "Momentum may accelerate rapidly",
                "High volatility expected",
                "Sentiment-driven movement",
                "Options premiums may spike"
            ]
        
        elif phase == PumpDumpPhase.EARLY_DUMP:
            risk_level = "HIGH"
            trade_recommendation = "AVOID_CATCHING_KNIFE"
            risk_factors = [
                "Further decline likely",
                "Sentiment turning negative",
                "Support levels breaking",
                "Volatile price action"
            ]
        
        elif phase == PumpDumpPhase.MAIN_DUMP:
            risk_level = "EXTREME"
            trade_recommendation = "AVOID"
            risk_factors = [
                "Major price decline in progress",
                "Panic selling likely",
                "Support levels broken",
                "Recovery uncertain"
            ]
        
        elif phase == PumpDumpPhase.BAGHOLDERS:
            risk_level = "HIGH"
            trade_recommendation = "LONG_TERM_ONLY"
            risk_factors = [
                "Extended decline period",
                "Negative sentiment entrenched",
                "Low probability of quick recovery",
                "Value trap potential"
            ]
        
        elif phase == PumpDumpPhase.DEAD:
            risk_level = "EXTREME"
            trade_recommendation = "AVOID_COMPLETELY"
            risk_factors = [
                "Delisting risk",
                "Potential total loss",
                "Liquidity issues",
                "Company distress likely"
            ]
        
        elif phase == PumpDumpPhase.RECOVERY_ATTEMPT:
            risk_level = "MEDIUM"
            trade_recommendation = "WAIT_FOR_CONFIRMATION"
            risk_factors = [
                "Could be dead cat bounce",
                "Recovery may fail",
                "Mixed signals present",
                "Patience required"
            ]
        
        elif phase == PumpDumpPhase.ACCUMULATION:
            risk_level = "LOW"
            trade_recommendation = "SAFEST_ENTRY"
            risk_factors = [
                "Relatively stable period",
                "Lower volatility",
                "Better risk/reward ratio",
                "Monitor for early signals"
            ]
        
        return {
            'risk_level': risk_level,
            'trade_recommendation': trade_recommendation,
            'risk_factors': risk_factors,
            'volatility_warning': metrics['volatility'] > 0.05,
            'volume_warning': metrics['volume_trend'] > 3.0,
            'sentiment_warning': abs(metrics['combined_sentiment']) > 0.5
        }
    
    def get_phase_timeline_estimate(self, phase: PumpDumpPhase, metrics: Dict) -> Dict:
        """
        Estimate timeline for phase transitions
        
        Args:
            phase: Current phase
            metrics: Phase metrics
            
        Returns:
            Dictionary with timeline estimates
        """
        current_intensity = self._calculate_phase_intensity(metrics)
        
        timeline_estimates = {
            PumpDumpPhase.ACCUMULATION: {
                'typical_duration': '2-8 weeks',
                'next_phase_probability': {
                    PumpDumpPhase.EARLY_PUMP: 0.3,
                    PumpDumpPhase.RECOVERY_ATTEMPT: 0.2,
                    PumpDumpPhase.ACCUMULATION: 0.5  # Stay in same phase
                },
                'catalyst_dependency': 'High - needs external trigger'
            },
            PumpDumpPhase.EARLY_PUMP: {
                'typical_duration': '3-10 days',
                'next_phase_probability': {
                    PumpDumpPhase.MAIN_PUMP: 0.4,
                    PumpDumpPhase.EARLY_DUMP: 0.3,
                    PumpDumpPhase.ACCUMULATION: 0.3
                },
                'catalyst_dependency': 'Medium - momentum dependent'
            },
            PumpDumpPhase.MAIN_PUMP: {
                'typical_duration': '1-5 days',
                'next_phase_probability': {
                    PumpDumpPhase.PEAK_FRENZY: 0.3,
                    PumpDumpPhase.EARLY_DUMP: 0.5,
                    PumpDumpPhase.MAIN_DUMP: 0.2
                },
                'catalyst_dependency': 'Low - unsustainable by nature'
            },
            PumpDumpPhase.PEAK_FRENZY: {
                'typical_duration': '1-3 days',
                'next_phase_probability': {
                    PumpDumpPhase.EARLY_DUMP: 0.4,
                    PumpDumpPhase.MAIN_DUMP: 0.6
                },
                'catalyst_dependency': 'None - imminent reversal'
            },
            PumpDumpPhase.EARLY_DUMP: {
                'typical_duration': '3-7 days',
                'next_phase_probability': {
                    PumpDumpPhase.MAIN_DUMP: 0.5,
                    PumpDumpPhase.RECOVERY_ATTEMPT: 0.3,
                    PumpDumpPhase.BAGHOLDERS: 0.2
                },
                'catalyst_dependency': 'Medium - sentiment dependent'
            },
            PumpDumpPhase.MAIN_DUMP: {
                'typical_duration': '1-4 weeks',
                'next_phase_probability': {
                    PumpDumpPhase.BAGHOLDERS: 0.6,
                    PumpDumpPhase.RECOVERY_ATTEMPT: 0.3,
                    PumpDumpPhase.DEAD: 0.1
                },
                'catalyst_dependency': 'High - needs stabilization'
            },
            PumpDumpPhase.BAGHOLDERS: {
                'typical_duration': '1-6 months',
                'next_phase_probability': {
                    PumpDumpPhase.ACCUMULATION: 0.4,
                    PumpDumpPhase.RECOVERY_ATTEMPT: 0.3,
                    PumpDumpPhase.DEAD: 0.3
                },
                'catalyst_dependency': 'Very High - needs major catalyst'
            },
            PumpDumpPhase.RECOVERY_ATTEMPT: {
                'typical_duration': '1-3 weeks',
                'next_phase_probability': {
                    PumpDumpPhase.EARLY_PUMP: 0.3,
                    PumpDumpPhase.ACCUMULATION: 0.4,
                    PumpDumpPhase.BAGHOLDERS: 0.3
                },
                'catalyst_dependency': 'High - needs sustained interest'
            },
            PumpDumpPhase.DEAD: {
                'typical_duration': 'Indefinite',
                'next_phase_probability': {
                    PumpDumpPhase.DEAD: 0.9,
                    PumpDumpPhase.RECOVERY_ATTEMPT: 0.1
                },
                'catalyst_dependency': 'Extreme - needs fundamental change'
            }
        }
        
        base_estimate = timeline_estimates.get(phase, {})
        
        # Adjust based on current intensity
        if current_intensity > 0.8:
            adjusted_duration = "Shorter than typical (high intensity)"
        elif current_intensity < 0.3:
            adjusted_duration = "Longer than typical (low intensity)"
        else:
            adjusted_duration = base_estimate.get('typical_duration', 'Unknown')
        
        return {
            'typical_duration': base_estimate.get('typical_duration', 'Unknown'),
            'adjusted_duration': adjusted_duration,
            'next_phase_probabilities': base_estimate.get('next_phase_probability', {}),
            'catalyst_dependency': base_estimate.get('catalyst_dependency', 'Unknown'),
            'intensity_score': current_intensity
        }
    
    def _calculate_phase_intensity(self, metrics: Dict) -> float:
        """Calculate intensity score for current phase (0-1)"""
        intensity = 0.0
        
        # Price change intensity
        price_change = abs(metrics['price_change_30d'])
        intensity += min(0.3, price_change / 2.0)  # Max 0.3 for 200% change
        
        # Volume intensity
        volume_ratio = metrics['volume_trend']
        if volume_ratio > 1:
            intensity += min(0.2, (volume_ratio - 1) / 4)  # Max 0.2 for 5x volume
        
        # Sentiment intensity
        sentiment_strength = abs(metrics['combined_sentiment'])
        intensity += min(0.2, sentiment_strength / 0.5)  # Max 0.2 for strong sentiment
        
        # Volatility intensity
        volatility = metrics['volatility']
        intensity += min(0.15, volatility / 0.1)  # Max 0.15 for 10% volatility
        
        # RSI extremes
        rsi = metrics['current_rsi']
        if rsi > 70 or rsi < 30:
            intensity += min(0.15, abs(rsi - 50) / 50)  # Max 0.15 for extreme RSI
        
        return min(1.0, intensity)