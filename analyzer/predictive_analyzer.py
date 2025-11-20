"""
Predictive analysis module for stock price movements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .enums import PumpDumpPhase
from .config import Config

class PredictiveAnalyzer:
    """Handles predictive analysis and visualization of stock movements"""
    
    def __init__(self):
        """Initialize predictive analyzer"""
        self.config = Config()
    
    def generate_price_predictions(self, stock_df: pd.DataFrame, phase: PumpDumpPhase, 
                                 phase_metrics: Dict, options_data: Dict, 
                                 reddit_metrics: Dict, news_sentiment: Dict) -> Dict:
        """
        Generate price predictions for 7, 14, and 21 day periods
        
        Args:
            stock_df: DataFrame with historical stock data
            phase: Current pump/dump phase
            phase_metrics: Phase detection metrics
            options_data: Options market data
            reddit_metrics: Reddit sentiment metrics
            news_sentiment: News sentiment analysis
            
        Returns:
            Dictionary with prediction scenarios and probabilities
        """
        if stock_df.empty or len(stock_df) < 30:
            return self._empty_prediction()
        
        current_price = stock_df['Close'].iloc[-1]
        
        # Calculate base volatility and trend
        base_metrics = self._calculate_base_metrics(stock_df)
        
        # Generate phase-specific scenarios
        scenarios = self._generate_phase_scenarios(phase, phase_metrics, base_metrics)
        
        # Adjust scenarios based on sentiment and options data
        adjusted_scenarios = self._adjust_scenarios_for_sentiment(
            scenarios, reddit_metrics, news_sentiment, options_data
        )
        
        # Generate price paths for each timeframe
        predictions = {}
        for days in [7, 14, 21]:
            predictions[f'{days}_day'] = self._generate_price_paths(
                current_price, adjusted_scenarios, days, base_metrics
            )
        
        # Calculate confidence levels
        confidence_metrics = self._calculate_prediction_confidence(
            stock_df, phase, phase_metrics, reddit_metrics
        )
        
        return {
            'predictions': predictions,
            'scenarios': adjusted_scenarios,
            'base_metrics': base_metrics,
            'confidence': confidence_metrics,
            'current_price': current_price,
            'generated_at': datetime.now()
        }
    
    def _empty_prediction(self) -> Dict:
        """Return empty prediction structure"""
        return {
            'predictions': {},
            'scenarios': [],
            'base_metrics': {},
            'confidence': {'overall': 0.0, 'factors': []},
            'current_price': 0,
            'generated_at': datetime.now()
        }
    
    def _calculate_base_metrics(self, stock_df: pd.DataFrame) -> Dict:
        """Calculate base statistical metrics from historical data"""
        recent_20d = stock_df.tail(20)
        recent_60d = stock_df.tail(60) if len(stock_df) >= 60 else stock_df
        
        # Price statistics
        returns = stock_df['daily_return'].dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Trend analysis
        prices = recent_20d['Close'].values
        days = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(days, prices)
        trend_strength = abs(r_value)
        
        # Support/resistance levels
        support_level = recent_60d['Low'].min()
        resistance_level = recent_60d['High'].max()
        current_price = stock_df['Close'].iloc[-1]
        
        # Volume patterns
        avg_volume = stock_df['Volume'].tail(20).mean()
        volume_volatility = stock_df['Volume'].tail(20).std() / avg_volume
        
        # RSI and momentum
        rsi = stock_df['rsi'].iloc[-1] if 'rsi' in stock_df.columns else 50
        momentum_5d = (current_price - stock_df['Close'].iloc[-6]) / stock_df['Close'].iloc[-6] if len(stock_df) >= 6 else 0
        
        return {
            'volatility': volatility,
            'daily_volatility': volatility / np.sqrt(252),
            'trend_slope': slope,
            'trend_strength': trend_strength,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'support_distance': (current_price - support_level) / current_price,
            'resistance_distance': (resistance_level - current_price) / current_price,
            'volume_volatility': volume_volatility,
            'rsi': rsi,
            'momentum_5d': momentum_5d,
            'mean_reversion_factor': self._calculate_mean_reversion(stock_df)
        }
    
    def _calculate_mean_reversion(self, stock_df: pd.DataFrame) -> float:
        """Calculate mean reversion tendency"""
        if len(stock_df) < 30:
            return 0.5
        
        # Calculate how often price returns to moving average
        sma_20 = stock_df['sma_20'].dropna()
        prices = stock_df['Close'][-len(sma_20):]
        
        deviations = (prices - sma_20) / sma_20
        
        # Count reversions (price moves back toward average)
        reversions = 0
        for i in range(1, len(deviations)):
            if abs(deviations.iloc[i]) < abs(deviations.iloc[i-1]):
                reversions += 1
        
        return reversions / max(len(deviations) - 1, 1)
    
    def _generate_phase_scenarios(self, phase: PumpDumpPhase, 
                                phase_metrics: Dict, base_metrics: Dict) -> List[Dict]:
        """Generate scenarios based on current pump/dump phase"""
        
        scenarios = []
        
        if phase == PumpDumpPhase.ACCUMULATION:
            scenarios = [
                {
                    'name': 'Continued Accumulation',
                    'probability': 0.5,
                    'direction': 'sideways',
                    'magnitude': 0.05,  # 5% max move
                    'volatility_multiplier': 0.8,
                    'description': 'Price continues to consolidate in range'
                },
                {
                    'name': 'Breakout to Upside',
                    'probability': 0.3,
                    'direction': 'up',
                    'magnitude': 0.25,
                    'volatility_multiplier': 1.5,
                    'description': 'Momentum builds, price breaks higher'
                },
                {
                    'name': 'Breakdown',
                    'probability': 0.2,
                    'direction': 'down',
                    'magnitude': 0.15,
                    'volatility_multiplier': 1.2,
                    'description': 'Support fails, price declines'
                }
            ]
        
        elif phase == PumpDumpPhase.EARLY_PUMP:
            scenarios = [
                {
                    'name': 'Pump Continues',
                    'probability': 0.4,
                    'direction': 'up',
                    'magnitude': 0.4,
                    'volatility_multiplier': 2.0,
                    'description': 'Momentum accelerates higher'
                },
                {
                    'name': 'Pullback/Consolidation',
                    'probability': 0.35,
                    'direction': 'down',
                    'magnitude': 0.2,
                    'volatility_multiplier': 1.5,
                    'description': 'Profit-taking causes temporary decline'
                },
                {
                    'name': 'Failed Pump',
                    'probability': 0.25,
                    'direction': 'down',
                    'magnitude': 0.35,
                    'volatility_multiplier': 1.8,
                    'description': 'Pump loses steam, significant reversal'
                }
            ]
        
        elif phase == PumpDumpPhase.MAIN_PUMP:
            scenarios = [
                {
                    'name': 'Peak and Dump',
                    'probability': 0.6,
                    'direction': 'down',
                    'magnitude': 0.5,
                    'volatility_multiplier': 2.5,
                    'description': 'Pump exhaustion leads to sharp decline'
                },
                {
                    'name': 'Blow-off Top',
                    'probability': 0.25,
                    'direction': 'up',
                    'magnitude': 0.3,
                    'volatility_multiplier': 3.0,
                    'description': 'Final parabolic move before collapse'
                },
                {
                    'name': 'Gradual Rollover',
                    'probability': 0.15,
                    'direction': 'down',
                    'magnitude': 0.25,
                    'volatility_multiplier': 1.5,
                    'description': 'Slow decline as interest wanes'
                }
            ]
        
        elif phase == PumpDumpPhase.PEAK_FRENZY:
            scenarios = [
                {
                    'name': 'Immediate Crash',
                    'probability': 0.7,
                    'direction': 'down',
                    'magnitude': 0.7,
                    'volatility_multiplier': 3.0,
                    'description': 'Panic selling, rapid price collapse'
                },
                {
                    'name': 'Dead Cat Bounce',
                    'probability': 0.2,
                    'direction': 'mixed',
                    'magnitude': 0.4,
                    'volatility_multiplier': 2.8,
                    'description': 'Brief recovery followed by continued decline'
                },
                {
                    'name': 'Extended Frenzy',
                    'probability': 0.1,
                    'direction': 'up',
                    'magnitude': 0.5,
                    'volatility_multiplier': 4.0,
                    'description': 'Irrational exuberance continues (very unlikely)'
                }
            ]
        
        elif phase in [PumpDumpPhase.EARLY_DUMP, PumpDumpPhase.MAIN_DUMP]:
            scenarios = [
                {
                    'name': 'Continued Decline',
                    'probability': 0.5,
                    'direction': 'down',
                    'magnitude': 0.3,
                    'volatility_multiplier': 1.8,
                    'description': 'Selling pressure continues'
                },
                {
                    'name': 'Bounce Attempt',
                    'probability': 0.3,
                    'direction': 'up',
                    'magnitude': 0.2,
                    'volatility_multiplier': 1.5,
                    'description': 'Oversold bounce from support'
                },
                {
                    'name': 'Capitulation',
                    'probability': 0.2,
                    'direction': 'down',
                    'magnitude': 0.5,
                    'volatility_multiplier': 2.2,
                    'description': 'Final panic selling wave'
                }
            ]
        
        elif phase == PumpDumpPhase.BAGHOLDERS:
            scenarios = [
                {
                    'name': 'Slow Grind Lower',
                    'probability': 0.4,
                    'direction': 'down',
                    'magnitude': 0.15,
                    'volatility_multiplier': 0.9,
                    'description': 'Gradual decline as hope fades'
                },
                {
                    'name': 'Range-bound',
                    'probability': 0.35,
                    'direction': 'sideways',
                    'magnitude': 0.1,
                    'volatility_multiplier': 0.7,
                    'description': 'Price stabilizes at low levels'
                },
                {
                    'name': 'Recovery Rally',
                    'probability': 0.25,
                    'direction': 'up',
                    'magnitude': 0.3,
                    'volatility_multiplier': 1.2,
                    'description': 'Surprise catalyst sparks recovery'
                }
            ]
        
        elif phase == PumpDumpPhase.RECOVERY_ATTEMPT:
            scenarios = [
                {
                    'name': 'Successful Recovery',
                    'probability': 0.3,
                    'direction': 'up',
                    'magnitude': 0.35,
                    'volatility_multiplier': 1.6,
                    'description': 'Recovery gains momentum'
                },
                {
                    'name': 'Failed Recovery',
                    'probability': 0.4,
                    'direction': 'down',
                    'magnitude': 0.25,
                    'volatility_multiplier': 1.4,
                    'description': 'Recovery fails, new lows made'
                },
                {
                    'name': 'Sideways Consolidation',
                    'probability': 0.3,
                    'direction': 'sideways',
                    'magnitude': 0.15,
                    'volatility_multiplier': 1.0,
                    'description': 'Price consolidates gains'
                }
            ]
        
        elif phase == PumpDumpPhase.DEAD:
            scenarios = [
                {
                    'name': 'Continued Decline',
                    'probability': 0.7,
                    'direction': 'down',
                    'magnitude': 0.4,
                    'volatility_multiplier': 1.5,
                    'description': 'Further deterioration toward zero'
                },
                {
                    'name': 'Zombie State',
                    'probability': 0.25,
                    'direction': 'sideways',
                    'magnitude': 0.2,
                    'volatility_multiplier': 0.8,
                    'description': 'Price remains near zero'
                },
                {
                    'name': 'Miracle Recovery',
                    'probability': 0.05,
                    'direction': 'up',
                    'magnitude': 1.0,
                    'volatility_multiplier': 3.0,
                    'description': 'Extremely unlikely turnaround'
                }
            ]
        
        return scenarios
    
    def _adjust_scenarios_for_sentiment(self, scenarios: List[Dict], reddit_metrics: Dict,
                                      news_sentiment: Dict, options_data: Dict) -> List[Dict]:
        """Adjust scenario probabilities based on sentiment and options data"""
        
        adjusted_scenarios = []
        
        # Calculate sentiment factors
        combined_sentiment = (reddit_metrics.get('avg_sentiment', 0) + 
                            news_sentiment.get('avg_sentiment', 0)) / 2
        
        # Options factors
        iv_level = 0.5
        call_put_ratio = 1.0
        if options_data:
            iv_level = max(options_data.get('atm_call_iv', 0), options_data.get('atm_put_iv', 0))
            call_put_ratio = options_data.get('call_put_volume_ratio', 1.0)
        
        # Reddit activity factor
        activity_multiplier = 1.0
        if reddit_metrics.get('activity_trend') == 'increasing':
            activity_multiplier = 1.2
        elif reddit_metrics.get('activity_trend') == 'decreasing':
            activity_multiplier = 0.8
        
        for scenario in scenarios:
            adjusted_scenario = scenario.copy()
            
            # Adjust probabilities based on sentiment
            if scenario['direction'] == 'up' and combined_sentiment > 0.2:
                adjusted_scenario['probability'] *= 1.3
            elif scenario['direction'] == 'up' and combined_sentiment < -0.2:
                adjusted_scenario['probability'] *= 0.7
            elif scenario['direction'] == 'down' and combined_sentiment < -0.2:
                adjusted_scenario['probability'] *= 1.3
            elif scenario['direction'] == 'down' and combined_sentiment > 0.2:
                adjusted_scenario['probability'] *= 0.7
            
            # Adjust for options activity
            if scenario['direction'] == 'up' and call_put_ratio > 2.0:
                adjusted_scenario['probability'] *= 1.2
            elif scenario['direction'] == 'down' and call_put_ratio < 0.5:
                adjusted_scenario['probability'] *= 1.2
            
            # Adjust volatility for high IV environment
            if iv_level > 0.8:
                adjusted_scenario['volatility_multiplier'] *= 1.3
            
            # Apply activity multiplier
            adjusted_scenario['probability'] *= activity_multiplier
            
            adjusted_scenarios.append(adjusted_scenario)
        
        # Normalize probabilities to sum to 1
        total_prob = sum(s['probability'] for s in adjusted_scenarios)
        if total_prob > 0:
            for scenario in adjusted_scenarios:
                scenario['probability'] /= total_prob
        
        return adjusted_scenarios
    
    def _generate_price_paths(self, current_price: float, scenarios: List[Dict], 
                            days: int, base_metrics: Dict) -> Dict:
        """Generate price paths for each scenario over specified days"""
        
        paths = {}
        
        for scenario in scenarios:
            # Generate daily price movements
            daily_returns = self._simulate_daily_returns(
                scenario, days, base_metrics
            )
            
            # Calculate price path
            price_path = [current_price]
            for daily_return in daily_returns:
                new_price = price_path[-1] * (1 + daily_return)
                price_path.append(max(0.01, new_price))  # Prevent negative prices
            
            paths[scenario['name']] = {
                'probability': scenario['probability'],
                'direction': scenario['direction'],
                'description': scenario['description'],
                'price_path': price_path,
                'final_price': price_path[-1],
                'total_return': (price_path[-1] - current_price) / current_price,
                'max_price': max(price_path),
                'min_price': min(price_path),
                'volatility': np.std(daily_returns) if len(daily_returns) > 1 else 0
            }
        
        # Calculate probability-weighted expected path
        expected_path = self._calculate_expected_path(paths, days)
        
        return {
            'scenarios': paths,
            'expected_path': expected_path,
            'summary': self._summarize_predictions(paths)
        }
    
    def _simulate_daily_returns(self, scenario: Dict, days: int, base_metrics: Dict) -> List[float]:
        """Simulate daily returns for a scenario"""
        
        # Base daily volatility
        daily_vol = base_metrics['daily_volatility'] * scenario['volatility_multiplier']
        
        # Target return based on scenario
        if scenario['direction'] == 'up':
            target_return = scenario['magnitude']
        elif scenario['direction'] == 'down':
            target_return = -scenario['magnitude']
        else:  # sideways
            target_return = np.random.uniform(-scenario['magnitude'], scenario['magnitude'])
        
        # Generate daily returns with drift toward target
        daily_drift = target_return / days
        
        returns = []
        for day in range(days):
            # Add mean reversion component
            mean_reversion = base_metrics['mean_reversion_factor'] * 0.01
            
            # Random component
            random_return = np.random.normal(0, daily_vol)
            
            # Combine components
            daily_return = daily_drift + random_return - (mean_reversion if day > days/2 else 0)
            
            returns.append(daily_return)
        
        return returns
    
    def _calculate_expected_path(self, paths: Dict, days: int) -> Dict:
        """Calculate probability-weighted expected price path"""
        
        expected_prices = []
        
        # Calculate expected price for each day
        for day in range(days + 1):
            expected_price = 0
            total_weight = 0
            
            for scenario_name, scenario_data in paths.items():
                if day < len(scenario_data['price_path']):
                    weight = scenario_data['probability']
                    price = scenario_data['price_path'][day]
                    expected_price += weight * price
                    total_weight += weight
            
            if total_weight > 0:
                expected_prices.append(expected_price / total_weight)
            else:
                expected_prices.append(expected_prices[-1] if expected_prices else 0)
        
        current_price = expected_prices[0] if expected_prices else 0
        final_price = expected_prices[-1] if expected_prices else 0
        
        return {
            'price_path': expected_prices,
            'expected_return': (final_price - current_price) / current_price if current_price > 0 else 0,
            'max_expected': max(expected_prices) if expected_prices else 0,
            'min_expected': min(expected_prices) if expected_prices else 0
        }
    
    def _summarize_predictions(self, paths: Dict) -> Dict:
        """Summarize prediction results"""
        
        bullish_prob = sum(data['probability'] for data in paths.values() 
                          if data['total_return'] > 0.1)
        bearish_prob = sum(data['probability'] for data in paths.values() 
                          if data['total_return'] < -0.1)
        neutral_prob = 1 - bullish_prob - bearish_prob
        
        # Expected return
        expected_return = sum(data['probability'] * data['total_return'] 
                            for data in paths.values())
        
        # Most likely scenario
        most_likely = max(paths.items(), key=lambda x: x[1]['probability'])
        
        return {
            'bullish_probability': bullish_prob,
            'bearish_probability': bearish_prob,
            'neutral_probability': neutral_prob,
            'expected_return': expected_return,
            'most_likely_scenario': most_likely[0],
            'most_likely_probability': most_likely[1]['probability'],
            'most_likely_return': most_likely[1]['total_return']
        }
    
    def _calculate_prediction_confidence(self, stock_df: pd.DataFrame, phase: PumpDumpPhase,
                                       phase_metrics: Dict, reddit_metrics: Dict) -> Dict:
        """Calculate confidence in predictions"""
        
        confidence_factors = []
        overall_confidence = 0.5  # Base confidence
        
        # Data quality factor
        data_points = len(stock_df)
        if data_points >= 60:
            confidence_factors.append(('Sufficient historical data', 0.1))
            overall_confidence += 0.1
        elif data_points >= 30:
            confidence_factors.append(('Adequate historical data', 0.05))
            overall_confidence += 0.05
        else:
            confidence_factors.append(('Limited historical data', -0.1))
            overall_confidence -= 0.1
        
        # Phase clarity factor
        phase_confidence = phase_metrics.get('volatility', 0.02)
        if phase_confidence > 0.1:
            confidence_factors.append(('High volatility reduces prediction accuracy', -0.15))
            overall_confidence -= 0.15
        elif phase_confidence < 0.03:
            confidence_factors.append(('Low volatility improves prediction accuracy', 0.1))
            overall_confidence += 0.1
        
        # Sentiment consistency
        reddit_sentiment = reddit_metrics.get('avg_sentiment', 0)
        if abs(reddit_sentiment) > 0.3:
            confidence_factors.append(('Strong sentiment provides directional bias', 0.1))
            overall_confidence += 0.1
        
        # Pattern recognition
        trend_strength = phase_metrics.get('trend_strength', 0)
        if trend_strength > 0.7:
            confidence_factors.append(('Clear trend pattern identified', 0.1))
            overall_confidence += 0.1
        
        # Phase-specific adjustments
        if phase in [PumpDumpPhase.PEAK_FRENZY, PumpDumpPhase.MAIN_DUMP]:
            confidence_factors.append(('Extreme phases have predictable outcomes', 0.15))
            overall_confidence += 0.15
        elif phase == PumpDumpPhase.ACCUMULATION:
            confidence_factors.append(('Accumulation phase is inherently unpredictable', -0.1))
            overall_confidence -= 0.1
        
        # Cap confidence between 0.1 and 0.9
        overall_confidence = max(0.1, min(0.9, overall_confidence))
        
        return {
            'overall': overall_confidence,
            'factors': confidence_factors,
            'level': self._get_confidence_level(overall_confidence)
        }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to descriptive level"""
        if confidence >= 0.8:
            return "VERY_HIGH"
        elif confidence >= 0.65:
            return "HIGH"
        elif confidence >= 0.5:
            return "MEDIUM"
        elif confidence >= 0.35:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def create_prediction_chart(self, prediction_data: Dict, symbol: str, 
                              days: int, save_path: str = None) -> str:
        """
        Create a visualization of price predictions
        
        Args:
            prediction_data: Prediction data dictionary
            symbol: Stock symbol
            days: Number of days to display
            save_path: Optional path to save chart
            
        Returns:
            Path to saved chart or empty string if display only
        """
        if not prediction_data.get('predictions') or f'{days}_day' not in prediction_data['predictions']:
            return ""
        
        day_data = prediction_data['predictions'][f'{days}_day']
        current_price = prediction_data['current_price']
        
        # Set up the plot
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                     gridspec_kw={'height_ratios': [3, 1]})
        
        # Create date range
        start_date = datetime.now()
        dates = [start_date + timedelta(days=i) for i in range(days + 1)]
        
        # Plot main prediction chart
        ax1.set_facecolor('#1e1e1e')
        
        # Plot expected path
        expected_path = day_data['expected_path']['price_path']
        ax1.plot(dates, expected_path, color='#00ff00', linewidth=3, 
                label='Expected Path', alpha=0.9)
        
        # Plot scenario paths with transparency based on probability
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3']
        
        for i, (scenario_name, scenario_data) in enumerate(day_data['scenarios'].items()):
            color = colors[i % len(colors)]
            alpha = 0.3 + (scenario_data['probability'] * 0.7)  # Higher probability = more opaque
            
            ax1.plot(dates, scenario_data['price_path'], color=color, 
                    linewidth=2, alpha=alpha, linestyle='--',
                    label=f"{scenario_name} ({scenario_data['probability']:.1%})")
        
        # Add current price line
        ax1.axhline(y=current_price, color='white', linestyle='-', alpha=0.8, 
                   label=f'Current Price: ${current_price:.2f}')
        
        # Formatting
        ax1.set_title(f'{symbol} - {days} Day Price Prediction\nConfidence: {prediction_data["confidence"]["level"]} ({prediction_data["confidence"]["overall"]:.1%})', 
                     fontsize=16, color='white', pad=20)
        ax1.set_ylabel('Price ($)', fontsize=12, color='white')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=9, framealpha=0.8)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days//7)))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Add probability distribution chart
        ax2.set_facecolor('#1e1e1e')
        
        scenario_names = []
        probabilities = []
        returns = []
        colors_bar = []
        
        for i, (name, data) in enumerate(day_data['scenarios'].items()):
            scenario_names.append(name.split(' ')[0])  # Shortened names
            probabilities.append(data['probability'])
            returns.append(data['total_return'])
            
            # Color based on return
            if data['total_return'] > 0.1:
                colors_bar.append('#00ff00')
            elif data['total_return'] < -0.1:
                colors_bar.append('#ff4444')
            else:
                colors_bar.append('#ffaa00')
        
        bars = ax2.bar(scenario_names, probabilities, color=colors_bar, alpha=0.7)
        
        # Add return percentages on bars
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{ret:+.1%}', ha='center', va='bottom', fontsize=9, color='white')
        
        ax2.set_title('Scenario Probabilities & Expected Returns', fontsize=12, color='white')
        ax2.set_ylabel('Probability', fontsize=10, color='white')
        ax2.set_ylim(0, max(probabilities) * 1.2)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, fontsize=9)
        
        # Add summary text
        summary = day_data['summary']
        summary_text = (f"Expected Return: {summary['expected_return']:+.1%}\n"
                       f"Bullish Probability: {summary['bullish_probability']:.1%}\n"
                       f"Bearish Probability: {summary['bearish_probability']:.1%}\n"
                       f"Most Likely: {summary['most_likely_scenario']}")
        
        plt.figtext(0.02, 0.02, summary_text, fontsize=10, color='white', 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8))
        
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='#1e1e1e', edgecolor='none')
            plt.close()
            return save_path
        else:
            plt.show()
            return ""
    
    def get_prediction_summary(self, prediction_data: Dict) -> str:
        """
        Get a human-readable summary of predictions
        
        Args:
            prediction_data: Prediction data dictionary
            
        Returns:
            Summary string
        """
        if not prediction_data.get('predictions'):
            return "No predictions available"
        
        confidence = prediction_data['confidence']
        current_price = prediction_data['current_price']
        
        summary_lines = [
            f"🔮 PRICE PREDICTION SUMMARY (Confidence: {confidence['level']})",
            "=" * 60
        ]
        
        for days in [7, 14, 21]:
            if f'{days}_day' not in prediction_data['predictions']:
                continue
                
            day_data = prediction_data['predictions'][f'{days}_day']
            summary = day_data['summary']
            expected_path = day_data['expected_path']
            
            expected_price = expected_path['price_path'][-1]
            expected_return = expected_path['expected_return']
            
            # Determine outlook
            if summary['bullish_probability'] > 0.5:
                outlook = "🟢 BULLISH"
            elif summary['bearish_probability'] > 0.5:
                outlook = "🔴 BEARISH"
            else:
                outlook = "🟡 NEUTRAL"
            
            summary_lines.extend([
                f"\n📅 {days}-Day Outlook: {outlook}",
                f"Expected Price: ${expected_price:.2f} ({expected_return:+.1%})",
                f"Most Likely: {summary['most_likely_scenario']} ({summary['most_likely_probability']:.1%})",
                f"Bull/Bear Split: {summary['bullish_probability']:.1%} / {summary['bearish_probability']:.1%}"
            ])
        
        # Add confidence factors
        summary_lines.extend([
            f"\n🎯 Confidence Factors:",
        ])
        
        for factor, impact in confidence['factors']:
            impact_symbol = "✅" if impact > 0 else "⚠️" if impact < 0 else "ℹ️"
            summary_lines.append(f"  {impact_symbol} {factor}")
        
        return "\n".join(summary_lines)