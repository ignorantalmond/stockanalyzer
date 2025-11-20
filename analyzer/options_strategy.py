"""
Options strategy recommendation module
"""

from typing import Dict, List
from .enums import PumpDumpPhase, OptionsStrategy, RiskLevel
from .config import Config

class OptionsStrategyRecommender:
    """Recommends options strategies based on pump/dump phase and market conditions"""
    
    def __init__(self):
        """Initialize options strategy recommender"""
        self.config = Config()
    
    def suggest_strategies(self, phase: PumpDumpPhase, reddit_sentiment: float, 
                          news_sentiment: Dict, options_data: Dict, stock_info: Dict,
                          phase_metrics: Dict) -> Dict:
        """
        Suggest options trading strategies based on current conditions
        
        Args:
            phase: Current pump/dump phase
            reddit_sentiment: Reddit sentiment score
            news_sentiment: News sentiment analysis
            options_data: Options market data
            stock_info: Stock information
            phase_metrics: Phase detection metrics
            
        Returns:
            Dictionary with strategy recommendations
        """
        combined_sentiment = (reddit_sentiment + news_sentiment.get('avg_sentiment', 0)) / 2
        current_price = options_data.get('current_price', 0) if options_data else stock_info.get('currentPrice', 0)
        iv = self._get_iv_level(options_data)
        
        # Get phase-specific strategies
        strategies = self._get_phase_strategies(phase, combined_sentiment, iv, phase_metrics)
        
        # Add market condition adjustments
        adjusted_strategies = self._adjust_for_market_conditions(strategies, options_data, phase_metrics)
        
        # Calculate overall risk assessment
        overall_risk = self._calculate_overall_risk(phase, iv, phase_metrics)
        
        return {
            'recommendations': adjusted_strategies,
            'overall_risk_level': overall_risk,
            'combined_sentiment': combined_sentiment,
            'iv_assessment': self._assess_iv_environment(iv),
            'market_conditions': self._assess_market_conditions(options_data, phase_metrics),
            'timing_considerations': self._get_timing_considerations(phase, phase_metrics)
        }
    
    def _get_iv_level(self, options_data: Dict) -> float:
        """Extract implied volatility level"""
        if not options_data:
            return 0.5  # Default moderate IV
        
        call_iv = options_data.get('atm_call_iv', 0)
        put_iv = options_data.get('atm_put_iv', 0)
        return max(call_iv, put_iv)
    
    def _get_phase_strategies(self, phase: PumpDumpPhase, sentiment: float, iv: float, 
                            metrics: Dict) -> List[Dict]:
        """Get strategies specific to the current phase"""
        
        strategies = []
        
        if phase == PumpDumpPhase.ACCUMULATION:
            strategies.extend(self._accumulation_strategies(sentiment, iv, metrics))
        elif phase == PumpDumpPhase.EARLY_PUMP:
            strategies.extend(self._early_pump_strategies(sentiment, iv, metrics))
        elif phase == PumpDumpPhase.MAIN_PUMP:
            strategies.extend(self._main_pump_strategies(sentiment, iv, metrics))
        elif phase == PumpDumpPhase.PEAK_FRENZY:
            strategies.extend(self._peak_frenzy_strategies(sentiment, iv, metrics))
        elif phase == PumpDumpPhase.EARLY_DUMP:
            strategies.extend(self._early_dump_strategies(sentiment, iv, metrics))
        elif phase == PumpDumpPhase.MAIN_DUMP:
            strategies.extend(self._main_dump_strategies(sentiment, iv, metrics))
        elif phase == PumpDumpPhase.BAGHOLDERS:
            strategies.extend(self._bagholders_strategies(sentiment, iv, metrics))
        elif phase == PumpDumpPhase.RECOVERY_ATTEMPT:
            strategies.extend(self._recovery_strategies(sentiment, iv, metrics))
        elif phase == PumpDumpPhase.DEAD:
            strategies.extend(self._dead_stock_strategies())
        
        return strategies
    
    def _accumulation_strategies(self, sentiment: float, iv: float, metrics: Dict) -> List[Dict]:
        """Strategies for accumulation phase"""
        strategies = []
        
        if sentiment > 0.2:  # Bullish sentiment
            strategies.append({
                'strategy': OptionsStrategy.BUY_CALLS,
                'reasoning': 'Accumulation phase with bullish sentiment - potential for upward breakout',
                'risk': RiskLevel.MEDIUM,
                'timeframe': '30-60 days',
                'strike_guidance': 'Slightly OTM calls (5-15% above current price)',
                'position_size': 'Small to medium',
                'profit_target': '50-100%',
                'stop_loss': '50% of premium paid'
            })
        
        if iv < 0.4:  # Low implied volatility
            strategies.append({
                'strategy': OptionsStrategy.STRADDLE,
                'reasoning': 'Low volatility in accumulation - expect big move in either direction',
                'risk': RiskLevel.MEDIUM,
                'timeframe': '30-45 days',
                'strike_guidance': 'ATM long straddle',
                'position_size': 'Medium',
                'profit_target': '25-50%',
                'stop_loss': '40% of premium paid'
            })
        
        if sentiment < -0.1:  # Bearish or neutral
            strategies.append({
                'strategy': OptionsStrategy.SELL_PUTS,
                'reasoning': 'Accumulation phase - sell puts if willing to own stock at lower prices',
                'risk': RiskLevel.MEDIUM,
                'timeframe': '30-45 days',
                'strike_guidance': 'OTM cash-secured puts (10-20% below current price)',
                'position_size': 'Small',
                'profit_target': '50-80% of premium collected',
                'stop_loss': 'Roll or close at 200% of premium collected'
            })
        
        return strategies
    
    def _early_pump_strategies(self, sentiment: float, iv: float, metrics: Dict) -> List[Dict]:
        """Strategies for early pump phase"""
        strategies = []
        
        if sentiment > 0.3:  # Strong bullish sentiment
            strategies.append({
                'strategy': OptionsStrategy.BUY_CALLS,
                'reasoning': 'Early pump with strong sentiment - ride the momentum',
                'risk': RiskLevel.HIGH,
                'timeframe': '14-30 days',
                'strike_guidance': 'ATM or slightly ITM calls for delta, OTM for leverage',
                'position_size': 'Medium',
                'profit_target': '100-300%',
                'stop_loss': '30-40% of premium paid'
            })
        
        if iv < 0.6:  # Moderate IV
            strategies.append({
                'strategy': OptionsStrategy.CALL_SPREADS,
                'reasoning': 'Early pump with reasonable IV - limit risk with spreads',
                'risk': RiskLevel.MEDIUM_HIGH,
                'timeframe': '14-30 days',
                'strike_guidance': 'Bull call spread (buy ATM, sell OTM)',
                'position_size': 'Medium to large',
                'profit_target': '50-75% of max profit',
                'stop_loss': '50% of net premium paid'
            })
        
        # Warning strategy
        strategies.append({
            'strategy': OptionsStrategy.AVOID_OPTIONS,
            'reasoning': 'Early pump can accelerate rapidly - consider taking profits if already positioned',
            'risk': RiskLevel.HIGH,
            'timeframe': 'N/A',
            'strike_guidance': 'Monitor closely for exit signals',
            'position_size': 'Reduce existing positions',
            'profit_target': 'Take profits on 50-100% gains',
            'stop_loss': 'Tighten stops to 20-30%'
        })
        
        return strategies
    
    def _main_pump_strategies(self, sentiment: float, iv: float, metrics: Dict) -> List[Dict]:
        """Strategies for main pump phase"""
        strategies = []
        
        # Primary recommendation - sell premium
        strategies.append({
            'strategy': OptionsStrategy.SELL_CALLS,
            'reasoning': 'Main pump phase - high probability of reversal, sell expensive premium',
            'risk': RiskLevel.HIGH,
            'timeframe': '7-21 days',
            'strike_guidance': 'OTM covered calls (if you own stock) or naked calls (experienced traders only)',
            'position_size': 'Small to medium',
            'profit_target': '50-80% of premium collected',
            'stop_loss': 'Close at 200% of premium collected'
        })
        
        if iv > 0.8:  # Very high IV
            strategies.append({
                'strategy': OptionsStrategy.IRON_CONDOR,
                'reasoning': 'Extremely high IV - sell premium with defined risk',
                'risk': RiskLevel.MEDIUM,
                'timeframe': '7-21 days',
                'strike_guidance': 'Wide iron condor around current price',
                'position_size': 'Medium',
                'profit_target': '25-50% of max profit',
                'stop_loss': 'Close at 200% of net credit received'
            })
        
        # Contrarian play
        strategies.append({
            'strategy': OptionsStrategy.BUY_PUTS,
            'reasoning': 'Contrarian play - pump may be exhausting, prepare for reversal',
            'risk': RiskLevel.HIGH,
            'timeframe': '14-30 days',
            'strike_guidance': 'ATM or slightly OTM puts',
            'position_size': 'Small',
            'profit_target': '100-500%',
            'stop_loss': '50% of premium paid'
        })
        
        return strategies
    
    def _peak_frenzy_strategies(self, sentiment: float, iv: float, metrics: Dict) -> List[Dict]:
        """Strategies for peak frenzy phase"""
        strategies = []
        
        # Primary recommendation - avoid or short
        strategies.append({
            'strategy': OptionsStrategy.BUY_PUTS,
            'reasoning': 'Peak frenzy - imminent dump likely, profit from decline',
            'risk': RiskLevel.HIGH,
            'timeframe': '7-21 days',
            'strike_guidance': 'ATM puts for high delta, OTM puts for leverage',
            'position_size': 'Medium',
            'profit_target': '200-1000%',
            'stop_loss': '40% of premium paid'
        })
        
        # Safety recommendation
        strategies.append({
            'strategy': OptionsStrategy.AVOID_OPTIONS,
            'reasoning': 'Extreme volatility and unpredictability - too dangerous for most traders',
            'risk': RiskLevel.EXTREME,
            'timeframe': 'N/A',
            'strike_guidance': 'Stay away completely until volatility subsides',
            'position_size': 'None',
            'profit_target': 'Capital preservation',
            'stop_loss': 'N/A'
        })
        
        return strategies
    
    def _early_dump_strategies(self, sentiment: float, iv: float, metrics: Dict) -> List[Dict]:
        """Strategies for early dump phase"""
        strategies = []
        
        strategies.append({
            'strategy': OptionsStrategy.BUY_PUTS,
            'reasoning': 'Early dump phase - more downside likely',
            'risk': RiskLevel.MEDIUM_HIGH,
            'timeframe': '14-30 days',
            'strike_guidance': 'ATM puts for momentum continuation',
            'position_size': 'Medium',
            'profit_target': '100-300%',
            'stop_loss': '40% of premium paid'
        })
        
        if sentiment < -0.3:  # Very bearish sentiment
            strategies.append({
                'strategy': OptionsStrategy.PUT_SPREADS,
                'reasoning': 'Strong negative sentiment - bear put spreads for defined risk',
                'risk': RiskLevel.MEDIUM,
                'timeframe': '14-30 days',
                'strike_guidance': 'Bear put spread (buy ATM put, sell OTM put)',
                'position_size': 'Medium to large',
                'profit_target': '50-75% of max profit',
                'stop_loss': '50% of net premium paid'
            })
        
        return strategies
    
    def _main_dump_strategies(self, sentiment: float, iv: float, metrics: Dict) -> List[Dict]:
        """Strategies for main dump phase"""
        strategies = []
        
        # Contrarian income strategy
        strategies.append({
            'strategy': OptionsStrategy.SELL_PUTS,
            'reasoning': 'Main dump - sell puts if willing to own stock at much lower prices',
            'risk': RiskLevel.HIGH,
            'timeframe': '30-45 days',
            'strike_guidance': 'Deep OTM cash-secured puts (30-50% below current price)',
            'position_size': 'Small',
            'profit_target': '50-80% of premium collected',
            'stop_loss': 'Roll down and out if tested'
        })
        
        # Safety first
        strategies.append({
            'strategy': OptionsStrategy.AVOID_OPTIONS,
            'reasoning': 'Falling knife - wait for stabilization before entering new positions',
            'risk': RiskLevel.EXTREME,
            'timeframe': 'N/A',
            'strike_guidance': 'Avoid until trend reverses or stabilizes',
            'position_size': 'None',
            'profit_target': 'Capital preservation',
            'stop_loss': 'N/A'
        })
        
        return strategies
    
    def _bagholders_strategies(self, sentiment: float, iv: float, metrics: Dict) -> List[Dict]:
        """Strategies for bagholders phase"""
        strategies = []
        
        # Income generation
        strategies.append({
            'strategy': OptionsStrategy.SELL_CALLS,
            'reasoning': 'If you own shares, sell covered calls to generate income while waiting',
            'risk': RiskLevel.MEDIUM,
            'timeframe': '30-60 days',
            'strike_guidance': 'OTM covered calls (20-30% above current price)',
            'position_size': 'Based on stock holdings',
            'profit_target': '50-80% of premium collected',
            'stop_loss': 'Roll up and out if stock recovers'
        })
        
        # Low probability plays
        strategies.append({
            'strategy': OptionsStrategy.AVOID_OPTIONS,
            'reasoning': 'Low probability of significant movement - better opportunities elsewhere',
            'risk': RiskLevel.LOW,
            'timeframe': 'N/A',
            'strike_guidance': 'Wait for better risk/reward setups',
            'position_size': 'Minimal',
            'profit_target': 'Focus on other opportunities',
            'stop_loss': 'N/A'
        })
        
        return strategies
    
    def _recovery_strategies(self, sentiment: float, iv: float, metrics: Dict) -> List[Dict]:
        """Strategies for recovery attempt phase"""
        strategies = []
        
        if sentiment > 0.1:  # Positive sentiment
            strategies.append({
                'strategy': OptionsStrategy.STRANGLE,
                'reasoning': 'Recovery attempt with positive sentiment - could break either way',
                'risk': RiskLevel.MEDIUM,
                'timeframe': '30-45 days',
                'strike_guidance': 'OTM long strangle (calls and puts)',
                'position_size': 'Medium',
                'profit_target': '50-100%',
                'stop_loss': '40% of premium paid'
            })
        
        strategies.append({
            'strategy': OptionsStrategy.CALL_SPREADS,
            'reasoning': 'Recovery attempt - limited upside potential with defined risk',
            'risk': RiskLevel.MEDIUM,
            'timeframe': '30-60 days',
            'strike_guidance': 'Bull call spread with reasonable strikes',
            'position_size': 'Medium',
            'profit_target': '50-75% of max profit',
            'stop_loss': '50% of net premium paid'
        })
        
        return strategies
    
    def _dead_stock_strategies(self) -> List[Dict]:
        """Strategies for dead/delisted risk stocks"""
        return [{
            'strategy': OptionsStrategy.AVOID_OPTIONS,
            'reasoning': 'Dead/delisted risk - no options activity recommended',
            'risk': RiskLevel.EXTREME,
            'timeframe': 'N/A',
            'strike_guidance': 'Completely avoid - focus capital elsewhere',
            'position_size': 'None',
            'profit_target': 'Capital preservation',
            'stop_loss': 'N/A'
        }]
    
    def _adjust_for_market_conditions(self, strategies: List[Dict], options_data: Dict, 
                                    metrics: Dict) -> List[Dict]:
        """Adjust strategies based on current market conditions"""
        
        adjusted_strategies = []
        
        for strategy in strategies:
            adjusted_strategy = strategy.copy()
            
            # Adjust for IV environment
            iv = self._get_iv_level(options_data)
            if iv > 0.8:  # Very high IV
                if strategy['strategy'] in [OptionsStrategy.BUY_CALLS, OptionsStrategy.BUY_PUTS]:
                    adjusted_strategy['reasoning'] += ' (Note: Very high IV - premium expensive)'
                    adjusted_strategy['risk'] = RiskLevel.HIGH
            elif iv < 0.3:  # Low IV
                if strategy['strategy'] in [OptionsStrategy.SELL_CALLS, OptionsStrategy.SELL_PUTS]:
                    adjusted_strategy['reasoning'] += ' (Note: Low IV - limited premium collection)'
            
            # Adjust for volume
            if options_data and options_data.get('total_call_volume', 0) + options_data.get('total_put_volume', 0) < 100:
                adjusted_strategy['reasoning'] += ' (Warning: Low options volume - wide spreads likely)'
                if adjusted_strategy['risk'] != RiskLevel.EXTREME:
                    adjusted_strategy['risk'] = RiskLevel.HIGH
            
            # Adjust position size for volatility
            volatility = metrics.get('volatility', 0.02)
            if volatility > 0.1:  # Very high volatility
                if adjusted_strategy['position_size'] == 'Large':
                    adjusted_strategy['position_size'] = 'Medium'
                elif adjusted_strategy['position_size'] == 'Medium':
                    adjusted_strategy['position_size'] = 'Small'
                adjusted_strategy['reasoning'] += ' (Reduced size due to high volatility)'
            
            adjusted_strategies.append(adjusted_strategy)
        
        return adjusted_strategies
    
    def _calculate_overall_risk(self, phase: PumpDumpPhase, iv: float, metrics: Dict) -> RiskLevel:
        """Calculate overall risk level"""
        
        # Base risk from phase
        phase_risk_scores = {
            PumpDumpPhase.ACCUMULATION: 2,
            PumpDumpPhase.EARLY_PUMP: 4,
            PumpDumpPhase.MAIN_PUMP: 5,
            PumpDumpPhase.PEAK_FRENZY: 6,
            PumpDumpPhase.EARLY_DUMP: 4,
            PumpDumpPhase.MAIN_DUMP: 6,
            PumpDumpPhase.BAGHOLDERS: 3,
            PumpDumpPhase.RECOVERY_ATTEMPT: 3,
            PumpDumpPhase.DEAD: 6
        }
        
        risk_score = phase_risk_scores.get(phase, 3)
        
        # Adjust for IV
        if iv > 1.0:
            risk_score += 1
        elif iv > 0.8:
            risk_score += 0.5
        
        # Adjust for volatility
        volatility = metrics.get('volatility', 0.02)
        if volatility > 0.15:
            risk_score += 1
        elif volatility > 0.1:
            risk_score += 0.5
        
        # Convert to risk level
        if risk_score >= 6:
            return RiskLevel.EXTREME
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 3.5:
            return RiskLevel.MEDIUM_HIGH
        elif risk_score >= 2.5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _assess_iv_environment(self, iv: float) -> Dict:
        """Assess the implied volatility environment"""
        
        if iv > 1.0:
            level = "EXTREMELY_HIGH"
            description = "IV crush risk very high - favor selling strategies"
            recommendation = "SELL_PREMIUM"
        elif iv > 0.8:
            level = "HIGH"
            description = "High IV - good for selling, expensive for buying"
            recommendation = "SELL_PREMIUM"
        elif iv > 0.6:
            level = "ELEVATED"
            description = "Elevated IV - mixed environment"
            recommendation = "NEUTRAL"
        elif iv > 0.4:
            level = "MODERATE"
            description = "Moderate IV - balanced environment"
            recommendation = "NEUTRAL"
        elif iv > 0.2:
            level = "LOW"
            description = "Low IV - good for buying, poor for selling"
            recommendation = "BUY_PREMIUM"
        else:
            level = "VERY_LOW"
            description = "Very low IV - excellent for buying volatility"
            recommendation = "BUY_PREMIUM"
        
        return {
            'level': level,
            'iv_value': iv,
            'description': description,
            'recommendation': recommendation
        }
    
    def _assess_market_conditions(self, options_data: Dict, metrics: Dict) -> Dict:
        """Assess overall market conditions for options trading"""
        
        conditions = {}
        
        # Liquidity assessment
        if options_data:
            total_volume = options_data.get('total_call_volume', 0) + options_data.get('total_put_volume', 0)
            if total_volume > 1000:
                conditions['liquidity'] = 'HIGH'
            elif total_volume > 100:
                conditions['liquidity'] = 'MODERATE'
            else:
                conditions['liquidity'] = 'LOW'
            
            # Call/Put flow
            call_put_ratio = options_data.get('call_put_volume_ratio', 1.0)
            if call_put_ratio > 2.0:
                conditions['flow_bias'] = 'VERY_BULLISH'
            elif call_put_ratio > 1.5:
                conditions['flow_bias'] = 'BULLISH'
            elif call_put_ratio < 0.5:
                conditions['flow_bias'] = 'VERY_BEARISH'
            elif call_put_ratio < 0.75:
                conditions['flow_bias'] = 'BEARISH'
            else:
                conditions['flow_bias'] = 'NEUTRAL'
        else:
            conditions['liquidity'] = 'UNKNOWN'
            conditions['flow_bias'] = 'UNKNOWN'
        
        # Volatility environment
        volatility = metrics.get('volatility', 0.02)
        if volatility > 0.15:
            conditions['volatility_regime'] = 'VERY_HIGH'
        elif volatility > 0.1:
            conditions['volatility_regime'] = 'HIGH'
        elif volatility > 0.05:
            conditions['volatility_regime'] = 'MODERATE'
        else:
            conditions['volatility_regime'] = 'LOW'
        
        return conditions
    
    def _get_timing_considerations(self, phase: PumpDumpPhase, metrics: Dict) -> Dict:
        """Get timing considerations for the current phase"""
        
        considerations = {}
        
        # Time decay considerations
        if phase in [PumpDumpPhase.PEAK_FRENZY, PumpDumpPhase.MAIN_PUMP]:
            considerations['time_decay'] = 'CRITICAL - Use short-dated options for directional plays'
        elif phase in [PumpDumpPhase.ACCUMULATION, PumpDumpPhase.RECOVERY_ATTEMPT]:
            considerations['time_decay'] = 'MODERATE - Use longer-dated options for patience'
        else:
            considerations['time_decay'] = 'IMPORTANT - Balance time and premium cost'
        
        # Entry timing
        volatility = metrics.get('volatility', 0.02)
        if volatility > 0.1:
            considerations['entry_timing'] = 'WAIT - Let volatility settle before entering'
        elif phase == PumpDumpPhase.ACCUMULATION:
            considerations['entry_timing'] = 'GOOD - Relatively stable entry environment'
        else:
            considerations['entry_timing'] = 'CAUTIOUS - Monitor closely for optimal entry'
        
        # Exit planning
        if phase in [PumpDumpPhase.MAIN_PUMP, PumpDumpPhase.PEAK_FRENZY]:
            considerations['exit_planning'] = 'AGGRESSIVE - Take profits quickly, use tight stops'
        elif phase in [PumpDumpPhase.MAIN_DUMP, PumpDumpPhase.EARLY_DUMP]:
            considerations['exit_planning'] = 'PATIENT - Let bearish moves develop, trail stops'
        else:
            considerations['exit_planning'] = 'STANDARD - Use 50% profit target, 50% loss stop'
        
        return considerations
    
    def get_strategy_summary(self, recommendations: Dict) -> str:
        """Get a human-readable summary of strategy recommendations"""
        
        strategies = recommendations.get('recommendations', [])
        risk_level = recommendations.get('overall_risk_level', RiskLevel.MEDIUM)
        iv_assessment = recommendations.get('iv_assessment', {})
        
        if not strategies:
            return "No strategies recommended - avoid trading this stock"
        
        primary_strategy = strategies[0]
        strategy_name = primary_strategy['strategy'].value
        risk_desc = risk_level.value
        
        summary = f"Primary Strategy: {strategy_name} ({risk_desc} Risk)"
        
        if len(strategies) > 1:
            summary += f" | Alternative: {strategies[1]['strategy'].value}"
        
        # Add IV context
        iv_level = iv_assessment.get('level', 'UNKNOWN')
        if iv_level in ['HIGH', 'EXTREMELY_HIGH']:
            summary += " | High IV environment favors premium selling"
        elif iv_level in ['LOW', 'VERY_LOW']:
            summary += " | Low IV environment favors premium buying"
        
        return summary