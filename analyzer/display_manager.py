"""
Display and user interface management module
"""

from typing import List, Dict
import pandas as pd

from .enums import PHASE_EMOJIS, STRATEGY_EMOJIS, RISK_COLORS, PumpDumpPhase, OptionsStrategy, RiskLevel
from .config import Config

class DisplayManager:
    """Handles all display and user interface functions"""
    
    def __init__(self):
        """Initialize display manager"""
        self.config = Config()
    
    def display_summary_table(self, results: List[Dict]) -> None:
        """
        Display enhanced summary table with options strategies
        
        Args:
            results: List of stock analysis results
        """
        if not results:
            print("No stocks to display")
            return
        
        print(f"\n{'='*150}")
        print("🎯 MEME STOCK ANALYSIS WITH OPTIONS STRATEGIES")
        print(f"{'='*150}")
        print(f"{'#':<3} {'Symbol':<6} {'Company':<22} {'Price':<8} {'1D %':<7} {'Vol':<6} "
              f"{'Phase':<17} {'Sentiment':<12} {'Options Strategy':<18} {'Risk':<8} {'Confidence':<10}")
        print(f"{'-'*150}")
        
        for i, result in enumerate(results, 1):
            # Get display data
            emoji = PHASE_EMOJIS.get(result['phase'], "❓")
            company_short = self._truncate_text(result['company_name'], 20)
            
            # Get primary options strategy
            strategies = result['options_strategy']['recommendations']
            primary_strategy = self._truncate_text(strategies[0]['strategy'].value, 16) if strategies else "No Strategy"
            
            # Combined sentiment with emoji
            combined_sentiment = result['options_strategy']['combined_sentiment']
            sentiment_emoji = "🟢" if combined_sentiment > 0.1 else "🔴" if combined_sentiment < -0.1 else "🟡"
            sentiment_display = f"{sentiment_emoji} {combined_sentiment:+.2f}"
            
            # Risk level with emoji
            risk_level = result['options_strategy']['overall_risk_level']
            risk_emoji = RISK_COLORS.get(risk_level, "❓")
            risk_display = f"{risk_emoji} {risk_level.value}"
            
            # Confidence with visual indicator
            confidence = result['phase_confidence']
            conf_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            conf_display = f"{conf_emoji} {confidence:.0%}"
            
            print(f"{i:<3} {result['symbol']:<6} {company_short:<22} "
                  f"${result['current_price']:<7.2f} {result['price_change_1d']:<6.1%} "
                  f"{result['volume_ratio']:<5.1f}x {emoji} {result['phase'].value:<15} "
                  f"{sentiment_display:<11} {primary_strategy:<18} {risk_display:<7} {conf_display:<9}")
        
        print(f"{'-'*150}")
        print(f"Total stocks analyzed: {len(results)}")
        self._display_quick_stats(results)
    
    def _display_quick_stats(self, results: List[Dict]) -> None:
        """Display quick statistics about the analyzed stocks"""
        if not results:
            return
        
        # Count by risk level
        risk_counts = {}
        phase_counts = {}
        
        for result in results:
            risk_level = result['options_strategy']['overall_risk_level'].value
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            
            phase = result['phase'].value
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        print(f"\n📊 Quick Stats:")
        print(f"Risk Levels: ", end="")
        for risk, count in risk_counts.items():
            emoji = {"LOW": "🟢", "MEDIUM": "🟡", "MEDIUM-HIGH": "🟠", "HIGH": "🔴", "EXTREME": "☠️"}.get(risk, "❓")
            print(f"{emoji}{risk}({count}) ", end="")
        
        print(f"\nTop Phases: ", end="")
        sorted_phases = sorted(phase_counts.items(), key=lambda x: x[1], reverse=True)
        for phase, count in sorted_phases[:3]:
            print(f"{phase}({count}) ", end="")
        print()
    
    def interactive_selector(self, results: List[Dict]) -> str:
        """
        Interactive interface for detailed stock analysis
        
        Args:
            results: List of stock analysis results
            
        Returns:
            Action string ('exit', 'refresh', or continues loop)
        """
        while True:
            print(f"\n{'='*100}")
            print("📊 SELECT A STOCK FOR DETAILED OPTIONS ANALYSIS")
            print(f"{'='*100}")
            
            for i, result in enumerate(results, 1):
                strategies = result['options_strategy']['recommendations']
                primary_strategy = self._truncate_text(strategies[0]['strategy'].value, 15) if strategies else "No Strategy"
                risk_level = result['options_strategy']['overall_risk_level']
                
                risk_color = RISK_COLORS.get(risk_level, "❓")
                phase_emoji = PHASE_EMOJIS.get(result['phase'], "❓")
                
                print(f"{i:2d}. {result['symbol']:<6} | {phase_emoji} {result['phase'].value:<15} | "
                      f"{primary_strategy:<15} | {risk_color} {risk_level.value}")
            
            print(f"{len(results)+1:2d}. 🔍 Analyze specific ticker")
            print(f"{len(results)+2:2d}. 🔄 Refresh/Rescan stocks")
            print(f"{len(results)+3:2d}. 📁 Export results to file")
            print(f"{len(results)+4:2d}. 📊 Show analysis summary")
            print(f"{len(results)+5:2d}. ❌ Exit")
            
            try:
                choice = input(f"\nEnter your choice (1-{len(results)+5}): ").strip()
                
                if choice == str(len(results)+1):
                    return "manual_ticker"
                elif choice == str(len(results)+2):
                    return "refresh"
                elif choice == str(len(results)+3):
                    return "export"
                elif choice == str(len(results)+4):
                    self._display_analysis_summary(results)
                    continue
                elif choice == str(len(results)+5):
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
    
    def display_detailed_analysis(self, stock_analysis: Dict) -> None:
        """
        Display comprehensive detailed analysis for a selected stock
        
        Args:
            stock_analysis: Complete stock analysis dictionary
        """
        symbol = stock_analysis['symbol']
        phase = stock_analysis['phase']
        
        print(f"\n{'='*120}")
        print(f"📈 COMPREHENSIVE ANALYSIS: {symbol} - {stock_analysis['company_name']}")
        print(f"{'='*120}")
        
        # Current Status Section
        self._display_current_status(stock_analysis)
        
        # Sentiment Analysis Section
        self._display_sentiment_analysis(stock_analysis)
        
        # Options Market Data Section
        self._display_options_data(stock_analysis)
        
        # Strategy Recommendations Section
        self._display_strategy_recommendations(stock_analysis)
        
        # Phase-Specific Warnings Section
        self._display_phase_warnings(stock_analysis)
        
        # Technical Analysis Section
        self._display_technical_analysis(stock_analysis)
        
        # News Highlights Section
        self._display_news_highlights(stock_analysis)
        
        # Risk Assessment Section
        self._display_risk_assessment(stock_analysis)
        
        # Timeline Estimates Section
        self._display_timeline_estimates(stock_analysis)
        
        # Disclaimer
        self._display_disclaimer()
        
        input("\nPress Enter to continue...")
    
    def _display_current_status(self, analysis: Dict) -> None:
        """Display current stock status"""
        print(f"\n{'💰 CURRENT STATUS':<60}")
        print("-" * 60)
        
        print(f"Current Price: ${analysis['current_price']:.2f}")
        print(f"24h Change: {analysis['price_change_1d']:+.2%}")
        print(f"Volume Ratio: {analysis['volume_ratio']:.1f}x normal")
        
        phase_emoji = PHASE_EMOJIS.get(analysis['phase'], "❓")
        print(f"Current Phase: {phase_emoji} {analysis['phase'].value}")
        print(f"Description: {analysis['phase_description']}")
        
        confidence = analysis['phase_confidence']
        conf_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
        print(f"Confidence: {conf_emoji} {confidence:.0%}")
    
    """
Add this enhanced sentiment display to your display_manager.py
Replace the existing _display_sentiment_analysis method with this version
"""

    def _display_sentiment_analysis(self, analysis: Dict) -> None:
        """Display sentiment analysis results with detailed positive/negative mention counts"""
        print(f"\n{'📊 SENTIMENT ANALYSIS':<60}")
        print("-" * 60)
        
        reddit_sentiment = analysis['reddit_sentiment']
        news_sentiment = analysis['news_sentiment']
        combined_sentiment = analysis['combined_sentiment']
        reddit_metrics = analysis['reddit_metrics']
        
        print(f"Reddit Sentiment: {reddit_sentiment:+.3f} {self._get_sentiment_desc(reddit_sentiment)}")
        print(f"News Sentiment: {news_sentiment.get('avg_sentiment', 0):+.3f} {self._get_sentiment_desc(news_sentiment.get('avg_sentiment', 0))}")
        print(f"Combined Sentiment: {combined_sentiment:+.3f} {self._get_sentiment_desc(combined_sentiment)}")
        
        # Detailed Reddit Mention Counts
        print(f"\n📝 DETAILED REDDIT MENTION ANALYSIS:")
        print("-" * 60)
        
        positive_mentions = reddit_metrics.get('positive_mentions', 0)
        negative_mentions = reddit_metrics.get('negative_mentions', 0)
        net_sentiment = reddit_metrics.get('net_sentiment', 0)
        
        # Display overall counts
        print(f"✅ Positive Mentions: {positive_mentions:,} (calls, bullish, moon, tendies, etc.)")
        print(f"❌ Negative Mentions: {negative_mentions:,} (puts, bearish, bagholder, dump, etc.)")
        print(f"📊 Net Sentiment: {net_sentiment:+,}")
        
        # Display ratios
        if positive_mentions + negative_mentions > 0:
            pos_ratio = reddit_metrics.get('positive_ratio', 0)
            neg_ratio = reddit_metrics.get('negative_ratio', 0)
            print(f"\n📈 Positive Ratio: {pos_ratio:.1%}")
            print(f"📉 Negative Ratio: {neg_ratio:.1%}")
            
            # Visual bar representation
            pos_bar = "█" * int(pos_ratio * 40)
            neg_bar = "█" * int(neg_ratio * 40)
            print(f"\nVisual Breakdown:")
            print(f"  Positive: {pos_bar} {pos_ratio:.1%}")
            print(f"  Negative: {neg_bar} {neg_ratio:.1%}")
        
        # Display top keywords
        print(f"\n🔝 TOP POSITIVE KEYWORDS:")
        top_positive = reddit_metrics.get('top_positive_keywords', [])
        if top_positive:
            for i, (keyword, count) in enumerate(top_positive[:5], 1):
                print(f"  {i}. '{keyword}': {count} mentions")
        else:
            print("  No positive keywords found")
        
        print(f"\n🔻 TOP NEGATIVE KEYWORDS:")
        top_negative = reddit_metrics.get('top_negative_keywords', [])
        if top_negative:
            for i, (keyword, count) in enumerate(top_negative[:5], 1):
                print(f"  {i}. '{keyword}': {count} mentions")
        else:
            print("  No negative keywords found")
        
        # Post classification
        bullish_posts = reddit_metrics.get('bullish_posts', 0)
        bearish_posts = reddit_metrics.get('bearish_posts', 0)
        neutral_posts = reddit_metrics.get('neutral_posts', 0)
        total_posts = reddit_metrics.get('post_count', 0)
        
        if total_posts > 0:
            print(f"\n📋 POST CLASSIFICATION:")
            print(f"  🟢 Bullish Posts: {bullish_posts} ({bullish_posts/total_posts:.1%})")
            print(f"  🔴 Bearish Posts: {bearish_posts} ({bearish_posts/total_posts:.1%})")
            print(f"  🟡 Neutral Posts: {neutral_posts} ({neutral_posts/total_posts:.1%})")
        
        # Reddit activity details
        print(f"\n📱 REDDIT ACTIVITY:")
        activity_trend = reddit_metrics.get('activity_trend', 'unknown')
        print(f"Total Posts: {reddit_metrics.get('post_count', 0)}")
        print(f"Activity Trend: {activity_trend.replace('_', ' ').title()}")
        
        # Subreddit distribution
        subreddit_dist = reddit_metrics.get('subreddit_distribution', {})
        if subreddit_dist:
            print(f"\n🌐 SUBREDDIT DISTRIBUTION:")
            sorted_subs = sorted(subreddit_dist.items(), key=lambda x: x[1], reverse=True)
            for sub, count in sorted_subs[:5]:
                print(f"  r/{sub}: {count} posts")
            if len(sorted_subs) > 5:
                other_count = sum(count for _, count in sorted_subs[5:])
                print(f"  Other ({len(sorted_subs) - 5} subreddits): {other_count} posts")
        
        # Check for coordination patterns
        penny_indicators = reddit_metrics.get('penny_stock_indicators', {})
        if penny_indicators.get('pump_pattern_detected'):
            print(f"\n⚠️  PUMP PATTERN WARNING:")
            print(f"  Coordination Score: {penny_indicators.get('coordination_score', 0)}/3")
            print(f"  Cross-Subreddit Spread: {penny_indicators.get('cross_subreddit_spread', 0)} subreddits")
            print(f"  🚨 This stock shows signs of coordinated promotion!")
        
        # News details
        if news_sentiment.get('news_count', 0) > 0:
            print(f"\n📰 NEWS ANALYSIS:")
            print(f"Recent News: {news_sentiment['bullish_news']} bullish, {news_sentiment['bearish_news']} bearish, {news_sentiment['neutral_news']} neutral")
    
    def _display_options_data(self, analysis: Dict) -> None:
        """Display options market data"""
        options_data = analysis.get('options_data')
        if not options_data:
            print(f"\n{'📈 OPTIONS DATA':<60}")
            print("-" * 60)
            print("❌ No options data available")
            return
        
        print(f"\n{'📈 OPTIONS MARKET DATA':<60}")
        print("-" * 60)
        
        print(f"ATM Call IV: {options_data['atm_call_iv']:.1%}")
        print(f"ATM Put IV: {options_data['atm_put_iv']:.1%}")
        print(f"Call/Put Volume Ratio: {options_data.get('call_put_volume_ratio', 1):.2f}")
        
        iv_assessment = analysis['options_strategy']['iv_assessment']
        print(f"IV Environment: {iv_assessment['level']} ({iv_assessment['description']})")
        print(f"Next Expiry: {options_data['nearest_expiry']}")
        
        # Options flow
        total_volume = options_data.get('total_call_volume', 0) + options_data.get('total_put_volume', 0)
        print(f"Total Options Volume: {total_volume:,}")
    
    def _display_strategy_recommendations(self, analysis: Dict) -> None:
        """Display options strategy recommendations"""
        print(f"\n{'🎯 OPTIONS STRATEGY RECOMMENDATIONS':<60}")
        print("-" * 60)
        
        options_strategy = analysis['options_strategy']
        strategies = options_strategy['recommendations']
        overall_risk = options_strategy['overall_risk_level']
        
        risk_emoji = RISK_COLORS.get(overall_risk, "❓")
        print(f"Overall Risk Level: {risk_emoji} {overall_risk.value}")
        print()
        
        for i, strategy in enumerate(strategies[:3], 1):  # Show top 3 strategies
            strategy_emoji = STRATEGY_EMOJIS.get(strategy['strategy'], "📋")
            risk_emoji = RISK_COLORS.get(strategy['risk'], "❓")
            
            print(f"{strategy_emoji} Strategy #{i}: {strategy['strategy'].value}")
            print(f"   Risk Level: {risk_emoji} {strategy['risk'].value}")
            print(f"   Timeframe: {strategy['timeframe']}")
            print(f"   Position Size: {strategy['position_size']}")
            print(f"   Reasoning: {strategy['reasoning']}")
            print(f"   Strike Guidance: {strategy['strike_guidance']}")
            print(f"   Profit Target: {strategy['profit_target']}")
            print(f"   Stop Loss: {strategy['stop_loss']}")
            print()
    
    def _display_phase_warnings(self, analysis: Dict) -> None:
        """Display phase-specific warnings and guidance"""
        print(f"\n{'🚨 PHASE-SPECIFIC WARNINGS & GUIDANCE':<60}")
        print("-" * 60)
        
        phase = analysis['phase']
        risk_assessment = analysis['risk_assessment']
        
        print(f"Trade Recommendation: {risk_assessment['trade_recommendation']}")
        print("Key Risk Factors:")
        for factor in risk_assessment['risk_factors']:
            print(f"  • {factor}")
        
        # Add specific warnings based on phase
        if phase == PumpDumpPhase.PEAK_FRENZY:
            print("\n🔴 EXTREME DANGER ZONE!")
            print("⚠️  This stock is likely at peak pump - dump imminent!")
            print("🛑 DO NOT BUY - Consider puts if experienced trader")
        
        elif phase == PumpDumpPhase.MAIN_PUMP:
            print("\n🟠 HIGH RISK ZONE!")
            print("⚠️  Active pump in progress - very dangerous for new positions")
            print("💡 If holding positions, consider taking profits")
        
        elif phase == PumpDumpPhase.DEAD:
            print("\n☠️  DEAD STOCK WARNING!")
            print("⚠️  Extreme delisting risk - avoid completely!")
            print("🚫 Total loss of investment likely")
    
    def _display_technical_analysis(self, analysis: Dict) -> None:
        """Display technical analysis indicators"""
        print(f"\n{'📊 TECHNICAL ANALYSIS':<60}")
        print("-" * 60)
        
        stock_summary = analysis['stock_summary']
        key_levels = analysis.get('key_levels', {})
        
        # Technical indicators
        rsi = stock_summary.get('rsi', 50)
        rsi_signal = stock_summary.get('rsi_signal', 'Neutral')
        macd_signal = stock_summary.get('macd_signal', 'Neutral')
        bb_signal = stock_summary.get('bb_signal', 'Neutral')
        
        print(f"RSI (14): {rsi:.1f} ({rsi_signal})")
        print(f"MACD Signal: {macd_signal}")
        print(f"Bollinger Bands: {bb_signal}")
        print(f"20-day Volatility: {stock_summary.get('volatility_20d', 0):.1%}")
        
        # Key levels
        if key_levels:
            print(f"Support Level: ${key_levels.get('support_60d', 0):.2f}")
            print(f"Resistance Level: ${key_levels.get('resistance_60d', 0):.2f}")
            print(f"Trend: {key_levels.get('trend', 'Unknown')}")
    
    def _display_news_highlights(self, analysis: Dict) -> None:
        """Display recent news highlights"""
        news_sentiment = analysis['news_sentiment']
        
        if news_sentiment.get('news_count', 0) == 0:
            return
        
        print(f"\n{'📰 RECENT NEWS HIGHLIGHTS':<60}")
        print("-" * 60)
        
        recent_news = news_sentiment.get('recent_news', [])
        for i, article in enumerate(recent_news[:3], 1):
            sentiment = article.get('sentiment', 0)
            sentiment_emoji = "🟢" if sentiment > 0.1 else "🔴" if sentiment < -0.1 else "🟡"
            
            title = self._truncate_text(article.get('title', 'No title'), 70)
            print(f"{i}. {sentiment_emoji} {title}")
            
            if article.get('date'):
                print(f"   📅 {article['date'].strftime('%Y-%m-%d %H:%M')}")
            print()
    
    def _display_risk_assessment(self, analysis: Dict) -> None:
        """Display comprehensive risk assessment"""
        risk_assessment = analysis['risk_assessment']
        
        print(f"\n{'⚠️  RISK ASSESSMENT':<60}")
        print("-" * 60)
        
        print(f"Risk Level: {risk_assessment['risk_level']}")
        print(f"Trade Recommendation: {risk_assessment['trade_recommendation']}")
        
        # Warning flags
        warnings = []
        if risk_assessment.get('volatility_warning'):
            warnings.append("🔴 High volatility detected")
        if risk_assessment.get('volume_warning'):
            warnings.append("🔴 Extreme volume spike")
        if risk_assessment.get('sentiment_warning'):
            warnings.append("🔴 Extreme sentiment levels")
        
        if warnings:
            print("Warning Flags:")
            for warning in warnings:
                print(f"  {warning}")
    
    def _display_timeline_estimates(self, analysis: Dict) -> None:
        """Display phase timeline estimates"""
        timeline = analysis.get('timeline_estimate', {})
        
        if not timeline:
            return
        
        print(f"\n{'⏰ PHASE TIMELINE ESTIMATES':<60}")
        print("-" * 60)
        
        print(f"Typical Duration: {timeline.get('typical_duration', 'Unknown')}")
        print(f"Adjusted Duration: {timeline.get('adjusted_duration', 'Unknown')}")
        print(f"Catalyst Dependency: {timeline.get('catalyst_dependency', 'Unknown')}")
        print(f"Intensity Score: {timeline.get('intensity_score', 0):.1f}/1.0")
        
        # Next phase probabilities
        next_phases = timeline.get('next_phase_probabilities', {})
        if next_phases:
            print("Next Phase Probabilities:")
            sorted_phases = sorted(next_phases.items(), key=lambda x: x[1], reverse=True)
            for phase_name, probability in sorted_phases[:3]:
                print(f"  {phase_name}: {probability:.0%}")
    
    def _display_disclaimer(self) -> None:
        """Display important disclaimer"""
        print(f"\n{'⚠️  IMPORTANT DISCLAIMER':<60}")
        print("-" * 60)
        print("This analysis is for educational purposes only and NOT financial advice.")
        print("Options trading involves significant risk and may result in total loss.")
        print("Meme stocks are highly volatile and unpredictable.")
        print("Always do your own research and consider consulting a financial advisor.")
        print("Past performance does not guarantee future results.")
    
    def _display_analysis_summary(self, results: List[Dict]) -> None:
        """Display overall analysis summary"""
        if not results:
            return
        
        print(f"\n{'='*80}")
        print("📊 ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        # Phase distribution
        phase_counts = {}
        risk_counts = {}
        
        high_risk_stocks = []
        opportunities = []
        
        for result in results:
            phase = result['phase'].value
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            
            risk_level = result['options_strategy']['overall_risk_level'].value
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            
            # Identify high-risk stocks
            if risk_level in ['HIGH', 'EXTREME']:
                high_risk_stocks.append(f"{result['symbol']} ({phase})")
            
            # Identify potential opportunities
            if (phase == 'Accumulation' and 
                result['combined_sentiment'] > 0.1 and 
                risk_level in ['LOW', 'MEDIUM']):
                opportunities.append(f"{result['symbol']} (sentiment: {result['combined_sentiment']:+.2f})")
        
        print(f"Total Stocks Analyzed: {len(results)}")
        print()
        
        print("Phase Distribution:")
        for phase, count in sorted(phase_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(results)) * 100
            print(f"  {phase}: {count} ({percentage:.1f}%)")
        print()
        
        print("Risk Distribution:")
        for risk, count in sorted(risk_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(results)) * 100
            emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "EXTREME": "☠️"}.get(risk, "❓")
            print(f"  {emoji} {risk}: {count} ({percentage:.1f}%)")
        print()
        
        if high_risk_stocks:
            print("🚨 High Risk Stocks (Avoid):")
            for stock in high_risk_stocks[:5]:
                print(f"  • {stock}")
            if len(high_risk_stocks) > 5:
                print(f"  ... and {len(high_risk_stocks) - 5} more")
            print()
        
        if opportunities:
            print("🎯 Potential Opportunities (Lower Risk):")
            for stock in opportunities[:5]:
                print(f"  • {stock}")
            if len(opportunities) > 5:
                print(f"  ... and {len(opportunities) - 5} more")
            print()
        
        print("Remember: This is analysis, not investment advice!")
    
    def _get_sentiment_desc(self, sentiment: float) -> str:
        """Get sentiment description with emoji"""
        if sentiment > 0.3:
            return "🟢 Very Bullish"
        elif sentiment > 0.1:
            return "🟢 Bullish"
        elif sentiment < -0.3:
            return "🔴 Very Bearish"
        elif sentiment < -0.1:
            return "🔴 Bearish"
        else:
            return "🟡 Neutral"
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to specified length"""
        if len(text) <= max_length:
            return text
        return text[:max_length-2] + ".."