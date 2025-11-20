"""
Main MemeStockAnalyzer class that coordinates all analysis modules
Updated version with predictive analysis integration
"""

import time
from typing import List, Dict, Optional

from .config import Config
from .reddit_analyzer import RedditAnalyzer
from .news_analyzer import NewsAnalyzer
from .stock_analyzer import StockAnalyzer
from .phase_detector import PhaseDetector
from .options_strategy import OptionsStrategyRecommender
from .display_manager import DisplayManager
from .predictive_analyzer import PredictiveAnalyzer
from .enums import PHASE_EMOJIS, STRATEGY_EMOJIS, RISK_COLORS

class MemeStockAnalyzer:
    """Main analyzer class that coordinates all analysis modules"""
    
    def __init__(self, reddit_client_id: str, reddit_client_secret: str, reddit_user_agent: str,
                 alpha_vantage_key: Optional[str] = None, finnhub_key: Optional[str] = None):
        """
        Initialize the meme stock analyzer
        
        Args:
            reddit_client_id: Reddit API client ID
            reddit_client_secret: Reddit API client secret
            reddit_user_agent: Reddit API user agent
            alpha_vantage_key: Optional Alpha Vantage API key for enhanced news
            finnhub_key: Optional Finnhub API key for enhanced news
        """
        self.config = Config()
        
        # Initialize all analysis modules
        self.reddit_analyzer = RedditAnalyzer(reddit_client_id, reddit_client_secret, reddit_user_agent)
        self.news_analyzer = NewsAnalyzer(alpha_vantage_key, self.config.FINNHUB_KEY)  # Pass the Finnhub key from config
        self.stock_analyzer = StockAnalyzer()
        self.phase_detector = PhaseDetector()
        self.options_recommender = OptionsStrategyRecommender()
        self.display_manager = DisplayManager()
        self.predictive_analyzer = PredictiveAnalyzer()
        
        # Validate configuration
        if not self.config.validate_config():
            raise ValueError("Invalid configuration - check your API credentials")
    
    def scan_trending_stocks(self) -> List[Dict]:
        """
        Scan for trending meme stocks and analyze each one
        
        Returns:
            List of analysis results for trending stocks
        """
        print("🔍 Scanning for trending meme stocks...")
        
        # Get trending stocks from Reddit
        trending_symbols = self.reddit_analyzer.get_trending_stocks()
        print(f"Found {len(trending_symbols)} trending stocks: {', '.join(trending_symbols)}")
        
        results = []
        for symbol in trending_symbols:
            try:
                print(f"  📊 Analyzing {symbol}...")
                analysis = self.analyze_stock(symbol)
                if analysis:
                    results.append(analysis)
                
                # Rate limiting
                time.sleep(self.config.API_RATE_LIMIT)
                
            except Exception as e:
                print(f"  ❌ Error analyzing {symbol}: {e}")
                continue
        
        return results
    
    def analyze_stock(self, symbol: str) -> Optional[Dict]:
        """
        Perform comprehensive analysis of a single stock
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with complete analysis results or None if analysis fails
        """
        try:
            # Step 1: Get stock data and basic info
            stock_df, stock_info = self.stock_analyzer.get_stock_data(symbol)
            if stock_df.empty:
                print(f"  ⚠️  No stock data available for {symbol}")
                return None
            
            # Step 2: Get Reddit sentiment and activity
            reddit_df = self.reddit_analyzer.get_posts_for_symbol(symbol)
            reddit_metrics = self.reddit_analyzer.calculate_reddit_metrics(reddit_df)
            
            # Step 3: Get news sentiment
            news_df = self.news_analyzer.get_stock_news(symbol)
            news_sentiment = self.news_analyzer.analyze_news_sentiment(news_df)
            
            # Step 4: Get options data
            options_data = self.stock_analyzer.get_options_data(symbol)
            
            # Step 5: Determine pump/dump phase
            phase_metrics = self.phase_detector._calculate_phase_metrics(
                stock_df.tail(30), reddit_metrics, news_sentiment, options_data or {}
            )
            phase, description, confidence = self.phase_detector.determine_phase(
                symbol, stock_df, reddit_metrics, news_sentiment, options_data or {}
            )
            
            # Step 6: Get options strategy recommendations
            reddit_sentiment_score = reddit_metrics.get('avg_sentiment', 0)
            options_strategy = self.options_recommender.suggest_strategies(
                phase, reddit_sentiment_score, news_sentiment, options_data or {}, 
                stock_info, phase_metrics
            )
            
            # Step 7: Calculate additional metrics
            stock_summary = self.stock_analyzer.get_stock_summary(stock_df, stock_info)
            key_levels = self.stock_analyzer.calculate_key_levels(stock_df)
            risk_assessment = self.phase_detector.get_phase_risk_assessment(phase, phase_metrics)
            timeline_estimate = self.phase_detector.get_phase_timeline_estimate(phase, phase_metrics)
            
            # Step 8: Compile results
            return {
                'symbol': symbol,
                'company_name': stock_info.get('longName', symbol),
                'current_price': stock_summary.get('current_price', 0),
                'price_change_1d': stock_summary.get('price_change_1d', 0),
                'volume_ratio': stock_summary.get('volume_ratio', 1),
                
                # Phase analysis
                'phase': phase,
                'phase_description': description,
                'phase_confidence': confidence,
                'phase_metrics': phase_metrics,
                'risk_assessment': risk_assessment,
                'timeline_estimate': timeline_estimate,
                
                # Sentiment analysis
                'reddit_sentiment': reddit_sentiment_score,
                'reddit_metrics': reddit_metrics,
                'news_sentiment': news_sentiment,
                'combined_sentiment': options_strategy['combined_sentiment'],
                
                # Options analysis
                'options_strategy': options_strategy,
                'options_data': options_data,
                
                # Technical analysis
                'stock_summary': stock_summary,
                'key_levels': key_levels,
                
                # Raw data
                'stock_data': stock_df,
                'reddit_data': reddit_df,
                'news_data': news_df,
                'stock_info': stock_info,
                
                # Prediction data (initially None, generated on demand)
                'prediction_data': None
            }
            
        except Exception as e:
            print(f"  ❌ Analysis failed for {symbol}: {e}")
            return None
    
    def generate_stock_predictions(self, stock_analysis: Dict) -> Dict:
        """
        Generate predictive analysis for a stock
        
        Args:
            stock_analysis: Complete stock analysis dictionary
            
        Returns:
            Dictionary with prediction data
        """
        try:
            prediction_data = self.predictive_analyzer.generate_price_predictions(
                stock_analysis['stock_data'],
                stock_analysis['phase'],
                stock_analysis['phase_metrics'],
                stock_analysis.get('options_data', {}),
                stock_analysis['reddit_metrics'],
                stock_analysis['news_sentiment']
            )
            return prediction_data
        except Exception as e:
            print(f"❌ Error generating predictions: {e}")
            return {}

    def create_prediction_chart(self, stock_analysis: Dict, days: int, save_path: str = None) -> str:
        """
        Create prediction chart for a stock
        
        Args:
            stock_analysis: Complete stock analysis dictionary
            days: Number of days to predict (7, 14, or 21)
            save_path: Optional path to save chart
            
        Returns:
            Path to saved chart or empty string
        """
        try:
            # Generate predictions if not already cached
            if 'prediction_data' not in stock_analysis or stock_analysis['prediction_data'] is None:
                stock_analysis['prediction_data'] = self.generate_stock_predictions(stock_analysis)
            
            prediction_data = stock_analysis['prediction_data']
            
            if not prediction_data:
                return ""
            
            return self.predictive_analyzer.create_prediction_chart(
                prediction_data, stock_analysis['symbol'], days, save_path
            )
        except Exception as e:
            print(f"❌ Error creating prediction chart: {e}")
            return ""

    def get_prediction_summary(self, stock_analysis: Dict) -> str:
        """
        Get prediction summary for a stock
        
        Args:
            stock_analysis: Complete stock analysis dictionary
            
        Returns:
            Summary string
        """
        try:
            # Generate predictions if not already cached
            if 'prediction_data' not in stock_analysis or stock_analysis['prediction_data'] is None:
                stock_analysis['prediction_data'] = self.generate_stock_predictions(stock_analysis)
            
            prediction_data = stock_analysis['prediction_data']
            
            if not prediction_data:
                return "No predictions available"
            
            return self.predictive_analyzer.get_prediction_summary(prediction_data)
        except Exception as e:
            print(f"❌ Error getting prediction summary: {e}")
            return "Error generating prediction summary"
    
    def display_stock_summary(self, results: List[Dict]) -> None:
        """Display summary table of analyzed stocks"""
        self.display_manager.display_summary_table(results)
    
    def interactive_stock_selector(self, results: List[Dict]) -> str:
        """Interactive interface for stock selection"""
        return self.display_manager.interactive_selector(results)
    
    def display_detailed_analysis(self, stock_analysis: Dict) -> None:
        """Display detailed analysis for a selected stock"""
        self.display_manager.display_detailed_analysis(stock_analysis)
    
    def run_interactive_analyzer(self) -> None:
        """
        Main interactive loop for the analyzer
        """
        print("\n🎯 Starting interactive analysis...")
        print("⚠️  Remember: This tool helps identify risks, not guarantee profits!")
        print()
        
        # Display API status
        api_status = self.config.get_api_status()
        print("📡 API Status:")
        for api, status in api_status.items():
            status_emoji = "✅" if status else "❌"
            print(f"  {status_emoji} {api.replace('_', ' ').title()}")
        print()
        
        while True:
            try:
                # Main menu
                print(f"\n{'='*80}")
                print("🎯 MEME STOCK & OPTIONS ANALYZER - MAIN MENU")
                print(f"{'='*80}")
                print("1. 🔍 Scan trending meme stocks from Reddit")
                print("2. 📊 Analyze specific ticker symbol")
                print("3. ❌ Exit")
                
                choice = input("\nSelect an option (1-3): ").strip()
                
                if choice == "1":
                    # Original trending stocks analysis
                    results = self.scan_trending_stocks()
                    
                    if not results:
                        print("❌ No stocks found. Please check your API credentials.")
                        continue
                    
                    # Display summary and handle interaction
                    self.display_stock_summary(results)
                    action = self.interactive_stock_selector(results)
                    
                    if action == "exit":
                        print("👋 Goodbye! Trade safely and remember: This is not financial advice!")
                        break
                    elif action == "refresh":
                        print("🔄 Refreshing data...")
                        continue
                    elif action == "export":
                        filename = self.export_analysis_results(results)
                        if filename:
                            print(f"📁 Results exported to: {filename}")
                        continue
                    elif action == "manual_ticker":
                        self.manual_ticker_analysis()
                        continue
                
                elif choice == "2":
                    # Manual ticker analysis
                    self.manual_ticker_analysis()
                
                elif choice == "3":
                    print("👋 Goodbye! Trade safely and remember: This is not financial advice!")
                    break
                
                else:
                    print("❌ Invalid choice. Please select 1, 2, or 3.")
                    
            except KeyboardInterrupt:
                print("\n👋 Analysis interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                print("Returning to main menu...")
                continue
    
    def manual_ticker_analysis(self) -> None:
        """
        Allow user to manually input a ticker symbol for analysis
        """
        while True:
            print(f"\n{'='*60}")
            print("📊 MANUAL TICKER ANALYSIS")
            print(f"{'='*60}")
            
            # Get ticker symbol from user
            symbol = input("Enter ticker symbol (or 'back' to return to main menu): ").strip().upper()
            
            if symbol.lower() == 'back':
                return
            
            if not symbol or len(symbol) > 10:
                print("❌ Please enter a valid ticker symbol (1-10 characters)")
                continue
            
            print(f"\n🔍 Analyzing {symbol}...")
            
            try:
                # Analyze the stock
                analysis = self.analyze_stock(symbol)
                
                if not analysis:
                    print(f"❌ Could not analyze {symbol}. Please check:")
                    print("  • Ticker symbol is correct")
                    print("  • Stock exists and is actively traded")
                    print("  • Your internet connection is stable")
                    
                    retry = input("\nTry another symbol? (y/n): ").strip().lower()
                    if retry != 'y':
                        return
                    continue
                
                # Create a single-item list for compatibility with existing display functions
                results = [analysis]
                
                # Display summary
                print(f"\n{'='*100}")
                print(f"📈 ANALYSIS RESULTS FOR {symbol}")
                print(f"{'='*100}")
                
                # Show quick summary
                phase_emoji = PHASE_EMOJIS.get(analysis['phase'], "❓")
                risk_level = analysis['options_strategy']['overall_risk_level']
                risk_emoji = RISK_COLORS.get(risk_level, "❓")
                sentiment = analysis['combined_sentiment']
                sentiment_emoji = "🟢" if sentiment > 0.1 else "🔴" if sentiment < -0.1 else "🟡"
                
                print(f"Company: {analysis['company_name']}")
                print(f"Current Price: ${analysis['current_price']:.2f}")
                print(f"24h Change: {analysis['price_change_1d']:+.2%}")
                print(f"Phase: {phase_emoji} {analysis['phase'].value}")
                print(f"Risk Level: {risk_emoji} {risk_level.value}")
                print(f"Sentiment: {sentiment_emoji} {sentiment:+.2f}")
                
                # Options for detailed analysis
                while True:
                    print(f"\n{'─'*60}")
                    print("What would you like to do?")
                    print("1. 📋 View detailed analysis")
                    print("2. 🔮 Generate predictive charts (NEW!)")
                    print("3. 📊 View options strategies")
                    print("4. 📈 View technical analysis")
                    print("5. 📰 View news & sentiment")
                    print("6. ⚠️  View risk assessment")
                    print("7. 📁 Export analysis")
                    print("8. 🔍 Analyze another ticker")
                    print("9. 🔙 Back to main menu")
                    
                    detail_choice = input("\nSelect option (1-9): ").strip()
                    
                    if detail_choice == "1":
                        # Full detailed analysis
                        self.display_detailed_analysis(analysis)
                    
                    elif detail_choice == "2":
                        # NEW: Predictive analysis
                        self._show_predictive_analysis_menu(analysis)
                    
                    elif detail_choice == "3":
                        # Options strategies only
                        self._display_options_strategies_only(analysis)
                    
                    elif detail_choice == "4":
                        # Technical analysis only
                        self._display_technical_only(analysis)
                    
                    elif detail_choice == "5":
                        # News and sentiment only
                        self._display_news_sentiment_only(analysis)
                    
                    elif detail_choice == "6":
                        # Risk assessment only
                        self._display_risk_only(analysis)
                    
                    elif detail_choice == "7":
                        # Export single analysis
                        filename = self.export_analysis_results(results, f"{symbol}_analysis_{int(time.time())}.json")
                        if filename:
                            print(f"📁 Analysis exported to: {filename}")
                    
                    elif detail_choice == "8":
                        # Analyze another ticker
                        break
                    
                    elif detail_choice == "9":
                        # Back to main menu
                        return
                    
                    else:
                        print("❌ Invalid choice. Please select 1-9.")
                
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {e}")
                retry = input("\nTry another symbol? (y/n): ").strip().lower()
                if retry != 'y':
                    return

    def _show_predictive_analysis_menu(self, stock_analysis: Dict) -> None:
        """
        Show predictive analysis menu for manual ticker analysis
        
        Args:
            stock_analysis: Complete stock analysis dictionary
        """
        symbol = stock_analysis['symbol']
        
        while True:
            print(f"\n{'='*70}")
            print(f"🔮 PREDICTIVE ANALYSIS: {symbol}")
            print(f"{'='*70}")
            print("1. 📊 Quick prediction summary (all timeframes)")
            print("2. 📈 Generate 7-day prediction chart")
            print("3. 📈 Generate 14-day prediction chart") 
            print("4. 📈 Generate 21-day prediction chart")
            print("5. 💾 Save all prediction charts")
            print("6. 📋 Export prediction data")
            print("7. 🔙 Back to analysis menu")
            
            pred_choice = input("\nSelect option (1-7): ").strip()
            
            if pred_choice == "1":
                self._show_quick_prediction_summary(stock_analysis)
            elif pred_choice == "2":
                self._generate_single_prediction_chart(stock_analysis, 7)
            elif pred_choice == "3":
                self._generate_single_prediction_chart(stock_analysis, 14)
            elif pred_choice == "4":
                self._generate_single_prediction_chart(stock_analysis, 21)
            elif pred_choice == "5":
                self._save_all_prediction_charts(stock_analysis)
            elif pred_choice == "6":
                self._export_prediction_data(stock_analysis)
            elif pred_choice == "7":
                break
            else:
                print("❌ Invalid choice. Please select 1-7.")

    def _show_quick_prediction_summary(self, stock_analysis: Dict) -> None:
        """Show quick prediction summary for all timeframes"""
        
        print(f"\n🔄 Generating predictions for {stock_analysis['symbol']}...")
        
        # Generate predictions if not cached
        if 'prediction_data' not in stock_analysis or stock_analysis['prediction_data'] is None:
            stock_analysis['prediction_data'] = self.generate_stock_predictions(stock_analysis)
        
        prediction_data = stock_analysis['prediction_data']
        
        if not prediction_data or not prediction_data.get('predictions'):
            print("❌ Unable to generate predictions - insufficient data")
            input("\nPress Enter to continue...")
            return
        
        # Display summary
        summary_text = self.get_prediction_summary(stock_analysis)
        print(f"\n{summary_text}")
        
        # Show phase-specific insights
        phase = stock_analysis['phase']
        print(f"\n🎯 PHASE-SPECIFIC INSIGHTS:")
        print(f"Current Phase: {phase.value}")
        
        phase_insights = {
            "Accumulation": "Low volatility period - good time for strategic positioning",
            "Early Pump": "Momentum building - watch for continuation or reversal signals",
            "Main Pump": "High risk period - consider profit-taking strategies",
            "Peak Frenzy": "EXTREME DANGER - dump highly likely, avoid new positions",
            "Early Dump": "Decline beginning - bearish strategies may be profitable",
            "Main Dump": "Active selling pressure - wait for stabilization",
            "Bagholders": "Low probability of quick recovery - patience required",
            "Recovery Attempt": "Mixed signals - wait for clear direction",
            "Dead": "Avoid completely - focus capital elsewhere"
        }
        
        insight = phase_insights.get(phase.value, "Monitor closely for phase transitions")
        print(f"Key Insight: {insight}")
        
        # Show confidence factors
        confidence = prediction_data['confidence']
        print(f"\nConfidence Level: {confidence['level']} ({confidence['overall']:.1%})")
        
        if confidence['overall'] < 0.5:
            print("⚠️  LOW CONFIDENCE WARNING:")
            print("   • Predictions are less reliable")
            print("   • Use smaller position sizes")
            print("   • Monitor closely for changes")
        
        input("\nPress Enter to continue...")

    def _generate_single_prediction_chart(self, stock_analysis: Dict, days: int) -> None:
        """Generate and display a single prediction chart"""
        
        symbol = stock_analysis['symbol']
        print(f"\n📊 Generating {days}-day prediction chart for {symbol}...")
        
        # Generate predictions if not cached
        if 'prediction_data' not in stock_analysis or stock_analysis['prediction_data'] is None:
            stock_analysis['prediction_data'] = self.generate_stock_predictions(stock_analysis)
        
        prediction_data = stock_analysis['prediction_data']
        
        if not prediction_data or not prediction_data.get('predictions'):
            print("❌ Unable to generate predictions - insufficient data")
            input("\nPress Enter to continue...")
            return
        
        # Show text summary first
        if f'{days}_day' in prediction_data['predictions']:
            day_data = prediction_data['predictions'][f'{days}_day']
            
            print(f"\n{'='*60}")
            print(f"📈 {days}-DAY PREDICTION SUMMARY")
            print(f"{'='*60}")
            
            current_price = prediction_data['current_price']
            expected_path = day_data['expected_path']
            expected_price = expected_path['price_path'][-1]
            expected_return = expected_path['expected_return']
            
            print(f"Current Price: ${current_price:.2f}")
            print(f"Expected Price: ${expected_price:.2f} ({expected_return:+.1%})")
            
            # Show top 3 scenarios
            sorted_scenarios = sorted(day_data['scenarios'].items(), 
                                    key=lambda x: x[1]['probability'], reverse=True)
            
            print(f"\nTop 3 Most Likely Scenarios:")
            for i, (name, data) in enumerate(sorted_scenarios[:3], 1):
                direction_emoji = "🟢" if data['total_return'] > 0.1 else "🔴" if data['total_return'] < -0.1 else "🟡"
                print(f"{i}. {direction_emoji} {name} ({data['probability']:.1%}) - {data['total_return']:+.1%}")
        
        # Ask if user wants visual chart
        show_chart = input(f"\nGenerate visual chart? (y/n): ").strip().lower()
        
        if show_chart == 'y':
            try:
                chart_path = self.create_prediction_chart(stock_analysis, days)
                if chart_path:
                    print(f"📊 Chart displayed and saved to: {chart_path}")
                else:
                    print("📊 Chart displayed (not saved)")
            except Exception as e:
                print(f"❌ Error generating chart: {e}")
        
        input("\nPress Enter to continue...")

    def _save_all_prediction_charts(self, stock_analysis: Dict) -> None:
        """Save prediction charts for all timeframes"""
        
        symbol = stock_analysis['symbol']
        print(f"\n💾 Saving all prediction charts for {symbol}...")
        
        # Generate predictions if not cached
        if 'prediction_data' not in stock_analysis or stock_analysis['prediction_data'] is None:
            stock_analysis['prediction_data'] = self.generate_stock_predictions(stock_analysis)
        
        prediction_data = stock_analysis['prediction_data']
        
        if not prediction_data or not prediction_data.get('predictions'):
            print("❌ Unable to generate predictions - insufficient data")
            input("\nPress Enter to continue...")
            return
        
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = []
        
        for days in [7, 14, 21]:
            if f'{days}_day' in prediction_data['predictions']:
                try:
                    filename = f"{symbol}_{days}day_prediction_{timestamp}.png"
                    chart_path = self.create_prediction_chart(stock_analysis, days, filename)
                    if chart_path:
                        saved_files.append(chart_path)
                        print(f"✅ Saved {days}-day chart: {filename}")
                except Exception as e:
                    print(f"❌ Error saving {days}-day chart: {e}")
        
        # Save text summary
        try:
            summary_filename = f"{symbol}_prediction_summary_{timestamp}.txt"
            summary_text = self.get_prediction_summary(stock_analysis)
            
            with open(summary_filename, 'w') as f:
                f.write(f"PREDICTION ANALYSIS FOR {symbol}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                f.write(summary_text)
                f.write(f"\n\n⚠️ DISCLAIMER: This analysis is for educational purposes only and NOT financial advice.")
                f.write(f"\nOptions trading involves significant risk. Past performance does not guarantee future results.")
            
            print(f"✅ Saved prediction summary: {summary_filename}")
            saved_files.append(summary_filename)
            
        except Exception as e:
            print(f"❌ Error saving summary: {e}")
        
        if saved_files:
            print(f"\n🎉 Successfully saved {len(saved_files)} files!")
        else:
            print("❌ No files were saved")
        
        input("\nPress Enter to continue...")

    def _export_prediction_data(self, stock_analysis: Dict) -> None:
        """Export prediction data to JSON file"""
        
        symbol = stock_analysis['symbol']
        print(f"\n📁 Exporting prediction data for {symbol}...")
        
        # Generate predictions if not cached
        if 'prediction_data' not in stock_analysis or stock_analysis['prediction_data'] is None:
            stock_analysis['prediction_data'] = self.generate_stock_predictions(stock_analysis)
        
        prediction_data = stock_analysis['prediction_data']
        
        if not prediction_data:
            print("❌ No prediction data to export")
            input("\nPress Enter to continue...")
            return
        
        try:
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_prediction_data_{timestamp}.json"
            
            # Prepare export data
            export_data = {
                'symbol': symbol,
                'company_name': stock_analysis['company_name'],
                'current_price': prediction_data['current_price'],
                'analysis_phase': stock_analysis['phase'].value,
                'generated_at': prediction_data['generated_at'].isoformat(),
                'confidence': prediction_data['confidence'],
                'predictions': {},
                'scenarios_summary': {}
            }
            
            # Add prediction data for each timeframe
            for days in [7, 14, 21]:
                if f'{days}_day' in prediction_data['predictions']:
                    day_data = prediction_data['predictions'][f'{days}_day']
                    
                    export_data['predictions'][f'{days}_day'] = {
                        'expected_return': day_data['expected_path']['expected_return'],
                        'expected_price': day_data['expected_path']['price_path'][-1],
                        'summary': day_data['summary']
                    }
                    
                    # Add scenario details
                    export_data['scenarios_summary'][f'{days}_day'] = {}
                    for scenario_name, scenario_data in day_data['scenarios'].items():
                        export_data['scenarios_summary'][f'{days}_day'][scenario_name] = {
                            'probability': scenario_data['probability'],
                            'expected_return': scenario_data['total_return'],
                            'final_price': scenario_data['final_price'],
                            'description': scenario_data['description']
                        }
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Prediction data exported to: {filename}")
            
            # Show summary of what was exported
            print(f"\nExported data includes:")
            print(f"  • Predictions for {len(export_data['predictions'])} timeframes")
            print(f"  • Confidence analysis: {prediction_data['confidence']['level']}")
            print(f"  • Scenario breakdowns for each timeframe")
            print(f"  • Current phase: {stock_analysis['phase'].value}")
            
        except Exception as e:
            print(f"❌ Error exporting prediction data: {e}")
        
        input("\nPress Enter to continue...")
    
    def _display_options_strategies_only(self, analysis: Dict) -> None:
        """Display only options strategies section"""
        print(f"\n{'='*80}")
        print(f"📈 OPTIONS STRATEGIES FOR {analysis['symbol']}")
        print(f"{'='*80}")
        
        self.display_manager._display_options_data(analysis)
        self.display_manager._display_strategy_recommendations(analysis)
        
        input("\nPress Enter to continue...")
    
    def _display_technical_only(self, analysis: Dict) -> None:
        """Display only technical analysis section"""
        print(f"\n{'='*80}")
        print(f"📊 TECHNICAL ANALYSIS FOR {analysis['symbol']}")
        print(f"{'='*80}")
        
        self.display_manager._display_current_status(analysis)
        self.display_manager._display_technical_analysis(analysis)
        
        input("\nPress Enter to continue...")
    
    def _display_news_sentiment_only(self, analysis: Dict) -> None:
        """Display only news and sentiment section"""
        print(f"\n{'='*80}")
        print(f"📰 NEWS & SENTIMENT FOR {analysis['symbol']}")
        print(f"{'='*80}")
        
        self.display_manager._display_sentiment_analysis(analysis)
        self.display_manager._display_news_highlights(analysis)
        
        input("\nPress Enter to continue...")
    
    def _display_risk_only(self, analysis: Dict) -> None:
        """Display only risk assessment section"""
        print(f"\n{'='*80}")
        print(f"⚠️  RISK ASSESSMENT FOR {analysis['symbol']}")
        print(f"{'='*80}")
        
        self.display_manager._display_phase_warnings(analysis)
        self.display_manager._display_risk_assessment(analysis)
        self.display_manager._display_timeline_estimates(analysis)
        
        input("\nPress Enter to continue...")
    
    def get_analysis_summary(self, results: List[Dict]) -> Dict:
        """
        Get overall analysis summary for all scanned stocks
        
        Args:
            results: List of stock analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}
        
        # Count stocks by phase
        phase_counts = {}
        risk_counts = {}
        total_stocks = len(results)
        
        high_risk_stocks = []
        opportunity_stocks = []
        
        for result in results:
            # Phase distribution
            phase = result['phase']
            phase_counts[phase.value] = phase_counts.get(phase.value, 0) + 1
            
            # Risk distribution
            risk_level = result['options_strategy']['overall_risk_level'].value
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            
            # High risk stocks
            if risk_level in ['HIGH', 'EXTREME']:
                high_risk_stocks.append({
                    'symbol': result['symbol'],
                    'phase': phase.value,
                    'risk': risk_level
                })
            
            # Opportunity stocks (accumulation phase with positive sentiment)
            if (phase.value == 'Accumulation' and 
                result['combined_sentiment'] > 0.1 and 
                risk_level in ['LOW', 'MEDIUM']):
                opportunity_stocks.append({
                    'symbol': result['symbol'],
                    'sentiment': result['combined_sentiment'],
                    'confidence': result['phase_confidence']
                })
        
        return {
            'total_stocks_analyzed': total_stocks,
            'phase_distribution': phase_counts,
            'risk_distribution': risk_counts,
            'high_risk_stocks': high_risk_stocks,
            'opportunity_stocks': opportunity_stocks,
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def export_analysis_results(self, results: List[Dict], filename: str = None) -> str:
        """
        Export analysis results to a file
        
        Args:
            results: List of analysis results
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Filename of exported file
        """
        import json
        from datetime import datetime
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"meme_stock_analysis_{timestamp}.json"
        
        # Prepare data for export (convert enums to strings)
        export_data = []
        for result in results:
            export_result = result.copy()
            
            # Convert enums to strings
            export_result['phase'] = result['phase'].value
            
            # Remove large DataFrames to keep file size manageable
            export_result.pop('stock_data', None)
            export_result.pop('reddit_data', None)
            export_result.pop('news_data', None)
            export_result.pop('prediction_data', None)  # Remove prediction data for main export
            
            export_data.append(export_result)
        
        # Add summary
        summary = self.get_analysis_summary(results)
        
        final_export = {
            'analysis_summary': summary,
            'stock_analyses': export_data,
            'export_timestamp': datetime.now().isoformat(),
            'config_used': {
                'reddit_enabled': bool(self.config.REDDIT_CLIENT_ID),
                'news_apis_enabled': bool(self.config.ALPHA_VANTAGE_KEY or self.config.FINNHUB_KEY),
                'analysis_version': '2.1',
                'predictive_analysis_available': True
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(final_export, f, indent=2, default=str)
            
            print(f"📁 Analysis exported to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return ""