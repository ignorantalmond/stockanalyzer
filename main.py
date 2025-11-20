#!/usr/bin/env python3
"""
Advanced Meme Stock & Options Analyzer
Main entry point for the application
"""

from analyzer.meme_stock_analyzer import MemeStockAnalyzer
from analyzer.config import Config
import sys

def main():
    """
    Main function to run the interactive analyzer
    """
    print("🚀 ADVANCED MEME STOCK & OPTIONS ANALYZER")
    print("=" * 60)
    print("Features: Pump/Dump Detection | Options Strategies | News Sentiment")
    print("=" * 60)
    
    try:
        # Initialize analyzer with configuration
        config = Config()
        analyzer = MemeStockAnalyzer(
            config.REDDIT_CLIENT_ID,
            config.REDDIT_CLIENT_SECRET, 
            config.REDDIT_USER_AGENT,
            config.ALPHA_VANTAGE_KEY,
            config.FINNHUB_KEY
        )
        
        print("🔥 Starting Advanced Meme Stock Analysis...")
        print("⚠️  Remember: This tool helps identify risks, not guarantee profits!")
        print()
        
        # Run the interactive analyzer
        analyzer.run_interactive_analyzer()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Trade safely!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting analyzer: {e}")
        print("Please check your API credentials and internet connection.")
        sys.exit(1)

if __name__ == "__main__":
    main()
