#!/usr/bin/env python3
"""
Fixed Setup Script for Meme Stock Analyzer (Windows Compatible)
Run this script to create all necessary files and directories
"""

import os

def create_file(filepath, content):
    """Create a file with the given content"""
    # Handle root directory files
    if os.path.dirname(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Created: {filepath}")

def setup_project():
    """Create the complete project structure"""
    
    print("🚀 Setting up Meme Stock Analyzer...")
    print("=" * 50)
    
    # Create main.py (root directory)
    main_py_content = '''#!/usr/bin/env python3
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
        print("\\n👋 Goodbye! Trade safely!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting analyzer: {e}")
        print("Please check your API credentials and internet connection.")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    create_file("main.py", main_py_content)

    # Create requirements.txt
    requirements_content = '''praw>=7.7.0
pandas>=1.5.0
yfinance>=0.2.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
textblob>=0.17.0
requests>=2.28.0
'''
    create_file("requirements.txt", requirements_content)

    # Create README.md
    readme_content = '''# Advanced Meme Stock & Options Analyzer

A comprehensive tool for analyzing meme stocks and generating options trading strategies.

## Features
- Pump/Dump phase detection
- Options strategy recommendations  
- Reddit sentiment analysis
- News sentiment analysis
- Technical indicators
- Risk assessment

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run analyzer: `python main.py`

## Configuration
Your Reddit API credentials are pre-configured. For enhanced news analysis, add:
- Alpha Vantage API key
- Finnhub API key

## Disclaimer
This tool is for educational purposes only and not financial advice.
'''
    create_file("README.md", readme_content)

    # Create analyzer directory and __init__.py
    os.makedirs("analyzer", exist_ok=True)
    
    init_py_content = '''"""
Advanced Meme Stock & Options Analyzer
A comprehensive tool for analyzing meme stocks and generating options strategies
"""

from .meme_stock_analyzer import MemeStockAnalyzer
from .config import Config
from .enums import PumpDumpPhase, OptionsStrategy, RiskLevel

__version__ = "2.0.0"
__all__ = [
    "MemeStockAnalyzer",
    "Config", 
    "PumpDumpPhase",
    "OptionsStrategy",
    "RiskLevel"
]
'''
    create_file("analyzer/__init__.py", init_py_content)

    # Create config.py
    config_py_content = '''"""
Configuration settings for the Meme Stock Analyzer
"""

import os
from typing import List

class Config:
    """Configuration class containing all API keys and settings"""
    
    # Reddit API credentials (provided)
    REDDIT_CLIENT_ID = "HQS2vBLyvf7KC1wu5XCYaQ"
    REDDIT_CLIENT_SECRET = "C_cfUlj-jhJzictGVvXaYTmX2gu2Dg"
    REDDIT_USER_AGENT = "MemeStockAnalyzer/2.0 by AdvancedTrader"
    
    # Optional: Enhanced news analysis APIs
    ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', None)
    FINNHUB_KEY = os.getenv('FINNHUB_KEY', None)
    
    # Analysis settings
    REDDIT_SEARCH_LIMIT = 100
    REDDIT_POST_LIMIT = 200
    STOCK_HISTORY_DAYS = 90
    NEWS_HISTORY_DAYS = 7
    TRENDING_STOCKS_LIMIT = 10
    API_RATE_LIMIT = 1.0
    
    # Known meme stock symbols for fallback
    KNOWN_MEME_STOCKS: List[str] = [
        'GME', 'AMC', 'CLOV', 'BB', 'WISH', 'SNDL', 'SPRT', 'IRNT',
        'BBBY', 'NOK', 'NAKD', 'EXPR', 'KOSS', 'RKT', 'UWMC', 
        'WKHS', 'RIDE', 'GOEV', 'PLTR', 'NIO', 'TSLA', 'MVIS', 'TLRY',
        'PROG', 'ATER', 'DWAC', 'PHUN', 'BKKT', 'MARK', 'GREE', 'PTON'
    ]
    
    # Analysis thresholds
    PUMP_THRESHOLD = 0.2
    DUMP_THRESHOLD = -0.3
    HIGH_VOLUME_THRESHOLD = 2.0
    HIGH_RSI_THRESHOLD = 70
    LOW_RSI_THRESHOLD = 30
    HIGH_IV_THRESHOLD = 0.6
    BULLISH_SENTIMENT_THRESHOLD = 0.1
    BEARISH_SENTIMENT_THRESHOLD = -0.1
    MAX_DISPLAY_STOCKS = 15
    MAX_NEWS_ARTICLES = 3
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present"""
        required_fields = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']
        for field in required_fields:
            if not getattr(cls, field):
                print(f"❌ Missing required config: {field}")
                return False
        return True
    
    @classmethod
    def get_api_status(cls) -> dict:
        """Get status of available APIs"""
        return {
            'reddit': bool(cls.REDDIT_CLIENT_ID and cls.REDDIT_CLIENT_SECRET),
            'alpha_vantage': bool(cls.ALPHA_VANTAGE_KEY),
            'finnhub': bool(cls.FINNHUB_KEY),
            'yahoo_finance': True
        }
'''
    create_file("analyzer/config.py", config_py_content)

    # Create helper script
    helper_content = '''#!/usr/bin/env python3
"""
This script shows you which files still need to be created.
"""

import os

REMAINING_FILES = [
    "analyzer/enums.py",
    "analyzer/reddit_analyzer.py", 
    "analyzer/news_analyzer.py",
    "analyzer/stock_analyzer.py",
    "analyzer/phase_detector.py",
    "analyzer/options_strategy.py",
    "analyzer/meme_stock_analyzer.py",
    "analyzer/display_manager.py"
]

def main():
    print("📋 REMAINING FILES TO CREATE:")
    print("=" * 50)
    
    missing_files = []
    for i, filename in enumerate(REMAINING_FILES, 1):
        if os.path.exists(filename):
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename}")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\\n💡 {len(missing_files)} files still needed")
        print("\\n🎯 Next: Ask the assistant for these files:")
        for i, filename in enumerate(missing_files[:3], 1):
            print(f"{i}. {filename}")
        
        if len(missing_files) > 3:
            print(f"... and {len(missing_files) - 3} more")
    else:
        print("\\n🎉 ALL FILES CREATED! Ready to run:")
        print("python main.py")

if __name__ == "__main__":
    main()
'''
    create_file("check_files.py", helper_content)

    print("\\n✅ Basic setup complete!")
    print("\\n📋 WHAT'S BEEN CREATED:")
    print("✅ main.py - Main entry point")
    print("✅ requirements.txt - Dependencies")
    print("✅ README.md - Documentation")
    print("✅ analyzer/__init__.py - Package init")
    print("✅ analyzer/config.py - Configuration (with your Reddit API keys)")
    print("✅ check_files.py - Helper script to check progress")
    
    print("\\n🔄 NEXT STEPS:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Check status: python check_files.py")
    print("3. Ask the assistant for the remaining 8 files")
    
    print("\\n💡 RECOMMENDED ORDER:")
    print("Group 1: analyzer/enums.py and analyzer/reddit_analyzer.py")
    print("Group 2: analyzer/news_analyzer.py and analyzer/stock_analyzer.py")
    print("Group 3: analyzer/phase_detector.py and analyzer/options_strategy.py")
    print("Group 4: analyzer/meme_stock_analyzer.py and analyzer/display_manager.py")

if __name__ == "__main__":
    setup_project()