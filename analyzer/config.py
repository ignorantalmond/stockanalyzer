"""
Configuration settings for the Meme Stock Analyzer
Diagnostic version to find what's overriding the API key
"""

import os
from typing import List

print("🔍 Config file loading...")
print(f"Environment FINNHUB_KEY: {repr(os.getenv('FINNHUB_KEY'))}")

#class Config:
#    """Configuration class containing all API keys and settings"""
class Config:
    def __init__(self):
        # Reddit API credentials - REMOVED (use user-provided keys)
        self.REDDIT_CLIENT_ID = None
        self.REDDIT_CLIENT_SECRET = None
        self.REDDIT_USER_AGENT = None
        
        # News API credentials - REMOVED (use user-provided keys)
        self.ALPHA_VANTAGE_KEY = None
        self.FINNHUB_KEY = None
        
        # Analysis settings
        self.REDDIT_POST_LIMIT = 100
        self.NEWS_LOOKBACK_DAYS = 7
        self.STOCK_HISTORY_DAYS = 90
        
    def validate_config(self):
        """
        Validate that required configuration is present
        Note: For Streamlit app, keys are provided by users at runtime
        """
        # For Streamlit, we don't validate here since keys come from user input
        return True
    
    def get_api_status(self):
        """
        Check which APIs are configured
        Returns dict of API name -> enabled status
        """
        return {
            'reddit_api': bool(self.REDDIT_CLIENT_ID and self.REDDIT_CLIENT_SECRET),
            'alpha_vantage': bool(self.ALPHA_VANTAGE_KEY),
            'finnhub': bool(self.FINNHUB_KEY)
        }    
    # Reddit API credentials (provided)
    FINNHUB_KEY = None
    #print(f"🔑 FINNHUB_KEY set to: {repr(FINNHUB_KEY)}")
    
    # Alpha Vantage API key (optional - from environment only)
    #ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', None)
    
    # Check if something is overriding
    #print(f"🔍 After initial assignment, FINNHUB_KEY is: {repr(FINNHUB_KEY)}")
    
    #def __init__(self):
    #    """Initialize config"""
    #    print("🏗️  Config.__init__ called")
    #    print(f"🔍 In __init__, FINNHUB_KEY is: {repr(self.FINNHUB_KEY)}")
    #    
    #    # Force set it again in init
    #    self.FINNHUB_KEY = None
    #    print(f"🔧 After forcing in __init__, FINNHUB_KEY is: {repr(self.FINNHUB_KEY)}")
    
    # Analysis settings
    REDDIT_SEARCH_LIMIT = 100
    REDDIT_POST_LIMIT = 200
    STOCK_HISTORY_DAYS = 90
    NEWS_HISTORY_DAYS = 7
    TRENDING_STOCKS_LIMIT = 10
    API_RATE_LIMIT = 1.0
    
    # Known meme stock symbols for fallback
    KNOWN_MEME_STOCKS = [
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
        
        print(f"🔍 In get_api_status, cls.FINNHUB_KEY is: {repr(cls.FINNHUB_KEY)}")
        
        return {
            'reddit': bool(cls.REDDIT_CLIENT_ID and cls.REDDIT_CLIENT_SECRET),
            'alpha_vantage': bool(cls.ALPHA_VANTAGE_KEY),
            'finnhub': bool(cls.FINNHUB_KEY),
            'yahoo_finance': True
        }

# Check class attribute after class definition
print(f"🔍 After class definition, Config.FINNHUB_KEY is: {repr(Config.FINNHUB_KEY)}")

# Force set it one more time
Config.FINNHUB_KEY = None
print(f"🔧 After forcing class attribute, Config.FINNHUB_KEY is: {repr(Config.FINNHUB_KEY)}")


print("✅ Config file loaded")
