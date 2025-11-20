"""
Finnhub API Test Script
Run this to verify your Finnhub connection is working
"""

def test_finnhub_connection():
    """Test Finnhub API connection"""
    
    # Test 1: Check if library is installed
    print("🧪 Testing Finnhub Integration...")
    print("=" * 50)
    
    try:
        import finnhub
        print("✅ Finnhub library imported successfully")
    except ImportError as e:
        print("❌ Finnhub library not found!")
        print("   Install with: pip install finnhub-python")
        return False
    
    # Test 2: Check API key from config
    try:
        from analyzer.config import Config
        config = Config()
        api_key = config.FINNHUB_KEY
        print(f"📋 API Key from config: {api_key}")
        print(f"📋 API Key valid format: {bool(api_key and len(api_key) > 10)}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False
    
    # Test 3: Initialize Finnhub client
    try:
        client = finnhub.Client(api_key=api_key)
        print("✅ Finnhub client created successfully")
    except Exception as e:
        print(f"❌ Error creating Finnhub client: {e}")
        return False
    
    # Test 4: Make a simple API call
    try:
        print("\n🔍 Testing API call with AAPL...")
        
        # Test basic company profile
        profile = client.company_profile2(symbol='AAPL')
        if profile and 'name' in profile:
            print(f"✅ Company profile: {profile['name']}")
        else:
            print("❌ Empty company profile response")
            return False
            
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False
    
    # Test 5: Test news API call
    try:
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        news = client.company_news('AAPL', 
                                  _from=start_date.strftime('%Y-%m-%d'), 
                                  to=end_date.strftime('%Y-%m-%d'))
        
        if news and len(news) > 0:
            print(f"✅ News API: Found {len(news)} articles for AAPL")
            print(f"   Sample article: {news[0].get('headline', 'No headline')[:50]}...")
        else:
            print("⚠️  News API: No articles found (might be normal)")
            
    except Exception as e:
        print(f"❌ News API call failed: {e}")
        print(f"   This might be due to API limits or invalid dates")
        
    print("\n🎉 Finnhub connection test completed!")
    print("If you see mostly ✅ marks above, Finnhub should work in your analyzer.")
    return True

if __name__ == "__main__":
    test_finnhub_connection()