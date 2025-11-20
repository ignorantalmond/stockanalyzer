"""
Direct Finnhub API test - no config dependencies
"""

import finnhub

def test_finnhub_direct():
    """Test Finnhub API directly"""
    
    api_key = "d2m76s9r01qq6fop9okgd2m76s9r01qq6fop9ol0"
    print(f"Testing API key: {api_key}")
    print(f"API key length: {len(api_key)}")
    
    try:
        # Create client
        client = finnhub.Client(api_key=api_key)
        print("✅ Finnhub client created")
        
        # Test API call
        profile = client.company_profile2(symbol='AAPL')
        print(f"✅ API call successful!")
        print(f"Company name: {profile.get('name', 'Unknown')}")
        print(f"Response keys: {list(profile.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        print(f"Error type: {type(e)}")
        return False

if __name__ == "__main__":
    test_finnhub_direct()