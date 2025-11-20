"""
Streamlit Web App for Meme Stock Analyzer
Users provide their own API keys - no storage needed
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Import your existing analyzer
from analyzer.meme_stock_analyzer import MemeStockAnalyzer

# Page config
st.set_page_config(
    page_title="Meme Stock Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🚀 Meme Stock & Options Analyzer")
st.markdown("### Analyze meme stocks with pump/dump detection and predictive analysis")

# Sidebar for API Keys
st.sidebar.header("🔑 API Configuration")
st.sidebar.markdown("Enter your own API keys to use your quota:")

# Reddit API Keys
with st.sidebar.expander("📱 Reddit API Keys (Required)", expanded=True):
    st.markdown("[Get Reddit API Keys](https://www.reddit.com/prefs/apps)")
    reddit_client_id = st.text_input(
        "Reddit Client ID", 
        type="password",
        help="Your Reddit application client ID",
        key="reddit_id"
    )
    reddit_client_secret = st.text_input(
        "Reddit Client Secret", 
        type="password",
        help="Your Reddit application secret",
        key="reddit_secret"
    )
    reddit_user_agent = st.text_input(
        "Reddit User Agent",
        value="MemeStockAnalyzer/2.0",
        help="Format: AppName/Version",
        key="reddit_agent"
    )

# Optional API Keys
with st.sidebar.expander("📊 Optional APIs", expanded=False):
    st.markdown("[Get Finnhub Key](https://finnhub.io/)")
    finnhub_key = st.text_input(
        "Finnhub API Key (Optional)", 
        type="password",
        help="For enhanced news analysis",
        key="finnhub"
    )
    
    st.markdown("[Get Alpha Vantage Key](https://www.alphavantage.co/)")
    alpha_vantage_key = st.text_input(
        "Alpha Vantage Key (Optional)", 
        type="password",
        help="For additional news sources",
        key="alpha_vantage"
    )

# Validation button
if st.sidebar.button("✅ Validate API Keys", use_container_width=True):
    if not reddit_client_id or not reddit_client_secret:
        st.sidebar.error("❌ Reddit API keys are required!")
    else:
        try:
            # Initialize analyzer with provided keys
            config = {
                'reddit_client_id': reddit_client_id,
                'reddit_client_secret': reddit_client_secret,
                'reddit_user_agent': reddit_user_agent,
                'alpha_vantage_key': alpha_vantage_key if alpha_vantage_key else None,
                'finnhub_key': finnhub_key if finnhub_key else None
            }
            
            analyzer = MemeStockAnalyzer(
                reddit_client_id=config['reddit_client_id'],
                reddit_client_secret=config['reddit_client_secret'],
                reddit_user_agent=config['reddit_user_agent'],
                alpha_vantage_key=config['alpha_vantage_key'],
                finnhub_key=config['finnhub_key']
            )
            
            st.session_state['analyzer'] = analyzer
            st.session_state['api_valid'] = True
            st.sidebar.success("✅ API Keys validated successfully!")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error validating keys: {str(e)}")
            st.session_state['api_valid'] = False

# Main interface tabs
if 'api_valid' in st.session_state and st.session_state['api_valid']:
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analyze Stock", 
        "🔥 Trending Stocks", 
        "📈 Predictions",
        "ℹ️ About"
    ])
    
    # TAB 1: Analyze Stock
    with tab1:
        st.header("Analyze Individual Stock")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            symbol_input = st.text_input(
                "Enter Stock Symbol:",
                placeholder="e.g., GME, AMC, TSLA",
                key="symbol_input"
            ).upper()
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_button = st.button("🔍 Analyze", use_container_width=True, type="primary")
        
        if analyze_button and symbol_input:
            with st.spinner(f"🔄 Analyzing {symbol_input}..."):
                try:
                    analyzer = st.session_state['analyzer']
                    result = analyzer.analyze_stock(symbol_input)
                    
                    if result:
                        st.session_state['current_analysis'] = result
                        st.success(f"✅ Analysis complete for {result['symbol']}")
                        
                        # Display results
                        st.markdown("---")
                        
                        # Key Metrics Row
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Current Price",
                                f"${result['current_price']:.2f}",
                                delta=f"{result.get('price_change_percent', 0):.2f}%"
                            )
                        
                        with col2:
                            phase = result['phase'].value if hasattr(result['phase'], 'value') else result['phase']
                            st.metric("Market Phase", phase)
                        
                        with col3:
                            risk = result.get('options_strategy', {}).get('overall_risk_level', 'UNKNOWN')
                            risk_val = risk.value if hasattr(risk, 'value') else risk
                            st.metric("Risk Level", risk_val)
                        with col4:
                            reddit_score = result.get('reddit_analysis', {}).get('overall_sentiment_score', 0)
                            st.metric("Reddit Sentiment", f"{reddit_score:.1f}/10")
                        
                        st.markdown("---")
                        
                        # Pump/Dump Detection
                        st.subheader("🚨 Pump/Dump Analysis")
                        pump_dump = result.get('pump_dump_indicators', {})
                    
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            is_pump = pump_dump.get('likely_pump', False)
                            if is_pump:
                                st.error("⚠️ **PUMP DETECTED** - High risk of sudden drop!")
                            else:
                                st.success("✅ No pump indicators detected")
                            
                            st.write("**Indicators:**")
                            for indicator, value in pump_dump.items():
                                if indicator != 'likely_pump':
                                    st.write(f"- {indicator}: {value}")
                        
                        with col2:
                            # Risk Assessment
                            st.markdown("**Risk Factors:**")
                            risk_factors = result.get('risk_factors', [])
                            if risk_factors:
                                for factor in risk_factors:
                                    st.warning(f"⚠️ {factor}")
                            else:
                                st.info("No major risk factors identified")
                        
                        st.markdown("---")
                        
                        # Reddit Analysis
                        st.subheader("📱 Reddit Analysis")
                        reddit = result.get('reddit_analysis', {})
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Mentions", reddit.get('total_mentions', 0))
                            st.metric("Unique Posts", reddit.get('unique_posts', 0))
                        
                        with col2:
                            st.metric("Positive Mentions", reddit.get('positive_mentions', 0))
                            st.metric("Negative Mentions", reddit.get('negative_mentions', 0))
                        
                        with col3:
                            sentiment_ratio = reddit.get('sentiment_ratio', 0)
                            st.metric("Sentiment Ratio", f"{sentiment_ratio:.2f}")
                            st.metric("Hype Score", f"{reddit.get('hype_score', 0):.1f}/10")
                        
                        # Top Keywords
                        if 'top_keywords' in reddit:
                            st.markdown("**Top Keywords:**")
                            keywords = reddit['top_keywords']
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("*Bullish:*")
                                for kw, count in keywords.get('bullish', [])[:5]:
                                    st.write(f"- {kw}: {count}")
                            
                            with col2:
                                st.markdown("*Bearish:*")
                                for kw, count in keywords.get('bearish', [])[:5]:
                                    st.write(f"- {kw}: {count}")
                        
                        st.markdown("---")
                        
                        # Options Strategy
                        st.subheader("📊 Recommended Options Strategy")
                        strategy = result.get('options_strategy', {})
                        
                        col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Primary Strategy:**")
                        primary = strategy.get('primary_strategy', 'N/A')
                        st.info(f"🎯 {primary}")
                        
                        st.markdown("**Alternative Strategy:**")
                        alternative = strategy.get('alternative_strategy', 'N/A')
                        st.info(f"🔄 {alternative}")
                    
                    with col2:
                        st.markdown("**Strategy Details:**")
                        details = strategy.get('strategy_details', {})
                        for key, value in details.items():
                            st.write(f"- **{key}:** {value}")
                        
                        # Warning box
                        if strategy.get('overall_risk_level') == 'HIGH':
                            st.error("⚠️ **HIGH RISK** - Exercise extreme caution with this stock!")
                        
                        st.markdown("---")
                        
                        # Price Chart
                        st.subheader("📈 Price History")
                        if 'price_history' in result:
                            price_data = result['price_history']
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=price_data['dates'],
                                y=price_data['prices'],
                                mode='lines',
                                name='Price',
                                line=dict(color='#00ff00', width=2)
                            ))
                            
                            fig.update_layout(
                                title=f"{symbol_input} Price History",
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                template="plotly_dark",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        else:
                            st.error(f"❌ Could not analyze {symbol_input}. Please check the symbol and try again.")
                            
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.exception(e)
    
    # TAB 2: Trending Stocks
    with tab2:
        st.header("🔥 Trending Meme Stocks")
        st.markdown("Scan Reddit for the most mentioned stocks right now")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            timeframe = st.selectbox(
                "Timeframe:",
                ["hour", "day", "week"],
                index=1
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            scan_button = st.button("🔍 Scan Trending", use_container_width=True, type="primary")
        
        if scan_button:
            with st.spinner(f"🔄 Scanning Reddit for trending stocks..."):
                try:
                    analyzer = st.session_state['analyzer']
                    trending = analyzer.scan_trending_stocks(timeframe=timeframe, limit=10)
                    
                    if trending:
                        st.success(f"✅ Found {len(trending)} trending stocks")
                        
                        # Create DataFrame for display
                        df_data = []
                        for stock in trending:
                            df_data.append({
                                'Symbol': stock['symbol'],
                                'Mentions': stock['mention_count'],
                                'Sentiment': f"{stock['sentiment_score']:.1f}/10",
								'Phase': stock.get('phase', 'N/A'),
                                'Risk': stock.get('risk_level', 'N/A'),
                                'Price': f"${stock.get('current_price', 0):.2f}"
                            })
                        
                        df = pd.DataFrame(df_data)
                        
                        # Display as interactive table
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Visualization
                        st.markdown("---")
                        st.subheader("📊 Mention Distribution")
                        
                        fig = px.bar(
                            df,
                            x='Symbol',
                            y='Mentions',
                            color='Sentiment',
                            title="Stock Mentions on Reddit",
                            template="plotly_dark"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Store trending data
                        st.session_state['trending_stocks'] = trending
                        
                    else:
                        st.warning("No trending stocks found. Try a different timeframe.")
                        
                except Exception as e:
                    st.error(f"❌ Error scanning trending stocks: {str(e)}")
                    st.exception(e)
        
        # Display previously scanned trending stocks
        if 'trending_stocks' in st.session_state:
            st.markdown("---")
            st.markdown("**💡 Tip:** Click 'Analyze Stock' tab and enter a symbol for detailed analysis")
    
    # TAB 3: Predictions
    with tab3:
        st.header("📈 Price Movement Predictions")
        st.markdown("Get AI-powered predictions for 7, 14, and 21-day price movements")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            predict_symbol = st.text_input(
                "Enter Stock Symbol for Prediction:",
                placeholder="e.g., GME, AMC, TSLA",
                key="predict_symbol"
            ).upper()
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_button = st.button("🔮 Predict", use_container_width=True, type="primary")
        
        if predict_button and predict_symbol:
            with st.spinner(f"🔄 Generating predictions for {predict_symbol}..."):
                try:
                    analyzer = st.session_state['analyzer']
                    
                    # First get full analysis
                    analysis = analyzer.analyze_stock(predict_symbol)
                    
                    if analysis:
                        # Get predictions
                        predictions = analyzer.get_price_predictions(predict_symbol, analysis)
                        
                        if predictions:
                            st.success(f"✅ Predictions generated for {predict_symbol}")
                            
                            # Current price info
                            st.metric(
                                "Current Price",
                                f"${analysis['current_price']:.2f}",
                                delta=f"{analysis.get('price_change_percent', 0):.2f}%"
                            )
                            
                            st.markdown("---")
                            
                            # Display predictions for each timeframe
                            for period in ['7_day', '14_day', '21_day']:
                                pred = predictions.get(period, {})
                                
                                st.subheader(f"📅 {period.replace('_', '-').upper()} Prediction")
                                
                                col1, col2, col3 = st.columns(3)
								with col1:
                                    st.markdown("**Bullish Scenario** 🚀")
                                    bullish = pred.get('bullish', {})
                                    st.metric(
                                        "Target Price",
                                        f"${bullish.get('price', 0):.2f}",
                                        delta=f"+{bullish.get('change_percent', 0):.1f}%"
                                    )
                                    st.write(f"Probability: {bullish.get('probability', 0):.0f}%")
                                
                                with col2:
                                    st.markdown("**Base Scenario** 📊")
                                    base = pred.get('base', {})
                                    st.metric(
                                        "Target Price",
                                        f"${base.get('price', 0):.2f}",
                                        delta=f"{base.get('change_percent', 0):+.1f}%"
                                    )
                                    st.write(f"Probability: {base.get('probability', 0):.0f}%")
                                
                                with col3:
                                    st.markdown("**Bearish Scenario** 🐻")
                                    bearish = pred.get('bearish', {})
                                    st.metric(
                                        "Target Price",
                                        f"${bearish.get('price', 0):.2f}",
                                        delta=f"{bearish.get('change_percent', 0):.1f}%"
                                    )
                                    st.write(f"Probability: {bearish.get('probability', 0):.0f}%")
                                
                                # Key factors
                                st.markdown("**Key Factors:**")
                                factors = pred.get('key_factors', [])
                                for factor in factors:
                                    st.write(f"• {factor}")
                                
                                # Confidence score
                                confidence = pred.get('confidence_score', 0)
                                st.progress(confidence / 10)
                                st.caption(f"Confidence: {confidence:.1f}/10")
                                
                                st.markdown("---")
                            
                            # Prediction Chart
                            st.subheader("📊 Visual Prediction")
                            
                            current_price = analysis['current_price']
                            
                            # Prepare data for chart
                            days = [0, 7, 14, 21]
                            bullish_prices = [current_price]
                            base_prices = [current_price]
                            bearish_prices = [current_price]
                            
                            for period in ['7_day', '14_day', '21_day']:
                                pred = predictions.get(period, {})
                                bullish_prices.append(pred.get('bullish', {}).get('price', current_price))
                                base_prices.append(pred.get('base', {}).get('price', current_price))
                                bearish_prices.append(pred.get('bearish', {}).get('price', current_price))
                            
                            # Create chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=days,
                                y=bullish_prices,
                                mode='lines+markers',
                                name='Bullish',
                                line=dict(color='#00ff00', width=3),
                                marker=dict(size=8)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=days,
                                y=base_prices,
                                mode='lines+markers',
                                name='Base',
                                line=dict(color='#ffaa00', width=3),
                                marker=dict(size=8)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=days,
                                y=bearish_prices,
                                mode='lines+markers',
                                name='Bearish',
								line=dict(color='#ff0000', width=3),
                                marker=dict(size=8)
                            ))
                            
                            fig.update_layout(
                                title=f"{predict_symbol} Price Predictions",
                                xaxis_title="Days from Today",
                                yaxis_title="Price ($)",
                                template="plotly_dark",
                                height=500,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Risk Warning
                            st.warning("""
                            ⚠️ **Disclaimer:** These predictions are based on current market data, 
                            sentiment analysis, and historical patterns. They are NOT financial advice. 
                            Market conditions can change rapidly, especially with volatile meme stocks.
                            """)
                            
                        else:
                            st.error("❌ Could not generate predictions. Please try again.")
                    else:
                        st.error(f"❌ Could not analyze {predict_symbol}. Please check the symbol.")
                        
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {str(e)}")
                    st.exception(e)
    
    # TAB 4: About
    with tab4:
        st.header("ℹ️ About Meme Stock Analyzer")
        
        st.markdown("""
        ### 🚀 What is this?
        
        This tool analyzes meme stocks using multiple data sources including:
        
        - **Reddit Analysis**: Scans 25+ subreddits for stock mentions and sentiment
        - **Price Data**: Real-time and historical price information
        - **News Analysis**: Recent news sentiment and impact assessment
        - **Pump/Dump Detection**: Identifies potential pump and dump schemes
        - **Predictive Analysis**: AI-powered price predictions for 7, 14, and 21 days
        - **Options Strategies**: Recommended trading strategies based on risk level
        
        ### 📊 Data Sources
        
        - **Reddit**: r/wallstreetbets, r/pennystocks, r/stocks, and 20+ other communities
        - **Stock Data**: Yahoo Finance, Alpha Vantage (optional)
        - **News**: Finnhub, Alpha Vantage (optional)
        
        ### 🔑 API Keys
        
        **Required:**
        - Reddit API (Free) - [Get it here](https://www.reddit.com/prefs/apps)
        
        **Optional (for enhanced features):**
        - Finnhub API (Free tier available) - [Get it here](https://finnhub.io/)
        - Alpha Vantage API (Free tier available) - [Get it here](https://www.alphavantage.co/)
        
        ### 🔒 Privacy & Security
        
        - Your API keys are stored ONLY in your browser session
        - No keys are saved to any server or database
        - All analysis happens in real-time using YOUR API quotas
        - When you close the browser, all keys are deleted
        
        ### ⚠️ Risk Disclaimer
        
        **IMPORTANT:** This tool is for educational and informational purposes only.
        
        - This is NOT financial advice
        - Meme stocks are extremely volatile and risky
        - You can lose your entire investment
        - Past performance does not guarantee future results
        - Options trading involves significant risk
        - Always do your own research (DYOR)
        - Consider consulting a licensed financial advisor
        
        ### 🛠️ Technical Details
        
        **Built with:**
        - Python 3.8+
        - Streamlit for web interface
        - Plotly for interactive charts
        - PRAW for Reddit API
		- yfinance for stock data
        - pandas for data analysis
        
        **Features:**
        - Real-time Reddit sentiment analysis
        - Multi-subreddit monitoring (25+ communities)
        - Pump/dump pattern detection
        - Phase-aware price predictions
        - Risk-adjusted options strategies
        - Interactive charts and visualizations
        
        ### 📝 How to Use
        
        1. **Enter API Keys**: Add your Reddit API credentials in the sidebar (required)
        2. **Validate Keys**: Click "Validate API Keys" to connect
        3. **Analyze Stocks**: 
           - Use "Analyze Stock" tab for detailed individual analysis
           - Use "Trending Stocks" tab to scan Reddit for hot stocks
           - Use "Predictions" tab for price movement forecasts
        
        ### 🤝 Subreddits Monitored
        
        - r/wallstreetbets
        - r/pennystocks
        - r/RobinHoodPennyStocks
        - r/stocks
        - r/investing
        - r/options
        - r/SecurityAnalysis
        - r/StockMarket
        - r/smallstreetbets
        - r/Daytrading
        - r/swingtrading
        - r/wallstreetbetsnew
        - r/Superstonk
        - r/GME
        - r/amcstock
        - r/SatoshiStreetBets
        - r/CryptoMoonShots
        - r/CryptoCurrency
        - And more...
        
        ### 📈 Analysis Components
        
        **Pump/Dump Detection:**
        - Sudden volume spikes
        - Coordinated posting patterns
        - New account activity
        - Cross-platform coordination
        - Penny stock patterns
        
        **Sentiment Analysis:**
        - Bullish keyword tracking (moon, rocket, calls, etc.)
        - Bearish keyword tracking (puts, crash, dump, etc.)
        - Post classification (bullish/bearish/neutral)
        - Weighted scoring by subreddit relevance
        
        **Predictive Modeling:**
        - Phase-aware scenarios (pre-pump, pumping, dumping, recovery)
        - Sentiment integration from Reddit and news
        - Historical pattern analysis
        - Probability-weighted outcomes
        - Confidence scoring
        
        ### 📧 Support & Feedback
        
        For questions, issues, or feature requests, please contact the developer.
        
        ### ⚖️ Legal
        
        By using this tool, you acknowledge that:
        - You are using it at your own risk
        - The creators are not responsible for any financial losses
        - This is not a recommendation to buy or sell securities
        - You should comply with all applicable laws and regulations
        """)
        
        st.markdown("---")
        st.markdown("**Version:** 2.0 | **Last Updated:** November 2025")

else:
    # Show instructions when API keys not validated
    st.info("👈 **Getting Started:** Enter your API keys in the sidebar to begin!")
    
    st.markdown("""
    ### 🚀 Quick Start Guide
    
    1. **Get Reddit API Keys** (Required):
       - Go to https://www.reddit.com/prefs/apps
       - Click "create another app" at the bottom
       - Choose "script" type
       - Name it anything (e.g., "MemeStockAnalyzer")
       - Copy your **Client ID** and **Client Secret**
    
    2. **Optional APIs** (for enhanced features):
       - Finnhub: https://finnhub.io/ (free tier available)
       - Alpha Vantage: https://www.alphavantage.co/ (free tier available)
    
    3. **Enter Keys**: Paste them in the sidebar and click "Validate"
    
    4. **Start Analyzing**: Use the tabs above to analyze stocks!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. 
    Not financial advice. Trading involves risk of loss.</p>
</div>
""", unsafe_allow_html=True)