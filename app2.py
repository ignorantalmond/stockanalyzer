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
                            # Fix Reddit sentiment calculation
                            reddit_metrics = result.get('reddit_metrics', {})
                            positive = reddit_metrics.get('positive_mentions', 0)
                            negative = reddit_metrics.get('negative_mentions', 0)
                            total = positive + negative
                            
                            if total > 0:
                                reddit_score = ((positive - negative) / total) * 10
                                # Normalize to 0-10 scale
                                reddit_score = max(0, min(10, (reddit_score + 10) / 2))
                            else:
                                reddit_score = reddit_metrics.get('overall_sentiment_score', 0)
                            
                            st.metric("Reddit Sentiment", f"{reddit_score:.1f}/10")
                        
                        st.markdown("---")
                        
                        # Pump/Dump Detection
                        st.subheader("🚨 Pump/Dump Analysis")
                        pump_dump = result.get('pump_dump_indicators', {})
                        
                        # If pump_dump_indicators doesn't exist, try getting it from phase_metrics
                        if not pump_dump:
                            phase_metrics = result.get('phase_metrics', {})
                            pump_dump = {
                                'likely_pump': phase_metrics.get('is_pumping', False),
                                'volume_spike': phase_metrics.get('volume_spike_ratio', 0) > 2.0,
                                'reddit_spike': phase_metrics.get('reddit_hype_spike', False),
                                'price_momentum': phase_metrics.get('price_momentum', 0)
                            }
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            is_pump = pump_dump.get('likely_pump', False)
                            if is_pump:
                                st.error("⚠️ **PUMP DETECTED** - High risk of sudden drop!")
                            else:
                                st.success("✅ No pump indicators detected")
                            
                            st.write("**Indicators:**")
                            if pump_dump:
                                for indicator, value in pump_dump.items():
                                    if indicator != 'likely_pump':
                                        # Format the display
                                        if isinstance(value, bool):
                                            display_value = "✓ Yes" if value else "✗ No"
                                        elif isinstance(value, (int, float)):
                                            display_value = f"{value:.2f}"
                                        else:
                                            display_value = str(value)
                                        st.write(f"- {indicator.replace('_', ' ').title()}: {display_value}")
                            else:
                                st.write("- No specific indicators available")
                        
                        with col2:
                            # Risk Assessment
                            st.markdown("**Risk Factors:**")
                            risk_assessment = result.get('risk_assessment', {})
                            risk_factors = risk_assessment.get('risk_factors', [])
                            
                            if risk_factors:
                                for factor in risk_factors:
                                    st.warning(f"⚠️ {factor}")
                            else:
                                st.info("No major risk factors identified")
                        
                        st.markdown("---")
                        
                        # Reddit Analysis
                        st.subheader("📱 Reddit Analysis")
                        reddit_metrics = result.get('reddit_metrics', {})
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Calculate total mentions from positive + negative if total_mentions is 0
                            total_mentions = reddit_metrics.get('total_mentions', 0)
                            positive = reddit_metrics.get('positive_mentions', 0)
                            negative = reddit_metrics.get('negative_mentions', 0)
                            
                            if total_mentions == 0 and (positive > 0 or negative > 0):
                                total_mentions = positive + negative
                            
                            unique_posts = reddit_metrics.get('unique_posts', 0)
                            # Calculate unique posts if it's 0
                            if unique_posts == 0 and total_mentions > 0:
                                unique_posts = reddit_metrics.get('post_count', total_mentions)
                            
                            st.metric("Total Mentions", total_mentions)
                            st.metric("Unique Posts", unique_posts)
                        
                        with col2:
                            st.metric("Positive Mentions", positive)
                            st.metric("Negative Mentions", negative)
                        
                        with col3:
                            # Calculate sentiment ratio
                            sentiment_ratio = reddit_metrics.get('sentiment_ratio', 0)
                            if sentiment_ratio == 0 and total_mentions > 0:
                                if negative > 0:
                                    sentiment_ratio = positive / negative
                                else:
                                    sentiment_ratio = positive if positive > 0 else 0
                            
                            # Calculate hype score
                            hype_score = reddit_metrics.get('hype_score', 0)
                            if hype_score == 0 and total_mentions > 0:
                                # Estimate hype score based on mentions
                                hype_score = min(10, (total_mentions / 50) * 5)
                            
                            st.metric("Sentiment Ratio", f"{sentiment_ratio:.2f}")
                            st.metric("Hype Score", f"{hype_score:.1f}/10")
                        
                        # Top Keywords
                        top_keywords = reddit_metrics.get('top_keywords', {})
                        if top_keywords and (top_keywords.get('bullish') or top_keywords.get('bearish')):
                            st.markdown("**Top Keywords:**")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("*Bullish:*")
                                bullish_kw = top_keywords.get('bullish', [])
                                if bullish_kw:
                                    for kw, count in bullish_kw[:5]:
                                        st.write(f"- {kw}: {count}")
                                else:
                                    st.write("- None found")
                            
                            with col2:
                                st.markdown("*Bearish:*")
                                bearish_kw = top_keywords.get('bearish', [])
                                if bearish_kw:
                                    for kw, count in bearish_kw[:5]:
                                        st.write(f"- {kw}: {count}")
                                else:
                                    st.write("- None found")
                        
                        # Only show the "limited activity" message if truly no activity
                        if total_mentions == 0:
                            st.info("💡 Limited Reddit activity detected for this stock.")
                        
                        st.markdown("---")
                        
                        # News Sentiment Analysis
                        st.subheader("📰 News Sentiment Analysis")
                        news_sentiment = result.get('news_sentiment', {})
                        news_data = result.get('news_data')
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            news_count = len(news_data) if news_data is not None and hasattr(news_data, '__len__') else 0
                            st.metric("News Articles", news_count)
                            
                            avg_sentiment = news_sentiment.get('avg_sentiment', 0)
                            sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
                            st.metric("Average Sentiment", sentiment_label)
                        
                        with col2:
                            positive_news = news_sentiment.get('positive_count', 0)
                            negative_news = news_sentiment.get('negative_count', 0)
                            st.metric("Positive News", positive_news)
                            st.metric("Negative News", negative_news)
                        
                        with col3:
                            neutral_news = news_sentiment.get('neutral_count', 0)
                            st.metric("Neutral News", neutral_news)
                            
                            if news_count > 0:
                                news_score = ((positive_news - negative_news) / news_count) * 10
                                news_score = max(0, min(10, (news_score + 10) / 2))
                                st.metric("News Score", f"{news_score:.1f}/10")
                        
                        # Show recent news headlines if available
                        if news_data is not None and hasattr(news_data, 'iterrows') and len(news_data) > 0:
                            with st.expander("📄 Recent News Headlines", expanded=False):
                                for idx, row in news_data.head(5).iterrows():
                                    headline = row.get('headline', row.get('title', 'No headline'))
                                    sentiment = row.get('sentiment', 0)
                                    date = row.get('date', row.get('published_date', 'Unknown'))
                                    
                                    sentiment_emoji = "🟢" if sentiment > 0.1 else "🔴" if sentiment < -0.1 else "🟡"
                                    st.write(f"{sentiment_emoji} **{headline}**")
                                    st.caption(f"Date: {date} | Sentiment: {sentiment:.2f}")
                                    st.markdown("---")
                        
                        st.markdown("---")
                        
                        # Options Data Analysis
                        st.subheader("📊 Options Data & Greeks")
                        options_data = result.get('options_data', {})
                        
                        if options_data and isinstance(options_data, dict) and len(options_data) > 1:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                # Use call_put_volume_ratio (not put_call_ratio)
                                cp_volume_ratio = options_data.get('call_put_volume_ratio', 0)
                                # Convert to put/call ratio (inverse)
                                put_call_ratio = 1 / cp_volume_ratio if cp_volume_ratio > 0 else 0
                                st.metric("Put/Call Ratio", f"{put_call_ratio:.2f}")
                                
                                if put_call_ratio > 1.5:
                                    st.caption("⬇️ Bearish (More Puts)")
                                elif put_call_ratio < 0.7:
                                    st.caption("⬆️ Bullish (More Calls)")
                                else:
                                    st.caption("↔️ Neutral")
                            
                            with col2:
                                # Use total_call_oi and total_put_oi
                                call_oi = options_data.get('total_call_oi', 0)
                                put_oi = options_data.get('total_put_oi', 0)
                                
                                # Handle numpy int types
                                if hasattr(call_oi, 'item'):
                                    call_oi = call_oi.item()
                                if hasattr(put_oi, 'item'):
                                    put_oi = put_oi.item()
                                
                                total_oi = call_oi + put_oi
                                st.metric("Total Open Interest", f"{total_oi:,.0f}")
                                st.caption(f"Calls: {call_oi:,.0f} | Puts: {put_oi:,.0f}")
                            
                            with col3:
                                # Use total_call_volume and total_put_volume
                                call_volume = options_data.get('total_call_volume', 0)
                                put_volume = options_data.get('total_put_volume', 0)
                                
                                # Handle numpy int types
                                if hasattr(call_volume, 'item'):
                                    call_volume = call_volume.item()
                                if hasattr(put_volume, 'item'):
                                    put_volume = put_volume.item()
                                
                                total_volume = call_volume + put_volume
                                st.metric("Options Volume", f"{total_volume:,.0f}")
                                st.caption(f"Calls: {call_volume:,.0f} | Puts: {put_volume:,.0f}")
                            
                            with col4:
                                # Use atm_call_iv and atm_put_iv (average them)
                                atm_call_iv = options_data.get('atm_call_iv', 0)
                                atm_put_iv = options_data.get('atm_put_iv', 0)
                                avg_iv = ((atm_call_iv + atm_put_iv) / 2) * 100  # Convert to percentage
                                
                                st.metric("Implied Volatility", f"{avg_iv:.1f}%")
                                
                                iv_rank = options_data.get('iv_rank', '')
                                if iv_rank == 'EXTREMELY_HIGH':
                                    st.caption("🔥 Extremely High")
                                elif iv_rank == 'HIGH':
                                    st.caption("⚠️ High")
                                elif iv_rank == 'MEDIUM':
                                    st.caption("📊 Medium")
                                else:
                                    st.caption("✅ Normal")
                            
                            # Options Flow Analysis
                            st.markdown("---")
                            st.markdown("**📊 Options Flow Analysis:**")
                            
                            options_flow = options_data.get('options_flow', {})
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                call_volume_pct = options_flow.get('call_volume_pct', 0)
                                put_volume_pct = options_flow.get('put_volume_pct', 0)
                                
                                st.write(f"**Call Volume:** {call_volume_pct:.1f}%")
                                st.write(f"**Put Volume:** {put_volume_pct:.1f}%")
                            
                            with col2:
                                volume_bias = options_flow.get('volume_bias', 'neutral')
                                oi_bias = options_flow.get('oi_bias', 'neutral')
                                
                                st.write(f"**Volume Bias:** {volume_bias.upper()}")
                                st.write(f"**OI Bias:** {oi_bias.upper()}")
                            
                            with col3:
                                unusual_volume = options_flow.get('unusual_volume', False)
                                
                                # Handle numpy bool
                                if hasattr(unusual_volume, 'item'):
                                    unusual_volume = unusual_volume.item()
                                
                                if unusual_volume:
                                    st.error("🚨 **UNUSUAL VOLUME DETECTED**")
                                else:
                                    st.success("✅ Normal Volume")
                            
                            # Strike Price Analysis
                            with st.expander("🎯 Additional Options Details", expanded=False):
                                nearest_expiry = options_data.get('nearest_expiry', 'Unknown')
                                st.write(f"**Nearest Expiration:** {nearest_expiry}")
                                
                                st.write(f"**ATM Call IV:** {atm_call_iv*100:.1f}%")
                                st.write(f"**ATM Put IV:** {atm_put_iv*100:.1f}%")
                                
                                atm_call_vol = options_data.get('atm_call_volume', 0)
                                atm_put_vol = options_data.get('atm_put_volume', 0)
                                
                                if hasattr(atm_call_vol, 'item'):
                                    atm_call_vol = atm_call_vol.item()
                                if hasattr(atm_put_vol, 'item'):
                                    atm_put_vol = atm_put_vol.item()
                                
                                st.write(f"**ATM Call Volume:** {atm_call_vol:,}")
                                st.write(f"**ATM Put Volume:** {atm_put_vol:,}")
                                
                                cp_oi_ratio = options_data.get('call_put_oi_ratio', 0)
                                st.write(f"**Call/Put OI Ratio:** {cp_oi_ratio:.2f}")
                                
                                if call_volume_pct > 70:
                                    st.info("💡 Heavy call buying suggests bullish sentiment")
                                elif put_volume_pct > 70:
                                    st.info("💡 Heavy put buying suggests bearish sentiment or hedging")
                        else:
                            st.info("📊 No options data available for this symbol, or options not actively traded")
                        
                        st.markdown("---")
                        
                        # Technical Indicators
                        st.subheader("📈 Technical Analysis")
                        stock_summary = result.get('stock_summary', {})
                        key_levels = result.get('key_levels', {})
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Price Metrics:**")
                            price_change_1d = stock_summary.get('price_change_1d', 0)
                            price_change_5d = stock_summary.get('price_change_5d', 0)
                            price_change_30d = stock_summary.get('price_change_30d', 0)
                            
                            st.metric("1-Day Change", f"{price_change_1d:+.2%}")
                            st.metric("5-Day Change", f"{price_change_5d:+.2%}")
                            st.metric("30-Day Change", f"{price_change_30d:+.2%}")
                        
                        with col2:
                            st.markdown("**Volume Analysis:**")
                            volume_ratio = stock_summary.get('volume_ratio', 1)
                            avg_volume = stock_summary.get('avg_volume', 0)
                            
                            st.metric("Volume vs Avg", f"{volume_ratio:.2f}x")
                            st.metric("Avg Volume (30d)", f"{avg_volume:,.0f}")
                            
                            if volume_ratio > 2:
                                st.caption("🔥 High volume spike")
                            elif volume_ratio > 1.5:
                                st.caption("⚠️ Above average")
                            else:
                                st.caption("✅ Normal")
                        
                        with col3:
                            st.markdown("**Key Levels:**")
                            support = key_levels.get('support', 0)
                            resistance = key_levels.get('resistance', 0)
                            
                            if support > 0:
                                st.metric("Support", f"${support:.2f}")
                            if resistance > 0:
                                st.metric("Resistance", f"${resistance:.2f}")
                            
                            current_price = result.get('current_price', 0)
                            if support > 0 and resistance > 0 and current_price > 0:
                                range_position = (current_price - support) / (resistance - support)
                                st.metric("Range Position", f"{range_position:.1%}")
                        
                        # Moving Averages
                        with st.expander("📊 Moving Averages", expanded=False):
                            ma_20 = stock_summary.get('ma_20', 0)
                            ma_50 = stock_summary.get('ma_50', 0)
                            ma_200 = stock_summary.get('ma_200', 0)
                            current_price = result.get('current_price', 0)
                            
                            if ma_20 > 0:
                                st.write(f"**20-Day MA:** ${ma_20:.2f} ({((current_price - ma_20) / ma_20 * 100):+.1f}%)")
                            if ma_50 > 0:
                                st.write(f"**50-Day MA:** ${ma_50:.2f} ({((current_price - ma_50) / ma_50 * 100):+.1f}%)")
                            if ma_200 > 0:
                                st.write(f"**200-Day MA:** ${ma_200:.2f} ({((current_price - ma_200) / ma_200 * 100):+.1f}%)")
                            
                            # Trend analysis
                            if ma_20 > 0 and ma_50 > 0:
                                if current_price > ma_20 > ma_50:
                                    st.success("✅ Bullish trend - price above key MAs")
                                elif current_price < ma_20 < ma_50:
                                    st.error("📉 Bearish trend - price below key MAs")
                                else:
                                    st.info("↔️ Mixed signals from moving averages")
                        
                        st.markdown("---")
                        
                        # Phase Analysis Details
                        st.subheader("🔄 Detailed Phase Analysis")
                        phase_metrics = result.get('phase_metrics', {})
                        timeline = result.get('timeline_estimate', {})
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Phase Confidence:**")
                            phase_confidence = result.get('phase_confidence', 0)
                            st.progress(phase_confidence)
                            st.caption(f"{phase_confidence:.1%} confidence")
                            
                            st.markdown("**Phase Metrics:**")
                            if phase_metrics:
                                for key, value in list(phase_metrics.items())[:5]:
                                    if isinstance(value, (int, float)):
                                        st.write(f"• {key.replace('_', ' ').title()}: {value:.2f}")
                                    elif isinstance(value, bool):
                                        st.write(f"• {key.replace('_', ' ').title()}: {'Yes' if value else 'No'}")
                        
                        with col2:
                            st.markdown("**Timeline Estimate:**")
                            if timeline:
                                current_phase_age = timeline.get('current_phase_age_days', 0)
                                typical_duration = timeline.get('typical_phase_duration', 'Unknown')
                                next_phase = timeline.get('likely_next_phase', 'Unknown')
                                
                                st.write(f"• Current phase age: {current_phase_age:.0f} days")
                                st.write(f"• Typical duration: {typical_duration}")
                                st.write(f"• Likely next phase: {next_phase}")
                            else:
                                st.info("Timeline data not available")
                        # Options Strategy
                        st.subheader("📊 Recommended Options Strategy")
                        strategy = result.get('options_strategy', {})
                        
                        if strategy and (strategy.get('primary_strategy') or strategy.get('recommended_strategies')):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Primary Strategy:**")
                                primary = strategy.get('primary_strategy', 'N/A')
                                
                                # Handle if primary_strategy is an object with value attribute
                                if hasattr(primary, 'value'):
                                    primary = primary.value
                                
                                if primary and primary != 'N/A':
                                    st.info(f"🎯 {primary}")
                                else:
                                    # Try to get from recommended_strategies list
                                    recommended = strategy.get('recommended_strategies', [])
                                    if recommended:
                                        st.info(f"🎯 {recommended[0]}")
                                    else:
                                        st.info("🎯 Based on market phase analysis")
                                
                                st.markdown("**Alternative Strategy:**")
                                alternative = strategy.get('alternative_strategy', 'N/A')
                                
                                # Handle if alternative_strategy is an object
                                if hasattr(alternative, 'value'):
                                    alternative = alternative.value
                                
                                if alternative and alternative != 'N/A':
                                    st.info(f"🔄 {alternative}")
                                else:
                                    # Try to get second from recommended_strategies
                                    recommended = strategy.get('recommended_strategies', [])
                                    if len(recommended) > 1:
                                        st.info(f"🔄 {recommended[1]}")
                                    else:
                                        st.info("🔄 Monitor for phase changes")
                            
                            with col2:
                                st.markdown("**Strategy Details:**")
                                details = strategy.get('strategy_details', {})
                                reasoning = strategy.get('reasoning', '')
                                
                                if details:
                                    for key, value in details.items():
                                        st.write(f"- **{key}:** {value}")
                                elif reasoning:
                                    st.write(f"- {reasoning}")
                                else:
                                    phase = result.get('phase')
                                    phase_name = phase.value if hasattr(phase, 'value') else str(phase)
                                    st.write(f"- Current Phase: {phase_name}")
                                    st.write(f"- Risk Level: {strategy.get('overall_risk_level', 'UNKNOWN')}")
                                    st.write("- Strategy aligned with accumulation phase")
                            
                            # Warning box
                            risk_level = strategy.get('overall_risk_level')
                            risk_value = risk_level.value if hasattr(risk_level, 'value') else str(risk_level)
                            
                            if 'HIGH' in risk_value.upper():
                                st.error("⚠️ **HIGH RISK** - Exercise extreme caution with this stock!")
                        else:
                            # Fallback display based on phase
                            phase = result.get('phase')
                            phase_name = phase.value if hasattr(phase, 'value') else str(phase)
                            
                            st.info(f"🎯 **Strategy based on {phase_name} phase:**")
                            
                            if 'ACCUMULATION' in phase_name.upper():
                                st.write("• Consider longer-term positions")
                                st.write("• Wait for clear momentum before aggressive strategies")
                                st.write("• Selling cash-secured puts at support levels")
                            elif 'PUMP' in phase_name.upper():
                                st.write("• High risk - consider profit taking")
                                st.write("• Protective puts for existing positions")
                                st.write("• Avoid new long positions")
                            else:
                                st.write("• Monitor closely for opportunities")
                                st.write("• Manage risk carefully")
                        
                        st.markdown("---")
                        
                        # Price Chart
                        st.subheader("📈 Price History")
                        
                        # Try to get price data from stock_data DataFrame
                        stock_data = result.get('stock_data')
                        
                        if stock_data is not None and not stock_data.empty:
                            try:
                                # Use the DataFrame directly
                                fig = go.Figure()
                                
                                # Get the close prices
                                if 'Close' in stock_data.columns:
                                    dates = stock_data.index
                                    prices = stock_data['Close']
                                    
                                    fig.add_trace(go.Scatter(
                                        x=dates,
                                        y=prices,
                                        mode='lines',
                                        name='Price',
                                        line=dict(color='#00ff00', width=2),
                                        fill='tozeroy',
                                        fillcolor='rgba(0, 255, 0, 0.1)'
                                    ))
                                    
                                    # Add volume as bar chart on secondary y-axis if available
                                    if 'Volume' in stock_data.columns:
                                        fig.add_trace(go.Bar(
                                            x=dates,
                                            y=stock_data['Volume'],
                                            name='Volume',
                                            yaxis='y2',
                                            marker=dict(color='rgba(100, 100, 255, 0.3)')
                                        ))
                                    
                                    fig.update_layout(
                                        title=f"{symbol_input} Price History (90 Days)",
                                        xaxis_title="Date",
                                        yaxis_title="Price ($)",
                                        yaxis2=dict(
                                            title="Volume",
                                            overlaying='y',
                                            side='right'
                                        ),
                                        template="plotly_dark",
                                        height=400,
                                        hovermode='x unified'
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Price data available but in unexpected format")
                                    
                            except Exception as e:
                                st.error(f"Error creating price chart: {e}")
                        else:
                            # Try alternative price_history field
                            price_history = result.get('price_history', {})
                            
                            if price_history and price_history.get('dates') and price_history.get('prices'):
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=price_history['dates'],
                                    y=price_history['prices'],
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
                                st.info("📊 Price chart data not available in current analysis")
                        
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
                        predictions = analyzer.generate_stock_predictions(analysis)
                        #st.write("DEBUG - Predictions structure:", predictions)
                        #st.write("DEBUG - Keys in predictions:", predictions.keys() if predictions else "None")
                        if predictions and predictions.get('predictions'):
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
                                pred = predictions['predictions'].get(period, {})
                                
                                if not pred:
                                    continue
                                
                                st.subheader(f"📅 {period.replace('_', '-').upper()} Prediction")
                                
                                # Get scenarios
                                scenarios = pred.get('scenarios', {})
                                summary = pred.get('summary', {})
                                
                                # Find bullish, base, and bearish scenarios
                                bullish_scenario = None
                                base_scenario = None
                                bearish_scenario = None
                                
                                for scenario_name, scenario_data in scenarios.items():
                                    direction = scenario_data.get('direction', '')
                                    if direction == 'up':
                                        bullish_scenario = scenario_data
                                    elif direction == 'sideways':
                                        base_scenario = scenario_data
                                    elif direction == 'down':
                                        bearish_scenario = scenario_data
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown("**Bullish Scenario** 🚀")
                                    if bullish_scenario:
                                        st.metric(
                                            "Target Price",
                                            f"${bullish_scenario.get('final_price', 0):.2f}",
                                            delta=f"+{bullish_scenario.get('total_return', 0)*100:.1f}%"
                                        )
                                        st.write(f"Probability: {bullish_scenario.get('probability', 0)*100:.0f}%")
                                    else:
                                        st.write("No bullish scenario")
                                
                                with col2:
                                    st.markdown("**Base Scenario** 📊")
                                    if base_scenario:
                                        st.metric(
                                            "Target Price",
                                            f"${base_scenario.get('final_price', 0):.2f}",
                                            delta=f"{base_scenario.get('total_return', 0)*100:+.1f}%"
                                        )
                                        st.write(f"Probability: {base_scenario.get('probability', 0)*100:.0f}%")
                                    else:
                                        st.write("No base scenario")
                                
                                with col3:
                                    st.markdown("**Bearish Scenario** 🐻")
                                    if bearish_scenario:
                                        st.metric(
                                            "Target Price",
                                            f"${bearish_scenario.get('final_price', 0):.2f}",
                                            delta=f"{bearish_scenario.get('total_return', 0)*100:.1f}%"
                                        )
                                        st.write(f"Probability: {bearish_scenario.get('probability', 0)*100:.0f}%")
                                    else:
                                        st.write("No bearish scenario")
                                
                                # Key factors - use the scenario descriptions
                                st.markdown("**Key Factors:**")
                                for scenario_name, scenario_data in scenarios.items():
                                    prob = scenario_data.get('probability', 0) * 100
                                    desc = scenario_data.get('description', '')
                                    st.write(f"• {scenario_name} ({prob:.0f}%): {desc}")
                                
                                # Confidence score
                                confidence = predictions['confidence']
                                confidence_value = confidence.get('overall', 0)
                                st.progress(confidence_value)
                                st.caption(f"Confidence: {confidence.get('level', 'UNKNOWN')} ({confidence_value*100:.0f}%)")
                                
                                st.markdown("---")
                            
                            # Prediction Chart
                            st.subheader("📊 Visual Prediction")
                            
                            current_price = predictions['current_price']
                            
                            # Prepare data for chart - use actual prediction data
                            days_7 = list(range(8))  # 0-7 days
                            days_14 = list(range(15))  # 0-14 days
                            days_21 = list(range(22))  # 0-21 days
                            
                            # Get the actual price paths from predictions
                            pred_7 = predictions['predictions'].get('7_day', {})
                            pred_14 = predictions['predictions'].get('14_day', {})
                            pred_21 = predictions['predictions'].get('21_day', {})
                            
                            # Determine which timeframe to show based on available data
                            # Default to 21-day if available, otherwise 14-day, otherwise 7-day
                            if pred_21.get('scenarios'):
                                days = days_21
                                scenarios = pred_21['scenarios']
                                title_days = 21
                            elif pred_14.get('scenarios'):
                                days = days_14
                                scenarios = pred_14['scenarios']
                                title_days = 14
                            else:
                                days = days_7
                                scenarios = pred_7['scenarios']
                                title_days = 7
                            
                            # Create chart
                            fig = go.Figure()
                            
                            # Find and plot each scenario type
                            for scenario_name, scenario_data in scenarios.items():
                                direction = scenario_data.get('direction', '')
                                price_path = scenario_data.get('price_path', [])
                                
                                if not price_path:
                                    continue
                                
                                # Determine color and name based on direction
                                if direction == 'up':
                                    color = '#00ff00'
                                    display_name = 'Bullish'
                                elif direction == 'sideways':
                                    color = '#ffaa00'
                                    display_name = 'Base'
                                elif direction == 'down':
                                    color = '#ff0000'
                                    display_name = 'Bearish'
                                else:
                                    color = '#888888'
                                    display_name = scenario_name
                                
                                # Add trace for this scenario
                                fig.add_trace(go.Scatter(
                                    x=list(range(len(price_path))),
                                    y=price_path,
                                    mode='lines+markers',
                                    name=display_name,
                                    line=dict(color=color, width=3),
                                    marker=dict(size=8),
                                    hovertemplate=f'<b>{display_name}</b><br>' +
                                                  'Day: %{x}<br>' +
                                                  'Price: $%{y:.2f}<br>' +
                                                  '<extra></extra>'
                                ))
                            
                            fig.update_layout(
                                title=f"{predict_symbol} Price Predictions ({title_days} Days)",
                                xaxis_title="Days from Today",
                                yaxis_title="Price ($)",
                                template="plotly_dark",
                                height=500,
                                hovermode='x unified',
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01
                                )
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