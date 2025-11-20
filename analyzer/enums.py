"""
Enumerations and constants for the Meme Stock Analyzer
"""

from enum import Enum

class PumpDumpPhase(Enum):
    """Enum for different phases of pump and dump cycle"""
    ACCUMULATION = "Accumulation"
    EARLY_PUMP = "Early Pump"
    MAIN_PUMP = "Main Pump"
    PEAK_FRENZY = "Peak Frenzy"
    EARLY_DUMP = "Early Dump"
    MAIN_DUMP = "Main Dump"
    BAGHOLDERS = "Bagholders"
    RECOVERY_ATTEMPT = "Recovery Attempt"
    DEAD = "Dead/Delisted Risk"

class OptionsStrategy(Enum):
    """Options trading strategies"""
    BUY_CALLS = "Buy Calls"
    BUY_PUTS = "Buy Puts"
    SELL_CALLS = "Sell Calls (Covered)"
    SELL_PUTS = "Sell Puts (Cash Secured)"
    CALL_SPREADS = "Call Spreads"
    PUT_SPREADS = "Put Spreads"
    STRADDLE = "Long Straddle"
    STRANGLE = "Long Strangle"
    IRON_CONDOR = "Iron Condor"
    AVOID_OPTIONS = "Avoid All Options"

class RiskLevel(Enum):
    """Risk levels for trading strategies"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    MEDIUM_HIGH = "MEDIUM-HIGH"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

class SentimentLevel(Enum):
    """Sentiment classification levels"""
    VERY_BEARISH = "Very Bearish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"
    BULLISH = "Bullish"
    VERY_BULLISH = "Very Bullish"

# Phase emojis mapping
PHASE_EMOJIS = {
    PumpDumpPhase.ACCUMULATION: "🔄",
    PumpDumpPhase.EARLY_PUMP: "📈",
    PumpDumpPhase.MAIN_PUMP: "🚀",
    PumpDumpPhase.PEAK_FRENZY: "🌋",
    PumpDumpPhase.EARLY_DUMP: "📉",
    PumpDumpPhase.MAIN_DUMP: "🔻",
    PumpDumpPhase.BAGHOLDERS: "💼",
    PumpDumpPhase.RECOVERY_ATTEMPT: "🔄",
    PumpDumpPhase.DEAD: "☠️"
}

# Strategy emojis mapping
STRATEGY_EMOJIS = {
    OptionsStrategy.BUY_CALLS: "📈",
    OptionsStrategy.BUY_PUTS: "📉",
    OptionsStrategy.SELL_CALLS: "💰",
    OptionsStrategy.SELL_PUTS: "💰",
    OptionsStrategy.CALL_SPREADS: "📊",
    OptionsStrategy.PUT_SPREADS: "📊",
    OptionsStrategy.STRADDLE: "🎯",
    OptionsStrategy.STRANGLE: "🎯",
    OptionsStrategy.IRON_CONDOR: "🦅",
    OptionsStrategy.AVOID_OPTIONS: "🚫"
}

# Risk level colors
RISK_COLORS = {
    RiskLevel.LOW: "🟢",
    RiskLevel.MEDIUM: "🟡",
    RiskLevel.MEDIUM_HIGH: "🟠",
    RiskLevel.HIGH: "🔴",
    RiskLevel.EXTREME: "☠️"
}

# Sentiment colors
SENTIMENT_COLORS = {
    "bullish": "🟢",
    "neutral": "🟡",
    "bearish": "🔴"
}

# Keywords for sentiment analysis
PUMP_KEYWORDS = [
    'moon', 'rocket', 'diamond hands', 'hold', 'hodl', 'squeeze', 
    'short squeeze', 'gamma squeeze', 'to the moon', 'buy the dip', 
    'yolo', 'ape', 'tendies', 'lambo', 'bullish', 'breakout',
    '🚀', '💎', '🦍', '📈', 'rally', 'momentum', 'calls', 'long'
]

DUMP_KEYWORDS = [
    'sell', 'dump', 'crash', 'falling', 'bearish', 'short', 
    'puts', 'bags', 'bagholders', 'dead', 'scam', 'pump and dump',
    'fraud', 'overvalued', 'bubble', '📉', 'rip', 'loss', 'exit'
]

OPTIONS_CALLS_KEYWORDS = [
    'calls', 'call option', 'buy calls', 'otm calls', 'itm calls'
]

OPTIONS_PUTS_KEYWORDS = [
    'puts', 'put option', 'buy puts', 'otm puts', 'itm puts'
]

# Common non-stock words to filter out
COMMON_WORDS = [
    'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 
    'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 
    'HIS', 'HOW', 'ITS', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 
    'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE',
    'WSB', 'APE', 'MOON', 'YOLO', 'HOLD', 'HODL'
]