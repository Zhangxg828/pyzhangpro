import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import sqlite3
from collections import defaultdict
import json
import requests
from config import PROXY_URL

from config import (
    DB_MEMORY,
    DB_VERIFY,
    LOG_LEVEL
)

# 避免在主界面显示情绪分析器的日志
# 不使用 basicConfig，而是创建独立的logger
logger = logging.getLogger(__name__)

# 如果需要记录到文件，可以添加文件处理器
# 但不添加控制台处理器以避免在主界面显示
if logger.handlers:
    logger.handlers.clear()

# 只保留错误级别的日志，避免INFO和WARNING显示在主界面
logger.setLevel(logging.ERROR)


@dataclass
class SentimentSource:
    """情绪数据源"""
    source_name: str
    sentiment_score: float  # -1 到 1，-1 极度看空，1 极度看多
    confidence: float  # 0 到 1，置信度
    timestamp: datetime
    metadata: Dict


@dataclass
class SentimentAnalysis:
    """情绪分析结果"""
    symbol: str
    overall_sentiment: float  # -1 到 1
    sentiment_trend: str  # 'improving', 'deteriorating', 'stable'
    fear_greed_index: float  # 0 到 100
    volatility_index: float  # 0 到 100
    sources: Dict[str, SentimentSource]
    extreme_sentiment: bool
    sentiment_signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    timestamp: datetime


class AdvancedSentimentAnalyzer:
    """高级情绪分析器"""
    
    def __init__(self, market_db: str = DB_MEMORY, 
                 verification_db: str = DB_VERIFY):
        # 设置日期时间适配器以避免弃用警告
        sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
        
        self.market_db = market_db
        self.verification_db = verification_db
        
        self.sentiment_weights = {
            'market_data': 0.25,    # 市场数据权重
            'social_media': 0.15,   # 社交媒体权重
            'news': 0.15,           # 新闻权重
            'order_flow': 0.15,     # 订单流权重
            'funding_rate': 0.15,   # 资金费率权重
            'open_interest': 0.10,  # 未平仓合约权重
            'put_call_ratio': 0.05  # 期权PCR权重
        }
        
        self.sentiment_thresholds = {
            'extreme_bullish': 0.7,
            'bullish': 0.3,
            'bearish': -0.3,
            'extreme_bearish': -0.7
        }
        
        self.fear_greed_thresholds = {
            'extreme_greed': 75,
            'greed': 55,
            'neutral': 45,
            'fear': 25,
            'extreme_fear': 0
        }
        
        logger.info("高级情绪分析器初始化完成")
    
    def get_funding_rate_sentiment(self, symbol: str) -> Tuple[float, float]:
        """
        通过资金费率获取情绪指标
        高正资金费率 → 贪婪（看涨）
        高负资金费率 → 恐惧（看跌）
        """
        try:
            # 使用Binance API获取资金费率
            url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
            
            # 如果配置了代理，使用代理
            proxies = {'http': PROXY_URL, 'https': PROXY_URL} if PROXY_URL else None
            
            response = requests.get(url, timeout=10, proxies=proxies)
            response.raise_for_status()
            
            data = response.json()
            
            if 'lastFundingRate' in data:
                funding_rate = float(data['lastFundingRate'])
                
                # 将资金费率转换为情绪分数
                # 资金费率通常在 -0.01 到 0.01 之间，需要标准化到 -1 到 1
                # 高正资金费率表示贪婪（情绪分数为正），高负资金费率表示恐惧（情绪分数为负）
                sentiment = np.clip(funding_rate * 100, -1, 1)  # 乘以100以放大信号
                
                # 计算置信度，资金费率绝对值越大，置信度越高
                confidence = min(1.0, abs(funding_rate) * 1000)  # 假设资金费率绝对值越大越可信
                
                logger.debug(f"{symbol} 资金费率: {funding_rate:.6f}, 情绪分数: {sentiment:.2f}, 置信度: {confidence:.2f}")
                
                return sentiment, confidence
            else:
                logger.debug(f"{symbol} 未找到资金费率数据")
                return 0.0, 0.0
                
        except Exception as e:
            logger.debug(f"获取 {symbol} 资金费率失败: {e}")
            return 0.0, 0.0
    
    def get_open_interest_sentiment(self, symbol: str) -> Tuple[float, float]:
        """
        通过未平仓合约变化获取情绪指标
        OI 快速上升 + 价格上涨 → 贪婪
        OI 下降 + 价格下跌 → 恐惧
        """
        try:
            # 使用CoinGlass API获取未平仓合约数据
            # 由于CoinGlass API可能需要API密钥，这里使用公开数据源
            # 首先尝试从我们的数据库获取最近的价格数据来辅助判断
            
            conn = sqlite3.connect(self.market_db)
            conn.execute("PRAGMA journal_mode=WAL")
            
            # 获取最近2小时的数据来判断OI变化趋势
            query = """
                SELECT 
                    recv_time as timestamp,
                    price as close,
                    volume
                FROM raw_ticker_stream
                WHERE symbol = ?
                AND recv_time >= datetime('now', '-2 hours')
                ORDER BY recv_time DESC
                LIMIT 120
            """
            
            # 确保符号是字符串类型
            if isinstance(symbol, tuple):
                actual_symbol = symbol[0] if symbol else 'BTCUSDT'
            else:
                actual_symbol = symbol
            
            df = pd.read_sql_query(query, conn, params=(actual_symbol,))
            conn.close()
            
            if df.empty or len(df) < 10:
                logger.debug(f"{symbol} 未平仓合约数据不足")
                return 0.0, 0.0
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # 计算价格变化和交易量变化
            price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
            volume_change = (df['volume'].iloc[-1] - df['volume'].iloc[0]) / (df['volume'].iloc[0] + 1e-10)
            
            # 模拟未平仓合约情绪指标
            # 如果价格上涨且交易量增加，可能是贪婪情绪
            oi_sentiment = 0.0
            if price_change > 0 and volume_change > 0:
                oi_sentiment = min(1.0, abs(price_change) * 2)  # 贪婪
            elif price_change < 0 and volume_change > 0:
                oi_sentiment = -min(1.0, abs(price_change) * 2)  # 恐惧
            else:
                oi_sentiment = (price_change + volume_change) / 2
            
            # 计算置信度
            confidence = min(1.0, (abs(price_change) + abs(volume_change)) / 2)
            
            logger.debug(f"{symbol} 未平仓合约情绪: {oi_sentiment:.2f}, 置信度: {confidence:.2f}")
            
            return oi_sentiment, confidence
            
        except Exception as e:
            logger.debug(f"获取 {symbol} 未平仓合约情绪失败: {e}")
            return 0.0, 0.0
    
    def get_put_call_ratio_sentiment(self, symbol: str) -> Tuple[float, float]:
        """
        通过期权PCR获取情绪指标
        Put/Call > 1 → 恐惧（看跌期权多）
        Put/Call < 1 → 贪婪（看涨期权多）
        """
        try:
            # 对于主要币种(BTC/ETH)，尝试获取期权数据
            # 由于直接的期权API可能需要付费，我们使用简化的逻辑
            
            # 检查是否为主要币种
            if symbol.startswith(('BTC', 'ETH')):
                # 模拟期权情绪分析（实际应用中需要接入期权API）
                # 这里使用市场价格波动作为替代指标
                
                conn = sqlite3.connect(self.market_db)
                conn.execute("PRAGMA journal_mode=WAL")
                
                query = """
                    SELECT 
                        recv_time as timestamp,
                        price as close
                    FROM raw_ticker_stream
                    WHERE symbol = ?
                    AND recv_time >= datetime('now', '-24 hours')
                    ORDER BY recv_time DESC
                    LIMIT 100
                """
                
                # 确保符号是字符串类型
                if isinstance(symbol, tuple):
                    actual_symbol = symbol[0] if symbol else 'BTCUSDT'
                else:
                    actual_symbol = symbol
                
                df = pd.read_sql_query(query, conn, params=(actual_symbol,))
                conn.close()
                
                if df.empty or len(df) < 10:
                    return 0.0, 0.0
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # 计算波动率作为情绪指标的替代
                returns = np.diff(np.log(df['close'].values))
                volatility = np.std(returns)
                
                # 高波动率可能表示恐惧
                sentiment = -np.clip(volatility * 10, -1, 1)
                confidence = min(1.0, volatility * 50)
                
                logger.debug(f"{symbol} 期权情绪(替代): {sentiment:.2f}, 置信度: {confidence:.2f}")
                
                return sentiment, confidence
            else:
                # 非主要币种不提供期权数据
                return 0.0, 0.0
                
        except Exception as e:
            logger.debug(f"获取 {symbol} 期权情绪失败: {e}")
            return 0.0, 0.0
    
    def load_market_data(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """加载市场数据"""
        try:
            conn = sqlite3.connect(self.market_db)
            conn.execute("PRAGMA journal_mode=WAL")
            
            query = """
                SELECT 
                    recv_time as timestamp,
                    price as close,
                    price as open,
                    price as high,
                    price as low,
                    volume
                FROM raw_ticker_stream
                WHERE symbol = ?
                AND recv_time >= datetime('now', '-{} hours')
                ORDER BY recv_time DESC
            """.format(hours)
            
            # 确保符号是字符串类型，如果不是则尝试提取
            if isinstance(symbol, tuple):
                # 如果符号是元组，取第一个元素作为实际符号
                actual_symbol = symbol[0] if symbol else 'BTCUSDT'
            else:
                actual_symbol = symbol
            
            df = pd.read_sql_query(query, conn, params=(actual_symbol,))
            conn.close()
            
            if df.empty:
                logger.warning(f"未找到 {symbol} 的市场数据")
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"加载市场数据失败: {e}")
            return pd.DataFrame()
    
    def load_sentiment_data(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """加载情绪数据"""
        try:
            conn = sqlite3.connect(self.verification_db)
            conn.execute("PRAGMA journal_mode=WAL")
            
            query = """
                SELECT 
                    timestamp,
                    overall_sentiment as sentiment_score,
                    confidence,
                    sources as source
                FROM sentiment_analysis
                WHERE symbol = ?
                AND timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            """.format(hours)
            
            # 确保符号是字符串类型，如果不是则尝试提取
            if isinstance(symbol, tuple):
                # 如果符号是元组，取第一个元素作为实际符号
                actual_symbol = symbol[0] if symbol else 'BTCUSDT'
            else:
                actual_symbol = symbol
            
            df = pd.read_sql_query(query, conn, params=(actual_symbol,))
            conn.close()
            
            if df.empty:
                logger.warning(f"未找到 {symbol} 的情绪数据")
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            # 如果有metadata列，则处理它；否则创建空的metadata列
            if 'metadata' in df.columns:
                df['metadata'] = df['metadata'].apply(lambda x: json.loads(x) if x else {})
            else:
                df['metadata'] = [{}] * len(df)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"加载情绪数据失败: {e}")
            return pd.DataFrame()
    
    def analyze_market_sentiment(self, df: pd.DataFrame) -> Tuple[float, float]:
        """分析市场情绪"""
        if df.empty or len(df) < 20:
            return 0.0, 0.0
        
        close = df['close'].values
        volume = df['volume'].values
        
        returns = np.diff(np.log(close))
        
        price_momentum = (close[-1] - close[-20]) / close[-20]
        
        volume_momentum = (volume[-1] - np.mean(volume[-20:])) / np.mean(volume[-20:])
        
        volatility = np.std(returns[-20:]) * np.sqrt(24 * 365)
        
        rsi = self._calculate_rsi(close, 14)
        current_rsi = rsi[-1] if len(rsi) > 0 else 50
        
        sentiment = 0.0
        sentiment += np.tanh(price_momentum * 2) * 0.3
        sentiment += np.tanh(volume_momentum * 2) * 0.2
        sentiment += (current_rsi - 50) / 50 * 0.2
        sentiment -= np.tanh(volatility * 10) * 0.3
        
        sentiment = np.clip(sentiment, -1, 1)
        
        volatility_index = min(100, volatility * 100)
        
        return sentiment, volatility_index
    
    def _calculate_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        """计算 RSI"""
        if len(close) < period + 1:
            return np.array([])
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(window=period).mean().values
        avg_loss = pd.Series(loss).rolling(window=period).mean().values
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def analyze_social_media_sentiment(self, symbol: str) -> Tuple[float, float]:
        """分析社交媒体情绪"""
        try:
            conn = sqlite3.connect(self.verification_db)
            conn.execute("PRAGMA journal_mode=WAL")
            
            query = """
                SELECT 
                    sentiment_score,
                    confidence,
                    timestamp
                FROM social_media_sentiment
                WHERE symbol = ?
                AND timestamp >= datetime('now', '-24 hours')
                ORDER BY timestamp DESC
                LIMIT 100
            """
            
            # 确保符号是字符串类型，如果不是则尝试提取
            if isinstance(symbol, tuple):
                # 如果符号是元组，取第一个元素作为实际符号
                actual_symbol = symbol[0] if symbol else 'BTCUSDT'
            else:
                actual_symbol = symbol
            
            df = pd.read_sql_query(query, conn, params=(actual_symbol,))
            conn.close()
            
            if df.empty:
                logger.debug(f"未找到 {symbol} 的社交媒体情绪数据，返回默认值")
                return 0.0, 0.0
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            
            weighted_sentiment = np.average(
                df['sentiment_score'],
                weights=df['confidence']
            )
            
            confidence = np.mean(df['confidence'])
            
            logger.debug(f"{symbol} 社交媒体情绪分析: {weighted_sentiment:.2f}, 置信度: {confidence:.2f}")
            
            return weighted_sentiment, confidence
            
        except Exception as e:
            logger.error(f"分析社交媒体情绪失败: {e}")
            return 0.0, 0.0
    
    def analyze_news_sentiment(self, symbol: str) -> Tuple[float, float]:
        """分析新闻情绪"""
        try:
            conn = sqlite3.connect(self.verification_db)
            conn.execute("PRAGMA journal_mode=WAL")
            
            query = """
                SELECT 
                    sentiment_score,
                    confidence,
                    timestamp
                FROM news_sentiment
                WHERE symbol = ?
                AND timestamp >= datetime('now', '-24 hours')
                ORDER BY timestamp DESC
                LIMIT 50
            """
            
            # 确保符号是字符串类型，如果不是则尝试提取
            if isinstance(symbol, tuple):
                # 如果符号是元组，取第一个元素作为实际符号
                actual_symbol = symbol[0] if symbol else 'BTCUSDT'
            else:
                actual_symbol = symbol
            
            df = pd.read_sql_query(query, conn, params=(actual_symbol,))
            conn.close()
            
            if df.empty:
                logger.debug(f"未找到 {symbol} 的新闻情绪数据，返回默认值")
                return 0.0, 0.0
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            
            recent_weight = np.exp(-np.arange(len(df)) / 10)
            
            weighted_sentiment = np.average(
                df['sentiment_score'],
                weights=df['confidence'] * recent_weight
            )
            
            confidence = np.mean(df['confidence'])
            
            logger.debug(f"{symbol} 新闻情绪分析: {weighted_sentiment:.2f}, 置信度: {confidence:.2f}")
            
            return weighted_sentiment, confidence
            
        except Exception as e:
            logger.error(f"分析新闻情绪失败: {e}")
            return 0.0, 0.0
    
    def analyze_order_flow_sentiment(self, symbol: str) -> Tuple[float, float]:
        """分析订单流情绪"""
        try:
            conn = sqlite3.connect(self.market_db)
            conn.execute("PRAGMA journal_mode=WAL")
            
            query = """
                SELECT 
                    buy_volume,
                    sell_volume,
                    recv_time as timestamp
                FROM raw_ticker_stream
                WHERE symbol = ?
                AND recv_time >= datetime('now', '-1 hours')
                ORDER BY recv_time DESC
                LIMIT 60
            """
            
            # 确保符号是字符串类型，如果不是则尝试提取
            if isinstance(symbol, tuple):
                # 如果符号是元组，取第一个元素作为实际符号
                actual_symbol = symbol[0] if symbol else 'BTCUSDT'
            else:
                actual_symbol = symbol
            
            df = pd.read_sql_query(query, conn, params=(actual_symbol,))
            conn.close()
            
            if df.empty:
                logger.debug(f"未找到 {symbol} 的订单流数据，返回默认值")
                return 0.0, 0.0
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            
            total_buy = df['buy_volume'].sum()
            total_sell = df['sell_volume'].sum()
            total_volume = total_buy + total_sell
            
            if total_volume == 0:
                logger.debug(f"{symbol} 买/卖量为0，返回默认值")
                return 0.0, 0.0
            
            buy_ratio = total_buy / total_volume
            sentiment = (buy_ratio - 0.5) * 2
            
            volume_trend = df['buy_volume'].values - df['sell_volume'].values
            confidence = min(1.0, np.std(volume_trend) / (np.mean(np.abs(volume_trend)) + 1e-10))
            
            logger.debug(f"{symbol} 订单流情绪分析: {sentiment:.2f}, 置信度: {confidence:.2f}")
            
            return sentiment, confidence
            
        except Exception as e:
            logger.error(f"分析订单流情绪失败: {e}")
            return 0.0, 0.0
    
    def calculate_fear_greed_index(self, sentiment_sources: Dict[str, float]) -> float:
        """计算恐惧贪婪指数"""
        try:
            if not sentiment_sources:
                return 50.0
            
            normalized_scores = {}
            
            for source, score in sentiment_sources.items():
                normalized = (score + 1) / 2 * 100
                normalized_scores[source] = normalized
            
            weighted_score = 0.0
            total_weight = 0.0
            
            # 使用完整的权重配置，包括衍生品数据源
            complete_weights = {
                'market_data': 0.25,    # 市场数据权重
                'social_media': 0.15,   # 社交媒体权重
                'news': 0.15,           # 新闻权重
                'order_flow': 0.15,     # 订单流权重
                'funding_rate': 0.15,   # 资金费率权重
                'open_interest': 0.10,  # 未平仓合约权重
                'put_call_ratio': 0.05  # 期权PCR权重
            }
            
            for source, score in normalized_scores.items():
                weight = complete_weights.get(source, 0.1)
                weighted_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                weighted_score /= total_weight
            
            fg_index = np.clip(weighted_score, 0, 100)
            
            return fg_index
            
        except Exception as e:
            logger.error(f"计算恐惧贪婪指数失败: {e}")
            return 50.0
    
    def detect_sentiment_trend(self, historical_sentiments: List[float]) -> str:
        """检测情绪趋势"""
        if len(historical_sentiments) < 10:
            return 'stable'
        
        recent_sentiments = historical_sentiments[-10:]
        
        slope = np.polyfit(range(len(recent_sentiments)), recent_sentiments, 1)[0]
        
        if slope > 0.02:
            return 'improving'
        elif slope < -0.02:
            return 'deteriorating'
        else:
            return 'stable'
    
    def detect_extreme_sentiment(self, sentiment: float, 
                                 fear_greed_index: float) -> bool:
        """检测极端情绪"""
        extreme_by_sentiment = (
            sentiment > self.sentiment_thresholds['extreme_bullish'] or
            sentiment < self.sentiment_thresholds['extreme_bearish']
        )
        
        extreme_by_fg = (
            fear_greed_index > self.fear_greed_thresholds['extreme_greed'] or
            fear_greed_index < self.fear_greed_thresholds['extreme_fear']
        )
        
        return extreme_by_sentiment or extreme_by_fg
    
    def generate_sentiment_signal(self, sentiment: float, 
                                   fear_greed_index: float,
                                   sentiment_trend: str,
                                   extreme_sentiment: bool) -> str:
        """生成情绪信号"""
        if extreme_sentiment:
            if sentiment > 0.5:
                return 'SELL'
            elif sentiment < -0.5:
                return 'BUY'
            else:
                return 'HOLD'
        
        if sentiment > self.sentiment_thresholds['bullish']:
            if sentiment_trend == 'improving':
                return 'BUY'
            else:
                return 'HOLD'
        elif sentiment < self.sentiment_thresholds['bearish']:
            if sentiment_trend == 'deteriorating':
                return 'SELL'
            else:
                return 'HOLD'
        else:
            return 'HOLD'
    
    def analyze_sentiment(self, symbol: str) -> Optional[SentimentAnalysis]:
        """综合情绪分析"""
        try:
            market_df = self.load_market_data(symbol)
            
            market_sentiment, volatility_index = self.analyze_market_sentiment(market_df)
            
            # 如果市场数据不足以生成情绪分数，使用默认值
            if market_df.empty or len(market_df) < 20:
                logger.debug(f"{symbol} 市场数据不足，使用默认情绪分数")
                market_sentiment = 0.0
                volatility_index = 50.0
            
            # 获取其他数据源的情绪分数
            social_sentiment, social_confidence = self.analyze_social_media_sentiment(symbol)
            news_sentiment, news_confidence = self.analyze_news_sentiment(symbol)
            order_sentiment, order_confidence = self.analyze_order_flow_sentiment(symbol)
            
            # 新增：获取衍生品数据源的情绪分数
            funding_sentiment, funding_confidence = self.get_funding_rate_sentiment(symbol)
            oi_sentiment, oi_confidence = self.get_open_interest_sentiment(symbol)
            pcr_sentiment, pcr_confidence = self.get_put_call_ratio_sentiment(symbol)
            
            # 构建情绪源字典
            sentiment_sources = {
                'market_data': market_sentiment,
                'social_media': social_sentiment,
                'news': news_sentiment,
                'order_flow': order_sentiment,
                'funding_rate': funding_sentiment,  # 新增资金费率情绪
                'open_interest': oi_sentiment,      # 新增未平仓合约情绪
                'put_call_ratio': pcr_sentiment    # 新增期权PCR情绪
            }
            
            # 创建情绪源对象，为每个源设置适当的置信度
            sources = {
                'market_data': SentimentSource(
                    source_name='market_data',
                    sentiment_score=market_sentiment,
                    confidence=0.8 if not market_df.empty and len(market_df) >= 20 else 0.3,  # 如果市场数据不足，降低置信度
                    timestamp=datetime.now(),
                    metadata={'volatility_index': volatility_index}
                ),
                'social_media': SentimentSource(
                    source_name='social_media',
                    sentiment_score=social_sentiment,
                    confidence=social_confidence,
                    timestamp=datetime.now(),
                    metadata={}
                ),
                'news': SentimentSource(
                    source_name='news',
                    sentiment_score=news_sentiment,
                    confidence=news_confidence,
                    timestamp=datetime.now(),
                    metadata={}
                ),
                'order_flow': SentimentSource(
                    source_name='order_flow',
                    sentiment_score=order_sentiment,
                    confidence=order_confidence,
                    timestamp=datetime.now(),
                    metadata={}
                ),
                # 新增衍生品数据源
                'funding_rate': SentimentSource(
                    source_name='funding_rate',
                    sentiment_score=funding_sentiment,
                    confidence=funding_confidence,
                    timestamp=datetime.now(),
                    metadata={'type': 'derivative'}
                ),
                'open_interest': SentimentSource(
                    source_name='open_interest',
                    sentiment_score=oi_sentiment,
                    confidence=oi_confidence,
                    timestamp=datetime.now(),
                    metadata={'type': 'derivative'}
                ),
                'put_call_ratio': SentimentSource(
                    source_name='put_call_ratio',
                    sentiment_score=pcr_sentiment,
                    confidence=pcr_confidence,
                    timestamp=datetime.now(),
                    metadata={'type': 'derivative'}
                )
            }
            
            # 计算加权综合情绪分数
            overall_sentiment = 0.0
            total_weight = 0.0
            
            # 使用不同的权重配置，包括衍生品数据源
            all_weights = {
                'market_data': 0.25,    # 市场数据权重
                'social_media': 0.15,   # 社交媒体权重
                'news': 0.15,           # 新闻权重
                'order_flow': 0.15,     # 订单流权重
                'funding_rate': 0.15,   # 资金费率权重
                'open_interest': 0.10,  # 未平仓合约权重
                'put_call_ratio': 0.05  # 期权PCR权重
            }
            
            for source, score in sentiment_sources.items():
                # 如果该数据源没有有效数据，降低其权重
                base_weight = all_weights.get(source, 0.05)
                
                # 检查是否为默认值（0.0），如果是，降低权重，但仍然保留一定影响
                if source not in ['market_data', 'funding_rate'] and score == 0.0:
                    # 如果不是市场数据或资金费率，且分数为0（表示没有数据），则降低权重但不设为0
                    effective_weight = base_weight * 0.05  # 保留5%的权重
                else:
                    effective_weight = base_weight
                
                overall_sentiment += score * effective_weight
                total_weight += effective_weight
            
            if total_weight > 0:
                overall_sentiment /= total_weight
            else:
                # 如果所有数据源都不可用，使用市场数据作为默认值
                overall_sentiment = market_sentiment
            
            overall_sentiment = np.clip(overall_sentiment, -1, 1)
            
            fear_greed_index = self.calculate_fear_greed_index(sentiment_sources)
            
            sentiment_df = self.load_sentiment_data(symbol, hours=48)
            historical_sentiments = sentiment_df['sentiment_score'].tolist() if not sentiment_df.empty else []
            sentiment_trend = self.detect_sentiment_trend(historical_sentiments)
            
            extreme_sentiment = self.detect_extreme_sentiment(overall_sentiment, fear_greed_index)
            
            sentiment_signal = self.generate_sentiment_signal(
                overall_sentiment,
                fear_greed_index,
                sentiment_trend,
                extreme_sentiment
            )
            
            # 计算整体置信度，考虑数据源的有效性
            valid_sources = [s for s in sources.values() if s.confidence > 0.1]
            confidence = np.mean([s.confidence for s in valid_sources]) if valid_sources else 0.3
            
            analysis = SentimentAnalysis(
                symbol=symbol,
                overall_sentiment=overall_sentiment,
                sentiment_trend=sentiment_trend,
                fear_greed_index=fear_greed_index,
                volatility_index=volatility_index,
                sources=sources,
                extreme_sentiment=extreme_sentiment,
                sentiment_signal=sentiment_signal,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            logger.info(f"{symbol} 情绪分析完成: {sentiment_signal} (情绪: {overall_sentiment:.2f}, 置信度: {confidence:.2f})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"情绪分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_sentiment_summary(self, analysis: SentimentAnalysis) -> Dict:
        """获取情绪分析摘要"""
        summary = {
            'symbol': analysis.symbol,
            'overall_sentiment': analysis.overall_sentiment,
            'sentiment_trend': analysis.sentiment_trend,
            'fear_greed_index': analysis.fear_greed_index,
            'volatility_index': analysis.volatility_index,
            'extreme_sentiment': analysis.extreme_sentiment,
            'sentiment_signal': analysis.sentiment_signal,
            'confidence': analysis.confidence,
            'timestamp': analysis.timestamp,
            'sources': {}
        }
        
        # 使用完整的权重配置，包括衍生品数据源
        complete_weights = {
            'market_data': 0.25,    # 市场数据权重
            'social_media': 0.15,   # 社交媒体权重
            'news': 0.15,           # 新闻权重
            'order_flow': 0.15,     # 订单流权重
            'funding_rate': 0.15,   # 资金费率权重
            'open_interest': 0.10,  # 未平仓合约权重
            'put_call_ratio': 0.05  # 期权PCR权重
        }
        
        for source_name, source in analysis.sources.items():
            summary['sources'][source_name] = {
                'sentiment_score': source.sentiment_score,
                'confidence': source.confidence,
                'weight': complete_weights.get(source_name, 0.1)
            }
        
        return summary
    
    def save_analysis_to_db(self, analysis: SentimentAnalysis) -> bool:
        """保存分析结果到数据库"""
        try:
            conn = sqlite3.connect(self.verification_db)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    overall_sentiment REAL NOT NULL,
                    sentiment_trend TEXT NOT NULL,
                    fear_greed_index REAL NOT NULL,
                    volatility_index REAL NOT NULL,
                    extreme_sentiment INTEGER NOT NULL,
                    sentiment_signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    sources TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            sources_json = json.dumps({
                name: {
                    'sentiment_score': s.sentiment_score,
                    'confidence': s.confidence,
                    'timestamp': s.timestamp.isoformat(),
                    'metadata': s.metadata
                }
                for name, s in analysis.sources.items()
            })
            
            cursor.execute("""
                INSERT INTO sentiment_analysis 
                (symbol, overall_sentiment, sentiment_trend, fear_greed_index, 
                 volatility_index, extreme_sentiment, sentiment_signal, 
                 confidence, sources, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis.symbol if isinstance(analysis.symbol, str) else str(analysis.symbol),
                analysis.overall_sentiment,
                analysis.sentiment_trend,
                analysis.fear_greed_index,
                analysis.volatility_index,
                1 if analysis.extreme_sentiment else 0,
                analysis.sentiment_signal,
                analysis.confidence,
                sources_json,
                analysis.timestamp
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"情绪分析结果已保存: {analysis.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"保存分析结果失败: {e}")
            return False
    
    def get_historical_sentiment(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """获取历史情绪数据"""
        try:
            conn = sqlite3.connect(self.verification_db)
            conn.execute("PRAGMA journal_mode=WAL")
            
            query = """
                SELECT 
                    timestamp,
                    overall_sentiment,
                    fear_greed_index,
                    sentiment_signal
                FROM sentiment_analysis
                WHERE symbol = ?
                AND timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            """.format(days)
            
            # 确保符号是字符串类型，如果不是则尝试提取
            if isinstance(symbol, tuple):
                # 如果符号是元组，取第一个元素作为实际符号
                actual_symbol = symbol[0] if symbol else 'BTCUSDT'
            else:
                actual_symbol = symbol
            
            df = pd.read_sql_query(query, conn, params=(actual_symbol,))
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"获取历史情绪数据失败: {e}")
            return pd.DataFrame()


def main():
    """测试函数 - 批量分析多个交易对"""
    analyzer = AdvancedSentimentAnalyzer()
    
    # 定义要分析的交易对列表
    symbols = [
        "BTCUSDT",
        "ETHUSDT", 
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "AVAXUSDT",
        "DOTUSDT"
    ]
    
    print(f"\n🔍 开始分析 {len(symbols)} 个交易对的情绪...")
    
    for symbol in symbols:
        print(f"\n{"="*60}")
        print(f"正在分析: {symbol}")
        print(f"{"="*60}")
        
        analysis = analyzer.analyze_sentiment(symbol)
        
        if analysis:
            summary = analyzer.get_sentiment_summary(analysis)
            
            print(f"\n=== 情绪分析结果 ===")
            print(f"交易对: {summary['symbol']}")
            print(f"综合情绪: {summary['overall_sentiment']:.2f}")
            print(f"情绪趋势: {summary['sentiment_trend']}")
            print(f"恐惧贪婪指数: {summary['fear_greed_index']:.2f}")
            print(f"波动率指数: {summary['volatility_index']:.2f}")
            print(f"极端情绪: {'是' if summary['extreme_sentiment'] else '否'}")
            print(f"情绪信号: {summary['sentiment_signal']}")
            print(f"置信度: {summary['confidence']:.2%}")
            print(f"分析时间: {summary['timestamp']}")
            
            print(f"\n=== 各数据源详情 ===")
            for source_name, data in summary['sources'].items():
                print(f"\n{source_name} (权重: {data['weight']:.2f}):")
                print(f"  情绪分数: {data['sentiment_score']:.2f}")
                print(f"  置信度: {data['confidence']:.2%}")
            
            analyzer.save_analysis_to_db(analysis)
        else:
            print(f"❌ 未能分析 {symbol}，可能是因为数据不足或网络问题")
    
    print(f"\n✅ 所有交易对的情绪分析完成！")


if __name__ == "__main__":
    main()
