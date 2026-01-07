import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import sqlite3

from config import (
    MARKET_MEMORY_DB,
    LOG_LEVEL,
    LOG_FORMAT
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)


@dataclass
class TimeframeSignal:
    """时间框架信号"""
    timeframe: str
    trend: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0-1, 信号强度
    indicators: Dict
    timestamp: datetime


@dataclass
class MultiTimeframeAnalysis:
    """多时间框架分析结果"""
    symbol: str
    signals: Dict[str, TimeframeSignal]
    overall_signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-1, 综合置信度
    trend_alignment: float  # 0-1, 趋势一致性
    timestamp: datetime


class MultiTimeframeAnalyzer:
    """多时间框架分析器"""
    
    def __init__(self, db_path: str = MARKET_MEMORY_DB):
        self.db_path = db_path
        
        self.timeframes = {
            '1m': {'period': 1, 'weight': 0.1},
            '5m': {'period': 5, 'weight': 0.15},
            '15m': {'period': 15, 'weight': 0.2},
            '1h': {'period': 60, 'weight': 0.25},
            '4h': {'period': 240, 'weight': 0.2},
            '1d': {'period': 1440, 'weight': 0.1}
        }
        
        self.indicator_periods = {
            'short': 7,
            'medium': 14,
            'long': 30
        }
        
        logger.info("多时间框架分析器初始化完成")
    
    def load_data(self, symbol: str, timeframe: str, 
                  limit: int = 500) -> pd.DataFrame:
        """从数据库加载指定时间框架的数据"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            
            period_minutes = self.timeframes[timeframe]['period']
            
            query = """
                SELECT 
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM raw_ticker_stream
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, limit))
            conn.close()
            
            if df.empty:
                logger.warning(f"未找到 {symbol} 的数据")
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            df = self._resample_to_timeframe(df, period_minutes)
            
            return df
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    def _resample_to_timeframe(self, df: pd.DataFrame, 
                               period_minutes: int) -> pd.DataFrame:
        """重采样到指定时间框架"""
        try:
            df = df.set_index('timestamp')
            
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            resampled = df.resample(f'{period_minutes}T').agg(agg_dict)
            resampled = resampled.dropna()
            
            return resampled.reset_index()
            
        except Exception as e:
            logger.error(f"重采样失败: {e}")
            return df
    
    def analyze_timeframe(self, df: pd.DataFrame, 
                         timeframe: str) -> Optional[TimeframeSignal]:
        """分析单个时间框架"""
        try:
            if len(df) < self.indicator_periods['long']:
                logger.warning(f"数据长度不足，无法分析 {timeframe} 时间框架")
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            indicators = self._calculate_indicators(close, high, low, volume)
            
            trend, strength = self._determine_trend(indicators)
            
            signal = TimeframeSignal(
                timeframe=timeframe,
                trend=trend,
                strength=strength,
                indicators=indicators,
                timestamp=df.iloc[-1]['timestamp']
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"分析时间框架 {timeframe} 失败: {e}")
            return None
    
    def _calculate_indicators(self, close: np.ndarray, high: np.ndarray,
                             low: np.ndarray, volume: np.ndarray) -> Dict:
        """计算技术指标"""
        indicators = {}
        
        short_period = self.indicator_periods['short']
        medium_period = self.indicator_periods['medium']
        long_period = self.indicator_periods['long']
        
        sma_short = self._calculate_sma(close, short_period)
        sma_medium = self._calculate_sma(close, medium_period)
        sma_long = self._calculate_sma(close, long_period)
        
        ema_short = self._calculate_ema(close, short_period)
        ema_medium = self._calculate_ema(close, medium_period)
        ema_long = self._calculate_ema(close, long_period)
        
        rsi = self._calculate_rsi(close, medium_period)
        macd, macd_signal, macd_hist = self._calculate_macd(close)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, medium_period)
        
        atr = self._calculate_atr(high, low, close, medium_period)
        
        adx, plus_di, minus_di = self._calculate_adx(high, low, close, medium_period)
        
        volume_ma = self._calculate_sma(volume, medium_period)
        volume_ratio = volume[-1] / volume_ma[-1] if volume_ma[-1] > 0 else 1
        
        indicators.update({
            'sma_short': sma_short[-1] if len(sma_short) > 0 else 0,
            'sma_medium': sma_medium[-1] if len(sma_medium) > 0 else 0,
            'sma_long': sma_long[-1] if len(sma_long) > 0 else 0,
            'ema_short': ema_short[-1] if len(ema_short) > 0 else 0,
            'ema_medium': ema_medium[-1] if len(ema_medium) > 0 else 0,
            'ema_long': ema_long[-1] if len(ema_long) > 0 else 0,
            'rsi': rsi[-1] if len(rsi) > 0 else 50,
            'macd': macd[-1] if len(macd) > 0 else 0,
            'macd_signal': macd_signal[-1] if len(macd_signal) > 0 else 0,
            'macd_hist': macd_hist[-1] if len(macd_hist) > 0 else 0,
            'bb_upper': bb_upper[-1] if len(bb_upper) > 0 else 0,
            'bb_middle': bb_middle[-1] if len(bb_middle) > 0 else 0,
            'bb_lower': bb_lower[-1] if len(bb_lower) > 0 else 0,
            'atr': atr[-1] if len(atr) > 0 else 0,
            'adx': adx[-1] if len(adx) > 0 else 0,
            'plus_di': plus_di[-1] if len(plus_di) > 0 else 0,
            'minus_di': minus_di[-1] if len(minus_di) > 0 else 0,
            'volume_ratio': volume_ratio,
            'current_price': close[-1]
        })
        
        return indicators
    
    def _calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算简单移动平均线"""
        if len(data) < period:
            return np.array([])
        return pd.Series(data).rolling(window=period).mean().values
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算指数移动平均线"""
        if len(data) < period:
            return np.array([])
        return pd.Series(data).ewm(span=period, adjust=False).mean().values
    
    def _calculate_rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        """计算相对强弱指标"""
        if len(close) < period + 1:
            return np.array([])
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(window=period).mean().values
        avg_loss = pd.Series(loss).rolling(window=period).mean().values
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, close: np.ndarray, 
                        fast: int = 12, slow: int = 26, 
                        signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算 MACD"""
        if len(close) < slow:
            return np.array([]), np.array([]), np.array([])
        
        ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
        ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
        
        macd = ema_fast - ema_slow
        macd_signal = pd.Series(macd).ewm(span=signal, adjust=False).mean().values
        macd_hist = macd - macd_signal
        
        return macd, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(self, close: np.ndarray, 
                                   period: int = 20, 
                                   std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算布林带"""
        if len(close) < period:
            return np.array([]), np.array([]), np.array([])
        
        sma = pd.Series(close).rolling(window=period).mean().values
        std = pd.Series(close).rolling(window=period).std().values
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, sma, lower
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, period: int) -> np.ndarray:
        """计算平均真实波幅"""
        if len(high) < period + 1:
            return np.array([])
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        tr = np.maximum(tr1, tr2, tr3)
        
        atr = pd.Series(tr).rolling(window=period).mean().values
        
        atr = np.concatenate([[0], atr])
        
        return atr
    
    def _calculate_adx(self, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算平均趋向指标"""
        if len(high) < period * 2:
            return np.array([]), np.array([]), np.array([])
        
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        tr = np.maximum(tr1, tr2, tr3)
        
        plus_di = 100 * (plus_dm / tr)
        minus_di = 100 * (minus_dm / tr)
        
        plus_di_smooth = pd.Series(plus_di).ewm(span=period, adjust=False).mean().values
        minus_di_smooth = pd.Series(minus_di).ewm(span=period, adjust=False).mean().values
        tr_smooth = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        
        dx = 100 * np.abs(plus_di_smooth - minus_di_smooth) / (plus_di_smooth + minus_di_smooth)
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
        
        adx = np.concatenate([[0] * period, adx])
        plus_di_smooth = np.concatenate([[0] * period, plus_di_smooth])
        minus_di_smooth = np.concatenate([[0] * period, minus_di_smooth])
        
        return adx, plus_di_smooth, minus_di_smooth
    
    def _determine_trend(self, indicators: Dict) -> Tuple[str, float]:
        """确定趋势方向和强度"""
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        sma_short = indicators.get('sma_short', 0)
        sma_medium = indicators.get('sma_medium', 0)
        sma_long = indicators.get('sma_long', 0)
        
        ema_short = indicators.get('ema_short', 0)
        ema_medium = indicators.get('ema_medium', 0)
        ema_long = indicators.get('ema_long', 0)
        
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_hist = indicators.get('macd_hist', 0)
        
        bb_upper = indicators.get('bb_upper', 0)
        bb_middle = indicators.get('bb_middle', 0)
        bb_lower = indicators.get('bb_lower', 0)
        current_price = indicators.get('current_price', 0)
        
        adx = indicators.get('adx', 0)
        plus_di = indicators.get('plus_di', 0)
        minus_di = indicators.get('minus_di', 0)
        
        volume_ratio = indicators.get('volume_ratio', 1)
        
        if sma_short > sma_medium > sma_long:
            bullish_signals += 2
        elif sma_short < sma_medium < sma_long:
            bearish_signals += 2
        total_signals += 2
        
        if ema_short > ema_medium > ema_long:
            bullish_signals += 2
        elif ema_short < ema_medium < ema_long:
            bearish_signals += 2
        total_signals += 2
        
        if rsi > 70:
            bearish_signals += 1
        elif rsi < 30:
            bullish_signals += 1
        total_signals += 1
        
        if macd > macd_signal and macd_hist > 0:
            bullish_signals += 2
        elif macd < macd_signal and macd_hist < 0:
            bearish_signals += 2
        total_signals += 2
        
        if current_price > bb_upper:
            bearish_signals += 1
        elif current_price < bb_lower:
            bullish_signals += 1
        total_signals += 1
        
        if adx > 25:
            if plus_di > minus_di:
                bullish_signals += 1
            else:
                bearish_signals += 1
        total_signals += 1
        
        if volume_ratio > 1.5:
            if bullish_signals > bearish_signals:
                bullish_signals += 1
            elif bearish_signals > bullish_signals:
                bearish_signals += 1
        total_signals += 1
        
        if total_signals == 0:
            return 'neutral', 0.0
        
        bullish_ratio = bullish_signals / total_signals
        bearish_ratio = bearish_signals / total_signals
        
        if bullish_ratio > 0.6:
            trend = 'bullish'
            strength = bullish_ratio
        elif bearish_ratio > 0.6:
            trend = 'bearish'
            strength = bearish_ratio
        else:
            trend = 'neutral'
            strength = 1 - abs(bullish_ratio - bearish_ratio)
        
        return trend, strength
    
    def analyze_multi_timeframe(self, symbol: str) -> Optional[MultiTimeframeAnalysis]:
        """多时间框架综合分析"""
        try:
            signals = {}
            
            for timeframe in self.timeframes.keys():
                df = self.load_data(symbol, timeframe)
                
                if not df.empty:
                    signal = self.analyze_timeframe(df, timeframe)
                    if signal:
                        signals[timeframe] = signal
            
            if not signals:
                logger.warning(f"未获取到 {symbol} 的任何时间框架信号")
                return None
            
            overall_signal, confidence, trend_alignment = self._combine_signals(signals)
            
            analysis = MultiTimeframeAnalysis(
                symbol=symbol,
                signals=signals,
                overall_signal=overall_signal,
                confidence=confidence,
                trend_alignment=trend_alignment,
                timestamp=datetime.now()
            )
            
            logger.info(f"{symbol} 多时间框架分析完成: {overall_signal} (置信度: {confidence:.2f})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"多时间框架分析失败: {e}")
            return None
    
    def _combine_signals(self, signals: Dict[str, TimeframeSignal]) -> Tuple[str, float, float]:
        """组合多个时间框架的信号"""
        bullish_score = 0.0
        bearish_score = 0.0
        neutral_score = 0.0
        
        trend_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        
        for timeframe, signal in signals.items():
            weight = self.timeframes[timeframe]['weight']
            strength = signal.strength
            
            if signal.trend == 'bullish':
                bullish_score += weight * strength
                trend_counts['bullish'] += 1
            elif signal.trend == 'bearish':
                bearish_score += weight * strength
                trend_counts['bearish'] += 1
            else:
                neutral_score += weight * strength
                trend_counts['neutral'] += 1
        
        total_score = bullish_score + bearish_score + neutral_score
        
        if total_score == 0:
            return 'HOLD', 0.0, 0.0
        
        bullish_ratio = bullish_score / total_score
        bearish_ratio = bearish_score / total_score
        
        trend_alignment = max(trend_counts.values()) / len(signals)
        
        if bullish_ratio > 0.6 and trend_alignment > 0.5:
            overall_signal = 'BUY'
            confidence = bullish_ratio * trend_alignment
        elif bearish_ratio > 0.6 and trend_alignment > 0.5:
            overall_signal = 'SELL'
            confidence = bearish_ratio * trend_alignment
        else:
            overall_signal = 'HOLD'
            confidence = trend_alignment
        
        return overall_signal, confidence, trend_alignment
    
    def get_timeframe_summary(self, analysis: MultiTimeframeAnalysis) -> Dict:
        """获取时间框架分析摘要"""
        summary = {
            'symbol': analysis.symbol,
            'overall_signal': analysis.overall_signal,
            'confidence': analysis.confidence,
            'trend_alignment': analysis.trend_alignment,
            'timestamp': analysis.timestamp,
            'timeframes': {}
        }
        
        for timeframe, signal in analysis.signals.items():
            summary['timeframes'][timeframe] = {
                'trend': signal.trend,
                'strength': signal.strength,
                'weight': self.timeframes[timeframe]['weight'],
                'rsi': signal.indicators.get('rsi', 0),
                'macd': signal.indicators.get('macd', 0),
                'adx': signal.indicators.get('adx', 0),
                'price': signal.indicators.get('current_price', 0)
            }
        
        return summary
    
    def save_analysis_to_db(self, analysis: MultiTimeframeAnalysis, 
                           db_path: Optional[str] = None) -> bool:
        """保存分析结果到数据库"""
        try:
            if db_path is None:
                db_path = self.db_path
            
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS multi_timeframe_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    overall_signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    trend_alignment REAL NOT NULL,
                    timeframe_signals TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            import json
            timeframe_signals = json.dumps({
                tf: {
                    'trend': s.trend,
                    'strength': s.strength,
                    'indicators': s.indicators,
                    'timestamp': s.timestamp.isoformat()
                }
                for tf, s in analysis.signals.items()
            })
            
            cursor.execute("""
                INSERT INTO multi_timeframe_analysis 
                (symbol, overall_signal, confidence, trend_alignment, timeframe_signals, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                analysis.symbol,
                analysis.overall_signal,
                analysis.confidence,
                analysis.trend_alignment,
                timeframe_signals,
                analysis.timestamp
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"多时间框架分析结果已保存: {analysis.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"保存分析结果失败: {e}")
            return False


def main():
    """测试函数"""
    analyzer = MultiTimeframeAnalyzer()
    
    symbol = "BTCUSDT"
    
    analysis = analyzer.analyze_multi_timeframe(symbol)
    
    if analysis:
        summary = analyzer.get_timeframe_summary(analysis)
        
        print(f"\n=== 多时间框架分析结果 ===")
        print(f"交易对: {summary['symbol']}")
        print(f"综合信号: {summary['overall_signal']}")
        print(f"置信度: {summary['confidence']:.2%}")
        print(f"趋势一致性: {summary['trend_alignment']:.2%}")
        print(f"分析时间: {summary['timestamp']}")
        
        print(f"\n=== 各时间框架详情 ===")
        for tf, data in summary['timeframes'].items():
            print(f"\n{tf} (权重: {data['weight']:.2f}):")
            print(f"  趋势: {data['trend']}")
            print(f"  强度: {data['strength']:.2f}")
            print(f"  RSI: {data['rsi']:.2f}")
            print(f"  MACD: {data['macd']:.4f}")
            print(f"  ADX: {data['adx']:.2f}")
            print(f"  价格: {data['price']:.2f}")
        
        analyzer.save_analysis_to_db(analysis)


if __name__ == "__main__":
    main()
