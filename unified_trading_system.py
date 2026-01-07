import sqlite3
import asyncio
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
import logging

from config import (
    DB_MEMORY, DB_VERIFY, DATA_DIR, setup_logger,
    RISK_CONTROL_CONFIG, TECHNICAL_INDICATORS_CONFIG, MARKET_REGIME_CONFIG,
    LIQUIDITY_MANAGER_CONFIG, SENTIMENT_ANALYSIS_CONFIG, ANOMALY_DETECTION_CONFIG,
    POSITION_SIZING_CONFIG, DYNAMIC_STOP_LOSS_CONFIG
)
from risk_manager import RiskManager
from technical_indicators import TechnicalIndicators
from market_regime_detector import MarketRegimeDetector
from liquidity_manager import LiquidityManager
from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer
from anomaly_detector import AnomalyDetector
from position_sizing import PositionSizing

logger = setup_logger('unified_trading_system', DATA_DIR / 'unified_trading_system.log')


@dataclass
class TradingSignal:
    symbol: str
    action: str
    entry_price: float
    take_profit: float
    stop_loss: float
    position_size: float
    confidence: float
    reasoning: str
    risk_level: str
    market_regime: str
    liquidity_score: float
    sentiment_score: float
    has_anomaly: bool
    timestamp: datetime


class UnifiedTradingSystem:
    def __init__(self):
        self.risk_manager = RiskManager(RISK_CONTROL_CONFIG)
        self.technical_indicators = TechnicalIndicators(TECHNICAL_INDICATORS_CONFIG)
        self.market_regime_detector = MarketRegimeDetector(MARKET_REGIME_CONFIG)
        self.liquidity_manager = LiquidityManager(LIQUIDITY_MANAGER_CONFIG)
        self.sentiment_analyzer = AdvancedSentimentAnalyzer(SENTIMENT_ANALYSIS_CONFIG)
        self.anomaly_detector = AnomalyDetector(ANOMALY_DETECTION_CONFIG)
        self.position_sizing = PositionSizing(POSITION_SIZING_CONFIG)
        
        self.is_running = False
        self.signal_callbacks: List[Callable] = []
        
        logger.info("统一交易系统初始化完成")

    def add_signal_callback(self, callback: Callable[[TradingSignal], None]):
        """
        添加信号回调函数
        
        Args:
            callback: 回调函数
        """
        self.signal_callbacks.append(callback)
        logger.info(f"添加信号回调函数: {callback.__name__}")

    def remove_signal_callback(self, callback: Callable[[TradingSignal], None]):
        """
        移除信号回调函数
        
        Args:
            callback: 回调函数
        """
        if callback in self.signal_callbacks:
            self.signal_callbacks.remove(callback)
            logger.info(f"移除信号回调函数: {callback.__name__}")

    def _notify_signal_callbacks(self, signal: TradingSignal):
        """
        通知所有信号回调函数
        
        Args:
            signal: 交易信号
        """
        for callback in self.signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"信号回调执行失败: {e}")

    def analyze_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """
        分析单个交易对并生成交易信号
        
        Args:
            symbol: 交易对
        
        Returns:
            TradingSignal: 交易信号
        """
        try:
            conn = sqlite3.connect(DB_MEMORY)
            conn.execute('PRAGMA journal_mode=WAL;')
            
            query = """
            SELECT price, volume, order_ratio, sar_value, sar_trend, volatility_24h
            FROM raw_ticker_stream
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """
            cur = conn.cursor()
            cur.execute(query, (symbol,))
            row = cur.fetchone()
            conn.close()
            
            if not row:
                logger.debug(f"未找到 {symbol} 的最新数据")
                return None
            
            price, volume, order_ratio, sar_value, sar_trend, volatility_24h = row
            
            risk_assessment = self.risk_manager.assess_market_risk(symbol, price, volatility_24h, order_ratio)
            risk_level = risk_assessment.get('risk_level', 'LOW')
            
            regime_result = self.market_regime_detector.detect_regime(symbol)
            market_regime = regime_result.get('regime', 'SIDEWAYS') if regime_result else 'SIDEWAYS'
            
            liquidity_result = self.liquidity_manager.analyze_liquidity(symbol)
            liquidity_score = liquidity_result.get('liquidity_score', 1.0) if liquidity_result else 1.0
            
            sentiment_result = self.sentiment_analyzer.get_sentiment_summary(symbol)
            sentiment_score = sentiment_result.get('overall_sentiment', 0.0) if sentiment_result else 0.0
            
            anomaly_result = self.anomaly_detector.get_anomaly_summary(symbol)
            has_anomaly = anomaly_result.get('has_anomaly', False) if anomaly_result else False
            
            if has_anomaly or risk_level == "HIGH":
                logger.info(f"{symbol} 检测到异常或高风险，跳过交易")
                return None
            
            action, confidence, reasoning = self._generate_trading_decision(
                symbol, price, sar_value, sar_trend, order_ratio, volatility_24h,
                risk_level, market_regime, liquidity_score, sentiment_score
            )
            
            if action == "WAIT":
                return None
            
            atr = self.technical_indicators.calculate_atr(symbol)
            stop_loss = self._calculate_stop_loss(price, action, atr)
            take_profit = self._calculate_take_profit(price, action, atr)
            
            conn = sqlite3.connect(DB_VERIFY)
            conn.execute('PRAGMA journal_mode=WAL;')
            cur = conn.cursor()
            cur.execute("SELECT balance FROM shadow_account WHERE id=1")
            account_balance = cur.fetchone()[0]
            conn.close()
            
            position_size_obj = self.position_sizing.calculate_position_size(
                symbol, price, stop_loss, account_balance
            )
            
            position_size = self.position_sizing.adjust_position_for_risk(
                position_size_obj, risk_level
            ).position_size
            
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                entry_price=price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                position_size=position_size,
                confidence=confidence,
                reasoning=reasoning,
                risk_level=risk_level,
                market_regime=market_regime,
                liquidity_score=liquidity_score,
                sentiment_score=sentiment_score,
                has_anomaly=has_anomaly,
                timestamp=datetime.now()
            )
            
            logger.info(f"生成交易信号: {symbol} {action} @ {price:.2f}")
            return signal
            
        except Exception as e:
            logger.error(f"分析 {symbol} 失败: {e}")
            return None

    def _generate_trading_decision(self, symbol: str, price: float, sar_value: float,
                                    sar_trend: str, order_ratio: float, volatility_24h: float,
                                    risk_level: str, market_regime: str,
                                    liquidity_score: float, sentiment_score: float) -> tuple:
        """
        生成交易决策
        
        Args:
            symbol: 交易对
            price: 当前价格
            sar_value: SAR值
            sar_trend: SAR趋势
            order_ratio: 买卖盘比
            volatility_24h: 24小时波动率
            risk_level: 风险水平
            market_regime: 市场环境
            liquidity_score: 流动性评分
            sentiment_score: 情绪分数
        
        Returns:
            tuple: (action, confidence, reasoning)
        """
        sar_diff = (price - sar_value) / price
        vol_adj = max(0.5, min(2.0, volatility_24h / 0.03))
        BUY_RATIO_THRESH = 1.6 * vol_adj
        SELL_RATIO_THRESH = 0.7 / vol_adj
        
        action = "WAIT"
        confidence = 0.0
        reasoning = ""
        
        if market_regime == "BEAR" and sar_trend == "BULL":
            action = "WAIT"
            confidence = 0.3
            reasoning = "市场环境不利，逆势交易风险高"
        
        elif market_regime == "BULL" and sar_trend == "BEAR":
            action = "LONG"
            confidence = 0.6
            reasoning = "市场环境有利，可考虑逢低做多"
        
        elif sar_trend == "BULL" and order_ratio > BUY_RATIO_THRESH and sar_diff > 0.005:
            action = "LONG"
            confidence = 0.8
            reasoning = "主升浪加速，趋势强劲"
        
        elif sar_trend == "BEAR" and order_ratio < SELL_RATIO_THRESH and sar_diff < -0.005:
            action = "SHORT"
            confidence = 0.8
            reasoning = "空头狙击，趋势向下"
        
        elif sentiment_score < -0.7 and sar_trend == "BULL":
            action = "LONG"
            confidence = 0.6
            reasoning = "情绪极度悲观，可能是抄底机会"
        
        elif sentiment_score > 0.7 and sar_trend == "BEAR":
            action = "SHORT"
            confidence = 0.6
            reasoning = "情绪极度乐观，可能是逃顶机会"
        
        else:
            action = "WAIT"
            confidence = 0.4
            reasoning = "市场信号不明确，等待更好的入场点"
        
        if liquidity_score < 0.5:
            action = "WAIT"
            confidence = 0.2
            reasoning = "流动性不足，避免交易"
        
        return action, confidence, reasoning

    def _calculate_stop_loss(self, price: float, action: str, atr: float) -> float:
        """
        计算止损价格
        
        Args:
            price: 当前价格
            action: 交易动作
            atr: ATR值
        
        Returns:
            float: 止损价格
        """
        atr_multiplier = DYNAMIC_STOP_LOSS_CONFIG.get('atr_multiplier', 2.0)
        
        if action == "LONG":
            return price - (atr * atr_multiplier)
        elif action == "SHORT":
            return price + (atr * atr_multiplier)
        else:
            return price

    def _calculate_take_profit(self, price: float, action: str, atr: float) -> float:
        """
        计算止盈价格
        
        Args:
            price: 当前价格
            action: 交易动作
            atr: ATR值
        
        Returns:
            float: 止盈价格
        """
        atr_multiplier = DYNAMIC_STOP_LOSS_CONFIG.get('atr_multiplier', 2.0)
        
        if action == "LONG":
            return price + (atr * atr_multiplier * 2)
        elif action == "SHORT":
            return price - (atr * atr_multiplier * 2)
        else:
            return price

    def analyze_all_symbols(self, symbols: List[str]) -> List[TradingSignal]:
        """
        分析所有交易对并生成交易信号
        
        Args:
            symbols: 交易对列表
        
        Returns:
            List[TradingSignal]: 交易信号列表
        """
        signals = []
        
        for symbol in symbols:
            signal = self.analyze_symbol(symbol)
            if signal:
                signals.append(signal)
        
        logger.info(f"生成 {len(signals)} 个交易信号")
        return signals

    def start(self, symbols: List[str], interval: int = 60):
        """
        启动交易系统
        
        Args:
            symbols: 交易对列表
            interval: 分析间隔（秒）
        """
        self.is_running = True
        logger.info(f"交易系统启动，监控 {len(symbols)} 个交易对，间隔 {interval} 秒")
        
        def run_loop():
            while self.is_running:
                try:
                    signals = self.analyze_all_symbols(symbols)
                    
                    for signal in signals:
                        self._notify_signal_callbacks(signal)
                    
                except Exception as e:
                    logger.error(f"交易系统运行异常: {e}")
                
                import time
                time.sleep(interval)
        
        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        logger.info("交易系统线程已启动")

    def stop(self):
        """
        停止交易系统
        """
        self.is_running = False
        logger.info("交易系统已停止")

    def get_system_status(self) -> Dict:
        """
        获取系统状态
        
        Returns:
            Dict: 系统状态
        """
        return {
            'is_running': self.is_running,
            'risk_manager': 'active',
            'technical_indicators': 'active',
            'market_regime_detector': 'active',
            'liquidity_manager': 'active',
            'sentiment_analyzer': 'active',
            'anomaly_detector': 'active',
            'position_sizing': 'active',
            'signal_callbacks': len(self.signal_callbacks),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


def main():
    """
    主函数示例
    """
    system = UnifiedTradingSystem()
    
    def signal_handler(signal: TradingSignal):
        print(f"收到交易信号: {signal.symbol} {signal.action} @ {signal.entry_price:.2f}")
        print(f"  止损: {signal.stop_loss:.2f}, 止盈: {signal.take_profit:.2f}")
        print(f"  仓位: {signal.position_size:.2%}, 置信度: {signal.confidence:.2f}")
        print(f"  风险: {signal.risk_level}, 环境: {signal.market_regime}")
        print(f"  流动性: {signal.liquidity_score:.2f}, 情绪: {signal.sentiment_score:.2f}")
        print(f"  异常: {signal.has_anomaly}")
        print(f"  理由: {signal.reasoning}")
        print()
    
    system.add_signal_callback(signal_handler)
    
    from config import CRYPTO_LIST
    system.start(CRYPTO_LIST[:10], interval=30)
    
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        system.stop()
        print("交易系统已停止")


if __name__ == "__main__":
    main()