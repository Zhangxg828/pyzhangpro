import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    市场环境识别模块
    
    功能：
    1. 识别牛市、熊市、震荡市
    2. 识别趋势市和震荡市
    3. 识别高波动和低波动环境
    4. 根据市场环境调整交易策略
    5. 市场情绪指数计算
    """
    
    def __init__(self, config: Optional[Dict] = None):
        if config:
            self.bull_threshold = config.get('bull_threshold', 0.02)
            self.bear_threshold = config.get('bear_threshold', -0.02)
            self.volatility_threshold = config.get('volatility_threshold', 0.03)
            self.trend_period = config.get('trend_period', 20)
            self.volatility_period = config.get('volatility_period', 20)
        else:
            self.bull_threshold = 0.02
            self.bear_threshold = -0.02
            self.volatility_threshold = 0.03
            self.trend_period = 20
            self.volatility_period = 20
        
        logger.info(f"市场环境识别器初始化完成 - 牛市阈值:{self.bull_threshold*100}%, 熊市阈值:{self.bear_threshold*100}%")
    
    def detect_regime(self, close: np.ndarray, high: Optional[np.ndarray] = None,
                     low: Optional[np.ndarray] = None, volume: Optional[np.ndarray] = None) -> Dict:
        """
        检测市场环境
        
        Args:
            close: 收盘价数组
            high: 最高价数组（可选）
            low: 最低价数组（可选）
            volume: 成交量数组（可选）
        
        Returns:
            市场环境字典
        """
        if len(close) < self.trend_period:
            logger.warning(f"数据长度不足，无法检测市场环境（需要至少 {self.trend_period} 个数据点）")
            return {
                'regime': 'UNKNOWN',
                'trend': 'UNKNOWN',
                'volatility': 'UNKNOWN',
                'confidence': 0.0
            }
        
        close = np.array(close)
        
        trend_regime = self._detect_trend_regime(close)
        volatility_regime = self._detect_volatility_regime(close, high, low)
        market_regime = self._combine_regimes(trend_regime, volatility_regime)
        
        sentiment_index = self._calculate_sentiment_index(close, volume)
        
        return {
            'regime': market_regime,
            'trend': trend_regime,
            'volatility': volatility_regime,
            'sentiment_index': sentiment_index,
            'confidence': self._calculate_confidence(close),
            'current_return': (close[-1] - close[-self.trend_period]) / close[-self.trend_period],
            'volatility_value': self._calculate_volatility(close)
        }
    
    def _detect_trend_regime(self, close: np.ndarray) -> str:
        """检测趋势环境"""
        recent_return = (close[-1] - close[-self.trend_period]) / close[-self.trend_period]
        
        if recent_return > self.bull_threshold:
            return 'BULL'
        elif recent_return < self.bear_threshold:
            return 'BEAR'
        else:
            return 'SIDEWAYS'
    
    def _detect_volatility_regime(self, close: np.ndarray, high: Optional[np.ndarray],
                                  low: Optional[np.ndarray]) -> str:
        """检测波动率环境"""
        volatility = self._calculate_volatility(close)
        
        if volatility > self.volatility_threshold:
            return 'HIGH'
        else:
            return 'LOW'
    
    def _combine_regimes(self, trend: str, volatility: str) -> str:
        """组合趋势和波动率环境"""
        if trend == 'BULL' and volatility == 'HIGH':
            return 'BULL_HIGH_VOL'
        elif trend == 'BULL' and volatility == 'LOW':
            return 'BULL_LOW_VOL'
        elif trend == 'BEAR' and volatility == 'HIGH':
            return 'BEAR_HIGH_VOL'
        elif trend == 'BEAR' and volatility == 'LOW':
            return 'BEAR_LOW_VOL'
        elif trend == 'SIDEWAYS' and volatility == 'HIGH':
            return 'SIDEWAYS_HIGH_VOL'
        elif trend == 'SIDEWAYS' and volatility == 'LOW':
            return 'SIDEWAYS_LOW_VOL'
        else:
            return 'UNKNOWN'
    
    def _calculate_volatility(self, close: np.ndarray) -> float:
        """计算波动率"""
        returns = np.diff(np.log(close))
        return np.std(returns[-self.volatility_period:])
    
    def _calculate_sentiment_index(self, close: np.ndarray, volume: Optional[np.ndarray]) -> float:
        """计算市场情绪指数"""
        if len(close) < self.trend_period:
            return 0.0
        
        returns = np.diff(close[-self.trend_period:]) / close[-self.trend_period:-1]
        
        positive_days = np.sum(returns > 0)
        total_days = len(returns)
        
        sentiment = (positive_days / total_days - 0.5) * 2
        
        if volume is not None and len(volume) >= self.trend_period:
            volume_trend = np.mean(volume[-5:]) / np.mean(volume[-self.trend_period:-5])
            sentiment *= volume_trend
        
        return np.clip(sentiment, -1.0, 1.0)
    
    def _calculate_confidence(self, close: np.ndarray) -> float:
        """计算市场环境识别的置信度"""
        if len(close) < self.trend_period * 2:
            return 0.5
        
        recent_returns = np.diff(close[-self.trend_period:]) / close[-self.trend_period:-1]
        earlier_returns = np.diff(close[-self.trend_period*2:-self.trend_period]) / close[-self.trend_period*2:-self.trend_period-1]
        
        recent_volatility = np.std(recent_returns)
        earlier_volatility = np.std(earlier_returns)
        
        volatility_ratio = recent_volatility / earlier_volatility if earlier_volatility > 0 else 1.0
        
        confidence = min(1.0, volatility_ratio)
        
        return confidence
    
    def adjust_strategy(self, regime: str) -> Dict:
        """
        根据市场环境调整策略
        
        Args:
            regime: 市场环境
        
        Returns:
            调整后的策略参数
        """
        strategy_adjustments = {
            'BULL_HIGH_VOL': {
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 1.2,
                'take_profit_multiplier': 1.5,
                'strategy_type': 'TREND_FOLLOWING',
                'recommended_action': '追涨杀跌，注意止损'
            },
            'BULL_LOW_VOL': {
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 1.2,
                'strategy_type': 'TREND_FOLLOWING',
                'recommended_action': '趋势跟踪，适当加仓'
            },
            'BEAR_HIGH_VOL': {
                'position_size_multiplier': 0.5,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 1.0,
                'strategy_type': 'MEAN_REVERSION',
                'recommended_action': '空仓或轻仓，反弹做空'
            },
            'BEAR_LOW_VOL': {
                'position_size_multiplier': 0.7,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 0.8,
                'strategy_type': 'MEAN_REVERSION',
                'recommended_action': '谨慎做空，快进快出'
            },
            'SIDEWAYS_HIGH_VOL': {
                'position_size_multiplier': 0.6,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'strategy_type': 'MEAN_REVERSION',
                'recommended_action': '区间交易，高抛低吸'
            },
            'SIDEWAYS_LOW_VOL': {
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 0.8,
                'strategy_type': 'MEAN_REVERSION',
                'recommended_action': '等待突破，避免频繁交易'
            },
            'UNKNOWN': {
                'position_size_multiplier': 0.5,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'strategy_type': 'CONSERVATIVE',
                'recommended_action': '观望为主，等待明确信号'
            }
        }
        
        return strategy_adjustments.get(regime, strategy_adjustments['UNKNOWN'])
    
    def detect_regime_change(self, close: np.ndarray, window: int = 5) -> Tuple[bool, str, str]:
        """
        检测市场环境变化
        
        Args:
            close: 收盘价数组
            window: 检测窗口（默认5）
        
        Returns:
            (是否变化, 旧环境, 新环境)
        """
        if len(close) < self.trend_period * 2 + window:
            return False, 'UNKNOWN', 'UNKNOWN'
        
        old_regime_data = self.detect_regime(close[-self.trend_period*2-window:-self.trend_period-window])
        new_regime_data = self.detect_regime(close[-self.trend_period-window:])
        
        old_regime = old_regime_data['regime']
        new_regime = new_regime_data['regime']
        
        has_changed = old_regime != new_regime
        
        if has_changed:
            logger.info(f"市场环境变化: {old_regime} -> {new_regime}")
        
        return has_changed, old_regime, new_regime
    
    def get_regime_description(self, regime: str) -> str:
        """
        获取市场环境描述
        
        Args:
            regime: 市场环境
        
        Returns:
            环境描述
        """
        descriptions = {
            'BULL_HIGH_VOL': '牛市高波动：上涨趋势但波动剧烈，适合趋势跟踪但需严格止损',
            'BULL_LOW_VOL': '牛市低波动：稳步上涨，适合趋势跟踪和加仓',
            'BEAR_HIGH_VOL': '熊市高波动：下跌趋势且波动剧烈，风险极高，建议空仓',
            'BEAR_LOW_VOL': '熊市低波动：缓慢下跌，可谨慎做空但需快进快出',
            'SIDEWAYS_HIGH_VOL': '震荡市高波动：区间震荡但波动大，适合区间交易',
            'SIDEWAYS_LOW_VOL': '震荡市低波动：窄幅震荡，建议观望等待突破',
            'UNKNOWN': '未知环境：数据不足或市场混乱，建议观望'
        }
        
        return descriptions.get(regime, '未知环境')
    
    def calculate_market_phase(self, close: np.ndarray) -> Dict:
        """
        计算市场阶段
        
        Args:
            close: 收盘价数组
        
        Returns:
            市场阶段字典
        """
        if len(close) < self.trend_period * 3:
            return {
                'phase': 'UNKNOWN',
                'description': '数据不足'
            }
        
        short_term_return = (close[-1] - close[-self.trend_period]) / close[-self.trend_period]
        medium_term_return = (close[-1] - close[-self.trend_period*2]) / close[-self.trend_period*2]
        long_term_return = (close[-1] - close[-self.trend_period*3]) / close[-self.trend_period*3]
        
        if short_term_return > 0 and medium_term_return > 0 and long_term_return > 0:
            phase = 'ACCUMULATION'
            description = '积累期：长期上涨，适合持有'
        elif short_term_return > 0 and medium_term_return < 0 and long_term_return > 0:
            phase = 'RECOVERY'
            description = '恢复期：短期反弹，中期调整'
        elif short_term_return < 0 and medium_term_return < 0 and long_term_return > 0:
            phase = 'DISTRIBUTION'
            description = '派发期：高位震荡，注意风险'
        elif short_term_return < 0 and medium_term_return < 0 and long_term_return < 0:
            phase = 'DECLINE'
            description = '衰退期：全面下跌，建议空仓'
        elif short_term_return > 0 and medium_term_return > 0 and long_term_return < 0:
            phase = 'EARLY_RECOVERY'
            description = '早期恢复：底部反弹，谨慎参与'
        else:
            phase = 'TRANSITION'
            description = '过渡期：市场方向不明'
        
        return {
            'phase': phase,
            'description': description,
            'short_term_return': short_term_return,
            'medium_term_return': medium_term_return,
            'long_term_return': long_term_return
        }
    
    def get_trading_signals(self, regime: str, indicators: Dict) -> List[str]:
        """
        根据市场环境和技术指标生成交易信号
        
        Args:
            regime: 市场环境
            indicators: 技术指标字典
        
        Returns:
            交易信号列表
        """
        signals = []
        
        strategy = self.adjust_strategy(regime)
        strategy_type = strategy['strategy_type']
        
        if strategy_type == 'TREND_FOLLOWING':
            if indicators.get('adx', 0) > 25:
                if indicators.get('plus_di', 0) > indicators.get('minus_di', 0):
                    signals.append('趋势跟随：做多信号')
                else:
                    signals.append('趋势跟随：做空信号')
            
            if indicators.get('macd', {}).get('histogram', 0) > 0:
                signals.append('MACD：多头排列')
            elif indicators.get('macd', {}).get('histogram', 0) < 0:
                signals.append('MACD：空头排列')
        
        elif strategy_type == 'MEAN_REVERSION':
            if indicators.get('rsi', 50) < 30:
                signals.append('均值回归：超卖，考虑做多')
            elif indicators.get('rsi', 50) > 70:
                signals.append('均值回归：超买，考虑做空')
            
            bb = indicators.get('bollinger_bands', {})
            if bb.get('percent_b', 0.5) < 0.2:
                signals.append('布林带：触及下轨，反弹概率高')
            elif bb.get('percent_b', 0.5) > 0.8:
                signals.append('布林带：触及上轨，回调概率高')
        
        elif strategy_type == 'CONSERVATIVE':
            signals.append('保守策略：等待明确信号')
        
        return signals