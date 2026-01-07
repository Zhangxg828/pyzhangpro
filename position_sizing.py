import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from config import (
    DB_MEMORY, POSITION_SIZING_CONFIG, TECHNICAL_INDICATORS_CONFIG,
    setup_logger, DATA_DIR
)

logger = setup_logger('position_sizing', DATA_DIR / 'position_sizing.log')


@dataclass
class PositionSize:
    symbol: str
    position_size: float
    position_value: float
    margin_required: float
    risk_amount: float
    risk_percentage: float
    sizing_method: str
    confidence: float
    timestamp: datetime


class PositionSizing:
    def __init__(self, config: Dict = None):
        self.config = config or POSITION_SIZING_CONFIG
        self.base_position_size = self.config.get('base_position_size', 0.05)
        self.max_position_size = self.config.get('max_position_size', 0.15)
        self.min_position_size = self.config.get('min_position_size', 0.01)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)
        self.kelly_criterion = self.config.get('kelly_criterion', False)
        self.volatility_adjustment = self.config.get('volatility_adjustment', True)
        self.correlation_adjustment = self.config.get('correlation_adjustment', True)
        self.atr_multiplier = self.config.get('atr_multiplier', 2.0)
        
        logger.info(f"仓位管理器初始化完成 - 基础仓位: {self.base_position_size:.2%}, 最大仓位: {self.max_position_size:.2%}")

    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float,
                                account_balance: float, win_rate: float = 0.5,
                                avg_win: float = 0.02, avg_loss: float = 0.01) -> PositionSize:
        """
        计算仓位大小
        
        Args:
            symbol: 交易对
            entry_price: 入场价格
            stop_loss: 止损价格
            account_balance: 账户余额
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损
        
        Returns:
            PositionSize: 仓位大小对象
        """
        try:
            risk_per_share = abs(entry_price - stop_loss) / entry_price
            
            if risk_per_share == 0:
                risk_per_share = 0.01
            
            if self.kelly_criterion:
                position_size = self._kelly_criterion(win_rate, avg_win, avg_loss)
            else:
                position_size = self._fixed_risk_position(account_balance, risk_per_share)
            
            if self.volatility_adjustment:
                volatility_factor = self._calculate_volatility_factor(symbol)
                position_size *= volatility_factor
            
            if self.correlation_adjustment:
                correlation_factor = self._calculate_correlation_factor(symbol)
                position_size *= correlation_factor
            
            position_size = max(self.min_position_size, min(position_size, self.max_position_size))
            
            position_value = account_balance * position_size
            margin_required = position_value / self.atr_multiplier
            risk_amount = position_value * risk_per_share
            risk_percentage = risk_amount / account_balance
            
            return PositionSize(
                symbol=symbol,
                position_size=position_size,
                position_value=position_value,
                margin_required=margin_required,
                risk_amount=risk_amount,
                risk_percentage=risk_percentage,
                sizing_method='kelly' if self.kelly_criterion else 'fixed_risk',
                confidence=0.8,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"计算仓位大小失败: {e}")
            return self._get_default_position(symbol, account_balance)

    def _fixed_risk_position(self, account_balance: float, risk_per_share: float) -> float:
        """
        固定风险仓位计算
        
        Args:
            account_balance: 账户余额
            risk_per_share: 每股风险
        
        Returns:
            float: 仓位大小
        """
        risk_amount = account_balance * self.risk_per_trade
        position_size = risk_amount / (account_balance * risk_per_share)
        return position_size

    def _kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        凯利公式计算最优仓位
        
        Args:
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损
        
        Returns:
            float: 最优仓位大小
        """
        if avg_loss == 0:
            return self.base_position_size
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        
        return kelly_fraction

    def _calculate_volatility_factor(self, symbol: str) -> float:
        """
        计算波动率调整因子
        
        Args:
            symbol: 交易对
        
        Returns:
            float: 波动率调整因子
        """
        try:
            conn = sqlite3.connect(DB_MEMORY)
            conn.execute('PRAGMA journal_mode=WAL;')
            
            query = """
            SELECT price FROM raw_ticker_stream
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 50
            """
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if len(df) < 20:
                return 1.0
            
            returns = df['price'].pct_change().dropna()
            volatility = returns.std()
            
            target_volatility = 0.02
            volatility_factor = target_volatility / volatility if volatility > 0 else 1.0
            volatility_factor = max(0.5, min(volatility_factor, 2.0))
            
            return volatility_factor
            
        except Exception as e:
            logger.debug(f"计算波动率因子失败: {e}")
            return 1.0

    def _calculate_correlation_factor(self, symbol: str) -> float:
        """
        计算相关性调整因子
        
        Args:
            symbol: 交易对
        
        Returns:
            float: 相关性调整因子
        """
        try:
            conn = sqlite3.connect(DB_MEMORY)
            conn.execute('PRAGMA journal_mode=WAL;')
            
            query = """
            SELECT symbol, price FROM raw_ticker_stream
            WHERE timestamp > datetime('now', '-1 day')
            ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) < 50:
                return 1.0
            
            pivot_df = df.pivot(columns='symbol', values='price')
            returns = pivot_df.pct_change().dropna()
            
            if symbol not in returns.columns:
                return 1.0
            
            correlations = returns.corrwith(returns[symbol])
            high_correlation_count = sum(1 for corr in correlations if abs(corr) > 0.7)
            
            correlation_factor = 1.0 / (1.0 + high_correlation_count * 0.1)
            correlation_factor = max(0.5, correlation_factor)
            
            return correlation_factor
            
        except Exception as e:
            logger.debug(f"计算相关性因子失败: {e}")
            return 1.0

    def _get_default_position(self, symbol: str, account_balance: float) -> PositionSize:
        """
        获取默认仓位大小
        
        Args:
            symbol: 交易对
            account_balance: 账户余额
        
        Returns:
            PositionSize: 默认仓位大小对象
        """
        position_size = self.base_position_size
        position_value = account_balance * position_size
        margin_required = position_value / self.atr_multiplier
        risk_amount = position_value * self.risk_per_trade
        risk_percentage = risk_amount / account_balance
        
        return PositionSize(
            symbol=symbol,
            position_size=position_size,
            position_value=position_value,
            margin_required=margin_required,
            risk_amount=risk_amount,
            risk_percentage=risk_percentage,
            sizing_method='default',
            confidence=0.5,
            timestamp=datetime.now()
        )

    def adjust_position_for_risk(self, position_size: PositionSize, risk_level: str) -> PositionSize:
        """
        根据风险水平调整仓位
        
        Args:
            position_size: 原始仓位大小
            risk_level: 风险水平 (HIGH/MEDIUM/LOW)
        
        Returns:
            PositionSize: 调整后的仓位大小
        """
        risk_adjustment = {
            'HIGH': 0.5,
            'MEDIUM': 0.75,
            'LOW': 1.0
        }
        
        adjustment_factor = risk_adjustment.get(risk_level, 1.0)
        position_size.position_size *= adjustment_factor
        position_size.position_value *= adjustment_factor
        position_size.margin_required *= adjustment_factor
        position_size.risk_amount *= adjustment_factor
        position_size.risk_percentage *= adjustment_factor
        
        return position_size

    def calculate_portfolio_exposure(self, positions: Dict[str, float]) -> float:
        """
        计算投资组合总暴露
        
        Args:
            positions: 持仓字典 {symbol: position_size}
        
        Returns:
            float: 总暴露比例
        """
        total_exposure = sum(positions.values())
        return total_exposure

    def check_exposure_limit(self, current_exposure: float, new_position_size: float) -> bool:
        """
        检查暴露限制
        
        Args:
            current_exposure: 当前暴露
            new_position_size: 新仓位大小
        
        Returns:
            bool: 是否在限制内
        """
        max_total_exposure = self.config.get('max_total_exposure', 0.5)
        new_exposure = current_exposure + new_position_size
        
        return new_exposure <= max_total_exposure

    def get_position_summary(self, position_size: PositionSize) -> Dict:
        """
        获取仓位摘要
        
        Args:
            position_size: 仓位大小对象
        
        Returns:
            Dict: 仓位摘要
        """
        return {
            'symbol': position_size.symbol,
            'position_size': position_size.position_size,
            'position_value': position_size.position_value,
            'margin_required': position_size.margin_required,
            'risk_amount': position_size.risk_amount,
            'risk_percentage': position_size.risk_percentage,
            'sizing_method': position_size.sizing_method,
            'confidence': position_size.confidence,
            'timestamp': position_size.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }