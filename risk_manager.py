import sqlite3
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    风险管理核心模块
    
    功能：
    1. 最大回撤控制
    2. 单笔交易止损限制
    3. 总仓位管理
    4. 相关性风险控制
    5. 动态风险调整
    """
    
    def __init__(self, db_path: str, config: Optional[Dict] = None):
        self.db_path = db_path
        
        if config:
            self.max_drawdown = config.get('max_drawdown', 0.15)
            self.max_single_loss = config.get('max_single_loss', 0.02)
            self.max_total_position = config.get('max_total_position', 0.5)
            self.max_correlated_position = config.get('max_correlated_position', 0.3)
            self.max_daily_loss = config.get('max_daily_loss', 0.05)
            self.max_positions_count = config.get('max_positions_count', 10)
            self.risk_free_rate = config.get('risk_free_rate', 0.02)
        else:
            self.max_drawdown = 0.15
            self.max_single_loss = 0.02
            self.max_total_position = 0.5
            self.max_correlated_position = 0.3
            self.max_daily_loss = 0.05
            self.max_positions_count = 10
            self.risk_free_rate = 0.02
        
        self.crypto_list = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
        self.stock_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        logger.info(f"风险管理器初始化完成 - 最大回撤:{self.max_drawdown*100}%, 单笔止损:{self.max_single_loss*100}%")
    
    def check_risk(self, positions: List[Dict], account_balance: float, 
                   new_trade: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        综合风险检查
        
        Args:
            positions: 当前持仓列表
            account_balance: 账户余额
            new_trade: 新交易信息（可选）
        
        Returns:
            (是否通过风险检查, 风险信息)
        """
        risk_checks = [
            self._check_total_position(positions, account_balance, new_trade),
            self._check_correlation_risk(positions, new_trade),
            self._check_single_trade_risk(new_trade),
            self._check_positions_count(positions, new_trade),
            self._check_daily_loss(account_balance)
        ]
        
        for passed, message in risk_checks:
            if not passed:
                logger.warning(f"风险检查失败: {message}")
                return False, message
        
        logger.info("所有风险检查通过")
        return True, "风险检查通过"
    
    def _check_total_position(self, positions: List[Dict], account_balance: float, 
                              new_trade: Optional[Dict] = None) -> Tuple[bool, str]:
        """检查总仓位"""
        total_position = sum(abs(p.get('margin', 0)) for p in positions) / account_balance
        
        if new_trade:
            total_position += abs(new_trade.get('margin', 0)) / account_balance
        
        if total_position > self.max_total_position:
            return False, f"总仓位超限: {total_position*100:.2f}% > {self.max_total_position*100:.2f}%"
        
        return True, "总仓位检查通过"
    
    def _check_correlation_risk(self, positions: List[Dict], 
                                new_trade: Optional[Dict] = None) -> Tuple[bool, str]:
        """检查相关性风险"""
        crypto_position = sum(abs(p.get('margin', 0)) for p in positions 
                             if p.get('symbol', '') in self.crypto_list)
        
        if new_trade and new_trade.get('symbol', '') in self.crypto_list:
            crypto_position += abs(new_trade.get('margin', 0))
        
        stock_position = sum(abs(p.get('margin', 0)) for p in positions 
                            if p.get('symbol', '') in self.stock_list)
        
        if new_trade and new_trade.get('symbol', '') in self.stock_list:
            stock_position += abs(new_trade.get('margin', 0))
        
        if crypto_position > self.max_correlated_position * len(positions + ([new_trade] if new_trade else [])):
            return False, f"加密货币相关性风险过高: {crypto_position}"
        
        if stock_position > self.max_correlated_position * len(positions + ([new_trade] if new_trade else [])):
            return False, f"股票相关性风险过高: {stock_position}"
        
        return True, "相关性风险检查通过"
    
    def _check_single_trade_risk(self, new_trade: Optional[Dict] = None) -> Tuple[bool, str]:
        """检查单笔交易风险"""
        if not new_trade:
            return True, "无新交易"
        
        entry = new_trade.get('entry_price', 0)
        stop = new_trade.get('stop_loss', 0)
        
        if entry > 0 and stop > 0:
            risk_pct = abs(entry - stop) / entry
            if risk_pct > self.max_single_loss:
                return False, f"单笔风险过高: {risk_pct*100:.2f}% > {self.max_single_loss*100:.2f}%"
        
        return True, "单笔风险检查通过"
    
    def _check_positions_count(self, positions: List[Dict], 
                               new_trade: Optional[Dict] = None) -> Tuple[bool, str]:
        """检查持仓数量"""
        current_count = len(positions)
        
        if new_trade:
            current_count += 1
        
        if current_count > self.max_positions_count:
            return False, f"持仓数量超限: {current_count} > {self.max_positions_count}"
        
        return True, "持仓数量检查通过"
    
    def _check_daily_loss(self, account_balance: float) -> Tuple[bool, str]:
        """检查当日亏损"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('PRAGMA journal_mode=WAL;')
            cursor = conn.cursor()
            
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute("""
                SELECT SUM(CASE WHEN type = 'SELL' THEN profit ELSE 0 END) as total_loss
                FROM shadow_portfolio_v7
                WHERE DATE(timestamp) = ?
            """, (today,))
            
            result = cursor.fetchone()
            total_loss = abs(result[0]) if result and result[0] else 0
            
            conn.close()
            
            if total_loss > account_balance * self.max_daily_loss:
                return False, f"当日亏损超限: {total_loss/account_balance*100:.2f}% > {self.max_daily_loss*100:.2f}%"
            
            return True, "当日亏损检查通过"
        
        except Exception as e:
            logger.error(f"检查当日亏损失败: {e}")
            return True, "当日亏损检查失败，允许交易"
    
    def calculate_drawdown(self, equity_curve: List[float]) -> float:
        """
        计算最大回撤
        
        Args:
            equity_curve: 权益曲线
        
        Returns:
            最大回撤比例
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        
        return abs(drawdown.min())
    
    def check_drawdown_limit(self, account_balance: float) -> Tuple[bool, float, str]:
        """
        检查是否超过最大回撤限制
        
        Args:
            account_balance: 当前账户余额
        
        Returns:
            (是否超限, 当前回撤, 信息)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('PRAGMA journal_mode=WAL;')
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT total_equity, timestamp
                FROM shadow_account
                ORDER BY timestamp ASC
            """)
            
            equity_data = cursor.fetchall()
            conn.close()
            
            if not equity_data or len(equity_data) < 2:
                return False, 0.0, "数据不足，无法计算回撤"
            
            equity_curve = [row[0] for row in equity_data]
            current_drawdown = self.calculate_drawdown(equity_curve)
            
            if current_drawdown > self.max_drawdown:
                return True, current_drawdown, f"最大回撤超限: {current_drawdown*100:.2f}% > {self.max_drawdown*100:.2f}%"
            
            return False, current_drawdown, f"当前回撤: {current_drawdown*100:.2f}%"
        
        except Exception as e:
            logger.error(f"检查回撤失败: {e}")
            return False, 0.0, f"检查回撤失败: {e}"
    
    def calculate_position_size(self, account_balance: float, entry_price: float, 
                               stop_loss: float, risk_per_trade: float = 0.01) -> float:
        """
        计算最优仓位大小（固定百分比法）
        
        Args:
            account_balance: 账户余额
            entry_price: 入场价格
            stop_loss: 止损价格
            risk_per_trade: 单笔风险比例（默认1%）
        
        Returns:
            仓位大小（数量）
        """
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0
        
        risk_amount = account_balance * risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return 0.0
        
        position_size = risk_amount / risk_per_share
        
        return position_size
    
    def calculate_kelly_position_size(self, account_balance: float, entry_price: float,
                                      stop_loss: float, win_rate: float, 
                                      avg_win: float, avg_loss: float,
                                      kelly_fraction: float = 0.25) -> float:
        """
        计算凯利公式仓位大小
        
        Args:
            account_balance: 账户余额
            entry_price: 入场价格
            stop_loss: 止损价格
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损
            kelly_fraction: 凯利公式系数（默认0.25，保守）
        
        Returns:
            仓位大小（数量）
        """
        if avg_loss == 0:
            return self.calculate_position_size(account_balance, entry_price, stop_loss)
        
        kelly_percentage = win_rate - (1 - win_rate) / (avg_win / avg_loss)
        
        if kelly_percentage <= 0:
            return 0.0
        
        risk_amount = account_balance * kelly_percentage * kelly_fraction
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return 0.0
        
        position_size = risk_amount / risk_per_share
        
        return position_size
    
    def calculate_risk_parity_position(self, account_balance: float, entry_price: float,
                                      stop_loss: float, volatility: float) -> float:
        """
        计算风险平价仓位大小
        
        Args:
            account_balance: 账户余额
            entry_price: 入场价格
            stop_loss: 止损价格
            volatility: 波动率
        
        Returns:
            仓位大小（数量）
        """
        if volatility <= 0:
            return self.calculate_position_size(account_balance, entry_price, stop_loss)
        
        base_position = self.calculate_position_size(account_balance, entry_price, stop_loss)
        
        adjusted_position = base_position / volatility
        
        return adjusted_position
    
    def get_risk_metrics(self, account_balance: float) -> Dict:
        """
        获取当前风险指标
        
        Args:
            account_balance: 账户余额
        
        Returns:
            风险指标字典
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('PRAGMA journal_mode=WAL;')
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM shadow_portfolio_v7 WHERE type != 'CLOSED'")
            positions = cursor.fetchall()
            
            conn.close()
            
            total_position = sum(abs(p[2] * p[4]) for p in positions) / account_balance if positions else 0
            
            crypto_position = sum(abs(p[2] * p[4]) for p in positions 
                                 if p[0] in self.crypto_list) / account_balance if positions else 0
            
            stock_position = sum(abs(p[2] * p[4]) for p in positions 
                                if p[0] in self.stock_list) / account_balance if positions else 0
            
            is_drawdown_exceeded, current_drawdown, drawdown_msg = self.check_drawdown_limit(account_balance)
            
            return {
                'total_position_ratio': total_position,
                'crypto_position_ratio': crypto_position,
                'stock_position_ratio': stock_position,
                'positions_count': len(positions),
                'current_drawdown': current_drawdown,
                'is_drawdown_exceeded': is_drawdown_exceeded,
                'max_drawdown_limit': self.max_drawdown,
                'max_total_position': self.max_total_position,
                'max_correlated_position': self.max_correlated_position,
                'risk_status': 'HIGH' if is_drawdown_exceeded else 'NORMAL'
            }
        
        except Exception as e:
            logger.error(f"获取风险指标失败: {e}")
            return {
                'error': str(e)
            }
    
    def should_stop_trading(self, account_balance: float) -> Tuple[bool, str]:
        """
        判断是否应该停止交易
        
        Args:
            account_balance: 账户余额
        
        Returns:
            (是否停止, 原因)
        """
        is_exceeded, current_drawdown, msg = self.check_drawdown_limit(account_balance)
        
        if is_exceeded:
            return True, f"最大回撤超限，停止交易: {msg}"
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('PRAGMA journal_mode=WAL;')
            cursor = conn.cursor()
            
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute("""
                SELECT SUM(CASE WHEN type = 'SELL' THEN profit ELSE 0 END) as total_loss
                FROM shadow_portfolio_v7
                WHERE DATE(timestamp) = ?
            """, (today,))
            
            result = cursor.fetchone()
            total_loss = abs(result[0]) if result and result[0] else 0
            
            conn.close()
            
            if total_loss > account_balance * self.max_daily_loss:
                return True, f"当日亏损超限，停止交易: {total_loss/account_balance*100:.2f}%"
        
        except Exception as e:
            logger.error(f"检查当日亏损失败: {e}")
        
        return False, "可以继续交易"
    
    def adjust_risk_parameters(self, current_drawdown: float) -> Dict:
        """
        根据当前回撤动态调整风险参数
        
        Args:
            current_drawdown: 当前回撤
        
        Returns:
            调整后的风险参数
        """
        drawdown_ratio = current_drawdown / self.max_drawdown
        
        if drawdown_ratio < 0.5:
            adjustment_factor = 1.0
        elif drawdown_ratio < 0.75:
            adjustment_factor = 0.75
        elif drawdown_ratio < 0.9:
            adjustment_factor = 0.5
        else:
            adjustment_factor = 0.25
        
        return {
            'adjusted_max_single_loss': self.max_single_loss * adjustment_factor,
            'adjusted_max_total_position': self.max_total_position * adjustment_factor,
            'adjusted_max_correlated_position': self.max_correlated_position * adjustment_factor,
            'adjustment_factor': adjustment_factor,
            'risk_level': 'HIGH' if drawdown_ratio > 0.75 else 'MEDIUM' if drawdown_ratio > 0.5 else 'LOW'
        }
    
    def check_position_risk(self, symbol: str, position_value: float, 
                            entry_price: float, stop_loss: float, 
                            action: str, account_balance: Optional[float] = None) -> Dict:
        """
        检查单个交易的风险（简化版本）
        
        Args:
            symbol: 交易对符号
            position_value: 仓位价值
            entry_price: 入场价格
            stop_loss: 止损价格
            action: 交易方向（LONG/SHORT）
            account_balance: 账户余额（可选）
        
        Returns:
            {'approved': bool, 'reason': str}
        """
        try:
            if account_balance is None:
                account_balance = 100000.0
            
            margin = position_value / 5.0
            
            risk_pct = abs(entry_price - stop_loss) / entry_price if entry_price > 0 else 0
            
            if risk_pct > self.max_single_loss:
                return {
                    'approved': False,
                    'reason': f"单笔风险过高: {risk_pct*100:.2f}% > {self.max_single_loss*100:.2f}%"
                }
            
            total_position_ratio = margin / account_balance
            
            if total_position_ratio > self.max_total_position:
                return {
                    'approved': False,
                    'reason': f"仓位超限: {total_position_ratio*100:.2f}% > {self.max_total_position*100:.2f}%"
                }
            
            return {
                'approved': True,
                'reason': '风险检查通过'
            }
        
        except Exception as e:
            logger.error(f"检查仓位风险失败: {e}")
            return {
                'approved': False,
                'reason': f'风险检查失败: {e}'
            }