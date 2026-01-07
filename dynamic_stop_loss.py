import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from config import DYNAMIC_STOP_LOSS_CONFIG, DB_MEMORY, setup_logger

logger = setup_logger('dynamic_stop_loss')


@dataclass
class StopLossLevel:
    stop_price: float
    stop_type: str
    reason: str
    timestamp: float


class DynamicStopLoss:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or DYNAMIC_STOP_LOSS_CONFIG
        self.atr_period = self.config.get('atr_period', 14)
        self.atr_multiplier = self.config.get('atr_multiplier', 2.0)
        self.trailing_stop = self.config.get('trailing_stop', True)
        self.trailing_activation = self.config.get('trailing_activation', 0.02)
        self.trailing_step = self.config.get('trailing_step', 0.01)
        self.time_based_exit = self.config.get('time_based_exit', False)
        self.max_hold_time = self.config.get('max_hold_time', 24)
        self.breakeven_threshold = self.config.get('breakeven_threshold', 0.015)
        
        self.positions: Dict[str, Dict] = {}
        self.stop_loss_levels: Dict[str, StopLossLevel] = {}
        
        logger.info(f"动态止损策略初始化完成 - ATR周期: {self.atr_period}, ATR倍数: {self.atr_multiplier}")

    def calculate_atr(self, df: pd.DataFrame) -> float:
        try:
            df = df.copy()
            df['high'] = df['price']
            df['low'] = df['price']
            df['close'] = df['price']
            
            df['prev_close'] = df['close'].shift(1)
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['prev_close']),
                    abs(df['low'] - df['prev_close'])
                )
            )
            
            atr = df['tr'].rolling(window=self.atr_period).mean().iloc[-1]
            return atr if not np.isnan(atr) else 0.01
        except Exception as e:
            logger.error(f"计算ATR失败: {e}")
            return 0.01

    def calculate_atr_stop(self, entry_price: float, df: pd.DataFrame, 
                          position_type: str = 'long') -> float:
        try:
            atr = self.calculate_atr(df)
            
            if position_type == 'long':
                stop_price = entry_price - (atr * self.atr_multiplier)
            else:
                stop_price = entry_price + (atr * self.atr_multiplier)
                
            return max(stop_price, 0)
        except Exception as e:
            logger.error(f"计算ATR止损失败: {e}")
            return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05

    def calculate_trailing_stop(self, current_price: float, entry_price: float,
                               highest_price: float, position_type: str = 'long') -> float:
        try:
            profit_pct = (current_price - entry_price) / entry_price if position_type == 'long' else (entry_price - current_price) / entry_price
            
            if profit_pct < self.trailing_activation:
                return None
                
            if position_type == 'long':
                stop_price = highest_price * (1 - self.trailing_step)
            else:
                stop_price = highest_price * (1 + self.trailing_step)
                
            return stop_price
        except Exception as e:
            logger.error(f"计算移动止损失败: {e}")
            return None

    def calculate_breakeven_stop(self, entry_price: float, current_price: float,
                                 position_type: str = 'long') -> Optional[float]:
        try:
            profit_pct = (current_price - entry_price) / entry_price if position_type == 'long' else (entry_price - current_price) / entry_price
            
            if profit_pct >= self.breakeven_threshold:
                return entry_price
            return None
        except Exception as e:
            logger.error(f"计算保本止损失败: {e}")
            return None

    def check_time_based_exit(self, entry_time: float, current_time: float) -> bool:
        try:
            if not self.time_based_exit:
                return False
                
            hold_time = (current_time - entry_time) / 3600
            
            return hold_time >= self.max_hold_time
        except Exception as e:
            logger.error(f"检查时间退出失败: {e}")
            return False

    def calculate_volatility_adjusted_stop(self, entry_price: float, df: pd.DataFrame,
                                          position_type: str = 'long') -> float:
        try:
            returns = df['price'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(len(returns))
            
            if volatility > 0.03:
                multiplier = self.atr_multiplier * 1.5
            elif volatility > 0.02:
                multiplier = self.atr_multiplier * 1.2
            else:
                multiplier = self.atr_multiplier * 0.8
                
            if position_type == 'long':
                stop_price = entry_price * (1 - multiplier * volatility)
            else:
                stop_price = entry_price * (1 + multiplier * volatility)
                
            return max(stop_price, 0)
        except Exception as e:
            logger.error(f"计算波动率调整止损失败: {e}")
            return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05

    def calculate_support_resistance_stop(self, entry_price: float, df: pd.DataFrame,
                                         position_type: str = 'long') -> float:
        try:
            prices = df['price'].values
            window = 20
            
            if len(prices) < window:
                return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05
                
            if position_type == 'long':
                support = np.min(prices[-window:])
                stop_price = support * 0.99
            else:
                resistance = np.max(prices[-window:])
                stop_price = resistance * 1.01
                
            return stop_price
        except Exception as e:
            logger.error(f"计算支撑阻力止损失败: {e}")
            return entry_price * 0.95 if position_type == 'long' else entry_price * 1.05

    def add_position(self, symbol: str, entry_price: float, position_type: str = 'long',
                    size: float = 0.0, entry_time: float = 0.0):
        try:
            self.positions[symbol] = {
                'entry_price': entry_price,
                'position_type': position_type,
                'size': size,
                'entry_time': entry_time,
                'highest_price': entry_price if position_type == 'long' else entry_price,
                'lowest_price': entry_price if position_type == 'short' else entry_price,
                'initial_stop': None
            }
            
            logger.info(f"添加仓位: {symbol}, 入场价: {entry_price}, 方向: {position_type}")
        except Exception as e:
            logger.error(f"添加仓位失败: {e}")

    def update_position(self, symbol: str, current_price: float, current_time: float) -> Optional[StopLossLevel]:
        try:
            if symbol not in self.positions:
                return None
                
            position = self.positions[symbol]
            position_type = position['position_type']
            
            if position_type == 'long':
                position['highest_price'] = max(position['highest_price'], current_price)
            else:
                position['lowest_price'] = min(position['lowest_price'], current_price)
                
            conn = sqlite3.connect(DB_MEMORY)
            conn.execute('PRAGMA journal_mode=WAL;')
            
            query = """
            SELECT price, timestamp
            FROM raw_ticker_stream
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 50
            """
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if len(df) < 10:
                return None
                
            df = df.sort_values('timestamp')
            
            stop_candidates = []
            
            atr_stop = self.calculate_atr_stop(
                position['entry_price'], df, position_type
            )
            stop_candidates.append(StopLossLevel(
                stop_price=atr_stop,
                stop_type='atr',
                reason=f'ATR止损 (ATR={atr_stop/position["entry_price"]:.4f})',
                timestamp=current_time
            ))
            
            if self.trailing_stop:
                trailing_stop = self.calculate_trailing_stop(
                    current_price, position['entry_price'],
                    position['highest_price'] if position_type == 'long' else position['lowest_price'],
                    position_type
                )
                if trailing_stop:
                    stop_candidates.append(StopLossLevel(
                        stop_price=trailing_stop,
                        stop_type='trailing',
                        reason=f'移动止损 (激活利润={self.trailing_activation*100:.1f}%)',
                        timestamp=current_time
                    ))
            
            breakeven_stop = self.calculate_breakeven_stop(
                position['entry_price'], current_price, position_type
            )
            if breakeven_stop:
                stop_candidates.append(StopLossLevel(
                    stop_price=breakeven_stop,
                    stop_type='breakeven',
                    reason=f'保本止损 (利润={self.breakeven_threshold*100:.1f}%)',
                    timestamp=current_time
                ))
            
            volatility_stop = self.calculate_volatility_adjusted_stop(
                position['entry_price'], df, position_type
            )
            stop_candidates.append(StopLossLevel(
                stop_price=volatility_stop,
                stop_type='volatility',
                reason='波动率调整止损',
                timestamp=current_time
            ))
            
            support_resistance_stop = self.calculate_support_resistance_stop(
                position['entry_price'], df, position_type
            )
            stop_candidates.append(StopLossLevel(
                stop_price=support_resistance_stop,
                stop_type='support_resistance',
                reason='支撑/阻力止损',
                timestamp=current_time
            ))
            
            if position_type == 'long':
                best_stop = max(stop_candidates, key=lambda x: x.stop_price)
            else:
                best_stop = min(stop_candidates, key=lambda x: x.stop_price)
                
            self.stop_loss_levels[symbol] = best_stop
            
            if self.check_time_based_exit(position['entry_time'], current_time):
                logger.info(f"{symbol} 达到最大持仓时间，建议退出")
                return StopLossLevel(
                    stop_price=current_price,
                    stop_type='time_exit',
                    reason=f'时间退出 (持仓时间={self.max_hold_time}小时)',
                    timestamp=current_time
                )
            
            return best_stop
        except Exception as e:
            logger.error(f"更新仓位失败: {e}")
            return None

    def check_stop_loss(self, symbol: str, current_price: float) -> Tuple[bool, Optional[StopLossLevel]]:
        try:
            if symbol not in self.stop_loss_levels:
                return False, None
                
            stop_level = self.stop_loss_levels[symbol]
            position = self.positions[symbol]
            position_type = position['position_type']
            
            if position_type == 'long':
                triggered = current_price <= stop_level.stop_price
            else:
                triggered = current_price >= stop_level.stop_price
                
            return triggered, stop_level
        except Exception as e:
            logger.error(f"检查止损失败: {e}")
            return False, None

    def remove_position(self, symbol: str):
        try:
            if symbol in self.positions:
                del self.positions[symbol]
            if symbol in self.stop_loss_levels:
                del self.stop_loss_levels[symbol]
            logger.info(f"移除仓位: {symbol}")
        except Exception as e:
            logger.error(f"移除仓位失败: {e}")

    def get_position_info(self, symbol: str) -> Optional[Dict]:
        try:
            if symbol not in self.positions:
                return None
                
            position = self.positions[symbol]
            stop_level = self.stop_loss_levels.get(symbol)
            
            return {
                'symbol': symbol,
                'entry_price': position['entry_price'],
                'position_type': position['position_type'],
                'size': position['size'],
                'entry_time': position['entry_time'],
                'highest_price': position['highest_price'],
                'lowest_price': position['lowest_price'],
                'stop_price': stop_level.stop_price if stop_level else None,
                'stop_type': stop_level.stop_type if stop_level else None,
                'stop_reason': stop_level.reason if stop_level else None
            }
        except Exception as e:
            logger.error(f"获取仓位信息失败: {e}")
            return None

    def get_all_positions(self) -> Dict[str, Dict]:
        try:
            positions_info = {}
            for symbol in self.positions:
                positions_info[symbol] = self.get_position_info(symbol)
            return positions_info
        except Exception as e:
            logger.error(f"获取所有仓位信息失败: {e}")
            return {}

    def calculate_risk_reward_ratio(self, entry_price: float, stop_price: float,
                                  target_price: float, position_type: str = 'long') -> float:
        try:
            if position_type == 'long':
                risk = abs(entry_price - stop_price)
                reward = abs(target_price - entry_price)
            else:
                risk = abs(stop_price - entry_price)
                reward = abs(entry_price - target_price)
                
            if risk == 0:
                return 0
                
            return reward / risk
        except Exception as e:
            logger.error(f"计算风险收益比失败: {e}")
            return 0

    def optimize_stop_loss(self, symbol: str, df: pd.DataFrame, 
                          position_type: str = 'long') -> Dict:
        try:
            results = {}
            
            for multiplier in [1.0, 1.5, 2.0, 2.5, 3.0]:
                self.atr_multiplier = multiplier
                
                stop_prices = []
                for i in range(len(df) - self.atr_period):
                    entry_price = df['price'].iloc[i]
                    window_df = df.iloc[i:i+self.atr_period+1]
                    stop_price = self.calculate_atr_stop(entry_price, window_df, position_type)
                    stop_prices.append(stop_price)
                
                results[multiplier] = {
                    'avg_stop_distance': np.mean([abs(df['price'].iloc[i] - sp) / df['price'].iloc[i] 
                                                 for i, sp in enumerate(stop_prices)]),
                    'stop_prices': stop_prices
                }
            
            self.atr_multiplier = self.config.get('atr_multiplier', 2.0)
            
            return results
        except Exception as e:
            logger.error(f"优化止损失败: {e}")
            return {}