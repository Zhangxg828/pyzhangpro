import sqlite3
import pandas as pd
import numpy as np
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from config import HIGH_FREQUENCY_CONFIG, DB_MEMORY, setup_logger

logger = setup_logger('high_frequency_trader')


@dataclass
class OrderBook:
    symbol: str
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    timestamp: float
    spread: float
    mid_price: float


@dataclass
class HFTrade:
    symbol: str
    side: str
    price: float
    size: float
    timestamp: float
    execution_time: float
    slippage: float
    status: str


@dataclass
class HFSignal:
    symbol: str
    signal_type: str
    strength: float
    entry_price: float
    target_price: float
    stop_price: float
    timestamp: float
    confidence: str


class HighFrequencyTrader:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or HIGH_FREQUENCY_CONFIG
        self.enabled = self.config.get('enabled', False)
        self.tick_interval = self.config.get('tick_interval', 0.1)
        self.order_book_depth = self.config.get('order_book_depth', 20)
        self.latency_threshold = self.config.get('latency_threshold', 0.01)
        self.position_duration = self.config.get('position_duration', 60)
        self.max_orders_per_minute = self.config.get('max_orders_per_minute', 100)
        self.slippage_tolerance = self.config.get('slippage_tolerance', 0.0001)
        
        self.order_books: Dict[str, OrderBook] = {}
        self.trades: List[HFTrade] = []
        self.signals: List[HFSignal] = []
        self.positions: Dict[str, Dict] = {}
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        
        self.order_count = 0
        self.last_order_time = 0
        self.is_running = False
        
        logger.info(f"高频交易器初始化完成 - 启用: {self.enabled}, Tick间隔: {self.tick_interval}秒")

    def update_order_book(self, symbol: str, bids: List[Tuple[float, float]], 
                         asks: List[Tuple[float, float]], timestamp: float):
        try:
            if not bids or not asks:
                return
                
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            
            order_book = OrderBook(
                symbol=symbol,
                bids=bids[:self.order_book_depth],
                asks=asks[:self.order_book_depth],
                timestamp=timestamp,
                spread=spread,
                mid_price=mid_price
            )
            
            self.order_books[symbol] = order_book
            
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=1000)
                
            self.price_history[symbol].append(mid_price)
            
        except Exception as e:
            logger.error(f"更新订单簿失败: {e}")

    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        try:
            return self.order_books.get(symbol)
        except Exception as e:
            logger.error(f"获取订单簿失败: {e}")
            return None

    def calculate_order_flow_imbalance(self, symbol: str) -> float:
        try:
            order_book = self.get_order_book(symbol)
            if not order_book:
                return 0
                
            bid_volume = sum(size for price, size in order_book.bids)
            ask_volume = sum(size for price, size in order_book.asks)
            
            if bid_volume + ask_volume == 0:
                return 0
                
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            return imbalance
            
        except Exception as e:
            logger.error(f"计算订单流不平衡失败: {e}")
            return 0

    def calculate_price_momentum(self, symbol: str, window: int = 10) -> float:
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < window:
                return 0
                
            prices = list(self.price_history[symbol])[-window:]
            momentum = (prices[-1] - prices[0]) / prices[0]
            
            return momentum
            
        except Exception as e:
            logger.error(f"计算价格动量失败: {e}")
            return 0

    def calculate_volatility(self, symbol: str, window: int = 20) -> float:
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < window:
                return 0
                
            prices = list(self.price_history[symbol])[-window:]
            returns = pd.Series(prices).pct_change().dropna()
            volatility = returns.std() * np.sqrt(len(returns))
            
            return volatility
            
        except Exception as e:
            logger.error(f"计算波动率失败: {e}")
            return 0

    def generate_signal(self, symbol: str) -> Optional[HFSignal]:
        try:
            if not self.enabled:
                return None
                
            order_book = self.get_order_book(symbol)
            if not order_book:
                return None
                
            imbalance = self.calculate_order_flow_imbalance(symbol)
            momentum = self.calculate_price_momentum(symbol)
            volatility = self.calculate_volatility(symbol)
            
            signal_strength = 0
            signal_type = 'hold'
            
            if imbalance > 0.3 and momentum > 0.001:
                signal_type = 'buy'
                signal_strength = min(imbalance + momentum * 100, 1.0)
            elif imbalance < -0.3 and momentum < -0.001:
                signal_type = 'sell'
                signal_strength = min(abs(imbalance) + abs(momentum) * 100, 1.0)
                
            if volatility > 0.02:
                signal_strength *= 0.5
                
            if signal_strength < 0.3:
                return None
                
            mid_price = order_book.mid_price
            spread_pct = order_book.spread / mid_price
            
            if signal_type == 'buy':
                target_price = mid_price * (1 + spread_pct * 5)
                stop_price = mid_price * (1 - spread_pct * 3)
            elif signal_type == 'sell':
                target_price = mid_price * (1 - spread_pct * 5)
                stop_price = mid_price * (1 + spread_pct * 3)
            else:
                return None
                
            if signal_strength >= 0.7:
                confidence = "高"
            elif signal_strength >= 0.5:
                confidence = "中"
            else:
                confidence = "低"
                
            signal = HFSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=signal_strength,
                entry_price=mid_price,
                target_price=target_price,
                stop_price=stop_price,
                timestamp=datetime.now().timestamp(),
                confidence=confidence
            )
            
            self.signals.append(signal)
            
            logger.info(f"生成高频信号: {symbol}, 类型: {signal_type}, 强度: {signal_strength:.4f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"生成信号失败: {e}")
            return None

    def execute_trade(self, signal: HFSignal, size: float = 0.001) -> Optional[HFTrade]:
        try:
            if not self.enabled:
                return None
                
            current_time = datetime.now().timestamp()
            
            if self.order_count >= self.max_orders_per_minute:
                if current_time - self.last_order_time < 60:
                    logger.warning("达到订单限制")
                    return None
                else:
                    self.order_count = 0
                    
            order_book = self.get_order_book(signal.symbol)
            if not order_book:
                return None
                
            start_time = datetime.now()
            
            if signal.signal_type == 'buy':
                execution_price = order_book.asks[0][0]
            elif signal.signal_type == 'sell':
                execution_price = order_book.bids[0][0]
            else:
                return None
                
            slippage = abs(execution_price - signal.entry_price) / signal.entry_price
            
            if slippage > self.slippage_tolerance:
                logger.warning(f"滑点过大: {slippage:.6f}")
                return None
                
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if execution_time > self.latency_threshold:
                logger.warning(f"执行延迟过高: {execution_time:.6f}秒")
                
            trade = HFTrade(
                symbol=signal.symbol,
                side=signal.signal_type,
                price=execution_price,
                size=size,
                timestamp=current_time,
                execution_time=execution_time,
                slippage=slippage,
                status='completed'
            )
            
            self.trades.append(trade)
            self.order_count += 1
            self.last_order_time = current_time
            
            if signal.symbol not in self.positions:
                self.positions[signal.symbol] = {
                    'entry_price': execution_price,
                    'size': size,
                    'side': signal.signal_type,
                    'entry_time': current_time,
                    'target_price': signal.target_price,
                    'stop_price': signal.stop_price
                }
            else:
                position = self.positions[signal.symbol]
                if position['side'] != signal.signal_type:
                    del self.positions[signal.symbol]
                else:
                    position['size'] += size
                    
            logger.info(f"执行高频交易: {signal.symbol}, 价格: {execution_price}, 大小: {size}")
            
            return trade
            
        except Exception as e:
            logger.error(f"执行交易失败: {e}")
            return None

    def check_positions(self):
        try:
            current_time = datetime.now().timestamp()
            symbols_to_close = []
            
            for symbol, position in self.positions.items():
                order_book = self.get_order_book(symbol)
                if not order_book:
                    continue
                    
                current_price = order_book.mid_price
                
                if position['side'] == 'buy':
                    if current_price >= position['target_price']:
                        symbols_to_close.append(symbol)
                    elif current_price <= position['stop_price']:
                        symbols_to_close.append(symbol)
                else:
                    if current_price <= position['target_price']:
                        symbols_to_close.append(symbol)
                    elif current_price >= position['stop_price']:
                        symbols_to_close.append(symbol)
                        
                if current_time - position['entry_time'] >= self.position_duration:
                    symbols_to_close.append(symbol)
                    
            for symbol in symbols_to_close:
                self.close_position(symbol)
                
        except Exception as e:
            logger.error(f"检查仓位失败: {e}")

    def close_position(self, symbol: str) -> Optional[HFTrade]:
        try:
            if symbol not in self.positions:
                return None
                
            position = self.positions[symbol]
            order_book = self.get_order_book(symbol)
            if not order_book:
                return None
                
            current_price = order_book.mid_price
            
            close_side = 'sell' if position['side'] == 'buy' else 'buy'
            
            start_time = datetime.now()
            
            if close_side == 'buy':
                execution_price = order_book.asks[0][0]
            else:
                execution_price = order_book.bids[0][0]
                
            execution_time = (datetime.now() - start_time).total_seconds()
            
            slippage = abs(execution_price - current_price) / current_price
            
            trade = HFTrade(
                symbol=symbol,
                side=close_side,
                price=execution_price,
                size=position['size'],
                timestamp=datetime.now().timestamp(),
                execution_time=execution_time,
                slippage=slippage,
                status='closed'
            )
            
            self.trades.append(trade)
            del self.positions[symbol]
            
            logger.info(f"平仓: {symbol}, 价格: {execution_price}")
            
            return trade
            
        except Exception as e:
            logger.error(f"平仓失败: {e}")
            return None

    def calculate_pnl(self) -> float:
        try:
            total_pnl = 0
            
            for symbol, position in self.positions.items():
                order_book = self.get_order_book(symbol)
                if not order_book:
                    continue
                    
                current_price = order_book.mid_price
                
                if position['side'] == 'buy':
                    pnl = (current_price - position['entry_price']) * position['size']
                else:
                    pnl = (position['entry_price'] - current_price) * position['size']
                    
                total_pnl += pnl
                
            return total_pnl
            
        except Exception as e:
            logger.error(f"计算盈亏失败: {e}")
            return 0

    def get_trading_statistics(self) -> Dict:
        try:
            if not self.trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'avg_execution_time': 0,
                    'avg_slippage': 0,
                    'total_pnl': 0
                }
                
            total_trades = len(self.trades)
            avg_execution_time = sum(t.execution_time for t in self.trades) / total_trades
            avg_slippage = sum(t.slippage for t in self.trades) / total_trades
            
            winning_trades = sum(1 for t in self.trades if t.status == 'completed')
            losing_trades = sum(1 for t in self.trades if t.status == 'closed')
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = self.calculate_pnl()
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_execution_time': avg_execution_time,
                'avg_slippage': avg_slippage,
                'total_pnl': total_pnl
            }
            
        except Exception as e:
            logger.error(f"获取交易统计失败: {e}")
            return {}

    async def run(self, symbols: List[str]):
        try:
            self.is_running = True
            logger.info(f"高频交易开始运行，监控 {len(symbols)} 个交易对")
            
            while self.is_running:
                start_time = datetime.now()
                
                for symbol in symbols:
                    try:
                        signal = self.generate_signal(symbol)
                        
                        if signal and signal.strength >= 0.5:
                            self.execute_trade(signal)
                            
                    except Exception as e:
                        logger.error(f"处理 {symbol} 失败: {e}")
                
                self.check_positions()
                
                elapsed = (datetime.now() - start_time).total_seconds()
                sleep_time = max(0, self.tick_interval - elapsed)
                
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"高频交易运行失败: {e}")
        finally:
            self.is_running = False
            logger.info("高频交易已停止")

    def stop(self):
        try:
            self.is_running = False
            
            for symbol in list(self.positions.keys()):
                self.close_position(symbol)
                
            logger.info("高频交易已停止并平仓所有仓位")
            
        except Exception as e:
            logger.error(f"停止高频交易失败: {e}")

    def clear_old_data(self, max_age_minutes: int = 60):
        try:
            current_time = datetime.now().timestamp()
            max_age = max_age_minutes * 60
            
            self.signals = [s for s in self.signals if current_time - s.timestamp < max_age]
            
            self.trades = [t for t in self.trades if current_time - t.timestamp < max_age]
            
            logger.info(f"清理旧数据完成，剩余 {len(self.signals)} 个信号，{len(self.trades)} 笔交易")
            
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")

    def get_active_positions(self) -> Dict[str, Dict]:
        try:
            positions_info = {}
            
            for symbol, position in self.positions.items():
                order_book = self.get_order_book(symbol)
                current_price = order_book.mid_price if order_book else 0
                
                if position['side'] == 'buy':
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                else:
                    unrealized_pnl = (position['entry_price'] - current_price) * position['size']
                    
                positions_info[symbol] = {
                    'entry_price': position['entry_price'],
                    'size': position['size'],
                    'side': position['side'],
                    'entry_time': position['entry_time'],
                    'current_price': current_price,
                    'unrealized_pnl': unrealized_pnl,
                    'target_price': position['target_price'],
                    'stop_price': position['stop_price']
                }
                
            return positions_info
            
        except Exception as e:
            logger.error(f"获取活跃仓位失败: {e}")
            return {}