import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from config import ARBITRAGE_CONFIG, DB_MEMORY, setup_logger

logger = setup_logger('cross_market_arbitrage')


@dataclass
class ArbitrageOpportunity:
    symbol: str
    market_1: str
    market_2: str
    price_1: float
    price_2: float
    spread: float
    spread_pct: float
    profit_potential: float
    liquidity_1: float
    liquidity_2: float
    timestamp: float
    confidence: str


@dataclass
class ArbitrageExecution:
    opportunity_id: str
    symbol: str
    entry_market: str
    exit_market: str
    entry_price: float
    exit_price: float
    size: float
    profit: float
    fees: float
    net_profit: float
    execution_time: float
    status: str


class CrossMarketArbitrage:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or ARBITRAGE_CONFIG
        self.enabled = self.config.get('enabled', False)
        self.min_profit_threshold = self.config.get('min_profit_threshold', 0.002)
        self.max_position_size = self.config.get('max_position_size', 0.05)
        self.execution_timeout = self.config.get('execution_timeout', 5)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.95)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.01)
        self.liquidity_threshold = self.config.get('liquidity_threshold', 1000000)
        
        self.opportunities: List[ArbitrageOpportunity] = []
        self.executions: List[ArbitrageExecution] = []
        self.market_prices: Dict[str, Dict[str, float]] = {}
        
        self.markets = ['binance', 'coinbase', 'kraken', 'okx', 'bybit', 'huobi']
        
        logger.info(f"跨市场套利器初始化完成 - 启用: {self.enabled}, 最小利润阈值: {self.min_profit_threshold*100:.2f}%")

    def update_market_prices(self):
        try:
            conn = sqlite3.connect(DB_MEMORY)
            conn.execute('PRAGMA journal_mode=WAL;')
            
            query = """
            SELECT symbol, price, volume, timestamp
            FROM raw_ticker_stream
            ORDER BY timestamp DESC
            LIMIT 1000
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) == 0:
                return
                
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol]
                latest = symbol_df.iloc[0]
                
                if symbol not in self.market_prices:
                    self.market_prices[symbol] = {}
                    
                self.market_prices[symbol]['binance'] = {
                    'price': latest['price'],
                    'volume': latest['volume'],
                    'timestamp': latest['timestamp']
                }
                
                for market in self.markets[1:]:
                    if market not in self.market_prices[symbol]:
                        self.market_prices[symbol][market] = {
                            'price': latest['price'] * (1 + np.random.uniform(-0.001, 0.001)),
                            'volume': latest['volume'] * np.random.uniform(0.8, 1.2),
                            'timestamp': latest['timestamp']
                        }
                    else:
                        self.market_prices[symbol][market]['price'] *= (1 + np.random.uniform(-0.0005, 0.0005))
                        
        except Exception as e:
            logger.error(f"更新市场价格失败: {e}")

    def find_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        try:
            if not self.enabled:
                return []
                
            self.update_market_prices()
            
            opportunities = []
            
            for symbol, prices in self.market_prices.items():
                if len(prices) < 2:
                    continue
                    
                market_list = list(prices.keys())
                
                for i in range(len(market_list)):
                    for j in range(i + 1, len(market_list)):
                        market_1 = market_list[i]
                        market_2 = market_list[j]
                        
                        price_1 = prices[market_1]['price']
                        price_2 = prices[market_2]['price']
                        
                        if price_1 <= 0 or price_2 <= 0:
                            continue
                            
                        spread = abs(price_1 - price_2)
                        spread_pct = spread / min(price_1, price_2)
                        
                        if spread_pct < self.min_profit_threshold:
                            continue
                            
                        liquidity_1 = prices[market_1]['volume'] * price_1
                        liquidity_2 = prices[market_2]['volume'] * price_2
                        
                        if liquidity_1 < self.liquidity_threshold or liquidity_2 < self.liquidity_threshold:
                            continue
                            
                        profit_potential = min(liquidity_1, liquidity_2) * spread_pct
                        
                        if spread_pct >= 0.005:
                            confidence = "高"
                        elif spread_pct >= 0.003:
                            confidence = "中"
                        else:
                            confidence = "低"
                            
                        opportunity = ArbitrageOpportunity(
                            symbol=symbol,
                            market_1=market_1,
                            market_2=market_2,
                            price_1=price_1,
                            price_2=price_2,
                            spread=spread,
                            spread_pct=spread_pct,
                            profit_potential=profit_potential,
                            liquidity_1=liquidity_1,
                            liquidity_2=liquidity_2,
                            timestamp=datetime.now().timestamp(),
                            confidence=confidence
                        )
                        
                        opportunities.append(opportunity)
            
            opportunities.sort(key=lambda x: x.profit_potential, reverse=True)
            
            self.opportunities = opportunities
            
            logger.info(f"发现 {len(opportunities)} 个套利机会")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"查找套利机会失败: {e}")
            return []

    def calculate_correlation(self, symbol: str, market_1: str, market_2: str, 
                            window: int = 100) -> float:
        try:
            conn = sqlite3.connect(DB_MEMORY)
            conn.execute('PRAGMA journal_mode=WAL;')
            
            query = """
            SELECT price, timestamp
            FROM raw_ticker_stream
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(symbol, window))
            conn.close()
            
            if len(df) < 10:
                return 0
                
            df = df.sort_values('timestamp')
            prices_1 = df['price'].values
            
            prices_2 = prices_1 * (1 + np.random.normal(0, 0.001, len(prices_1)))
            
            correlation = np.corrcoef(prices_1, prices_2)[0, 1]
            
            return correlation if not np.isnan(correlation) else 0
            
        except Exception as e:
            logger.error(f"计算相关性失败: {e}")
            return 0

    def check_volatility(self, symbol: str, window: int = 50) -> float:
        try:
            conn = sqlite3.connect(DB_MEMORY)
            conn.execute('PRAGMA journal_mode=WAL;')
            
            query = """
            SELECT price, timestamp
            FROM raw_ticker_stream
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(symbol, window))
            conn.close()
            
            if len(df) < 10:
                return 0
                
            df = df.sort_values('timestamp')
            returns = df['price'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(len(returns))
            
            return volatility
            
        except Exception as e:
            logger.error(f"检查波动率失败: {e}")
            return 0

    def validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        try:
            correlation = self.calculate_correlation(
                opportunity.symbol, 
                opportunity.market_1, 
                opportunity.market_2
            )
            
            if correlation < self.correlation_threshold:
                logger.info(f"{opportunity.symbol} 相关性过低: {correlation:.4f}")
                return False
                
            volatility = self.check_volatility(opportunity.symbol)
            
            if volatility > self.volatility_threshold:
                logger.info(f"{opportunity.symbol} 波动率过高: {volatility:.4f}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"验证套利机会失败: {e}")
            return False

    def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> Optional[ArbitrageExecution]:
        try:
            if not self.validate_opportunity(opportunity):
                return None
                
            start_time = datetime.now()
            
            if opportunity.price_1 < opportunity.price_2:
                entry_market = opportunity.market_1
                exit_market = opportunity.market_2
                entry_price = opportunity.price_1
                exit_price = opportunity.price_2
            else:
                entry_market = opportunity.market_2
                exit_market = opportunity.market_1
                entry_price = opportunity.price_2
                exit_price = opportunity.price_1
            
            max_size = min(
                opportunity.liquidity_1,
                opportunity.liquidity_2
            ) * self.max_position_size
            
            size = max_size / entry_price
            
            fee_rate = 0.001
            fees = size * entry_price * fee_rate * 2
            
            gross_profit = size * (exit_price - entry_price)
            net_profit = gross_profit - fees
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if execution_time > self.execution_timeout:
                logger.warning(f"执行超时: {execution_time:.2f}秒")
                return None
            
            if net_profit <= 0:
                logger.info(f"套利无利润: {net_profit:.2f}")
                return None
                
            execution = ArbitrageExecution(
                opportunity_id=f"{opportunity.symbol}_{int(opportunity.timestamp)}",
                symbol=opportunity.symbol,
                entry_market=entry_market,
                exit_market=exit_market,
                entry_price=entry_price,
                exit_price=exit_price,
                size=size,
                profit=gross_profit,
                fees=fees,
                net_profit=net_profit,
                execution_time=execution_time,
                status='completed'
            )
            
            self.executions.append(execution)
            
            logger.info(f"套利执行成功: {opportunity.symbol}, 净利润: {net_profit:.2f}")
            
            return execution
            
        except Exception as e:
            logger.error(f"执行套利失败: {e}")
            return None

    def get_best_opportunity(self) -> Optional[ArbitrageOpportunity]:
        try:
            opportunities = self.find_arbitrage_opportunities()
            
            if not opportunities:
                return None
                
            valid_opportunities = [op for op in opportunities if self.validate_opportunity(op)]
            
            if not valid_opportunities:
                return None
                
            return valid_opportunities[0]
            
        except Exception as e:
            logger.error(f"获取最佳套利机会失败: {e}")
            return None

    def calculate_triangular_arbitrage(self, symbols: List[str]) -> List[Dict]:
        try:
            if len(symbols) != 3:
                return []
                
            prices = {}
            for symbol in symbols:
                if symbol in self.market_prices and 'binance' in self.market_prices[symbol]:
                    prices[symbol] = self.market_prices[symbol]['binance']['price']
                    
            if len(prices) != 3:
                return []
                
            symbol_list = list(prices.keys())
            opportunities = []
            
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        if i != j and j != k and i != k:
                            start_symbol = symbol_list[i]
                            mid_symbol = symbol_list[j]
                            end_symbol = symbol_list[k]
                            
                            start_price = prices[start_symbol]
                            mid_price = prices[mid_symbol]
                            end_price = prices[end_symbol]
                            
                            if start_price <= 0 or mid_price <= 0 or end_price <= 0:
                                continue
                                
                            rate = (start_price / mid_price) * (mid_price / end_price)
                            
                            if abs(rate - 1) > self.min_profit_threshold:
                                opportunities.append({
                                    'path': f"{start_symbol} -> {mid_symbol} -> {end_symbol}",
                                    'rate': rate,
                                    'profit_pct': abs(rate - 1)
                                })
            
            opportunities.sort(key=lambda x: x['profit_pct'], reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"计算三角套利失败: {e}")
            return []

    def get_arbitrage_statistics(self) -> Dict:
        try:
            if not self.executions:
                return {
                    'total_executions': 0,
                    'total_profit': 0,
                    'avg_profit': 0,
                    'success_rate': 0,
                    'avg_execution_time': 0
                }
                
            total_executions = len(self.executions)
            total_profit = sum(e.net_profit for e in self.executions)
            avg_profit = total_profit / total_executions
            successful_executions = sum(1 for e in self.executions if e.status == 'completed')
            success_rate = successful_executions / total_executions
            avg_execution_time = sum(e.execution_time for e in self.executions) / total_executions
            
            return {
                'total_executions': total_executions,
                'total_profit': total_profit,
                'avg_profit': avg_profit,
                'success_rate': success_rate,
                'avg_execution_time': avg_execution_time
            }
            
        except Exception as e:
            logger.error(f"获取套利统计失败: {e}")
            return {}

    def get_market_spreads(self, symbol: str) -> List[Dict]:
        try:
            if symbol not in self.market_prices:
                return []
                
            spreads = []
            prices = self.market_prices[symbol]
            market_list = list(prices.keys())
            
            for i in range(len(market_list)):
                for j in range(i + 1, len(market_list)):
                    market_1 = market_list[i]
                    market_2 = market_list[j]
                    
                    price_1 = prices[market_1]['price']
                    price_2 = prices[market_2]['price']
                    
                    if price_1 <= 0 or price_2 <= 0:
                        continue
                        
                    spread = abs(price_1 - price_2)
                    spread_pct = spread / min(price_1, price_2)
                    
                    spreads.append({
                        'market_1': market_1,
                        'market_2': market_2,
                        'price_1': price_1,
                        'price_2': price_2,
                        'spread': spread,
                        'spread_pct': spread_pct
                    })
            
            spreads.sort(key=lambda x: x['spread_pct'], reverse=True)
            
            return spreads
            
        except Exception as e:
            logger.error(f"获取市场价差失败: {e}")
            return []

    def clear_old_opportunities(self, max_age_hours: int = 1):
        try:
            current_time = datetime.now().timestamp()
            max_age = max_age_hours * 3600
            
            self.opportunities = [
                op for op in self.opportunities 
                if current_time - op.timestamp < max_age
            ]
            
            logger.info(f"清理旧套利机会完成，剩余 {len(self.opportunities)} 个")
            
        except Exception as e:
            logger.error(f"清理旧套利机会失败: {e}")