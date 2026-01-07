import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    回测引擎模块
    
    功能：
    1. 历史数据回测
    2. 计算绩效指标
    3. 生成回测报告
    4. 策略优化
    5. 风险分析
    """
    
    def __init__(self, db_path: str, config: Optional[Dict] = None):
        self.db_path = db_path
        
        if config:
            self.initial_capital = config.get('initial_capital', 100000)
            self.commission = config.get('commission', 0.001)
            self.slippage = config.get('slippage', 0.0005)
            self.risk_free_rate = config.get('risk_free_rate', 0.02)
        else:
            self.initial_capital = 100000
            self.commission = 0.001
            self.slippage = 0.0005
            self.risk_free_rate = 0.02
        
        logger.info(f"回测引擎初始化完成 - 初始资金:{self.initial_capital}, 手续费:{self.commission*100}%")
    
    def load_historical_data(self, symbol: str, start_date: str, 
                             end_date: str) -> pd.DataFrame:
        """
        加载历史数据
        
        Args:
            symbol: 交易对
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            历史数据 DataFrame
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('PRAGMA journal_mode=WAL;')
            
            query = """
                SELECT * FROM history
                WHERE symbol = ? 
                AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
            conn.close()
            
            if df.empty:
                logger.warning(f"未找到 {symbol} 的历史数据")
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"加载 {symbol} 历史数据: {len(df)} 条记录")
            return df
        
        except Exception as e:
            logger.error(f"加载历史数据失败: {e}")
            return pd.DataFrame()
    
    def backtest_strategy(self, strategy: Callable, symbol: str, 
                          start_date: str, end_date: str,
                          params: Optional[Dict] = None) -> Dict:
        """
        回测策略
        
        Args:
            strategy: 策略函数
            symbol: 交易对
            start_date: 开始日期
            end_date: 结束日期
            params: 策略参数
        
        Returns:
            回测结果字典
        """
        df = self.load_historical_data(symbol, start_date, end_date)
        
        if df.empty:
            return {'error': '无历史数据'}
        
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = [capital]
        peak_equity = capital
        max_drawdown = 0
        
        for i in range(len(df)):
            current_data = df.iloc[:i+1]
            
            signal = strategy(current_data, params)
            
            if signal == 'BUY' and position == 0:
                price = df.iloc[i]['price'] * (1 + self.slippage)
                quantity = (capital * 0.95) / price
                cost = quantity * price * (1 + self.commission)
                
                if cost <= capital:
                    capital -= cost
                    position = quantity
                    trades.append({
                        'type': 'BUY',
                        'timestamp': df.index[i],
                        'price': price,
                        'quantity': quantity,
                        'cost': cost
                    })
            
            elif signal == 'SELL' and position > 0:
                price = df.iloc[i]['price'] * (1 - self.slippage)
                revenue = position * price * (1 - self.commission)
                
                profit = revenue - trades[-1]['cost']
                profit_pct = profit / trades[-1]['cost']
                
                capital += revenue
                trades[-1]['exit_price'] = price
                trades[-1]['exit_timestamp'] = df.index[i]
                trades[-1]['profit'] = profit
                trades[-1]['profit_pct'] = profit_pct
                
                position = 0
            
            current_equity = capital + position * df.iloc[i]['price']
            equity_curve.append(current_equity)
            
            if current_equity > peak_equity:
                peak_equity = current_equity
            
            drawdown = (peak_equity - current_equity) / peak_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        if position > 0:
            price = df.iloc[-1]['price'] * (1 - self.slippage)
            revenue = position * price * (1 - self.commission)
            profit = revenue - trades[-1]['cost']
            profit_pct = profit / trades[-1]['cost']
            
            capital += revenue
            trades[-1]['exit_price'] = price
            trades[-1]['exit_timestamp'] = df.index[-1]
            trades[-1]['profit'] = profit
            trades[-1]['profit_pct'] = profit_pct
        
        metrics = self.calculate_metrics(trades, equity_curve)
        
        return {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': (capital - self.initial_capital) / self.initial_capital,
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': metrics
        }
    
    def calculate_metrics(self, trades: List[Dict], equity_curve: List[float]) -> Dict:
        """
        计算绩效指标
        
        Args:
            trades: 交易列表
            equity_curve: 权益曲线
        
        Returns:
            绩效指标字典
        """
        if not trades:
            return {}
        
        completed_trades = [t for t in trades if 'profit' in t]
        
        if not completed_trades:
            return {}
        
        profits = [t['profit'] for t in completed_trades]
        win_trades = [p for p in profits if p > 0]
        loss_trades = [p for p in profits if p < 0]
        
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        peak_equity = np.maximum.accumulate(equity_curve)
        drawdown = (peak_equity - np.array(equity_curve)) / peak_equity
        max_drawdown = np.max(drawdown)
        
        win_rate = len(win_trades) / len(completed_trades) if completed_trades else 0
        
        avg_win = np.mean(win_trades) if win_trades else 0
        avg_loss = np.mean(loss_trades) if loss_trades else 0
        
        profit_factor = abs(sum(win_trades) / sum(loss_trades)) if loss_trades else float('inf')
        
        avg_trade_duration = np.mean([(t['exit_timestamp'] - t['timestamp']).total_seconds() / 3600 
                                      for t in completed_trades])
        
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        sortino_ratio = np.mean(returns) / np.std([r for r in returns if r < 0]) * np.sqrt(252) \
                       if len([r for r in returns if r < 0]) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade_duration': avg_trade_duration,
            'total_trades': len(completed_trades),
            'winning_trades': len(win_trades),
            'losing_trades': len(loss_trades)
        }
    
    def generate_report(self, backtest_result: Dict) -> str:
        """
        生成回测报告
        
        Args:
            backtest_result: 回测结果
        
        Returns:
            回测报告字符串
        """
        if 'error' in backtest_result:
            return f"回测失败: {backtest_result['error']}"
        
        metrics = backtest_result['metrics']
        
        report = f"""
{'='*60}
回测报告
{'='*60}

交易对: {backtest_result['symbol']}
回测期间: {backtest_result['start_date']} 至 {backtest_result['end_date']}
初始资金: ${backtest_result['initial_capital']:,.2f}
最终资金: ${backtest_result['final_capital']:,.2f}
总收益率: {backtest_result['total_return']*100:.2f}%

{'='*60}
绩效指标
{'='*60}

夏普比率: {metrics.get('sharpe_ratio', 0):.2f}
索提诺比率: {metrics.get('sortino_ratio', 0):.2f}
卡玛比率: {metrics.get('calmar_ratio', 0):.2f}
最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%

{'='*60}
交易统计
{'='*60}

总交易次数: {metrics.get('total_trades', 0)}
盈利交易: {metrics.get('winning_trades', 0)}
亏损交易: {metrics.get('losing_trades', 0)}
胜率: {metrics.get('win_rate', 0)*100:.2f}%
盈亏比: {metrics.get('profit_factor', 0):.2f}
平均盈利: ${metrics.get('avg_win', 0):.2f}
平均亏损: ${metrics.get('avg_loss', 0):.2f}
平均持仓时间: {metrics.get('avg_trade_duration', 0):.2f} 小时

{'='*60}
"""
        return report
    
    def optimize_strategy(self, strategy: Callable, symbol: str,
                         start_date: str, end_date: str,
                         param_grid: Dict) -> Dict:
        """
        策略参数优化
        
        Args:
            strategy: 策略函数
            symbol: 交易对
            start_date: 开始日期
            end_date: 结束日期
            param_grid: 参数网格
        
        Returns:
            优化结果
        """
        best_result = None
        best_sharpe = -float('inf')
        best_params = None
        
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        
        logger.info(f"开始参数优化，共 {total_combinations} 种组合")
        
        from itertools import product
        keys = param_grid.keys()
        values = param_grid.values()
        
        for i, combination in enumerate(product(*values)):
            params = dict(zip(keys, combination))
            
            try:
                result = self.backtest_strategy(strategy, symbol, start_date, end_date, params)
                
                if 'metrics' in result:
                    sharpe = result['metrics'].get('sharpe_ratio', -float('inf'))
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_result = result
                        best_params = params
                
                if (i + 1) % 10 == 0:
                    logger.info(f"已完成 {i+1}/{total_combinations} 种组合")
            
            except Exception as e:
                logger.error(f"参数组合 {params} 回测失败: {e}")
                continue
        
        logger.info(f"参数优化完成，最佳夏普比率: {best_sharpe:.2f}")
        
        return {
            'best_params': best_params,
            'best_result': best_result,
            'best_sharpe': best_sharpe
        }
    
    def monte_carlo_simulation(self, backtest_result: Dict, 
                             num_simulations: int = 1000) -> Dict:
        """
        蒙特卡洛模拟
        
        Args:
            backtest_result: 回测结果
            num_simulations: 模拟次数（默认1000）
        
        Returns:
            模拟结果
        """
        trades = backtest_result.get('trades', [])
        completed_trades = [t for t in trades if 'profit' in t]
        
        if not completed_trades:
            return {'error': '无交易数据'}
        
        profits = [t['profit'] for t in completed_trades]
        mean_profit = np.mean(profits)
        std_profit = np.std(profits)
        
        final_capitals = []
        
        for _ in range(num_simulations):
            simulated_profits = np.random.normal(mean_profit, std_profit, len(completed_trades))
            final_capital = self.initial_capital + np.sum(simulated_profits)
            final_capitals.append(final_capital)
        
        final_capitals = np.array(final_capitals)
        
        percentiles = np.percentile(final_capitals, [5, 25, 50, 75, 95])
        
        return {
            'mean_final_capital': np.mean(final_capitals),
            'std_final_capital': np.std(final_capitals),
            'min_final_capital': np.min(final_capitals),
            'max_final_capital': np.max(final_capitals),
            'percentiles': {
                '5%': percentiles[0],
                '25%': percentiles[1],
                '50%': percentiles[2],
                '75%': percentiles[3],
                '95%': percentiles[4]
            },
            'probability_of_loss': np.sum(final_capitals < self.initial_capital) / num_simulations
        }
    
    def analyze_risk(self, backtest_result: Dict) -> Dict:
        """
        风险分析
        
        Args:
            backtest_result: 回测结果
        
        Returns:
            风险分析结果
        """
        equity_curve = backtest_result.get('equity_curve', [])
        trades = backtest_result.get('trades', [])
        
        if not equity_curve:
            return {'error': '无权益曲线数据'}
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        cvar_95 = np.mean([r for r in returns if r <= var_95])
        cvar_99 = np.mean([r for r in returns if r <= var_99])
        
        peak_equity = np.maximum.accumulate(equity_curve)
        drawdown = (peak_equity - np.array(equity_curve)) / peak_equity
        
        avg_drawdown = np.mean(drawdown[drawdown > 0]) if np.any(drawdown > 0) else 0
        max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown)
        
        completed_trades = [t for t in trades if 'profit' in t]
        if completed_trades:
            profits = [t['profit'] for t in completed_trades]
            max_consecutive_losses = self._calculate_max_consecutive_losses(profits)
        else:
            max_consecutive_losses = 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_drawdown': np.max(drawdown),
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'max_consecutive_losses': max_consecutive_losses
        }
    
    def _calculate_max_drawdown_duration(self, drawdown: np.ndarray) -> int:
        """计算最大回撤持续时间"""
        in_drawdown = False
        current_duration = 0
        max_duration = 0
        
        for dd in drawdown:
            if dd > 0:
                if not in_drawdown:
                    in_drawdown = True
                current_duration += 1
            else:
                if in_drawdown:
                    max_duration = max(max_duration, current_duration)
                    current_duration = 0
                    in_drawdown = False
        
        return max_duration
    
    def _calculate_max_consecutive_losses(self, profits: List[float]) -> int:
        """计算最大连续亏损次数"""
        max_consecutive = 0
        current_consecutive = 0
        
        for profit in profits:
            if profit < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive