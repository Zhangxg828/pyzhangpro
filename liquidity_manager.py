import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LiquidityManager:
    """
    流动性管理模块
    
    功能：
    1. 检查市场流动性
    2. 计算预期滑点
    3. 计算最优交易量
    4. 检测流动性风险
    5. 优化订单执行
    """
    
    def __init__(self, config: Optional[Dict] = None):
        if config:
            self.min_depth = config.get('min_depth', 100000)
            self.max_slippage = config.get('max_slippage', 0.005)
            self.large_order_threshold = config.get('large_order_threshold', 100)
            self.liquidity_ratio_threshold = config.get('liquidity_ratio_threshold', 0.1)
        else:
            self.min_depth = 100000
            self.max_slippage = 0.005
            self.large_order_threshold = 100
            self.liquidity_ratio_threshold = 0.1
        
        logger.info(f"流动性管理器初始化完成 - 最小深度:{self.min_depth}, 最大滑点:{self.max_slippage*100}%")
    
    def check_liquidity(self, order_book: Dict, symbol: str, 
                       order_size: Optional[float] = None) -> Tuple[bool, str, Dict]:
        """
        检查流动性
        
        Args:
            order_book: 订单簿数据
            symbol: 交易对
            order_size: 订单大小（可选）
        
        Returns:
            (是否通过, 信息, 流动性指标)
        """
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return False, "订单簿数据为空", {}
        
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        if not bids or not asks:
            return False, "订单簿无数据", {}
        
        bid_depth = sum(amount for price, amount in bids[:10])
        ask_depth = sum(amount for price, amount in asks[:10])
        total_depth = bid_depth + ask_depth
        
        liquidity_metrics = {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'total_depth': total_depth,
            'spread': asks[0][0] - bids[0][0],
            'spread_pct': (asks[0][0] - bids[0][0]) / bids[0][0],
            'mid_price': (bids[0][0] + asks[0][0]) / 2
        }
        
        if total_depth < self.min_depth:
            return False, f"流动性不足: {total_depth} < {self.min_depth}", liquidity_metrics
        
        if liquidity_metrics['spread_pct'] > 0.01:
            return False, f"价差过大: {liquidity_metrics['spread_pct']*100:.2f}%", liquidity_metrics
        
        if order_size:
            slippage = self.calculate_expected_slippage(order_book, order_size, 'buy')
            if slippage > self.max_slippage:
                return False, f"预期滑点过大: {slippage*100:.2f}% > {self.max_slippage*100:.2f}%", liquidity_metrics
        
        logger.info(f"{symbol} 流动性检查通过 - 深度:{total_depth:.2f}, 价差:{liquidity_metrics['spread_pct']*100:.4f}%")
        return True, "流动性检查通过", liquidity_metrics
    
    def calculate_expected_slippage(self, order_book: Dict, order_size: float, 
                                    side: str) -> float:
        """
        计算预期滑点
        
        Args:
            order_book: 订单簿数据
            order_size: 订单大小
            side: 交易方向 ('buy' 或 'sell')
        
        Returns:
            预期滑点比例
        """
        if side == 'buy':
            orders = order_book.get('asks', [])
            base_price = orders[0][0] if orders else 0
        else:
            orders = order_book.get('bids', [])
            base_price = orders[0][0] if orders else 0
        
        if base_price == 0 or not orders:
            return 0.0
        
        remaining_size = order_size
        total_cost = 0.0
        filled_size = 0.0
        
        for price, amount in orders:
            if remaining_size <= 0:
                break
            
            fill_amount = min(remaining_size, amount)
            total_cost += fill_amount * price
            filled_size += fill_amount
            remaining_size -= fill_amount
        
        if filled_size == 0:
            return 0.0
        
        avg_price = total_cost / filled_size
        slippage = abs(avg_price - base_price) / base_price
        
        return slippage
    
    def calculate_optimal_size(self, order_book: Dict, max_slippage: float,
                              account_balance: float, risk_per_trade: float = 0.01) -> float:
        """
        计算最优交易量
        
        Args:
            order_book: 订单簿数据
            max_slippage: 最大允许滑点
            account_balance: 账户余额
            risk_per_trade: 单笔风险比例（默认1%）
        
        Returns:
            最优交易量
        """
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return 0.0
        
        mid_price = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
        
        risk_amount = account_balance * risk_per_trade
        base_size = risk_amount / mid_price
        
        low = 0
        high = base_size * 2
        optimal_size = base_size
        
        for _ in range(10):
            mid = (low + high) / 2
            slippage = self.calculate_expected_slippage(order_book, mid, 'buy')
            
            if slippage <= max_slippage:
                low = mid
                optimal_size = mid
            else:
                high = mid
        
        return optimal_size
    
    def calculate_real_order_ratio(self, order_book: Dict, depth: int = 5) -> Dict:
        """
        基于真实订单簿计算买卖盘比
        
        Args:
            order_book: 订单簿数据
            depth: 深度（默认5）
        
        Returns:
            买卖盘比字典
        """
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return {}
        
        bids = order_book.get('bids', [])[:depth]
        asks = order_book.get('asks', [])[:depth]
        
        if not bids or not asks:
            return {}
        
        simple_ratio = sum(amount for price, amount in bids) / sum(amount for price, amount in asks)
        
        bid_weighted = sum(price * amount for price, amount in bids) / sum(amount for price, amount in bids)
        ask_weighted = sum(price * amount for price, amount in asks) / sum(amount for price, amount in asks)
        weighted_ratio = bid_weighted / ask_weighted if ask_weighted > 0 else 1.0
        
        large_bid_amount = sum(amount for price, amount in bids if amount > self.large_order_threshold)
        large_ask_amount = sum(amount for price, amount in asks if amount > self.large_order_threshold)
        large_order_ratio = large_bid_amount / large_ask_amount if large_ask_amount > 0 else 1.0
        
        return {
            'simple_ratio': simple_ratio,
            'weighted_ratio': weighted_ratio,
            'large_order_ratio': large_order_ratio,
            'bid_depth': sum(amount for price, amount in bids),
            'ask_depth': sum(amount for price, amount in asks)
        }
    
    def detect_liquidity_risk(self, order_book: Dict, symbol: str) -> Tuple[bool, str, str]:
        """
        检测流动性风险
        
        Args:
            order_book: 订单簿数据
            symbol: 交易对
        
        Returns:
            (是否有风险, 风险等级, 风险描述)
        """
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return True, 'CRITICAL', '订单簿数据缺失'
        
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        if not bids or not asks:
            return True, 'CRITICAL', '订单簿无数据'
        
        bid_depth = sum(amount for price, amount in bids[:10])
        ask_depth = sum(amount for price, amount in asks[:10])
        total_depth = bid_depth + ask_depth
        
        spread = (asks[0][0] - bids[0][0]) / bids[0][0]
        
        liquidity_ratio = min(bid_depth, ask_depth) / max(bid_depth, ask_depth)
        
        if total_depth < self.min_depth * 0.5:
            return True, 'CRITICAL', f'流动性严重不足: {total_depth:.2f}'
        elif total_depth < self.min_depth:
            return True, 'HIGH', f'流动性不足: {total_depth:.2f}'
        elif spread > 0.02:
            return True, 'HIGH', f'价差过大: {spread*100:.2f}%'
        elif spread > 0.01:
            return True, 'MEDIUM', f'价差偏大: {spread*100:.2f}%'
        elif liquidity_ratio < self.liquidity_ratio_threshold:
            return True, 'MEDIUM', f'买卖盘不平衡: {liquidity_ratio:.2f}'
        else:
            return False, 'LOW', '流动性正常'
    
    def optimize_order_execution(self, order_book: Dict, order_size: float, 
                                 side: str) -> Dict:
        """
        优化订单执行
        
        Args:
            order_book: 订单簿数据
            order_size: 订单大小
            side: 交易方向 ('buy' 或 'sell')
        
        Returns:
            优化建议
        """
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return {'error': '订单簿数据缺失'}
        
        slippage = self.calculate_expected_slippage(order_book, order_size, side)
        
        if slippage > self.max_slippage * 2:
            return {
                'recommendation': '分批执行',
                'splits': 4,
                'size_per_split': order_size / 4,
                'expected_slippage': slippage / 4,
                'reason': '订单过大，建议分批执行以减少滑点'
            }
        elif slippage > self.max_slippage:
            return {
                'recommendation': '限价单',
                'limit_price': order_book['asks'][0][0] * 1.002 if side == 'buy' else order_book['bids'][0][0] * 0.998,
                'expected_slippage': slippage,
                'reason': '滑点较大，建议使用限价单'
            }
        else:
            return {
                'recommendation': '市价单',
                'expected_slippage': slippage,
                'reason': '流动性良好，可以使用市价单'
            }
    
    def calculate_liquidity_score(self, order_book: Dict) -> float:
        """
        计算流动性评分（0-100）
        
        Args:
            order_book: 订单簿数据
        
        Returns:
            流动性评分
        """
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return 0.0
        
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        if not bids or not asks:
            return 0.0
        
        depth_score = min(100, (sum(amount for price, amount in bids[:10]) + 
                              sum(amount for price, amount in asks[:10])) / self.min_depth * 100)
        
        spread = (asks[0][0] - bids[0][0]) / bids[0][0]
        spread_score = max(0, 100 - spread * 10000)
        
        balance_ratio = min(sum(amount for price, amount in bids[:10]), 
                           sum(amount for price, amount in asks[:10])) / \
                       max(sum(amount for price, amount in bids[:10]), 
                           sum(amount for price, amount in asks[:10]))
        balance_score = balance_ratio * 100
        
        liquidity_score = (depth_score * 0.5 + spread_score * 0.3 + balance_score * 0.2)
        
        return liquidity_score
    
    def get_liquidity_report(self, order_book: Dict, symbol: str) -> Dict:
        """
        获取流动性报告
        
        Args:
            order_book: 订单簿数据
            symbol: 交易对
        
        Returns:
            流动性报告
        """
        has_risk, risk_level, risk_description = self.detect_liquidity_risk(order_book, symbol)
        liquidity_score = self.calculate_liquidity_score(order_book)
        order_ratio = self.calculate_real_order_ratio(order_book)
        
        return {
            'symbol': symbol,
            'liquidity_score': liquidity_score,
            'risk_level': risk_level,
            'risk_description': risk_description,
            'has_risk': has_risk,
            'order_ratio': order_ratio,
            'recommendation': self._get_liquidity_recommendation(liquidity_score, risk_level)
        }
    
    def _get_liquidity_recommendation(self, score: float, risk_level: str) -> str:
        """获取流动性建议"""
        if risk_level == 'CRITICAL':
            return '避免交易，流动性严重不足'
        elif risk_level == 'HIGH':
            return '谨慎交易，使用限价单并减少仓位'
        elif risk_level == 'MEDIUM':
            return '正常交易，注意滑点成本'
        elif score > 80:
            return '流动性良好，可以正常交易'
        elif score > 60:
            return '流动性一般，建议使用限价单'
        else:
            return '流动性较差，建议减少交易量'