import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import sqlite3
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import (
    DB_MEMORY,
    DB_VERIFY,
    LOG_LEVEL
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """异常事件"""
    anomaly_type: str  # 'price_spike', 'volume_spike', 'volatility_spike', 'order_flow_anomaly', 'sentiment_anomaly'
    severity: float  # 0-1, 严重程度
    description: str
    timestamp: datetime
    metadata: Dict


@dataclass
class AnomalyDetectionResult:
    """异常检测结果"""
    symbol: str
    has_anomaly: bool
    anomalies: List[Anomaly]
    overall_risk_score: float  # 0-1, 整体风险分数
    recommended_action: str  # 'MONITOR', 'REDUCE_POSITION', 'CLOSE_POSITION', 'NO_ACTION'
    confidence: float
    timestamp: datetime


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, market_db: str = DB_MEMORY,
                 verification_db: str = DB_VERIFY):
        self.market_db = market_db
        self.verification_db = verification_db
        
        self.price_spike_threshold = 3.0
        self.volume_spike_threshold = 3.0
        self.volatility_spike_threshold = 3.0
        
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        self.scaler = StandardScaler()
        
        self.anomaly_weights = {
            'price_spike': 0.3,
            'volume_spike': 0.2,
            'volatility_spike': 0.2,
            'order_flow_anomaly': 0.15,
            'sentiment_anomaly': 0.15
        }
        
        logger.info("异常检测器初始化完成")
    
    def load_market_data(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """加载市场数据"""
        try:
            conn = sqlite3.connect(self.market_db)
            conn.execute("PRAGMA journal_mode=WAL")
            
            query = """
                SELECT 
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    buy_volume,
                    sell_volume
                FROM raw_ticker_stream
                WHERE symbol = ?
                AND timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            """.format(hours)
            
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if df.empty:
                logger.warning(f"未找到 {symbol} 的市场数据")
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"加载市场数据失败: {e}")
            return pd.DataFrame()
    
    def load_sentiment_data(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """加载情绪数据"""
        try:
            conn = sqlite3.connect(self.verification_db)
            conn.execute("PRAGMA journal_mode=WAL")
            
            query = """
                SELECT 
                    timestamp,
                    overall_sentiment,
                    fear_greed_index
                FROM sentiment_analysis
                WHERE symbol = ?
                AND timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            """.format(hours)
            
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"加载情绪数据失败: {e}")
            return pd.DataFrame()
    
    def detect_price_spike(self, df: pd.DataFrame) -> Optional[Anomaly]:
        """检测价格异常波动"""
        try:
            if df.empty or len(df) < 20:
                return None
            
            close = df['close'].values
            returns = np.diff(np.log(close))
            
            recent_return = returns[-1]
            mean_return = np.mean(returns[:-1])
            std_return = np.std(returns[:-1])
            
            if std_return == 0:
                return None
            
            z_score = abs((recent_return - mean_return) / std_return)
            
            if z_score > self.price_spike_threshold:
                severity = min(1.0, (z_score - self.price_spike_threshold) / 5)
                
                direction = "上涨" if recent_return > 0 else "下跌"
                description = f"价格异常{direction}: {direction}幅度 {abs(recent_return)*100:.2f}%, Z-score: {z_score:.2f}"
                
                anomaly = Anomaly(
                    anomaly_type='price_spike',
                    severity=severity,
                    description=description,
                    timestamp=df.iloc[-1]['timestamp'],
                    metadata={
                        'return': recent_return,
                        'z_score': z_score,
                        'price': close[-1]
                    }
                )
                
                logger.warning(f"检测到价格异常: {description}")
                return anomaly
            
            return None
            
        except Exception as e:
            logger.error(f"检测价格异常失败: {e}")
            return None
    
    def detect_volume_spike(self, df: pd.DataFrame) -> Optional[Anomaly]:
        """检测成交量异常"""
        try:
            if df.empty or len(df) < 20:
                return None
            
            volume = df['volume'].values
            
            recent_volume = volume[-1]
            mean_volume = np.mean(volume[:-1])
            std_volume = np.std(volume[:-1])
            
            if std_volume == 0:
                return None
            
            z_score = (recent_volume - mean_volume) / std_volume
            
            if z_score > self.volume_spike_threshold:
                severity = min(1.0, (z_score - self.volume_spike_threshold) / 5)
                
                description = f"成交量异常: 成交量 {recent_volume:.2f}, Z-score: {z_score:.2f}"
                
                anomaly = Anomaly(
                    anomaly_type='volume_spike',
                    severity=severity,
                    description=description,
                    timestamp=df.iloc[-1]['timestamp'],
                    metadata={
                        'volume': recent_volume,
                        'mean_volume': mean_volume,
                        'z_score': z_score
                    }
                )
                
                logger.warning(f"检测到成交量异常: {description}")
                return anomaly
            
            return None
            
        except Exception as e:
            logger.error(f"检测成交量异常失败: {e}")
            return None
    
    def detect_volatility_spike(self, df: pd.DataFrame) -> Optional[Anomaly]:
        """检测波动率异常"""
        try:
            if df.empty or len(df) < 30:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            true_range = np.maximum(
                high - low,
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )[1:]
            
            atr_window = 14
            atr = pd.Series(true_range).rolling(window=atr_window).mean().values
            
            recent_atr = atr[-1]
            mean_atr = np.mean(atr[:-1])
            std_atr = np.std(atr[:-1])
            
            if std_atr == 0:
                return None
            
            z_score = (recent_atr - mean_atr) / std_atr
            
            if z_score > self.volatility_spike_threshold:
                severity = min(1.0, (z_score - self.volatility_spike_threshold) / 5)
                
                description = f"波动率异常: ATR {recent_atr:.2f}, Z-score: {z_score:.2f}"
                
                anomaly = Anomaly(
                    anomaly_type='volatility_spike',
                    severity=severity,
                    description=description,
                    timestamp=df.iloc[-1]['timestamp'],
                    metadata={
                        'atr': recent_atr,
                        'mean_atr': mean_atr,
                        'z_score': z_score
                    }
                )
                
                logger.warning(f"检测到波动率异常: {description}")
                return anomaly
            
            return None
            
        except Exception as e:
            logger.error(f"检测波动率异常失败: {e}")
            return None
    
    def detect_order_flow_anomaly(self, df: pd.DataFrame) -> Optional[Anomaly]:
        """检测订单流异常"""
        try:
            if df.empty or len(df) < 30:
                return None
            
            buy_volume = df['buy_volume'].values
            sell_volume = df['sell_volume'].values
            
            buy_ratio = buy_volume / (buy_volume + sell_volume + 1e-10)
            
            recent_buy_ratio = buy_ratio[-1]
            mean_buy_ratio = np.mean(buy_ratio[:-1])
            std_buy_ratio = np.std(buy_ratio[:-1])
            
            if std_buy_ratio == 0:
                return None
            
            z_score = abs((recent_buy_ratio - mean_buy_ratio) / std_buy_ratio)
            
            if z_score > self.price_spike_threshold:
                severity = min(1.0, (z_score - self.price_spike_threshold) / 5)
                
                direction = "买入" if recent_buy_ratio > 0.5 else "卖出"
                description = f"订单流异常: {direction}比例异常 {recent_buy_ratio:.2%}, Z-score: {z_score:.2f}"
                
                anomaly = Anomaly(
                    anomaly_type='order_flow_anomaly',
                    severity=severity,
                    description=description,
                    timestamp=df.iloc[-1]['timestamp'],
                    metadata={
                        'buy_ratio': recent_buy_ratio,
                        'mean_buy_ratio': mean_buy_ratio,
                        'z_score': z_score
                    }
                )
                
                logger.warning(f"检测到订单流异常: {description}")
                return anomaly
            
            return None
            
        except Exception as e:
            logger.error(f"检测订单流异常失败: {e}")
            return None
    
    def detect_sentiment_anomaly(self, sentiment_df: pd.DataFrame) -> Optional[Anomaly]:
        """检测情绪异常"""
        try:
            if sentiment_df.empty or len(sentiment_df) < 10:
                return None
            
            sentiment = sentiment_df['overall_sentiment'].values
            
            recent_sentiment = sentiment[-1]
            mean_sentiment = np.mean(sentiment[:-1])
            std_sentiment = np.std(sentiment[:-1])
            
            if std_sentiment == 0:
                return None
            
            z_score = abs((recent_sentiment - mean_sentiment) / std_sentiment)
            
            if z_score > self.price_spike_threshold:
                severity = min(1.0, (z_score - self.price_spike_threshold) / 5)
                
                direction = "看多" if recent_sentiment > 0 else "看空"
                description = f"情绪异常: {direction}情绪异常 {recent_sentiment:.2f}, Z-score: {z_score:.2f}"
                
                anomaly = Anomaly(
                    anomaly_type='sentiment_anomaly',
                    severity=severity,
                    description=description,
                    timestamp=sentiment_df.iloc[-1]['timestamp'],
                    metadata={
                        'sentiment': recent_sentiment,
                        'mean_sentiment': mean_sentiment,
                        'z_score': z_score
                    }
                )
                
                logger.warning(f"检测到情绪异常: {description}")
                return anomaly
            
            return None
            
        except Exception as e:
            logger.error(f"检测情绪异常失败: {e}")
            return None
    
    def detect_multivariate_anomaly(self, df: pd.DataFrame) -> Optional[Anomaly]:
        """使用机器学习检测多变量异常"""
        try:
            if df.empty or len(df) < 50:
                return None
            
            features = []
            
            close = df['close'].values
            volume = df['volume'].values
            high = df['high'].values
            low = df['low'].values
            
            returns = np.diff(np.log(close))
            returns = np.concatenate([[0], returns])
            
            price_change = (close[-1] - close[-20]) / close[-20]
            volume_change = (volume[-1] - np.mean(volume[-20:])) / np.mean(volume[-20:])
            volatility = np.std(returns[-20:])
            
            true_range = np.maximum(
                high - low,
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
            atr = pd.Series(true_range).rolling(window=14).mean().values[-1]
            
            features = [[price_change, volume_change, volatility, atr]]
            
            features_scaled = self.scaler.fit_transform(features)
            
            anomaly_score = self.isolation_forest.fit_predict(features_scaled)
            
            if anomaly_score[0] == -1:
                severity = 0.7
                
                description = f"多变量异常: 检测到综合市场异常模式"
                
                anomaly = Anomaly(
                    anomaly_type='multivariate_anomaly',
                    severity=severity,
                    description=description,
                    timestamp=df.iloc[-1]['timestamp'],
                    metadata={
                        'price_change': price_change,
                        'volume_change': volume_change,
                        'volatility': volatility,
                        'atr': atr
                    }
                )
                
                logger.warning(f"检测到多变量异常: {description}")
                return anomaly
            
            return None
            
        except Exception as e:
            logger.error(f"检测多变量异常失败: {e}")
            return None
    
    def detect_anomalies(self, symbol: str) -> Optional[AnomalyDetectionResult]:
        """综合异常检测"""
        try:
            market_df = self.load_market_data(symbol)
            sentiment_df = self.load_sentiment_data(symbol)
            
            anomalies = []
            
            price_anomaly = self.detect_price_spike(market_df)
            if price_anomaly:
                anomalies.append(price_anomaly)
            
            volume_anomaly = self.detect_volume_spike(market_df)
            if volume_anomaly:
                anomalies.append(volume_anomaly)
            
            volatility_anomaly = self.detect_volatility_spike(market_df)
            if volatility_anomaly:
                anomalies.append(volatility_anomaly)
            
            order_flow_anomaly = self.detect_order_flow_anomaly(market_df)
            if order_flow_anomaly:
                anomalies.append(order_flow_anomaly)
            
            sentiment_anomaly = self.detect_sentiment_anomaly(sentiment_df)
            if sentiment_anomaly:
                anomalies.append(sentiment_anomaly)
            
            multivariate_anomaly = self.detect_multivariate_anomaly(market_df)
            if multivariate_anomaly:
                anomalies.append(multivariate_anomaly)
            
            has_anomaly = len(anomalies) > 0
            
            overall_risk_score = self._calculate_overall_risk(anomalies)
            
            recommended_action = self._determine_action(overall_risk_score, anomalies)
            
            confidence = min(1.0, len(anomalies) / 5)
            
            result = AnomalyDetectionResult(
                symbol=symbol,
                has_anomaly=has_anomaly,
                anomalies=anomalies,
                overall_risk_score=overall_risk_score,
                recommended_action=recommended_action,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            if has_anomaly:
                logger.warning(f"{symbol} 检测到 {len(anomalies)} 个异常, 风险分数: {overall_risk_score:.2f}")
            else:
                logger.info(f"{symbol} 未检测到异常")
            
            return result
            
        except Exception as e:
            logger.error(f"异常检测失败: {e}")
            return None
    
    def _calculate_overall_risk(self, anomalies: List[Anomaly]) -> float:
        """计算整体风险分数"""
        if not anomalies:
            return 0.0
        
        weighted_risk = 0.0
        total_weight = 0.0
        
        for anomaly in anomalies:
            weight = self.anomaly_weights.get(anomaly.anomaly_type, 0.1)
            weighted_risk += anomaly.severity * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_risk /= total_weight
        
        return min(1.0, weighted_risk)
    
    def _determine_action(self, risk_score: float, 
                         anomalies: List[Anomaly]) -> str:
        """确定推荐操作"""
        if risk_score > 0.8:
            return 'CLOSE_POSITION'
        elif risk_score > 0.5:
            return 'REDUCE_POSITION'
        elif risk_score > 0.3:
            return 'MONITOR'
        else:
            return 'NO_ACTION'
    
    def get_anomaly_summary(self, result: AnomalyDetectionResult) -> Dict:
        """获取异常检测摘要"""
        summary = {
            'symbol': result.symbol,
            'has_anomaly': result.has_anomaly,
            'anomaly_count': len(result.anomalies),
            'overall_risk_score': result.overall_risk_score,
            'recommended_action': result.recommended_action,
            'confidence': result.confidence,
            'timestamp': result.timestamp,
            'anomalies': []
        }
        
        for anomaly in result.anomalies:
            summary['anomalies'].append({
                'type': anomaly.anomaly_type,
                'severity': anomaly.severity,
                'description': anomaly.description,
                'timestamp': anomaly.timestamp
            })
        
        return summary
    
    def save_anomaly_to_db(self, result: AnomalyDetectionResult) -> bool:
        """保存异常检测结果到数据库"""
        try:
            conn = sqlite3.connect(self.verification_db)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_detection (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    has_anomaly INTEGER NOT NULL,
                    anomaly_count INTEGER NOT NULL,
                    overall_risk_score REAL NOT NULL,
                    recommended_action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    anomalies TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            import json
            anomalies_json = json.dumps([
                {
                    'type': a.anomaly_type,
                    'severity': a.severity,
                    'description': a.description,
                    'timestamp': a.timestamp.isoformat(),
                    'metadata': a.metadata
                }
                for a in result.anomalies
            ])
            
            cursor.execute("""
                INSERT INTO anomaly_detection 
                (symbol, has_anomaly, anomaly_count, overall_risk_score, 
                 recommended_action, confidence, anomalies, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.symbol,
                1 if result.has_anomaly else 0,
                len(result.anomalies),
                result.overall_risk_score,
                result.recommended_action,
                result.confidence,
                anomalies_json,
                result.timestamp
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"异常检测结果已保存: {result.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"保存异常检测结果失败: {e}")
            return False
    
    def get_historical_anomalies(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """获取历史异常数据"""
        try:
            conn = sqlite3.connect(self.verification_db)
            conn.execute("PRAGMA journal_mode=WAL")
            
            query = """
                SELECT 
                    timestamp,
                    has_anomaly,
                    anomaly_count,
                    overall_risk_score,
                    recommended_action
                FROM anomaly_detection
                WHERE symbol = ?
                AND timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            """.format(days)
            
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"获取历史异常数据失败: {e}")
            return pd.DataFrame()


def main():
    """测试函数"""
    detector = AnomalyDetector()
    
    symbol = "BTCUSDT"
    
    result = detector.detect_anomalies(symbol)
    
    if result:
        summary = detector.get_anomaly_summary(result)
        
        print(f"\n=== 异常检测结果 ===")
        print(f"交易对: {summary['symbol']}")
        print(f"是否有异常: {'是' if summary['has_anomaly'] else '否'}")
        print(f"异常数量: {summary['anomaly_count']}")
        print(f"整体风险分数: {summary['overall_risk_score']:.2f}")
        print(f"推荐操作: {summary['recommended_action']}")
        print(f"置信度: {summary['confidence']:.2%}")
        print(f"检测时间: {summary['timestamp']}")
        
        if summary['anomalies']:
            print(f"\n=== 异常详情 ===")
            for i, anomaly in enumerate(summary['anomalies'], 1):
                print(f"\n{i}. {anomaly['type']}:")
                print(f"   严重程度: {anomaly['severity']:.2f}")
                print(f"   描述: {anomaly['description']}")
                print(f"   时间: {anomaly['timestamp']}")
        
        detector.save_anomaly_to_db(result)


if __name__ == "__main__":
    main()
