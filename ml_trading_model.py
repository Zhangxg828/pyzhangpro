import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
import os
from config import ML_MODEL_CONFIG, DB_MEMORY, DATA_DIR, setup_logger

logger = setup_logger('ml_trading_model')

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn未安装，机器学习功能将受限")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost未安装，XGBoost模型将不可用")


@dataclass
class PredictionResult:
    symbol: str
    prediction: int
    probability: float
    confidence: str
    timestamp: float
    features: Dict[str, float]


class MLTradingModel:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or ML_MODEL_CONFIG
        self.model_type = self.config.get('model_type', 'random_forest')
        self.feature_window = self.config.get('feature_window', 50)
        self.prediction_horizon = self.config.get('prediction_horizon', 5)
        self.train_test_split = self.config.get('train_test_split', 0.8)
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 10)
        self.min_samples_split = self.config.get('min_samples_split', 5)
        self.retrain_interval = self.config.get('retrain_interval', 168)
        self.feature_importance_threshold = self.config.get('feature_importance_threshold', 0.05)
        
        self.models: Dict[str, object] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        self.last_train_time: Dict[str, float] = {}
        
        self.model_dir = DATA_DIR / "ml_models"
        self.model_dir.mkdir(exist_ok=True)
        
        logger.info(f"机器学习交易模型初始化完成 - 模型类型: {self.model_type}, 特征窗口: {self.feature_window}")

    def create_model(self) -> object:
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn未安装，无法创建模型")
            
        if self.model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=42
            )
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1
            )
        else:
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=42,
                n_jobs=-1
            )
        
        return model

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            df = df.sort_values('timestamp')
            
            if len(df) < self.feature_window:
                logger.warning(f"数据不足，需要至少{self.feature_window}条记录")
                return pd.DataFrame()
            
            df['returns'] = df['price'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            
            df['ma_5'] = df['price'].rolling(window=5).mean()
            df['ma_10'] = df['price'].rolling(window=10).mean()
            df['ma_20'] = df['price'].rolling(window=20).mean()
            df['ma_50'] = df['price'].rolling(window=50).mean()
            
            df['std_5'] = df['price'].rolling(window=5).std()
            df['std_10'] = df['price'].rolling(window=10).std()
            df['std_20'] = df['price'].rolling(window=20).std()
            
            df['rsi'] = self._calculate_rsi(df['price'], 14)
            
            df['momentum_5'] = df['price'] / df['price'].shift(5) - 1
            df['momentum_10'] = df['price'] / df['price'].shift(10) - 1
            
            df['volatility_5'] = df['returns'].rolling(window=5).std()
            df['volatility_10'] = df['returns'].rolling(window=10).std()
            
            df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_5']
            
            df['price_position'] = (df['price'] - df['price'].rolling(window=20).min()) / \
                                  (df['price'].rolling(window=20).max() - df['price'].rolling(window=20).min())
            
            df['order_ratio_ma'] = df['order_ratio'].rolling(window=10).mean()
            
            df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp'], unit='s').dt.dayofweek
            
            feature_cols = [
                'returns', 'volume_change', 'ma_5', 'ma_10', 'ma_20', 'ma_50',
                'std_5', 'std_10', 'std_20', 'rsi', 'momentum_5', 'momentum_10',
                'volatility_5', 'volatility_10', 'volume_ratio', 'price_position',
                'order_ratio_ma', 'hour', 'day_of_week'
            ]
            
            features = df[feature_cols].dropna()
            
            return features
        except Exception as e:
            logger.error(f"提取特征失败: {e}")
            return pd.DataFrame()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.error(f"计算RSI失败: {e}")
            return pd.Series([50] * len(prices), index=prices.index)

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        try:
            df = df.copy()
            df = df.sort_values('timestamp')
            
            future_returns = df['price'].shift(-self.prediction_horizon) / df['price'] - 1
            
            labels = pd.Series(0, index=df.index)
            labels[future_returns > 0.02] = 1
            labels[future_returns < -0.02] = -1
            
            return labels
        except Exception as e:
            logger.error(f"创建标签失败: {e}")
            return pd.Series()

    def prepare_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            conn = sqlite3.connect(DB_MEMORY)
            conn.execute('PRAGMA journal_mode=WAL;')
            
            query = """
            SELECT price, volume, order_ratio, timestamp
            FROM raw_ticker_stream
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 1000
            """
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if len(df) < self.feature_window + self.prediction_horizon:
                logger.warning(f"{symbol} 数据不足")
                return pd.DataFrame(), pd.Series()
            
            df = df.sort_values('timestamp')
            
            features = self.extract_features(df)
            labels = self.create_labels(df)
            
            min_len = min(len(features), len(labels))
            features = features.iloc[:min_len]
            labels = labels.iloc[:min_len]
            
            return features, labels
        except Exception as e:
            logger.error(f"准备数据失败: {e}")
            return pd.DataFrame(), pd.Series()

    def train_model(self, symbol: str) -> bool:
        try:
            if not SKLEARN_AVAILABLE:
                logger.error("sklearn未安装，无法训练模型")
                return False
                
            features, labels = self.prepare_data(symbol)
            
            if len(features) == 0 or len(labels) == 0:
                logger.warning(f"{symbol} 数据不足，无法训练模型")
                return False
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=1-self.train_test_split, random_state=42
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = self.create_model()
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            try:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                auc = 0
            
            logger.info(f"{symbol} 模型训练完成 - 准确率: {accuracy:.4f}, 精确率: {precision:.4f}, "
                       f"召回率: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.last_train_time[symbol] = datetime.now().timestamp()
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_names = features.columns
                self.feature_importance[symbol] = dict(zip(feature_names, importance))
                
                important_features = {k: v for k, v in self.feature_importance[symbol].items() 
                                     if v >= self.feature_importance_threshold}
                logger.info(f"{symbol} 重要特征: {important_features}")
            
            self.save_model(symbol)
            
            return True
        except Exception as e:
            logger.error(f"训练模型失败: {e}")
            return False

    def predict(self, symbol: str) -> Optional[PredictionResult]:
        try:
            if symbol not in self.models:
                logger.warning(f"{symbol} 模型未训练，开始训练...")
                if not self.train_model(symbol):
                    return None
            
            model = self.models[symbol]
            scaler = self.scalers[symbol]
            
            conn = sqlite3.connect(DB_MEMORY)
            conn.execute('PRAGMA journal_mode=WAL;')
            
            query = """
            SELECT price, volume, order_ratio, timestamp
            FROM raw_ticker_stream
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 100
            """
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if len(df) < self.feature_window:
                logger.warning(f"{symbol} 数据不足，无法预测")
                return None
            
            df = df.sort_values('timestamp')
            features = self.extract_features(df)
            
            if len(features) == 0:
                logger.warning(f"{symbol} 特征提取失败")
                return None
            
            latest_features = features.iloc[-1:].values
            latest_features_scaled = scaler.transform(latest_features)
            
            prediction = model.predict(latest_features_scaled)[0]
            probabilities = model.predict_proba(latest_features_scaled)[0]
            max_prob = np.max(probabilities)
            
            if max_prob >= 0.7:
                confidence = "高"
            elif max_prob >= 0.5:
                confidence = "中"
            else:
                confidence = "低"
            
            feature_values = dict(zip(features.columns, latest_features[0]))
            
            result = PredictionResult(
                symbol=symbol,
                prediction=int(prediction),
                probability=float(max_prob),
                confidence=confidence,
                timestamp=datetime.now().timestamp(),
                features=feature_values
            )
            
            logger.info(f"{symbol} 预测结果: {prediction}, 概率: {max_prob:.4f}, 置信度: {confidence}")
            
            return result
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return None

    def should_retrain(self, symbol: str) -> bool:
        try:
            if symbol not in self.last_train_time:
                return True
                
            last_train = datetime.fromtimestamp(self.last_train_time[symbol])
            hours_since_train = (datetime.now() - last_train).total_seconds() / 3600
            
            return hours_since_train >= self.retrain_interval
        except Exception as e:
            logger.error(f"检查是否需要重新训练失败: {e}")
            return False

    def save_model(self, symbol: str):
        try:
            model_path = self.model_dir / f"{symbol}_model.pkl"
            scaler_path = self.model_dir / f"{symbol}_scaler.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[symbol], f)
                
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[symbol], f)
                
            logger.info(f"{symbol} 模型已保存")
        except Exception as e:
            logger.error(f"保存模型失败: {e}")

    def load_model(self, symbol: str) -> bool:
        try:
            model_path = self.model_dir / f"{symbol}_model.pkl"
            scaler_path = self.model_dir / f"{symbol}_scaler.pkl"
            
            if not model_path.exists() or not scaler_path.exists():
                return False
                
            with open(model_path, 'rb') as f:
                self.models[symbol] = pickle.load(f)
                
            with open(scaler_path, 'rb') as f:
                self.scalers[symbol] = pickle.load(f)
                
            logger.info(f"{symbol} 模型已加载")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False

    def get_feature_importance(self, symbol: str) -> Optional[Dict[str, float]]:
        try:
            if symbol not in self.feature_importance:
                return None
                
            return self.feature_importance[symbol]
        except Exception as e:
            logger.error(f"获取特征重要性失败: {e}")
            return None

    def evaluate_model(self, symbol: str) -> Optional[Dict[str, float]]:
        try:
            if symbol not in self.models:
                return None
                
            features, labels = self.prepare_data(symbol)
            
            if len(features) == 0:
                return None
                
            model = self.models[symbol]
            scaler = self.scalers[symbol]
            
            X_scaled = scaler.transform(features)
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)
            
            metrics = {
                'accuracy': accuracy_score(labels, y_pred),
                'precision': precision_score(labels, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(labels, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(labels, y_pred, average='weighted', zero_division=0)
            }
            
            try:
                metrics['auc'] = roc_auc_score(labels, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                metrics['auc'] = 0
            
            return metrics
        except Exception as e:
            logger.error(f"评估模型失败: {e}")
            return None

    def batch_train(self, symbols: List[str]) -> Dict[str, bool]:
        try:
            results = {}
            for symbol in symbols:
                results[symbol] = self.train_model(symbol)
            return results
        except Exception as e:
            logger.error(f"批量训练失败: {e}")
            return {}

    def batch_predict(self, symbols: List[str]) -> Dict[str, Optional[PredictionResult]]:
        try:
            results = {}
            for symbol in symbols:
                results[symbol] = self.predict(symbol)
            return results
        except Exception as e:
            logger.error(f"批量预测失败: {e}")
            return {}