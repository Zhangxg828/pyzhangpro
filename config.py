import os
import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

DB_MEMORY = os.getenv('DB_MEMORY', './data/market_memory.db')
DB_VERIFY = os.getenv('DB_VERIFY', './data/verification_pro.db')

VLLM_API = os.getenv('VLLM_API', 'https://zhangmodes.ddnsto.com:443/v1')
MODEL_NAME = os.getenv('MODEL_NAME', '/models')

PROXY_URL = os.getenv('PROXY_URL', 'socks5h://127.0.0.1:1080')

DATASET_PATH = os.getenv('DATASET_PATH', str(DATA_DIR / "datasets"))

QWEN_LOG = os.getenv('QWEN_LOG', str(DATA_DIR / "qwen_logic.txt"))
STATE_JSON = os.getenv('STATE_JSON', str(DATA_DIR / "market_state.json"))
SUMMARY_OUT = os.getenv('SUMMARY_OUT', str(DATA_DIR / "long_term_memory.txt"))

DINGTALK_WEBHOOK = os.getenv('DINGTALK_WEBHOOK', '')

TELEGRAM_API_ID = os.getenv('TELEGRAM_API_ID', '25555585')
TELEGRAM_API_HASH = os.getenv('TELEGRAM_API_HASH', '00349140ddf5aa4b5a1c2b1474278f4e')

BENZINGA_KEY = os.getenv('BENZINGA_KEY', 'bz.HFTLINLDAQI2Y722GH22ASUM4DDVPEKM')

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', str(DATA_DIR / "system.log"))


def setup_logger(name, log_file=None, level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name):
    return logging.getLogger(name)


system_logger = setup_logger('system', LOG_FILE, getattr(logging, LOG_LEVEL.upper()))

CRYPTO_LIST = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'DOT/USDT',
               'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT', 'UNI/USDT', 'AAVE/USDT', 'LDO/USDT', 'ARB/USDT', 'OP/USDT',
               'BASE/USDT', 'TON/USDT', 'SUI/USDT', 'SEI/USDT', 'TIA/USDT', 'RNDR/USDT', 'FET/USDT', 'CRV/USDT',
               'MKR/USDT', 'MORPHO/USDT']

STOCK_A_LIST = ['601318', '600519', '601398', '601939', '601288', '600036', '600900', '000001', '601988', '601857',
                '300750', '600030', '000858', '601166', '600887', '600809', '601328', '002594', '601601', '600276',
                '600309', '000002', '601088', '600050', '601668', '600028', '601012', '000333', '600941', '601138',
                '300274', '601899', '600690', '601211', '600703', '600000', '600104', '600585', '600438', '600048',
                '601888', '600999', '601816', '600346', '601669', '600919', '601006', '600837', '601628', '002415',
                '300760', '688981', '300957', '300982', '688041', '688036', '300014', '002371', '002049']

STOCK_HK_LIST = ['00700', '09988', '00941', '01810', '03690', '09618', '01299', '0388', '00005', '01801', '01398',
                 '02318', '00883', '01211', '00288', '00016', '00011', '01988', '02388', '01109', '02007', '00027',
                 '00066', '00316', '00001', '00002', '00003', '00006', '00012', '00017', '00019', '00101', '00151',
                 '00241', '00267', '00291', '00386', '00688', '00762', '00857', '00881', '00939', '00960', '01088',
                 '01113', '01177', '01288', '01336', '01359', '01658', '01766', '01772', '01816', '01918', '01928',
                 '01997', '02015', '02018', '02269', '02319', '02331', '02333', '02382', '02688', '02799', '02888',
                 '03328', '03698', '03759', '03968', '03988', '06618', '06862', '06881', '06969', '09626', '09633',
                 '09688', '09698', '09888', '09961', '09987', '09999', '01888', '00384', '00669', '00836', '01099',
                 '01171', '01378', '01638', '01833', '01910', '01958', '02020', '02202', '02359', '02378', '02628',
                 '02638', '03319', '03618', '06886', '09658', '09868', '09930']

STOCK_US_LIST = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'JPM', 'LLY', 'V', 'WMT', 'XOM',
                 'UNH', 'MA', 'PG', 'JNJ', 'HD', 'ORCL', 'MRK', 'COST', 'ABBV', 'CRM', 'NFLX', 'AMD', 'BAC', 'KO',
                 'TMO', 'ACN', 'LIN', 'MCD', 'CSCO', 'GE', 'ABT', 'PEP', 'DHR', 'WFC', 'TXN', 'INTU', 'VZ', 'ADBE',
                 'NOW', 'QCOM', 'IBM', 'CAT', 'UBER', 'PM', 'RTX', 'PFE', 'SPGI', 'UNP', 'NEE', 'GS', 'ISRG', 'BKNG',
                 'HON', 'MU', 'LOW', 'PGR', 'BLK', 'ELV', 'SYK', 'LRCX', 'BA', 'MDT', 'DE', 'PLD', 'ADI', 'TJX', 'VRTX',
                 'KLAC', 'REGN', 'CB', 'LMT', 'ETN', 'BSX', 'MMC', 'PANW', 'ADP', 'FI', 'SBUX', 'MDLZ', 'SO', 'BMY',
                 'MO', 'GILD', 'DUK', 'ZTS', 'ICE', 'CL', 'ITW', 'CME', 'SHW', 'PLTR', 'MCO', 'APH', 'WM', 'TT', 'EOG',
                 'FCX', 'TGT', 'NOC', 'BDX', 'SLB', 'FDX', 'EMR', 'ROP', 'MSI', 'PCAR', 'OXY']

TARGET_CHANNELS = [
    -1001526765830, -1001456088978, -1001525105897, -1001515681710, -1001574411937,
    -1001525379130, -1001748596288, -1001648734310, -1001380566582, -1001387109317
]

TIMEFRAME = '15m'
THRESHOLD_PERCENT = 0.1
DEVICE_ID = 0

FINBERT_MODEL_PATH = os.getenv('FINBERT_MODEL_PATH', '/root/.cache/huggingface/transformers/finbert-tone-chinese')

HISTORY_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    timestamp TEXT,
    symbol TEXT,
    price REAL,
    rsi REAL,
    sentiment REAL,
    volume REAL,
    source TEXT
)
"""

HISTORY_TABLE_COLUMNS = ['timestamp', 'symbol', 'price', 'rsi', 'sentiment', 'volume', 'source']

RISK_CONTROL_CONFIG = {
    'max_drawdown': 0.15,
    'max_position_size': 0.1,
    'max_total_exposure': 0.5,
    'max_correlation_exposure': 0.3,
    'max_single_loss': 0.02,
    'daily_loss_limit': 0.05,
    'correlation_threshold': 0.7,
    'volatility_threshold': 0.02,
    'risk_free_rate': 0.02
}

TECHNICAL_INDICATORS_CONFIG = {
    'atr_period': 14,
    'atr_multiplier': 2.0,
    'adx_period': 14,
    'adx_threshold': 25,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_period': 20,
    'bollinger_std': 2,
    'volume_ma_period': 20,
    'order_ratio_window': 10
}

MARKET_REGIME_CONFIG = {
    'lookback_period': 50,
    'trend_period': 5,
    'trend_threshold': 0.02,
    'volatility_window': 20,
    'volume_window': 20,
    'momentum_period': 14,
    'regime_change_threshold': 0.3
}

LIQUIDITY_MANAGER_CONFIG = {
    'liquidity_window': 20,
    'spread_threshold': 0.001,
    'volume_threshold': 1000000,
    'depth_levels': 5,
    'slippage_threshold': 0.005,
    'min_liquidity_ratio': 0.1
}

BACKTEST_CONFIG = {
    'initial_capital': 100000,
    'commission': 0.001,
    'slippage': 0.0005,
    'lookback_days': 365,
    'warmup_period': 30,
    'min_trades': 10,
    'confidence_level': 0.95
}

MULTI_TIMEFRAME_CONFIG = {
    'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
    'primary_timeframe': '15m',
    'alignment_method': 'close',
    'signal_weight': {
        '1m': 0.1,
        '5m': 0.15,
        '15m': 0.3,
        '1h': 0.25,
        '4h': 0.15,
        '1d': 0.05
    }
}

SENTIMENT_ANALYSIS_CONFIG = {
    'window_size': 24,
    'confidence_threshold': 0.7,
    'update_interval': 5,
    'sentiment_weights': {
        'market_data': 0.3,
        'social_media': 0.25,
        'news': 0.25,
        'order_flow': 0.2
    },
    'extreme_threshold': 0.8,
    'fear_greed_threshold': 80,
    'trend_window': 48
}

ANOMALY_DETECTION_CONFIG = {
    'price_spike_threshold': 3.0,
    'volume_spike_threshold': 3.0,
    'volatility_spike_threshold': 3.0,
    'contamination': 0.1,
    'window_size': 20,
    'z_score_threshold': 3.0,
    'isolation_forest_n_estimators': 100,
    'isolation_forest_max_samples': 'auto',
    'multivariate_threshold': 0.95
}

POSITION_SIZING_CONFIG = {
    'base_position_size': 0.05,
    'max_position_size': 0.15,
    'min_position_size': 0.01,
    'risk_per_trade': 0.02,
    'kelly_criterion': False,
    'volatility_adjustment': True,
    'correlation_adjustment': True,
    'atr_multiplier': 2.0
}

DYNAMIC_STOP_LOSS_CONFIG = {
    'atr_period': 14,
    'atr_multiplier': 2.0,
    'trailing_stop': True,
    'trailing_activation': 0.02,
    'trailing_step': 0.01,
    'time_based_exit': False,
    'max_hold_time': 24,
    'breakeven_threshold': 0.015
}

ML_MODEL_CONFIG = {
    'model_type': 'random_forest',
    'feature_window': 50,
    'prediction_horizon': 5,
    'train_test_split': 0.8,
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'retrain_interval': 168,
    'feature_importance_threshold': 0.05
}

HIGH_FREQUENCY_CONFIG = {
    'enabled': False,
    'tick_interval': 0.1,
    'order_book_depth': 20,
    'latency_threshold': 0.01,
    'position_duration': 60,
    'max_orders_per_minute': 100,
    'slippage_tolerance': 0.0001
}

ARBITRAGE_CONFIG = {
    'enabled': False,
    'min_profit_threshold': 0.002,
    'max_position_size': 0.05,
    'execution_timeout': 5,
    'correlation_threshold': 0.95,
    'volatility_threshold': 0.01,
    'liquidity_threshold': 1000000
}

SMART_CONTRACT_CONFIG = {
    'enabled': False,
    'network': 'mainnet',
    'gas_limit': 300000,
    'gas_price_gwei': 20,
    'max_gas_price_gwei': 100,
    'contract_address': '',
    'private_key': '',
    'rpc_endpoint': ''
}
