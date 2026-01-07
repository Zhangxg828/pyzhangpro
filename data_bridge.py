import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from config import DB_MEMORY, DB_VERIFY, LOG_LEVEL, LOG_FILE, CRYPTO_LIST

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def calculate_sar(high, low, acceleration=0.02, maximum=0.2):
    if len(high) < 2:
        return 0.0, 'NEUTRAL'
    
    sar = np.zeros(len(high))
    ep = np.zeros(len(high))
    af = np.zeros(len(high))
    
    sar[0] = low[0]
    ep[0] = high[0]
    af[0] = acceleration
    is_up_trend = True
    
    for i in range(1, len(high)):
        if is_up_trend:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = min(sar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
            
            if high[i] > ep[i-1]:
                ep[i] = high[i]
                af[i] = min(af[i-1] + acceleration, maximum)
            else:
                ep[i] = ep[i-1]
                af[i] = af[i-1]
            
            if low[i] < sar[i]:
                is_up_trend = False
                sar[i] = ep[i-1]
                ep[i] = low[i]
                af[i] = acceleration
        else:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
            
            if low[i] < ep[i-1]:
                ep[i] = low[i]
                af[i] = min(af[i-1] + acceleration, maximum)
            else:
                ep[i] = ep[i-1]
                af[i] = af[i-1]
            
            if high[i] > sar[i]:
                is_up_trend = True
                sar[i] = ep[i-1]
                ep[i] = high[i]
                af[i] = acceleration
    
    current_sar = sar[-1]
    current_price = high[-1]
    
    if current_price > current_sar:
        trend = 'BULL'
    elif current_price < current_sar:
        trend = 'BEAR'
    else:
        trend = 'NEUTRAL'
    
    return current_sar, trend


def calculate_order_ratio(price_history, volume_history):
    if len(price_history) < 5:
        return 1.0
    
    try:
        price_changes = np.diff(price_history[-5:])
        volume_changes = np.diff(volume_history[-5:])
        
        buy_pressure = np.sum((price_changes > 0) * volume_changes[1:])
        sell_pressure = np.sum((price_changes < 0) * volume_changes[1:])
        
        if sell_pressure == 0:
            return 2.0
        ratio = buy_pressure / sell_pressure
        return max(0.1, min(5.0, ratio))
    except:
        return 1.0


def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    
    try:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except:
        return 50.0


def sync_data_to_verify_db():
    try:
        conn_memory = sqlite3.connect(DB_MEMORY, timeout=30)
        conn_memory.execute('PRAGMA journal_mode=WAL;')
        
        conn_verify = sqlite3.connect(DB_VERIFY, timeout=30)
        conn_verify.execute('PRAGMA journal_mode=WAL;')
        
        cursor_memory = conn_memory.cursor()
        cursor_verify = conn_verify.cursor()
        
        five_minutes_ago = (datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
        
        cursor_memory.execute('''
            SELECT symbol, price, volume, rsi, sentiment, timestamp
            FROM history
            WHERE timestamp >= ?
            ORDER BY symbol, timestamp DESC
        ''', (five_minutes_ago,))
        
        rows = cursor_memory.fetchall()
        
        if not rows:
            logger.info("📭 没有新数据需要同步")
            conn_memory.close()
            conn_verify.close()
            return
        
        symbol_data = {}
        for row in rows:
            symbol, price, volume, rsi, sentiment, timestamp = row
            if symbol not in symbol_data:
                symbol_data[symbol] = {'prices': [], 'volumes': [], 'latest': None}
            symbol_data[symbol]['prices'].append(price)
            symbol_data[symbol]['volumes'].append(volume)
            if symbol_data[symbol]['latest'] is None:
                symbol_data[symbol]['latest'] = {
                    'price': price,
                    'volume': volume,
                    'rsi': rsi,
                    'sentiment': sentiment,
                    'timestamp': timestamp
                }
        
        sync_count = 0
        for symbol, data in symbol_data.items():
            if len(data['prices']) < 2:
                continue
            
            prices = data['prices']
            volumes = data['volumes']
            latest = data['latest']
            
            sar_value, sar_trend = calculate_sar(prices, prices)
            order_ratio = calculate_order_ratio(prices, volumes)
            rsi_value = calculate_rsi(prices)
            
            cursor_verify.execute('''
                INSERT OR REPLACE INTO verify_pro_ticker 
                (symbol, price, order_ratio, sar_value, sar_trend, rsi, sentiment, timestamp, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                symbol,
                latest['price'],
                order_ratio,
                sar_value,
                sar_trend,
                rsi_value,
                latest['sentiment'],
                latest['timestamp']
            ))
            
            sync_count += 1
        
        conn_verify.commit()
        conn_memory.close()
        conn_verify.close()
        
        logger.info(f"✅ 数据同步完成: {sync_count} 个标的")
        return sync_count
        
    except Exception as e:
        logger.error(f"❌ 数据同步失败: {e}")
        return 0


def get_latest_ticker_data():
    try:
        conn_verify = sqlite3.connect(DB_VERIFY, timeout=30)
        conn_verify.execute('PRAGMA journal_mode=WAL;')
        
        cursor = conn_verify.cursor()
        cursor.execute('''
            SELECT symbol, price, order_ratio, sar_value, sar_trend, rsi, sentiment, timestamp
            FROM verify_pro_ticker
            ORDER BY updated_at DESC
        ''')
        
        rows = cursor.fetchall()
        conn_verify.close()
        
        return rows
        
    except Exception as e:
        logger.error(f"❌ 获取行情数据失败: {e}")
        return []


if __name__ == "__main__":
    logger.info("🚀 开始数据同步...")
    count = sync_data_to_verify_db()
    if count > 0:
        logger.info(f"✅ 同步成功: {count} 个标的")
    else:
        logger.warning("⚠️ 没有数据被同步")
