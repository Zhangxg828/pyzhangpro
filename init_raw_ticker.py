import sqlite3
import os
from config import DB_MEMORY, DATA_DIR, setup_logger

logger = setup_logger('init_raw_ticker', os.path.join(DATA_DIR, 'init_raw_ticker.log'))

def init_raw_ticker_table():
    try:
        logger.info("开始初始化 raw_ticker_stream 表...")
        conn = sqlite3.connect(DB_MEMORY)
        conn.execute('PRAGMA journal_mode=WAL;')
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_ticker_stream
            (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recv_time TEXT,
                event_time INTEGER,
                symbol TEXT,
                price REAL,
                volume REAL,
                change_pct REAL,
                source TEXT,
                buy_volume REAL DEFAULT 0,
                sell_volume REAL DEFAULT 0
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_time_raw ON raw_ticker_stream (symbol, recv_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recv_time_raw ON raw_ticker_stream (recv_time)")

        conn.commit()
        conn.close()
        logger.info("raw_ticker_stream 表初始化完成")
    except Exception as e:
        logger.error(f"初始化 raw_ticker_stream 表失败: {e}")
        raise

if __name__ == "__main__":
    init_raw_ticker_table()
