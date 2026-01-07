import sqlite3
import logging
from config import DB_VERIFY, LOG_LEVEL, LOG_FILE
from pathlib import Path

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def init_verification_db():
    try:
        Path(DB_VERIFY).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(DB_VERIFY)
        cursor = conn.cursor()
        
        cursor.execute('PRAGMA journal_mode=WAL;')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verify_pro_ticker (
                symbol TEXT PRIMARY KEY,
                price REAL NOT NULL,
                order_ratio REAL DEFAULT 1.0,
                sar_value REAL DEFAULT 0.0,
                sar_trend TEXT DEFAULT 'NEUTRAL',
                volume_24h_usd REAL DEFAULT 0.0,
                rsi REAL DEFAULT 50.0,
                sentiment REAL DEFAULT 0.0,
                timestamp TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_verify_ticker_symbol ON verify_pro_ticker(symbol);')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shadow_portfolio_v7 (
                symbol TEXT PRIMARY KEY,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                type TEXT NOT NULL CHECK(type IN ('LONG', 'SHORT')),
                leverage INTEGER DEFAULT 5 CHECK(leverage > 0),
                timestamp TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shadow_account (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                balance REAL NOT NULL DEFAULT 100000.0,
                total_equity REAL DEFAULT 100000.0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            INSERT OR IGNORE INTO shadow_account (id, balance, total_equity)
            VALUES (1, 100000.0, 100000.0)
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ verification_pro.db 初始化成功: {DB_VERIFY}")
        return True
        
    except Exception as e:
        logger.error(f"❌ verification_pro.db 初始化失败: {e}")
        return False


def get_db_connection():
    try:
        conn = sqlite3.connect(DB_VERIFY, timeout=30)
        conn.execute('PRAGMA journal_mode=WAL;')
        return conn
    except Exception as e:
        logger.error(f"❌ 数据库连接失败: {e}")
        raise


if __name__ == "__main__":
    if init_verification_db():
        print("✅ 数据库初始化完成")
    else:
        print("❌ 数据库初始化失败")
