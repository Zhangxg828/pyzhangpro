import sqlite3
import os
from config import DB_MEMORY, DB_VERIFY, DATA_DIR, setup_logger

logger = setup_logger('init_sentiment_tables', os.path.join(DATA_DIR, 'init_sentiment_tables.log'))

def init_sentiment_tables():
    """初始化情绪分析所需的所有表"""
    try:
        # 初始化市场数据库中的表
        logger.info("开始初始化市场数据库中的情绪相关表...")
        conn_memory = sqlite3.connect(DB_MEMORY)
        conn_memory.execute('PRAGMA journal_mode=WAL;')
        cursor_memory = conn_memory.cursor()
        
        # 创建 social_media_sentiment 表
        cursor_memory.execute("""
            CREATE TABLE IF NOT EXISTS social_media_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建 news_sentiment 表
        cursor_memory.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建 sentiment_analysis 表
        cursor_memory.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                overall_sentiment REAL NOT NULL,
                sentiment_trend TEXT NOT NULL,
                fear_greed_index REAL NOT NULL,
                volatility_index REAL NOT NULL,
                extreme_sentiment INTEGER NOT NULL,
                sentiment_signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                sources TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引以提高查询性能
        cursor_memory.execute("CREATE INDEX IF NOT EXISTS idx_social_symbol_time ON social_media_sentiment (symbol, timestamp)")
        cursor_memory.execute("CREATE INDEX IF NOT EXISTS idx_news_symbol_time ON news_sentiment (symbol, timestamp)")
        cursor_memory.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_time ON sentiment_analysis (symbol, timestamp)")
        
        conn_memory.commit()
        conn_memory.close()
        logger.info("市场数据库中的情绪相关表初始化完成")
        
        # 初始化验证数据库中的表（如果需要）
        logger.info("开始初始化验证数据库中的情绪相关表...")
        conn_verify = sqlite3.connect(DB_VERIFY)
        conn_verify.execute('PRAGMA journal_mode=WAL;')
        cursor_verify = conn_verify.cursor()
        
        # 创建 social_media_sentiment 表
        cursor_verify.execute("""
            CREATE TABLE IF NOT EXISTS social_media_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建 news_sentiment 表
        cursor_verify.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建 sentiment_analysis 表
        cursor_verify.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                overall_sentiment REAL NOT NULL,
                sentiment_trend TEXT NOT NULL,
                fear_greed_index REAL NOT NULL,
                volatility_index REAL NOT NULL,
                extreme_sentiment INTEGER NOT NULL,
                sentiment_signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                sources TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引以提高查询性能
        cursor_verify.execute("CREATE INDEX IF NOT EXISTS idx_social_symbol_time ON social_media_sentiment (symbol, timestamp)")
        cursor_verify.execute("CREATE INDEX IF NOT EXISTS idx_news_symbol_time ON news_sentiment (symbol, timestamp)")
        cursor_verify.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_time ON sentiment_analysis (symbol, timestamp)")
        
        conn_verify.commit()
        conn_verify.close()
        logger.info("验证数据库中的情绪相关表初始化完成")
        
    except Exception as e:
        logger.error(f"初始化情绪分析表失败: {e}")
        raise

if __name__ == "__main__":
    init_sentiment_tables()
    print("✅ 情绪分析相关表初始化完成")