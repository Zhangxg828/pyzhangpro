import pandas as pd
import sqlite3
import os
import glob
from config import DB_MEMORY, DATASET_PATH, HISTORY_TABLE_COLUMNS, setup_logger, DATA_DIR

logger = setup_logger('data_importer', os.path.join(DATA_DIR, 'data_importer.log'))

DB_PATH = DB_MEMORY


def upgrade_table_structure(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(history)")
        existing_cols = [info[1] for info in cursor.fetchall()]

        standard_cols = {
            "timestamp": "TEXT",
            "symbol": "TEXT",
            "price": "REAL",
            "volume": "REAL",
            "rsi": "REAL",
            "sentiment": "REAL",
            "source": "TEXT"
        }

        if not existing_cols:
            logger.info("正在创建标准 history 表...")
            cols_str = ", ".join([f"{k} {v}" for k, v in standard_cols.items()])
            cursor.execute(f"CREATE TABLE history ({cols_str})")
        else:
            for col, col_type in standard_cols.items():
                if col not in existing_cols:
                    logger.info(f"正在补齐缺失列: {col}")
                    cursor.execute(f"ALTER TABLE history ADD COLUMN {col} {col_type} DEFAULT 0")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_time ON history (symbol, timestamp)")
        conn.commit()
        logger.info("表结构升级完成")
    except Exception as e:
        logger.error(f"表结构升级失败: {e}")
        raise


def import_historical_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('PRAGMA journal_mode=WAL;')
        upgrade_table_structure(conn)

        csv_files = glob.glob(os.path.join(DATASET_PATH, "*.csv"))
        if not csv_files:
            logger.error(f"未找到 CSV 文件，路径: {DATASET_PATH}")
            return

        logger.info(f"找到 {len(csv_files)} 个 CSV 文件")
        total_rows = 0
        for file in csv_files:
            logger.info(f"正在加载: {os.path.basename(file)}")
            try:
                chunk_iter = pd.read_csv(file, chunksize=200000)

                for chunk in chunk_iter:
                    try:
                        rename_map = {
                            'date': 'timestamp',
                            'close': 'price',
                            'last': 'price',
                            'vol': 'volume',
                            'amount': 'volume'
                        }
                        chunk = chunk.rename(columns=rename_map)

                        for col in ['volume', 'rsi', 'sentiment']:
                            if col not in chunk.columns:
                                chunk[col] = 0.0

                        target_cols = ['timestamp', 'symbol', 'price', 'volume', 'rsi', 'sentiment']
                        valid_chunk = chunk[[c for c in target_cols if c in chunk.columns]]

                        valid_chunk.to_sql('history', conn, if_exists='append', index=False)
                        total_rows += len(valid_chunk)
                        logger.debug(f"累计导入: {total_rows} 条数据")
                    except Exception as e:
                        logger.error(f"处理数据块失败: {e}")
                        continue
            except Exception as e:
                logger.error(f"读取文件 {file} 失败: {e}")
                continue

        conn.commit()
        conn.close()
        logger.info(f"数据导入完成！共计 {total_rows} 条历史记录")
    except Exception as e:
        logger.error(f"导入历史数据失败: {e}")
        raise


if __name__ == "__main__":
    import_historical_data()