import pandas as pd
import sqlite3
import os
from config import DB_MEMORY, DATASET_PATH, HISTORY_TABLE_SCHEMA, setup_logger, DATA_DIR

logger = setup_logger('importer_3060', os.path.join(DATA_DIR, 'importer_3060.log'))

DB_PATH = DB_MEMORY
CSV_PATH = os.path.join(DATASET_PATH, "chinese-stock-dataset.csv")


def smart_import():
    try:
        if not os.path.exists(CSV_PATH):
            logger.error(f"找不到数据集文件: {CSV_PATH}")
            return

        logger.info("正在重置并优化数据库表结构...")
        conn = sqlite3.connect(DB_PATH)
        conn.execute('PRAGMA journal_mode=WAL;')
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS history")
        cursor.execute(HISTORY_TABLE_SCHEMA)
        conn.commit()

        mapping = {
            'date': 'timestamp',
            'close': 'price',
            'stock_code': 'symbol'
        }

        logger.info(f"映射配置: {mapping}")

        chunksize = 100000
        total_rows = 0

        try:
            for chunk in pd.read_csv(CSV_PATH, chunksize=chunksize):
                try:
                    df = chunk[list(mapping.keys())].rename(columns=mapping)
                    df['symbol'] = df['symbol'].astype(str).str.strip()
                    df['rsi'] = 50.0
                    df['sentiment'] = 0.0
                    df['volume'] = 0.0
                    df['source'] = 'IMPORT'

                    df.to_sql('history', conn, if_exists='append', index=False)
                    total_rows += len(df)
                    logger.debug(f"已处理 {total_rows} 条数据")
                except Exception as e:
                    logger.error(f"处理数据块失败: {e}")
                    continue

        except Exception as e:
            logger.error(f"读取 CSV 文件失败: {e}")
            raise
        finally:
            conn.close()
            logger.info(f"导入完成！共计 {total_rows} 条历史记录")
    except Exception as e:
        logger.error(f"智能导入失败: {e}")
        raise


if __name__ == "__main__":
    smart_import()