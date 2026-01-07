import sqlite3
import pandas as pd
import os
from datetime import datetime, timedelta
from config import DB_MEMORY, DATA_DIR, setup_logger

logger = setup_logger('auto_summarizer', os.path.join(DATA_DIR, 'auto_summarizer.log'))

DB_PATH = DB_MEMORY
SUMMARY_OUT = os.path.join(DATA_DIR, "long_term_memory.txt")


def generate_automated_summary():
    try:
        if not os.path.exists(DB_PATH):
            logger.error(f"数据库尚未建立: {DB_PATH}")
            return "数据库尚未建立。"

        logger.info("开始生成自动摘要...")
        conn = sqlite3.connect(DB_PATH)
        conn.execute('PRAGMA journal_mode=WAL;')
        now = datetime.now()

        seven_days_ago = (now - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
        df = pd.read_sql_query(f"SELECT * FROM history WHERE timestamp >= '{seven_days_ago}'", conn)
        conn.close()

        if df.empty:
            logger.warning("数据量不足，无法生成摘要")
            return "数据量不足，无法生成摘要。"

        logger.info(f"找到 {len(df)} 条记录，正在生成摘要...")
        
        try:
            with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
                f.write(f"### [3060 自动归档记忆] 生成时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("> 提示：此文件由 3060 提取自 L2 SQLite 库，包含过去 7 天的宏观数据。\n\n")

                f.write("#### 1. 过去 7 天处于'超卖区间'的标的列表\n")
                oversold = df.groupby('symbol')['rsi'].mean()
                oversold_targets = oversold[oversold < 35].sort_values()
                if oversold_targets.empty:
                    f.write("- 无明显超卖标的。\n")
                for sym, rsi in oversold_targets.items():
                    f.write(f"- {sym}: 平均 RSI {rsi:.2f}\n")

                f.write("\n#### 2. 市场活跃度（波动率前 5）\n")
                volatility = df.groupby('symbol')['price'].apply(lambda x: (x.max() - x.min()) / x.min() * 100)
                for sym, vol in volatility.sort_values(ascending=False).head(5).items():
                    f.write(f"- {sym}: 7日波幅 {vol:.2f}%\n")

                f.write("\n#### 3. 关键资产锚点\n")
                sol_data = df[df['symbol'] == 'SOL/USDT']
                if not sol_data.empty:
                    f.write(f"- SOL 7日平均价: ${sol_data['price'].mean():.2f}\n")
                    f.write(f"- SOL 历史高/低: ${sol_data['price'].max():.2f} / ${sol_data['price'].min():.2f}\n")

            logger.info(f"长期记忆已更新至: {SUMMARY_OUT}")
            return f"摘要生成成功，共 {len(df)} 条记录"
        except IOError as e:
            logger.error(f"写入摘要文件失败: {e}")
            return "写入摘要文件失败"
    except Exception as e:
        logger.error(f"生成自动摘要失败: {e}")
        raise


if __name__ == "__main__":
    generate_automated_summary()