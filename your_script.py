import sys
import os
import time
import sqlite3
import threading
import requests
import json
from datetime import datetime
from openai import OpenAI
from config import (DB_MEMORY, DB_VERIFY, DATA_DIR, VLLM_API, MODEL_NAME,
                    SYMBOLS, TIMEFRAME, THRESHOLD_PERCENT, QWEN_LOG,
                    HISTORY_TABLE_SCHEMA, setup_logger)

logger = setup_logger('your_script', os.path.join(DATA_DIR, 'your_script.log'))

client = OpenAI(
    api_key="EMPTY",
    base_url=VLLM_API
)

print_lock = threading.Lock()
last_analysis_time = {symbol: 0 for symbol in SYMBOLS}


def init_db():
    try:
        conn = sqlite3.connect(DB_MEMORY)
        conn.execute('PRAGMA journal_mode=WAL;')
        cur = conn.cursor()
        cur.execute(HISTORY_TABLE_SCHEMA)
        cur.execute("""
                    CREATE TABLE IF NOT EXISTS qwen_analysis_history
                    (
                        timestamp TEXT,
                        raw_news TEXT,
                        reasoning_output TEXT
                    )
                    """)
        conn.commit()
        conn.close()
        logger.info("数据库初始化成功")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise


def fetch_binance_ohlcv(symbol, timeframe='15m', limit=10):
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 Master-Quant-2026'})
    session.proxies = {'http': 'socks5h://127.0.0.1:1080', 'https': 'socks5h://127.0.0.1:1080'}

    symbol_clean = symbol.replace('/', '')
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol_clean, 'interval': timeframe, 'limit': limit}

    try:
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        raw_data = response.json()
        logger.debug(f"成功获取 {symbol} 的 K 线数据")
        return [[r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in raw_data]
    except requests.exceptions.Timeout:
        logger.warning(f"网络超时 ({symbol})")
        with print_lock:
            print(f"❌ 网络超时 ({symbol})")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"网络请求异常 ({symbol}): {e}")
        with print_lock:
            print(f"❌ 网络异常 ({symbol}): {e}")
        return None
    except Exception as e:
        logger.error(f"未知异常 ({symbol}): {e}")
        with print_lock:
            print(f"❌ 未知异常 ({symbol}): {e}")
        return None
    finally:
        session.close()


# --- [5. 核心逻辑：深度推演与数据沉淀] ---
def ask_qwen_analysis(symbol, price, change, data):
    ts_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prompt = f"""
    [指令：首席量化研究员决策模式 - 深度逻辑链版]
    标的资产：{symbol} | 当前价格：{price} | 异常波动：{change:.2f}%

    [输入数据：5周期K线快照]
    {data[-5:]}

    [任务：基于 NoFx 框架进行严谨推演]
    1. 🔍 [VSA 量价分析] 2. 🧠 [博弈心理] 3. ⚖️ [反向逻辑自检] 4. 🎯 [实战指令]
    请在 </think> 标签内展示完整逻辑。
    """

    with print_lock:
        print(f"\n{'=' * 70}\n🧠 [4x4070 Cluster] 唤醒 2026 逻辑阵列: {symbol} ({change:.2f}%)\n{'=' * 70}")

    full_response = []
    try:
        logger.info(f"开始分析 {symbol}，波动率: {change:.2f}%")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
            temperature=0.15,
            max_tokens=3072
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end='', flush=True)
                full_response.append(content)

        final_logic = "".join(full_response)

        if final_logic:
            try:
                conn = sqlite3.connect(DB_MEMORY)
                conn.execute('PRAGMA journal_mode=WAL;')
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO qwen_analysis_history (timestamp, raw_news, reasoning_output) VALUES (?, ?, ?)",
                    (ts_str, f"异动监测: {symbol} 波动{change:.2f}%", final_logic)
                )
                conn.commit()
                conn.close()
                logger.info(f"{symbol} 分析结果已保存到数据库")
                with open(QWEN_LOG, "a", encoding="utf-8") as f:
                    f.write(f"\n[{ts_str}] {symbol} 推演结论已入库。\n")
            except Exception as e:
                logger.error(f"保存分析结果失败: {e}")

    except Exception as e:
        logger.error(f"算力矩阵异常 ({symbol}): {e}")
        with print_lock:
            print(f"❌ [算力矩阵异常]: {e}")


def monitor_symbol(symbol):
    global last_analysis_time
    with print_lock:
        print(f"✅ {symbol} 监控矩阵点火成功...")
    logger.info(f"启动 {symbol} 监控线程")

    while True:
        try:
            ohlcv_data = fetch_binance_ohlcv(symbol, timeframe=TIMEFRAME, limit=10)
            if not ohlcv_data or len(ohlcv_data) < 2:
                logger.debug(f"{symbol} 数据不足，等待重试")
                time.sleep(20)
                continue

            current_p = ohlcv_data[-1][4]
            last_p = ohlcv_data[-2][4]
            delta_p = ((current_p - last_p) / last_p) * 100
            ts_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            try:
                conn = sqlite3.connect(DB_MEMORY)
                conn.execute('PRAGMA journal_mode=WAL;')
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO history (timestamp, symbol, price, sentiment, source) VALUES (?, ?, ?, ?, ?)",
                    (ts_now, f"CRY_{symbol.replace('/', '')}", current_p, 0.0, "WATCHER")
                )
                conn.commit()
                conn.close()
                logger.debug(f"{symbol} 数据已保存: 价格={current_p}, 波动={delta_p:.2f}%")
            except Exception as e:
                logger.error(f"保存 {symbol} 数据失败: {e}")

            if abs(delta_p) >= THRESHOLD_PERCENT:
                now_ts = time.time()
                if now_ts - last_analysis_time[symbol] > 600:
                    logger.info(f"{symbol} 触发分析阈值，波动率: {delta_p:.2f}%")
                    ask_qwen_analysis(symbol, current_p, delta_p, ohlcv_data)
                    last_analysis_time[symbol] = now_ts

            time.sleep(15)
        except Exception as e:
            logger.error(f"{symbol} 监控线程异常: {e}")
            time.sleep(10)


# --- [7. 主程序入口] ---
def main():
    try:
        init_db()
        logger.info("NoFx-Alpha 2026 旗舰级监控系统启动")
        print(f"🚀 [NoFx-Alpha 2026] 旗舰级监控系统启动")
        print(f"数据持久化已开启: {DB_MEMORY}")
        print("-" * 50)

        threads = []
        for s in SYMBOLS:
            t = threading.Thread(target=monitor_symbol, args=(s,), daemon=True)
            t.start()
            threads.append(t)

        logger.info(f"已启动 {len(threads)} 个监控线程")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在关闭系统...")
            print("\n[系统下线] 正在释放资源...")
    except Exception as e:
        logger.critical(f"系统启动失败: {e}")
        raise


if __name__ == "__main__":
    main()