import os
import time
import sqlite3
import akshare as ak
import pandas as pd
import requests
import json
import feedparser
import hashlib
import re
from bs4 import BeautifulSoup
from datetime import datetime
from threading import Thread, Lock
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed

# 🔌 新增依赖：WebSocket + SOCKS5
import asyncio
import websockets
import socks

# 🛡️ 环境配置
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# 📦 导入统一配置
from config import (
    CRYPTO_LIST, STOCK_A_LIST, STOCK_HK_LIST, STOCK_US_LIST,
    DB_MEMORY, VLLM_API, BENZINGA_KEY, QWEN_LOG, STATE_JSON,
    DEVICE_ID, FINBERT_MODEL_PATH, HISTORY_TABLE_SCHEMA,
    setup_logger, get_logger
)

# 📦 导入高级分析模块
from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer
from anomaly_detector import AnomalyDetector

# 📊 日志配置
logger = setup_logger('alpha_processor', level=20)

def clean_html(raw_html):
    if not raw_html or not isinstance(raw_html, str): return ""
    try:
        text = BeautifulSoup(raw_html, "html.parser").get_text(separator=' ', strip=True)
        return ' '.join(text.split())
    except: return str(raw_html).strip() if raw_html else ""

# === 🧠 Qwen3 推理引擎 (4x4070 适配版) ===
class QwenThinker:
    def __init__(self, model_name="qwen3-thinking"):
        self.model_name = model_name
        self.trigger_keywords = ["美联储", "加息", "降息", "巨鲸", "匿名地址", "NVDA", "BTC", "ETH", "ETF", "SEC"]

    def should_think(self, text):
        return any(kw in text for kw in self.trigger_keywords)

    def ask_qwen(self, news_text):
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": f"[首席研究员指令]：请基于下述情报进行定性推演。禁止回答数据缺失。\n情报内容：\"{news_text}\""}
            ],
            "max_tokens": 1000,
            "temperature": 0.4
        }
        try:
            r = requests.post(VLLM_API, json=payload, timeout=120)
            res = r.json()['choices'][0]['message']['content'].strip()
            if "</think>" in res:
                output = res.split("</think>")[-1].strip()
                if len(output) < 20: 
                    output = "【深度推演报告】:\n" + res.replace("<think>","").replace("</think>","").strip()
            else:
                output = "【逻辑外推全文】:\n" + res.strip()
            return output
        except: return "推理请求超时"

# === 📡 情报炼金术士 (全量复刻 + 市场标签版) ===
class NewsAlchemist:
    def __init__(self, db_path):
        self.db_path = db_path
        self.bz_last_ts = int(time.time()) - 3600
        self.seen_hashes = set()

    def _get_fp(self, text): return hashlib.md5(text.encode()).hexdigest()[:16]

    def _fetch_tg(self):
        items = []
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cur = conn.cursor()
            limit = (datetime.now() - pd.Timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
            cur.execute("SELECT source, content FROM telegram_news WHERE timestamp > ? ORDER BY id DESC LIMIT 5", (limit,))
            for s, c in cur.fetchall():
                txt = clean_html(c)
                fp = self._get_fp(txt)
                if fp not in self.seen_hashes:
                    items.append((f"[TG_{s}]", txt))
                    self.seen_hashes.add(fp)
            conn.close()
        except Exception as e:
            logger.warning(f"获取Telegram新闻失败: {e}")
        return items

    def _fetch_cls(self):
        items = []
        try:
            r = requests.get("https://www.cls.cn/nodeapi/telegraphList?rn=5", timeout=8)
            for i in r.json().get('data', {}).get('roll_data', []):
                c = clean_html(i.get('content', ''))
                fp = self._get_fp(c)
                if fp not in self.seen_hashes:
                    items.append(("[CLS]", c))
                    self.seen_hashes.add(fp)
        except Exception as e:
            logger.warning(f"获取财联社新闻失败: {e}")
        return items

    def _fetch_bz(self):
        items = []
        try:
            url = f"https://api.benzinga.com/api/v2/news?token={BENZINGA_KEY}&pageSize=3"
            r = requests.get(url, timeout=10)
            for i in r.json():
                c = clean_html(i.get('title', ''))
                fp = self._get_fp(c)
                if fp not in self.seen_hashes:
                    items.append(("[BZ]", c))
                    self.seen_hashes.add(fp)
        except Exception as e:
            logger.warning(f"获取Benzinga新闻失败: {e}")
        return items

    def fetch_all(self):
        all_news = []
        with ThreadPoolExecutor(max_workers=4) as exec:
            tg_fut = exec.submit(self._fetch_tg)
            cls_fut = exec.submit(self._fetch_cls)
            bz_fut = exec.submit(self._fetch_bz)

            for s, c in tg_fut.result():
                all_news.append((f"{s} {c}", "crypto"))
            for s, c in cls_fut.result():
                all_news.append((f"{s} {c}", "stock"))
            for s, c in bz_fut.result():
                all_news.append((f"{s} {c}", "stock"))

        return all_news[-20:]

# === 🌐 Binance WebSocket 引擎（✅ 完整继承原版 SOCKS5 代理 + WAL模式优化）===
class BinanceWebSocket:
    def __init__(self, crypto_list, db_path, proxy_host="127.0.0.1", proxy_port=1080):
        self.crypto_list = crypto_list
        self.db_path = db_path
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.ticker_cache = {}
        self.cache_lock = Lock()
        self.running = True
        self.conn = None
        self.cursor = None
        self._init_db()
        self.ws_thread = Thread(target=self._run_async_loop, daemon=True)
        self.ws_thread.start()
        logger.info(f"Binance WebSocket 初始化完成，代理: socks5://{proxy_host}:{proxy_port}")

    def _init_db(self):
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute('PRAGMA journal_mode=WAL;')
            self.conn.execute('PRAGMA synchronous=NORMAL;')
            self.conn.execute('PRAGMA cache_size=-64000;')
            self.cursor = self.conn.cursor()
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_ticker_stream
                (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recv_time TEXT,
                    event_time INTEGER,
                    symbol TEXT,
                    price REAL,
                    volume REAL,
                    change_pct REAL,
                    source TEXT
                )
            """)
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_time_raw ON raw_ticker_stream (symbol, recv_time)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_recv_time_raw ON raw_ticker_stream (recv_time)")
            self.conn.commit()
            logger.info("raw_ticker_stream 表初始化完成")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise

    def _get_streams(self):
        streams = []
        for pair in self.crypto_list:
            symbol = pair.replace("/", "").lower()
            streams.append(f"{symbol}@ticker")
        return "/".join(streams)

    def _create_socks_tcp_socket(self, host, port):
        sock = socks.socksocket()
        sock.set_proxy(
            proxy_type=socks.SOCKS5,
            addr=self.proxy_host,
            port=self.proxy_port,
            rdns=True
        )
        sock.connect((host, port))
        return sock

    def _run_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._websocket_client())

    async def _websocket_client(self):
        host = "stream.binance.com"
        port = 443
        path = f"/stream?streams={self._get_streams()}"
        uri = f"wss://{host}{path}"

        while self.running:
            raw_tcp_sock = None
            try:
                raw_tcp_sock = self._create_socks_tcp_socket(host, port)
                async with websockets.connect(
                    uri,
                    sock=raw_tcp_sock,
                    ssl=True,
                    close_timeout=5,
                    max_size=None,
                    ping_interval=20,
                    ping_timeout=10
                ) as ws:
                    logger.info(f"WebSocket 通过 socks5://{self.proxy_host}:{self.proxy_port} 隧道连接 Binance 成功")
                    async for msg in ws:
                        if not self.running:
                            break
                        try:
                            data = json.loads(msg)
                            symbol = data['s']
                            recv_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                            event_time = data['E']
                            price = float(data['c'])
                            volume = float(data['v'])
                            change_pct = float(data['P'])

                            self.cursor.execute("""
                                INSERT INTO raw_ticker_stream 
                                (recv_time, event_time, symbol, price, volume, change_pct, source)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (recv_time, event_time, symbol, price, volume, change_pct, "binance"))
                            self.conn.commit()

                            with self.cache_lock:
                                self.ticker_cache[symbol] = {
                                    'price': price,
                                    'change_pct': change_pct,
                                    'volume': volume
                                }
                        except Exception as e:
                            logger.error(f"消息解析/写入错误: {e}")
            except Exception as e:
                logger.warning(f"WebSocket 隧道断开 ({e})，5秒后重连...")
                if self.running:
                    await asyncio.sleep(5)
            finally:
                if raw_tcp_sock:
                    try:
                        raw_tcp_sock.close()
                    except:
                        pass

    def get_ticker(self, symbol):
        with self.cache_lock:
            return self.ticker_cache.get(symbol)

    def stop(self):
        logger.info("正在关闭 Binance WebSocket...")
        self.running = False
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Binance WebSocket 已关闭")

# === 🏗️ 全球市场炼金术士 (4x4070 旗舰整合版 + 双情绪引擎 + WebSocket 快照 + REST降级) ===
class GlobalMarketAlchemist:
    def __init__(self, use_websocket=True, proxy_host="127.0.0.1", proxy_port=1080):
        logger.info("4x4070 旗舰级全情报引擎初始化...")
        logger.info(f"使用本地 FinBERT 模型: {FINBERT_MODEL_PATH}")
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=FINBERT_MODEL_PATH,
            tokenizer=FINBERT_MODEL_PATH,
            device=DEVICE_ID,
            local_files_only=True
        )
        self.news_tool = NewsAlchemist(DB_MEMORY)
        self.qwen_expert = QwenThinker()
        self.conn = sqlite3.connect(DB_MEMORY, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self.cursor = self.conn.cursor()
        self.crypto_sentiment = 0.0
        self.stock_sentiment = 0.0
        self.latest_news = []
        
        self.use_websocket = use_websocket
        if use_websocket:
            self.binance_ws = BinanceWebSocket(CRYPTO_LIST, DB_MEMORY, proxy_host, proxy_port)
            logger.info("Binance WebSocket 已启用")
        else:
            self.binance_ws = None
            logger.info("使用 REST API 获取加密货币数据")
        
        self.cursor.execute(HISTORY_TABLE_SCHEMA)
        self.conn.commit()
        
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        logger.info("高级情绪分析器和异常检测器已初始化")
        
        logger.info("4x4070 集群初始化完成")

    def _calc_sentiment(self, news_list):
        if not news_list:
            return 0.0
        try:
            res = self.sentiment_pipe(news_list[:5], truncation=True)
            scores = [({'Positive': 1, 'Negative': -1, 'Neutral': 0}[r['label']]) * r['score'] for r in res]
            return sum(scores) / len(scores)
        except Exception as e:
            logger.error(f"情绪分析异常: {e}")
            return 0.0

    def _fetch_crypto_rest(self):
        crypto_data = []
        try:
            for pair in CRYPTO_LIST:
                symbol = pair.replace("/", "")
                url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    crypto_data.append({
                        'pair': pair,
                        'price': float(data['lastPrice']),
                        'change_pct': float(data['priceChangePercent']),
                        'volume': float(data['volume'])
                    })
            logger.debug(f"REST API 获取 {len(crypto_data)} 个加密货币数据")
        except Exception as e:
            logger.error(f"REST API 获取加密货币数据失败: {e}")
        return crypto_data

    def stop(self):
        logger.info("正在关闭 GlobalMarketAlchemist...")
        if self.binance_ws:
            self.binance_ws.stop()
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("GlobalMarketAlchemist 已关闭")

    def run_cycle(self):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql_batch = []
        snapshot_data = []

        # 1. 采集全量情报并按市场分类
        labeled_news = self.news_tool.fetch_all()
        if labeled_news:
            self.latest_news = [text for text, _ in labeled_news]
            crypto_texts = [text for text, mkt in labeled_news if mkt == "crypto"]
            stock_texts = [text for text, mkt in labeled_news if mkt == "stock"]
            self.crypto_sentiment = self._calc_sentiment(crypto_texts)
            self.stock_sentiment = self._calc_sentiment(stock_texts)

            for news in self.latest_news[:2]:
                if self.qwen_expert.should_think(news):
                    logic = self.qwen_expert.ask_qwen(news)
                    try:
                        with open(QWEN_LOG, "a", encoding="utf-8") as f:
                            f.write(f"\n[{ts}] 深度研报：\n原文：{news}\n结论：{logic}\n" + "-"*30)
                    except Exception as e:
                        logger.error(f"写入Qwen日志失败: {e}")
                    break

        # 2. 采集行情 —— WebSocket优先，REST API降级
        # 加密货币 (WebSocket优先，REST API降级)
        if self.use_websocket and self.binance_ws:
            try:
                for pair in CRYPTO_LIST:
                    symbol = pair.replace("/", "")
                    ticker = self.binance_ws.get_ticker(symbol)
                    if ticker:
                        p = ticker['price']
                        change_pct = ticker['change_pct']
                        vol = ticker['volume']
                        sql_batch.append((ts, f"CRY_{pair}", p, 50.0, self.crypto_sentiment, vol, "4070_Cluster"))
                        snapshot_data.append({"s": f"CRY_{pair}", "p": p, "rsi": 50.0, "change_pct": change_pct, "volume": vol})
                logger.debug(f"从 WebSocket 获取 {len([d for d in snapshot_data if d['s'].startswith('CRY_')])} 个加密货币数据")
            except Exception as e:
                logger.error(f"WebSocket 数据获取失败，降级到 REST API: {e}")
                self.use_websocket = False
        
        if not self.use_websocket or not self.binance_ws:
            crypto_rest_data = self._fetch_crypto_rest()
            for data in crypto_rest_data:
                p = data['price']
                change_pct = data['change_pct']
                vol = data['volume']
                sql_batch.append((ts, f"CRY_{data['pair']}", p, 50.0, self.crypto_sentiment, vol, "REST_API"))
                snapshot_data.append({"s": f"CRY_{data['pair']}", "p": p, "rsi": 50.0, "change_pct": change_pct, "volume": vol})
            logger.debug(f"从 REST API 获取 {len(crypto_rest_data)} 个加密货币数据")

        # A股 (AkShare)
        try:
            df = ak.stock_zh_a_spot_em()
            for c in STOCK_A_LIST:
                m = df[df['代码'] == c]
                if not m.empty:
                    p = float(m['最新价'].iloc[0])
                    sql_batch.append((ts, f"A_{c}", p, 50.0, self.stock_sentiment, 0, "4070_Cluster"))
                    snapshot_data.append({"s": f"A_{c}", "p": p, "rsi": 50.0})
            logger.debug(f"获取 {len([d for d in snapshot_data if d['s'].startswith('A_')])} 个A股数据")
        except Exception as e:
            logger.error(f"A股获取失败: {e}")

        # 港股 & 美股 (AkShare)
        try:
            df_hk = ak.stock_hk_spot_em()
            for c in STOCK_HK_LIST:
                m = df_hk[df_hk['代码'] == c]
                p = float(m['最新价'].iloc[0]) if not m.empty else None
                if p:
                    sql_batch.append((ts, f"HK_{c}", p, 50.0, self.stock_sentiment, 0, "4070_Cluster"))
                    snapshot_data.append({"s": f"HK_{c}", "p": p, "rsi": 50.0})
            
            df_us = ak.stock_us_spot_em()
            for c in STOCK_US_LIST:
                m = df_us[df_us['代码'] == c]
                p = float(m['最新价'].iloc[0]) if not m.empty else None
                if p:
                    sql_batch.append((ts, f"US_{c}", p, 50.0, self.stock_sentiment, 0, "4070_Cluster"))
                    snapshot_data.append({"s": f"US_{c}", "p": p, "rsi": 50.0})
            logger.debug(f"获取 {len([d for d in snapshot_data if d['s'].startswith('HK_')])} 个港股数据，{len([d for d in snapshot_data if d['s'].startswith('US_')])} 个美股数据")
        except Exception as e:
            logger.error(f"港美股获取失败: {e}")

        # 3. 存储与同步
        if sql_batch:
            try:
                self.cursor.executemany("INSERT INTO history VALUES (?,?,?,?,?,?,?)", sql_batch)
                self.conn.commit()
            except Exception as e:
                logger.error(f"数据库写入失败: {e}")

        # 4. 高级情绪分析
        advanced_sentiment_results = {}
        try:
            for pair in CRYPTO_LIST:
                symbol = pair.replace("/", "")
                sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(symbol)
                if sentiment_analysis:
                    self.sentiment_analyzer.save_analysis_to_db(sentiment_analysis)
                    summary = self.sentiment_analyzer.get_sentiment_summary(sentiment_analysis)
                    advanced_sentiment_results[symbol] = summary
                    logger.info(f"{symbol} 高级情绪分析: {summary['sentiment_signal']} (情绪: {summary['overall_sentiment']:.2f})")
        except Exception as e:
            logger.error(f"高级情绪分析失败: {e}")

        # 5. 异常检测
        anomaly_results = {}
        try:
            for pair in CRYPTO_LIST:
                symbol = pair.replace("/", "")
                anomaly_detection = self.anomaly_detector.detect_anomalies(symbol)
                if anomaly_detection:
                    self.anomaly_detector.save_anomaly_to_db(anomaly_detection)
                    summary = self.anomaly_detector.get_anomaly_summary(anomaly_detection)
                    anomaly_results[symbol] = summary
                    if summary['has_anomaly']:
                        logger.warning(f"{symbol} 检测到异常: {summary['anomaly_count']} 个, 风险分数: {summary['overall_risk_score']:.2f}")
        except Exception as e:
            logger.error(f"异常检测失败: {e}")

        try:
            with open(STATE_JSON, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": ts, 
                    "sentiment": {
                        "crypto": self.crypto_sentiment,
                        "stock": self.stock_sentiment
                    },
                    "advanced_sentiment": advanced_sentiment_results,
                    "anomaly_detection": anomaly_results,
                    "news": self.latest_news[:10], 
                    "data": snapshot_data
                }, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"状态JSON写入失败: {e}")

        logger.info(f"[{ts}] 4070集群运行 | 情报:{len(self.latest_news)} | 资产:{len(snapshot_data)} | Crypto情绪:{self.crypto_sentiment:+.4f} | Stock情绪:{self.stock_sentiment:+.4f}")

if __name__ == "__main__":
    logger.info("启动 4x4070 旗舰级全情报引擎...")
    agent = GlobalMarketAlchemist(use_websocket=True, proxy_host="127.0.0.1", proxy_port=1080)
    
    try:
        logger.info("主循环启动，每30秒执行一次...")
        while True:
            try:
                agent.run_cycle()
                time.sleep(30)
            except KeyboardInterrupt:
                logger.info("收到键盘中断信号")
                break
            except Exception as e:
                logger.error(f"主循环异常: {e}", exc_info=True)
                time.sleep(15)
    except KeyboardInterrupt:
        logger.info("收到键盘中断信号")
    except Exception as e:
        logger.error(f"系统级异常: {e}", exc_info=True)
    finally:
        logger.info("正在关闭系统...")
        agent.stop()
        logger.info("系统已安全退出")