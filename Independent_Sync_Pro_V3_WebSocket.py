import sqlite3, time, pandas as pd
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    print("pandas-ta 未安装，将使用替代方法")
    PANDAS_TA_AVAILABLE = False
    ta = None
import asyncio
import websockets
import socks
import json
from datetime import datetime
from threading import Thread, Lock

# === 📁 核心对齐配置 ===
from config import DB_MEMORY, DB_VERIFY
PROXY_URL = 'socks5h://127.0.0.1:1080'
# 💡 确保这里包含驾驶舱显示的所有币种
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'DOGE/USDT', 'APT/USDT', 'XRP/USDT', 'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT']

class BinanceWebSocket:
    def __init__(self, symbols, proxy_url):
        self.symbols = symbols
        self.proxy_url = proxy_url
        self.ticker_cache = {}
        self.order_book_cache = {}
        self.cache_lock = Lock()
        self.running = True
        
        self.proxy_host = None
        self.proxy_port = None
        
        if self.proxy_url and self.proxy_url != 'None':
            self._parse_proxy()
        
        self.ws_thread = Thread(target=self._run_async_loop, daemon=True)
        self.ws_thread.start()
    
    def _parse_proxy(self):
        try:
            proxy_url = self.proxy_url
            if proxy_url.startswith('socks5h://'):
                url = proxy_url.replace('socks5h://', '')
            elif proxy_url.startswith('socks5://'):
                url = proxy_url.replace('socks5://', '')
            else:
                url = proxy_url
            
            if ':' in url:
                host, port = url.split(':')
                self.proxy_host = host
                self.proxy_port = int(port)
        except Exception as e:
            print(f"解析代理地址失败: {e}")
    
    def _create_socks_tcp_socket(self, host, port):
        try:
            sock = socks.socksocket()
            sock.set_proxy(
                proxy_type=socks.SOCKS5,
                addr=self.proxy_host,
                port=self.proxy_port,
                rdns=True
            )
            sock.connect((host, port))
            return sock
        except Exception as e:
            print(f"创建SOCKS5代理socket失败: {e}")
            raise
    
    async def _websocket_client(self):
        try:
            streams = []
            for symbol in self.symbols:
                binance_symbol = symbol.replace("/", "").lower()
                streams.append(f"{binance_symbol}@ticker")
                streams.append(f"{binance_symbol}@depth5")
            
            combined_stream = "/".join(streams)
            uri = f"wss://stream.binance.com:9443/ws/{combined_stream}"
            
            raw_tcp_sock = None
            if self.proxy_host and self.proxy_port:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(uri)
                    raw_tcp_sock = self._create_socks_tcp_socket(parsed.hostname, 443)
                except Exception as e:
                    print(f"创建代理socket失败: {e}")
                    return
            
            async with websockets.connect(
                uri,
                sock=raw_tcp_sock,
                ssl=True,
                close_timeout=10,
                max_size=None,
                ping_interval=30,
                ping_timeout=20
            ) as ws:
                print(f"WebSocket连接成功，订阅 {len(self.symbols)} 个交易对")
                
                while self.running:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=60.0)
                        data = json.loads(message)
                        event_type = data.get('e', '')
                        
                        if event_type == '24hrTicker':
                            self._process_ticker(data)
                        elif event_type == 'depthUpdate':
                            self._process_depth(data)
                    except asyncio.TimeoutError:
                        continue
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误: {e}")
                        continue
                    except Exception as e:
                        print(f"接收消息错误: {e}")
                        break
        
        except websockets.exceptions.WebSocketException as e:
            print(f"WebSocket连接异常: {e}")
        except Exception as e:
            print(f"WebSocket客户端错误: {e}")
    
    def _run_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._websocket_client())
        finally:
            loop.close()
    
    def _process_ticker(self, data):
        try:
            symbol = data.get('s', '')
            
            if not symbol:
                return
            
            with self.cache_lock:
                self.ticker_cache[symbol] = {
                    'symbol': symbol,
                    'price': float(data.get('c', 0)),
                    'volume': float(data.get('v', 0)),
                    'quote_volume': float(data.get('q', 0)),
                    'change_pct': float(data.get('P', 0)),
                    'high': float(data.get('h', 0)),
                    'low': float(data.get('l', 0)),
                    'open': float(data.get('o', 0)),
                    'timestamp': data.get('E', 0)
                }
        except Exception as e:
            print(f"处理ticker数据失败: {e}")
    
    def _process_depth(self, data):
        try:
            symbol = data.get('s', '')
            bids = data.get('b', [])
            asks = data.get('a', [])
            
            if not symbol or not bids or not asks:
                return
            
            bid_volume = sum(float(b[1]) for b in bids)
            ask_volume = sum(float(a[1]) for a in asks)
            order_ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0
            
            with self.cache_lock:
                self.order_book_cache[symbol] = {
                    'order_ratio': order_ratio,
                    'bids': bids,
                    'asks': asks,
                    'timestamp': data.get('E', 0)
                }
        except Exception as e:
            print(f"处理depth数据失败: {e}")
    
    def get_ticker(self, symbol):
        try:
            binance_symbol = symbol.replace("/", "")
            with self.cache_lock:
                if binance_symbol in self.ticker_cache:
                    return self.ticker_cache[binance_symbol]
        except Exception as e:
            print(f"获取 {symbol} ticker 失败: {e}")
        return None
    
    def get_order_book(self, symbol):
        try:
            binance_symbol = symbol.replace("/", "")
            with self.cache_lock:
                if binance_symbol in self.order_book_cache:
                    return self.order_book_cache[binance_symbol]['order_ratio']
        except Exception as e:
            print(f"获取 {symbol} 订单簿失败: {e}")
        return 1.0
    
    def stop(self):
        self.running = False
        if self.ws_thread:
            self.ws_thread.join(timeout=5)

class MarketSyncV3:
    def __init__(self):
        self.ws = BinanceWebSocket(SYMBOLS, PROXY_URL)
        self._init_db()
        
        import requests
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 Master-Quant-2026'})
        
        proxy = PROXY_URL if PROXY_URL and PROXY_URL != 'None' else None
        if proxy:
            self.session.proxies = {
                'http': proxy,
                'https': proxy
            }
    
    def _init_db(self):
        conn = sqlite3.connect(DB_VERIFY)
        conn.execute('''CREATE TABLE IF NOT EXISTS verify_pro_ticker (
            symbol TEXT PRIMARY KEY, price REAL, change_24h REAL, 
            volume_24h_usd REAL, order_ratio REAL, sar_value REAL, 
            sar_trend TEXT, last_update TEXT)''')
        
        # 确保表结构兼容（添加缺失的列，如果需要）
        try:
            conn.execute('ALTER TABLE verify_pro_ticker ADD COLUMN sar_value REAL')
        except sqlite3.OperationalError:
            pass  # 列已存在
        try:
            conn.execute('ALTER TABLE verify_pro_ticker ADD COLUMN sar_trend TEXT')
        except sqlite3.OperationalError:
            pass  # 列已存在
        conn.close()
    
    def get_sar(self, symbol):
        try:
            binance_symbol = symbol.replace("/", "")
            url = f"https://api.binance.com/api/v3/klines?symbol={binance_symbol}&interval=1h&limit=100"
            r = self.session.get(url, timeout=10)
            if r.status_code != 200:
                return None, "ERROR"
            
            bars = r.json()
            # 创建DataFrame，包含K线数据的12个字段
            # [open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
            df = pd.DataFrame(bars, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            
            if PANDAS_TA_AVAILABLE and ta is not None:
                psar_df = ta.psar(df['high'], df['low'], df['close'], af=0.02, max_af=0.2)
                if psar_df is not None and not psar_df.empty:
                    # 获取PSAR指标的最后一个值
                    last_row = psar_df.iloc[-1]
                    
                    # 查找包含SAR值的列名
                    sar_col = None
                    for col in psar_df.columns:
                        if 'psar' in col.lower():
                            sar_col = col
                            break
                    
                    if sar_col and sar_col in last_row:
                        sar_value = last_row[sar_col]
                        if not pd.isna(sar_value):
                            # 简单判断趋势：如果SAR值小于收盘价，则为上涨趋势(BULL)，否则为下跌趋势(BEAR)
                            close_price = float(df['close'].iloc[-1])
                            trend = "BULL" if float(sar_value) < close_price else "BEAR"
                            return float(sar_value), trend
            else:
                # 使用简单的替代方法：返回最近收盘价的移动平均值作为趋势参考
                close_prices = df['close'].values
                if len(close_prices) >= 5:
                    # 使用最近5个收盘价的平均值作为参考点
                    avg_price = sum(close_prices[-5:]) / 5
                    current_price = close_prices[-1]
                    trend = "BULL" if current_price > avg_price else "BEAR"
                    return avg_price, trend
            
            return None, "ERROR"
        except Exception as e:
            print(f"获取SAR失败: {e}")
            return None, "ERROR"
    
    def sync(self):
        conn_verify = sqlite3.connect(DB_VERIFY)
        conn_memory = sqlite3.connect(DB_MEMORY)
        conn_memory.execute('PRAGMA journal_mode=WAL;')
        
        for sym in SYMBOLS:
            ticker = self.ws.get_ticker(sym)
            if not ticker:
                continue
            
            order_ratio = self.ws.get_order_book(sym)
            sar_val, trend = self.get_sar(sym)
            
            # 更新验证数据库中的ticker表
            conn_verify.execute("""INSERT OR REPLACE INTO verify_pro_ticker (symbol, price, order_ratio, sar_value, sar_trend, volume_24h_usd, rsi, sentiment, timestamp) VALUES (?,?,?,?,?,?,?,?,?)""",
                (sym, ticker['price'], round(order_ratio, 4), sar_val, trend, 
                 ticker['quote_volume'], 0.0, 0.0, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            # 同时更新市场内存数据库中的raw_ticker_stream表，供情绪分析器使用
            # 尝试插入完整字段（包括buy_volume和sell_volume），如果字段不存在则插入基础字段
            try:
                conn_memory.execute("""INSERT INTO raw_ticker_stream (recv_time, event_time, symbol, price, volume, change_pct, source, buy_volume, sell_volume) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ticker['timestamp'], sym.replace('/', ''), 
                     ticker['price'], ticker['volume'], ticker['change_pct'], 'binance_ws', ticker['volume']/2, ticker['volume']/2))
            except sqlite3.OperationalError:
                # 如果表结构不匹配，则只插入现有字段
                conn_memory.execute("""INSERT INTO raw_ticker_stream (recv_time, event_time, symbol, price, volume, change_pct, source) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ticker['timestamp'], sym.replace('/', ''), 
                     ticker['price'], ticker['volume'], ticker['change_pct'], 'binance_ws'))
        
        conn_verify.commit()
        conn_verify.close()
        
        # 只保留最近1小时的数据以避免表过大
        conn_memory.execute("""DELETE FROM raw_ticker_stream WHERE recv_time < datetime('now', '-1 hour')""")
        conn_memory.commit()
        conn_memory.close()

if __name__ == "__main__":
    sync_engine = MarketSyncV3()
    print("等待WebSocket连接...")
    time.sleep(5)
    
    while True:
        try:
            sync_engine.sync()
            print(f"✅ {datetime.now().strftime('%H:%M:%S')} 同步成功")
            time.sleep(5)
        except Exception as e:
            print(f"❌ 同步异常: {e}"); time.sleep(5)
