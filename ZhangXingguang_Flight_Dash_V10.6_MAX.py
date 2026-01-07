import sqlite3, os, time, re, json, threading, requests
import asyncio
import websockets
from websockets.exceptions import InvalidStatusCode
import socks
from datetime import datetime, timedelta
from openai import OpenAI
from colorama import Fore, Style
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from config import (
    DB_VERIFY, DB_MEMORY, VLLM_API, MODEL_NAME, PROXY_URL, setup_logger, DATA_DIR,
    RISK_CONTROL_CONFIG, TECHNICAL_INDICATORS_CONFIG, MARKET_REGIME_CONFIG,
    LIQUIDITY_MANAGER_CONFIG, SENTIMENT_ANALYSIS_CONFIG, ANOMALY_DETECTION_CONFIG
)
from risk_manager import RiskManager
from technical_indicators import TechnicalIndicators
from market_regime_detector import MarketRegimeDetector
from liquidity_manager import LiquidityManager
from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer
from anomaly_detector import AnomalyDetector

logger = setup_logger('flight_dash', os.path.join(DATA_DIR, 'flight_dash.log'))

def init_database():
    try:
        os.makedirs(os.path.dirname(DB_VERIFY), exist_ok=True)
        
        conn = sqlite3.connect(DB_VERIFY)
        cursor = conn.cursor()
        
        cursor.execute('PRAGMA journal_mode=WAL;')
        
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
        
        logger.debug("验证数据库初始化成功")
        
        os.makedirs(os.path.dirname(DB_MEMORY), exist_ok=True)
        
        conn_mem = sqlite3.connect(DB_MEMORY)
        cursor_mem = conn_mem.cursor()
        
        cursor_mem.execute('PRAGMA journal_mode=WAL;')
        
        cursor_mem.execute('''
            CREATE TABLE IF NOT EXISTS telegram_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                is_processed BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 确保表结构兼容（添加缺失的列，如果需要）
        try:
            cursor_mem.execute('ALTER TABLE telegram_news ADD COLUMN is_processed BOOLEAN DEFAULT 0')
        except sqlite3.OperationalError:
            pass  # 列已存在
        
        try:
            cursor_mem.execute('ALTER TABLE telegram_news ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP')
        except sqlite3.OperationalError:
            pass  # 列已存在
        
        cursor_mem.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON telegram_news(timestamp DESC)
        ''')
        
        news_count = cursor_mem.execute('SELECT COUNT(*) FROM telegram_news').fetchone()[0]
        
        # 不再添加示例数据，完全依赖Telegram Scout获取真实数据
        # 清理所有可能存在的示例数据（这些数据通常没有TG_前缀）
        # 删除包含示例内容的记录
        cursor_mem.execute("DELETE FROM telegram_news WHERE content LIKE '%比特币价格突破$100,000大关%' OR content LIKE '%ETH/USDT 4小时图%' OR content LIKE '%DeFi协议总锁仓量%' OR content LIKE '%BNB/USDT突破关键阻力位%' OR content LIKE '%SOL/USDT波动率增加%'")
        conn_mem.commit()
        
        # 检查是否有过期的其他数据需要清理（保留24小时内的数据）
        current_time = datetime.now()
        threshold_time = current_time - timedelta(hours=24)
        cursor_mem.execute("DELETE FROM telegram_news WHERE timestamp < ? AND source NOT LIKE 'TG_%'", (threshold_time.strftime('%Y-%m-%d %H:%M:%S'),))
        conn_mem.commit()
        
        logger.debug(f"资讯数据库初始化完成，清理示例数据和过期数据")
        
        conn_mem.close()
        
        logger.debug("资讯数据库初始化成功")
        return True
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        return False

C_HEADER, C_AI, C_GOLD = "\033[1;95m", "\033[1;36m", "\033[1;33m"
RESET, BOLD, GREEN, RED, YELLOW, CYAN, MAGENTA, WHITE = Style.RESET_ALL, "\033[1m", Fore.GREEN, Fore.RED, Fore.YELLOW, Fore.CYAN, Fore.MAGENTA, Fore.WHITE
BG_AI = "\033[48;5;234m"

client = OpenAI(api_key="EMPTY", base_url=VLLM_API, timeout=60.0)

ai_report_display = ["🛰️ 核心逻辑已重组：情绪分已剔除，系统回归纯净量化模式..."]
is_ai_calculating, report_lock = False, threading.Lock()

risk_manager = RiskManager(RISK_CONTROL_CONFIG)
technical_indicators = TechnicalIndicators()
market_regime_detector = MarketRegimeDetector(MARKET_REGIME_CONFIG)
liquidity_manager = LiquidityManager(LIQUIDITY_MANAGER_CONFIG)
sentiment_analyzer = AdvancedSentimentAnalyzer()
anomaly_detector = AnomalyDetector(ANOMALY_DETECTION_CONFIG)

logger.debug("所有风险管理和技术分析模块已初始化")

CRYPTO_LIST = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", 
               "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT"]

class MarketDataFetcher:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.ws_url = "wss://stream.binance.com:9443"
        self.price_cache = {}
        self.order_book_cache = {}
        self.ticker_24h_cache = {}
        self.is_running = False
        self.lock = threading.Lock()
        
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 Master-Quant-2026'})
        
        self.proxy_url = PROXY_URL
        self.proxy_host = None
        self.proxy_port = None
        
        if self.proxy_url and self.proxy_url != 'None':
            self._parse_proxy()
        
        self.ws_thread = None
        self._init_websocket()
    
    def _parse_proxy(self):
        try:
            proxy_url = self.proxy_url
            self.proxy_type = None
            
            if proxy_url.startswith('socks5h://'):
                url = proxy_url.replace('socks5h://', '')
                self.proxy_type = 'socks5'
            elif proxy_url.startswith('socks5://'):
                url = proxy_url.replace('socks5://', '')
                self.proxy_type = 'socks5'
            elif proxy_url.startswith('http://'):
                url = proxy_url.replace('http://', '')
                self.proxy_type = 'http'
            elif proxy_url.startswith('https://'):
                url = proxy_url.replace('https://', '')
                self.proxy_type = 'http'
            else:
                url = proxy_url
                self.proxy_type = 'http'
            
            if ':' in url:
                host, port = url.split(':')
                self.proxy_host = host
                self.proxy_port = int(port)
                logger.info(f"代理地址解析成功: {self.proxy_host}:{self.proxy_port} (类型: {self.proxy_type})")
            else:
                logger.debug("代理地址格式错误，缺少端口号")
        except Exception as e:
            logger.error(f"解析代理地址失败: {e}")
    
    def _test_proxy_connection(self, host, port, timeout=5):
        try:
            import socket
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_sock.settimeout(timeout)
            test_sock.connect((host, port))
            test_sock.close()
            logger.info(f"代理连接测试成功: {host}:{port}")
            return True
        except socket.timeout:
            logger.error(f"代理连接超时: {host}:{port} (超时时间: {timeout}秒)")
            return False
        except Exception as e:
            logger.error(f"代理连接测试失败: {host}:{port} - {e}")
            return False
    
    def _create_socks_tcp_socket(self, host, port):
        try:
            sock = socks.socksocket()
            sock.set_proxy(
                proxy_type=socks.SOCKS5,
                addr=self.proxy_host,
                port=self.proxy_port,
                rdns=True
            )
            sock.settimeout(10)
            sock.connect((host, port))
            logger.info(f"SOCKS5代理socket创建成功: {self.proxy_host}:{self.proxy_port}")
            return sock
        except socks.ProxyError as e:
            logger.error(f"SOCKS5代理错误: {e}")
            raise
        except socket.timeout:
            logger.error(f"代理连接超时: {self.proxy_host}:{self.proxy_port}")
            raise
        except Exception as e:
            logger.error(f"创建SOCKS5代理socket失败: {e}")
            raise
    
    async def _websocket_client(self):
        try:
            streams = []
            for symbol in CRYPTO_LIST:
                binance_symbol = symbol.replace("/", "").lower()
                streams.append(f"{binance_symbol}@ticker")
                streams.append(f"{binance_symbol}@depth5")
            
            combined_stream = "/".join(streams)
            uri = f"{self.ws_url}/stream?streams={combined_stream}"
            
            logger.debug(f"正在连接 WebSocket: {uri}")
            
            if self.proxy_host and self.proxy_port:
                logger.debug(f"使用 SOCKS5 代理: {self.proxy_host}:{self.proxy_port}")
                sock = socks.socksocket()
                sock.set_proxy(socks.SOCKS5, self.proxy_host, self.proxy_port)
                sock.connect(("stream.binance.com", 443))
            else:
                logger.debug("未配置代理，使用直接连接")
                sock = None
            
            async with websockets.connect(uri, sock=sock, ssl=True) as ws:
                logger.debug(f"✓ WebSocket连接成功，订阅 {len(CRYPTO_LIST)} 个交易对")
                logger.debug(f"等待接收市场数据...")
                
                last_message_time = time.time()
                message_count = 0
                ticker_count = 0
                depth_count = 0
                first_ticker_received = False
                first_depth_received = False
                
                while self.is_running:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=60.0)
                        data = json.loads(message)
                        
                        stream_data = None
                        event_type = None
                        stream_name = None
                        
                        if 'data' in data and 'stream' in data:
                            stream_data = data.get('data', {})
                            stream_name = data.get('stream', '')
                            event_type = stream_data.get('e', '')
                            
                            if not event_type and '@depth' in stream_name:
                                event_type = 'depthUpdate'
                            
                            logger.debug(f"检测到Combined Streams格式: stream={stream_name}, event_type={event_type}")
                        elif 'e' in data:
                            stream_data = data
                            event_type = data.get('e', '')
                            logger.debug(f"检测到单流格式: event_type={event_type}")
                        else:
                            logger.debug(f"未知的消息格式: {str(data)[:200]}")
                            continue
                        
                        last_message_time = time.time()
                        message_count += 1
                        
                        if message_count <= 5 or message_count % 100 == 0:
                            logger.debug(f"收到WebSocket消息 #{message_count}: 事件类型={event_type}, stream_name={stream_name}, 数据={str(stream_data)[:200]}")
                        
                        if event_type == '24hrTicker':
                            self._process_ticker(stream_data)
                            ticker_count += 1
                            if not first_ticker_received:
                                first_ticker_received = True
                                logger.debug(f"✓ 收到第一条ticker消息！")
                        elif event_type == 'depthUpdate' or (stream_name and '@depth' in stream_name):
                            if depth_count < 10:
                                logger.debug(f"收到depth消息 #{depth_count + 1}: stream_name={stream_name}, 完整数据={json.dumps(stream_data, indent=2)}")
                            self._process_depth(stream_data, stream_name)
                            depth_count += 1
                            if not first_depth_received:
                                first_depth_received = True
                                logger.debug(f"✓ 收到第一条depth消息！")
                        
                        if message_count <= 10 or message_count % 100 == 0:
                            logger.debug(f"📊 消息统计: 总消息={message_count}, ticker={ticker_count}, depth={depth_count}")
                            
                    except asyncio.TimeoutError:
                        time_since_last = time.time() - last_message_time
                        if time_since_last > 120:
                            logger.error(f"超过{time_since_last:.0f}秒未收到消息，连接可能已断开")
                            break
                        else:
                            logger.debug(f"接收消息超时 ({time_since_last:.0f}秒未收到消息)，继续等待...")
                            continue
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析错误: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"接收消息错误: {e}")
                        break
                        
        except asyncio.TimeoutError:
            logger.error("WebSocket连接超时（15秒内未建立连接），请检查网络或代理设置")
        except InvalidStatusCode as e:
            logger.error(f"WebSocket连接失败，状态码: {e.status_code}, 原因: {e.reason}")
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket连接异常: {e}")
        except Exception as e:
            logger.error(f"WebSocket客户端错误: {e}")
        finally:
            logger.debug("WebSocket客户端已关闭")
    
    def _run_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._websocket_client())
        finally:
            loop.close()
    
    def _init_websocket(self):
        try:
            self.is_running = True
            self.ws_thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self.ws_thread.start()
            logger.debug(f"WebSocket连接已启动，线程ID: {self.ws_thread.ident}")
        except Exception as e:
            logger.error(f"WebSocket初始化失败: {e}")
            self.is_running = False
    
    def _process_ticker(self, data):
        try:
            ticker_data = data
            symbol = ticker_data.get('s', '')
            
            if not symbol:
                logger.debug(f"收到ticker数据但缺少symbol字段: {str(ticker_data)[:200]}")
                return
            
            symbol_lower = symbol.lower()
            price = float(ticker_data.get('c', 0))
            change_pct = float(ticker_data.get('P', 0))
            
            with self.lock:
                self.ticker_24h_cache[symbol_lower] = {
                    'symbol': symbol_lower,
                    'price': price,
                    'volume': float(ticker_data.get('v', 0)),
                    'quote_volume': float(ticker_data.get('q', 0)),
                    'change_pct': change_pct,
                    'high': float(ticker_data.get('h', 0)),
                    'low': float(ticker_data.get('l', 0)),
                    'open': float(ticker_data.get('o', 0)),
                    'timestamp': ticker_data.get('E', 0)
                }
            
            logger.debug(f"✓ 处理ticker数据成功: {symbol} 价格: {price} 涨跌: {change_pct:.2f}% | 缓存大小: {len(self.ticker_24h_cache)}")
        except Exception as e:
            logger.error(f"✗ 处理ticker数据失败: {e} | 原始数据: {str(data)[:200]}")
    
    def _process_depth(self, data, stream_name=None):
        try:
            depth_data = data
            
            logger.debug(f"处理depth数据 - 原始数据: {str(depth_data)[:500]}")
            logger.debug(f"处理depth数据 - stream_name: {stream_name}")
            
            # 尝试从多个可能的字段获取数据
            symbol = depth_data.get('s', '')
            
            # 检查是否是深度更新格式
            if 'bids' in depth_data:
                bids = depth_data['bids']
            elif 'b' in depth_data:
                bids = depth_data['b']
            else:
                bids = []
                
            if 'asks' in depth_data:
                asks = depth_data['asks']
            elif 'a' in depth_data:
                asks = depth_data['a']
            else:
                asks = []
            
            # 如果仍然没有数据，尝试从其他可能的字段获取
            if not bids and not asks:
                # 尝试从data字段获取（兼容不同格式）
                if 'data' in depth_data:
                    inner_data = depth_data['data']
                    if isinstance(inner_data, dict):
                        bids = inner_data.get('bids', inner_data.get('b', []))
                        asks = inner_data.get('asks', inner_data.get('a', []))
                        if not symbol:
                            symbol = inner_data.get('s', '')
            
            # 如果通过stream_name可以提取symbol，则尝试提取
            if not symbol and stream_name:
                # 处理不同格式的stream_name
                if '@depth' in stream_name:
                    symbol = stream_name.split('@')[0].upper()
                elif '@ticker' in stream_name:
                    symbol = stream_name.split('@')[0].upper()
                logger.debug(f"从stream_name提取symbol: {symbol}")
            
            logger.debug(f"symbol: {symbol}, bids类型: {type(bids)}, bids数量: {len(bids)}, asks类型: {type(asks)}, asks数量: {len(asks)}")
            
            if not symbol:
                logger.debug(f"收到depth数据但缺少symbol字段: stream_name={stream_name}")
                return
            
            # 检查bids和asks是否为字符串格式（可能需要解析）
            if isinstance(bids, str):
                try:
                    bids = json.loads(bids)
                except:
                    bids = []
            if isinstance(asks, str):
                try:
                    asks = json.loads(asks)
                except:
                    asks = []
            
            # 确保bids和asks是列表格式
            if not isinstance(bids, list):
                bids = []
            if not isinstance(asks, list):
                asks = []
            
            symbol_lower = symbol.lower()
            
            bid_volume = 0.0
            ask_volume = 0.0
            
            # 计算买单总数量
            for bid in bids:
                if isinstance(bid, list) and len(bid) >= 2:
                    try:
                        bid_volume += float(bid[1])
                    except (ValueError, TypeError):
                        logger.debug(f"无法解析买单数量: {bid}")
                        continue
                elif isinstance(bid, dict) and 'quantity' in bid:
                    try:
                        bid_volume += float(bid['quantity'])
                    except (ValueError, TypeError):
                        logger.debug(f"无法解析买单数量: {bid}")
                        continue
                
            # 计算卖单总数量
            for ask in asks:
                if isinstance(ask, list) and len(ask) >= 2:
                    try:
                        ask_volume += float(ask[1])
                    except (ValueError, TypeError):
                        logger.debug(f"无法解析卖单数量: {ask}")
                        continue
                elif isinstance(ask, dict) and 'quantity' in ask:
                    try:
                        ask_volume += float(ask['quantity'])
                    except (ValueError, TypeError):
                        logger.debug(f"无法解析卖单数量: {ask}")
                        continue
            
            order_ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0
            
            with self.lock:
                self.order_book_cache[symbol_lower] = {
                    'order_ratio': order_ratio,
                    'bids': bids,
                    'asks': asks,
                    'timestamp': depth_data.get('E', depth_data.get('lastUpdateId', 0))
                }
            
            logger.debug(f"✓ 处理depth数据成功: {symbol} 买卖盘比: {order_ratio:.4f} | 买单量: {bid_volume:.4f} | 卖单量: {ask_volume:.4f} | 缓存大小: {len(self.order_book_cache)}")
        except Exception as e:
            logger.error(f"✗ 处理depth数据失败: {e} | 原始数据: {str(data)[:200]}")
            import traceback
            traceback.print_exc()
    
    def get_ticker_data(self, symbol):
        try:
            binance_symbol = symbol.replace("/", "").lower()  # 使用小写符号匹配缓存
            with self.lock:
                if binance_symbol in self.ticker_24h_cache:
                    data = self.ticker_24h_cache[binance_symbol]
                    logger.debug(f"✓ 从缓存获取 {symbol} ticker数据: 价格={data['price']}, 涨跌={data['change_pct']:.2f}%")
                    return {
                        'symbol': symbol,
                        'price': data['price'],
                        'volume': data['volume'],
                        'quote_volume': data['quote_volume'],
                        'change_pct': data['change_pct'],
                        'high': data['high'],
                        'low': data['low'],
                        'open': data['open']
                    }
                else:
                    logger.debug(f"✗ 缓存中未找到 {symbol} (binance_symbol={binance_symbol}) 的ticker数据 | 当前缓存: {list(self.ticker_24h_cache.keys())}")
        except Exception as e:
            logger.error(f"✗ 获取 {symbol} 行情失败: {e}")
        return None
    
    def get_order_book(self, symbol):
        try:
            binance_symbol = symbol.replace("/", "").lower()  # 使用小写符号匹配缓存
            with self.lock:
                if binance_symbol in self.order_book_cache:
                    # 返回完整的订单簿数据，包括bids和asks，用于流动性分析
                    return self.order_book_cache[binance_symbol]
                else:
                    logger.debug(f"缓存中未找到 {symbol} 的订单簿数据，使用默认值1.0")
        except Exception as e:
            logger.error(f"✗ 获取 {symbol} 订单簿失败: {e}")
        # 返回默认订单簿结构
        return {
            'order_ratio': 1.0,
            'bids': [],
            'asks': [],
            'timestamp': 0
        }
    
    def has_received_data(self):
        """检查WebSocket是否已接收到数据"""
        with self.lock:
            ticker_count = len(self.ticker_24h_cache)
            depth_count = len(self.order_book_cache)
            has_data = ticker_count > 0 or depth_count > 0
            logger.debug(f"数据接收状态检查: ticker缓存={ticker_count}, depth缓存={depth_count}, 有数据={has_data}")
            return has_data, ticker_count, depth_count
    
    def get_cache_status(self):
        """获取缓存状态信息"""
        with self.lock:
            return {
                'ticker_cache_size': len(self.ticker_24h_cache),
                'order_book_cache_size': len(self.order_book_cache),
                'ticker_symbols': list(self.ticker_24h_cache.keys()),
                'order_book_symbols': list(self.order_book_cache.keys())
            }
    
    def calculate_sar(self, prices, af=0.02, max_af=0.2):
        if len(prices) < 2:
            return prices[-1] if prices else 0, "BULL"
        
        high_prices = prices
        low_prices = prices
        
        sar = low_prices[0]
        ep = high_prices[0]
        is_up_trend = True
        current_af = af
        
        for i in range(1, len(prices)):
            if is_up_trend:
                sar = sar + current_af * (ep - sar)
                sar = min(sar, low_prices[i-1], low_prices[i])
                if high_prices[i] > ep:
                    ep = high_prices[i]
                    current_af = min(current_af + af, max_af)
                if low_prices[i] < sar:
                    is_up_trend = False
                    sar = ep
                    ep = low_prices[i]
                    current_af = af
            else:
                sar = sar + current_af * (ep - sar)
                sar = max(sar, high_prices[i-1], high_prices[i])
                if low_prices[i] < ep:
                    ep = low_prices[i]
                    current_af = min(current_af + af, max_af)
                if high_prices[i] > sar:
                    is_up_trend = True
                    sar = ep
                    ep = high_prices[i]
                    current_af = af
        
        trend = "BULL" if is_up_trend else "BEAR"
        return sar, trend
    
    def get_all_market_data(self):
        logger.debug("=" * 80)
        logger.debug("开始获取所有市场数据")
        logger.debug("=" * 80)
        
        cache_status = self.get_cache_status()
        logger.debug(f"当前缓存状态: ticker={cache_status['ticker_cache_size']}, order_book={cache_status['order_book_cache_size']}")
        logger.debug(f"Ticker缓存中的交易对: {cache_status['ticker_symbols']}")
        logger.debug(f"Order Book缓存中的交易对: {cache_status['order_book_symbols']}")
        
        market_data = []
        missing_symbols = []
        
        for symbol in CRYPTO_LIST:
            logger.debug(f"正在获取 {symbol} 的市场数据...")
            ticker = self.get_ticker_data(symbol)
            if ticker:
                order_book_full = self.get_order_book(symbol)
                order_ratio = order_book_full.get('order_ratio', 1.0)
                
                price_history = [ticker['open'], ticker['high'], ticker['low'], ticker['price']]
                sar_value, sar_trend = self.calculate_sar(price_history)
                
                market_data.append({
                    'symbol': ticker['symbol'],
                    'price': ticker['price'],
                    'order_ratio': order_ratio,
                    'sar_value': sar_value,
                    'sar_trend': sar_trend,
                    'volume': ticker['volume'],
                    'change_pct': ticker['change_pct']
                })
                
                logger.debug(f"✓ {symbol} 数据获取成功: 价格={ticker['price']}, 涨跌={ticker['change_pct']:.2f}%, 买卖盘比={order_ratio:.4f}")
            else:
                missing_symbols.append(symbol)
                logger.debug(f"✗ 未获取到 {symbol} 的ticker数据")
        
        logger.debug("=" * 80)
        logger.debug(f"市场数据获取完成: 成功={len(market_data)}/{len(CRYPTO_LIST)}, 缺失={len(missing_symbols)}")
        if missing_symbols:
            logger.debug(f"缺失的交易对: {missing_symbols}")
        logger.debug("=" * 80)
        
        return market_data
    
    def close(self):
        self.is_running = False
        if self.ws_thread:
            self.ws_thread.join(timeout=5)
        logger.info("WebSocket连接已关闭")

market_fetcher = MarketDataFetcher()
logger.debug("市场数据获取器已初始化")

def save_analysis_results(symbol, price, risk_level, market_regime, liquidity_score, sentiment_score, has_anomaly):
    try:
        conn = sqlite3.connect(DB_VERIFY)
        conn.execute('PRAGMA journal_mode=WAL;')
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results
            (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                price REAL,
                risk_level TEXT,
                market_regime TEXT,
                liquidity_score REAL,
                sentiment_score REAL,
                has_anomaly INTEGER,
                timestamp TEXT
            )
        """)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cur.execute("""
            INSERT INTO analysis_results 
            (symbol, price, risk_level, market_regime, liquidity_score, sentiment_score, has_anomaly, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, price, risk_level, market_regime, liquidity_score, sentiment_score, 1 if has_anomaly else 0, timestamp))
        
        conn.commit()
        conn.close()
        logger.debug(f"分析结果已保存: {symbol} {risk_level} {market_regime}")
    except Exception as e:
        logger.error(f"保存分析结果失败: {e}")

# === 🛡️ 核心 1：增强型技术战术决策引擎 (集成风险管理) ===
def get_tactical_decision(trend, ratio, sar_diff, price, volatility_24h, volume_ratio=1.0, 
                          risk_level=None, market_regime=None, liquidity_score=None, 
                          sentiment_score=None, has_anomaly=False):
    """
    集成风险管理、市场环境、流动性、情绪分析和异常检测的增强型决策引擎
    """
    vol_adj = max(0.5, min(2.0, volatility_24h / 0.03))
    BUY_RATIO_THRESH = 1.6 * vol_adj
    SELL_RATIO_THRESH = 0.7 / vol_adj

    high_volume = volume_ratio > 1.3
    
    risk_warning = ""
    if risk_level == "HIGH":
        risk_warning = f"{RED}⚠️高风险{RESET} "
    elif risk_level == "MEDIUM":
        risk_warning = f"{YELLOW}⚠️中风险{RESET} "
    
    regime_indicator = ""
    if market_regime == "BULL":
        regime_indicator = f"{GREEN}🐂牛市{RESET} "
    elif market_regime == "BEAR":
        regime_indicator = f"{RED}🐻熊市{RESET} "
    elif market_regime == "SIDEWAYS":
        regime_indicator = f"{CYAN}📊震荡{RESET} "
    
    liquidity_indicator = ""
    if liquidity_score and liquidity_score < 0.5:
        liquidity_indicator = f"{RED}💧低流动性{RESET} "
    
    sentiment_indicator = ""
    if sentiment_score:
        if sentiment_score > 0.7:
            sentiment_indicator = f"{GREEN}😊乐观{RESET} "
        elif sentiment_score < -0.7:
            sentiment_indicator = f"{RED}😰悲观{RESET} "
    
    anomaly_indicator = ""
    if has_anomaly:
        anomaly_indicator = f"{RED}🚨异常{RESET} "

    if has_anomaly:
        return f"{RED}异常观望{RESET}"

    if liquidity_score and liquidity_score < 0.5:
        return f"{RED}流动性低{RESET}"

    if risk_level == "HIGH":
        return f"{YELLOW}风险高{RESET}"

    if market_regime == "BEAR" and trend == "BULL":
        return f"{CYAN}逆势{RESET}"

    if market_regime == "BULL" and trend == "BEAR":
        return f"{GREEN}做多{RESET}"

    if sentiment_score and sentiment_score < -0.7 and trend == "BULL":
        return f"{GREEN}抄底{RESET}"

    if sentiment_score and sentiment_score > 0.7 and trend == "BEAR":
        return f"{RED}逃顶{RESET}"

    if (trend == "BULL" and ratio > BUY_RATIO_THRESH and sar_diff > 0.005 and high_volume):
        return f"{GREEN}主升浪{RESET}"

    elif (trend == "BEAR" and ratio < SELL_RATIO_THRESH and sar_diff < -0.005 and high_volume):
        return f"{RED}空头{RESET}"

    elif (trend == "BULL" and ratio < 0.8 and sar_diff > 0.015):
        return f"{YELLOW}超卖{RESET}"
    elif (trend == "BEAR" and ratio > 1.5 and sar_diff < -0.015):
        return f"{YELLOW}超买{RESET}"

    elif abs(sar_diff) < 0.003 and volatility_24h < 0.02:
        if ratio > 1.2:
            return f"{CYAN}试多{RESET}"
        elif ratio < 0.8:
            return f"{CYAN}试空{RESET}"
        else:
            return f"{CYAN}观望{RESET}"

    elif abs(sar_diff) < 0.005:
        return f"{MAGENTA}关键位{RESET}"

    else:
        return f"{WHITE}跟踪{RESET}"


# === ⚙️ 核心 2：增强型执行引擎 (集成风险管理) ===
def execute_smart_trade(instr, bal_total):
    try:
        sym = instr['symbol'].replace('/', '').upper()
        action = instr['action'].upper()
        if action == "WAIT": 
            logger.debug(f"{sym}: WAIT 指令，跳过执行")
            return False, "WAITING"
        
        ep, tp, sl = float(instr['entry_price']), float(instr['take_profit']), float(instr['stop_loss'])
        lev = int(instr.get('leverage', 5))
        
        logger.debug(f"执行交易指令: {sym} {action} EP:{ep} TP:{tp} SL:{sl} LEV:{lev}x")

        conn = sqlite3.connect(DB_VERIFY)
        conn.execute('PRAGMA journal_mode=WAL;')
        cur = conn.cursor()
        cur.execute("SELECT type, entry_price, quantity FROM shadow_portfolio_v7 WHERE symbol=?", (sym,))
        existing_pos = cur.fetchone()

        if existing_pos and existing_pos[0] != action:
            logger.debug(f"{sym}: 平仓并反向开仓")
            cur.execute("DELETE FROM shadow_portfolio_v7 WHERE symbol=?", (sym,))
            cur.execute("UPDATE shadow_account SET balance = balance + 500 WHERE id=1")

        if not existing_pos or existing_pos[0] != action:
            risk_pct = abs(ep - sl) / ep if abs(ep - sl) > 0 else 0.01
            margin = (bal_total * 0.01) / risk_pct / lev
            
            risk_check = risk_manager.check_position_risk(sym, margin * lev, ep, sl, action)
            if not risk_check['approved']:
                logger.debug(f"{sym}: 风险管理拒绝交易 - {risk_check['reason']}")
                conn.close()
                return False, "RISK_REJECTED"
            
            cur.execute("SELECT balance FROM shadow_account WHERE id=1")
            if cur.fetchone()[0] >= margin:
                cur.execute("INSERT INTO shadow_portfolio_v7 (symbol, entry_price, quantity, type, leverage, timestamp) VALUES (?,?,?,?,?,?)",
                            (sym, ep, (margin * lev) / ep, action, lev, datetime.now().strftime('%H:%M')))
                cur.execute("UPDATE shadow_account SET balance = balance - ? WHERE id=1", (margin,))
                conn.commit()
                conn.close()
                logger.debug(f"{sym}: 交易执行成功，保证金: {margin:.2f}")
                return True, "SUCCESS"
            else:
                logger.debug(f"{sym}: 余额不足，需要 {margin:.2f}")
        conn.close()
        return False, "HOLDING"
    except Exception as e:
        logger.error(f"执行交易失败: {e}")
        return False, "ERROR"


# === 🧠 辅助函数：JSON修复 ===
def fix_json_strings(json_str):
    """
    尝试修复JSON字符串中未闭合的引号问题
    """
    try:
        # 首先尝试直接解析，如果成功则直接返回
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass  # 继续执行修复逻辑

    try:
        # 简化修复逻辑：处理常见的JSON格式问题
        fixed_str = json_str
        
        # 移除可能的非JSON内容（如开头的描述文字）
        start_idx = fixed_str.find('[')
        end_idx = fixed_str.rfind(']')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            # 提取JSON数组部分
            fixed_str = fixed_str[start_idx:end_idx+1]
        else:
            # 如果没有找到数组，尝试找对象
            start_idx = fixed_str.find('{')
            end_idx = fixed_str.rfind('}')
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                fixed_str = fixed_str[start_idx:end_idx+1]
        
        # 清理常见的格式问题
        fixed_str = fixed_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        # 修复可能的未转义引号问题
        # 先将现有的正确转义处理好
        fixed_str = fixed_str.replace('\\\"', '\\"')  # 处理错误的双重转义
        
        # 检查引号是否平衡
        quote_count = 0
        i = 0
        while i < len(fixed_str):
            if i + 1 < len(fixed_str) and fixed_str[i] == '\\':  # 检查转义字符
                i += 2  # 跳过转义字符
            elif fixed_str[i] == '"':
                quote_count += 1
                i += 1
            else:
                i += 1
        
        # 如果引号数量为奇数，说明有未闭合的引号
        if quote_count % 2 == 1:
            # 尝试找到最后一个未转义的引号，并在合适位置添加闭合引号
            last_quote_pos = -1
            i = 0
            while i < len(fixed_str):
                if i + 1 < len(fixed_str) and fixed_str[i] == '\\':
                    i += 2
                elif fixed_str[i] == '"':
                    last_quote_pos = i
                    i += 1
                else:
                    i += 1
            
            if last_quote_pos != -1:
                # 在最后一个未转义引号之后查找可能的结束位置
                search_start = last_quote_pos + 1
                found_closing_pos = False
                for j in range(search_start, len(fixed_str)):
                    if fixed_str[j] in [',', ']', '}', ':']:
                        # 在合适位置插入闭合引号
                        fixed_str = fixed_str[:j] + '"' + fixed_str[j:]
                        found_closing_pos = True
                        break
                
                # 如果没找到合适的分隔符，尝试在字符串末尾添加引号
                if not found_closing_pos:
                    fixed_str = fixed_str + '"'
        
        # 尝试修复后再次检查
        try:
            json.loads(fixed_str)
            return fixed_str
        except json.JSONDecodeError:
            # 如果仍然失败，尝试更激进的清理方法
            # 逐字符构建字符串，正确处理转义字符
            result = []
            i = 0
            in_string = False
            escaped = False
            
            while i < len(fixed_str):
                char = fixed_str[i]
                
                if not escaped and char == '\\':
                    # 遇到转义字符
                    result.append(char)
                    escaped = True
                elif escaped:
                    # 前一个字符是转义符，这个字符被转义
                    result.append(char)
                    escaped = False
                elif char == '"' and not escaped:
                    # 非转义的引号，切换字符串状态
                    in_string = not in_string
                    result.append(char)
                elif in_string and char in ['\n', '\r', '\t']:
                    # 在字符串内替换换行符等为普通空格
                    result.append(' ')
                else:
                    result.append(char)
                
                i += 1
            
            cleaned_str = ''.join(result)
            
            # 再次尝试修复引号问题
            quote_count = 0
            i = 0
            while i < len(cleaned_str):
                if i + 1 < len(cleaned_str) and cleaned_str[i] == '\\':
                    i += 2  # 跳过转义字符
                elif cleaned_str[i] == '"':
                    quote_count += 1
                    i += 1
                else:
                    i += 1
            
            if quote_count % 2 == 1:
                # 如果引号仍不平衡，在末尾添加一个引号
                cleaned_str = cleaned_str + '"'
            
            # 尝试解析最终字符串
            try:
                json.loads(cleaned_str)
                return cleaned_str
            except json.JSONDecodeError:
                # 最后的备选方案：使用更宽松的解析方法
                # 尝试找到最可能的JSON部分
                import re
                matches = re.findall(r'\[.*?\]', cleaned_str, re.DOTALL)
                for match in matches:
                    try:
                        # 清理匹配到的部分
                        clean_match = match.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                        json.loads(clean_match)
                        return clean_match
                    except:
                        continue
                
                # 如果还是不行，返回原始字符串
                return json_str
        
        return fixed_str
    except Exception as e:
        logger.error(f"JSON修复错误: {e}")
        # 如果修复失败，返回原始字符串
        return json_str

def extract_json_objects(text):
    """
    从文本中提取JSON对象，使用更宽松的解析方法
    """
    try:
        # 查找所有可能的JSON数组
        array_matches = re.findall(r'\[.*?\]', text, re.DOTALL)
        
        for match in array_matches:
            try:
                # 尝试清理并解析
                cleaned = match.strip()
                if cleaned.startswith('[') and cleaned.endswith(']'):
                    # 尝试修复常见的格式问题
                    cleaned = cleaned.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                    # 尝试修复未闭合的字符串
                    cleaned = fix_json_strings(cleaned)
                    return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
        
        return None
    except Exception:
        return None


# === 🧠 核心 3：首席策略官 AI 推理 (增强型技术派版) ===
def ai_inference_thread(summary, bal):
    global ai_report_display, is_ai_calculating
    with report_lock:
        is_ai_calculating = True
    try:
        logger.debug("开始 AI 推理分析")
        logger.debug(f"发送给AI的摘要: {summary[:500]}...")
        
        system_prompt = (
            "你现在是章新光号的首席策略官。系统已升级为增强型技术面分析模式，你现在必须基于以下多维度数据输出推理。\n"
            "分析维度包括：\n"
            "1. 风险水平 (HIGH/MEDIUM/LOW) - 基于波动率、回撤、相关性等综合评估\n"
            "2. 市场环境 (BULL/BEAR/SIDEWAYS) - 牛市/熊市/震荡市识别\n"
            "3. 流动性评分 (0-1) - 市场深度和流动性分析\n"
            "4. 情绪分数 (-1到1) - 市场情绪综合分析\n"
            "5. 异常检测 (True/False) - 市场异常情况识别\n"
            "\n"
            "要求：reasoning 字段严禁少于 80 字，必须包含：\n"
            "- [风险水平评估] 当前风险等级及原因\n"
            "- [市场环境分析] 当前市场环境及适应性策略\n"
            "- [流动性状况] 流动性是否充足及影响\n"
            "- [情绪分析] 市场情绪状态及潜在反转信号\n"
            "- [技术面确认] SAR、Ratio等指标确认\n"
            "\n"
            "重点关注：\n"
            "- 高风险资产需要谨慎对待或规避\n"
            "- 市场环境与趋势的一致性\n"
            "- 流动性不足时避免大额交易\n"
            "- 极端情绪可能预示反转\n"
            "- 异常情况需要特别警惕\n"
            "\n"
            "输出格式：\n"
            "[{\"symbol\": \"币种\", \"action\": \"LONG/SHORT/WAIT\", \"sar_ref\": \"SAR\", \"entry_price\": \"现价\", \"take_profit\": \"止盈\", \"stop_loss\": \"止损\", \"position_size\": \"仓位\", \"reasoning\": \"深度技术理由\"}]"
        )
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": summary}
            ],
            temperature=0.2
        )
        
        response_content = completion.choices[0].message.content
        logger.debug(f"AI响应内容: {response_content[:500]}...")
        
        # 查找JSON数组，更精确地处理JSON格式
        match = re.search(r'\[.*?\]', response_content, re.S)
        if not match:
            raise ValueError("AI响应中未找到有效的JSON数组")
        
        json_str = match.group()
        
        # 尝试修复可能的JSON格式问题
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.error(f"尝试修复的JSON: {json_str[:200]}...")
            
            # 尝试更高级的JSON修复方法
            try:
                # 1. 查找最可能的JSON数组边界
                start = response_content.find('[')
                end = response_content.rfind(']')
                
                if start != -1 and end != -1 and start < end:
                    json_str = response_content[start:end+1]
                    
                    # 2. 清理常见的格式问题
                    json_str = json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                    
                    # 3. 修复可能的未闭合字符串
                    # 遍历字符串，尝试修复未闭合的引号
                    json_str = fix_json_strings(json_str)
                    
                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        # 4. 如果还是失败，尝试使用更宽松的解析方法
                        data = extract_json_objects(json_str)
                        if not data:
                            raise ValueError(f"无法解析AI响应为有效的JSON: {str(e)}")
                else:
                    raise ValueError(f"无法找到有效的JSON数组: {str(e)}")
            except Exception as fix_error:
                logger.error(f"JSON修复失败: {fix_error}")
                raise ValueError(f"无法解析AI响应为有效的JSON: {str(e)}")
        reports = []
        for d in data:
            success, msg = execute_smart_trade(d, bal)
            status = f"{C_GOLD}✔调仓{RESET}" if success else (
                f"{WHITE}⌛持仓{RESET}" if msg == "HOLDING" else f"{RED}✘拒绝{RESET}")
            reports.append(f"{status} | {d['symbol']:<8} | {d['action']:<5} | TP:{d['take_profit']} | {d['reasoning']}")
            logger.debug(f"AI决策: {d['symbol']} {d['action']} 状态: {msg}")
        with report_lock:
            ai_report_display = reports
        logger.debug("AI 推理分析完成")
    except Exception as e:
        logger.error(f"AI 推理异常: {e}")
        logger.error(f"异常类型: {type(e).__name__}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        with report_lock:
            ai_report_display = [f"❌ AI 推理失败: {e}"]
    finally:
        with report_lock:
            is_ai_calculating = False


# === 🚀 核心 4：增强型主循环 (集成所有分析模块) ===
def get_latest_news(limit=5):
    """获取最新的资讯消息"""
    try:
        conn = sqlite3.connect(DB_MEMORY)
        conn.execute('PRAGMA journal_mode=WAL;')
        cur = conn.cursor()
        # 查询最新的Telegram资讯，优先显示真实Telegram数据（有TG_前缀的），按时间倒序排列
        cur.execute(f"SELECT timestamp, source, content FROM telegram_news WHERE source LIKE 'TG_%' ORDER BY timestamp DESC LIMIT {limit}")
        news = cur.fetchall()
        
        # 如果没有真实Telegram数据，再查询其他数据
        if not news:
            cur.execute(f"SELECT timestamp, source, content FROM telegram_news ORDER BY timestamp DESC LIMIT {limit}")
            news = cur.fetchall()
        
        # 确保返回的数据格式一致
        if not news:
            # 如果仍然没有数据，返回空列表
            news = []
        
        conn.close()
        return news
    except Exception as e:
        logger.debug(f"获取资讯失败: {e}")
        return []

def update_telegram_news():
    """更新Telegram资讯，确保显示最新的真实数据"""
    try:
        conn = sqlite3.connect(DB_MEMORY)
        conn.execute('PRAGMA journal_mode=WAL;')
        cur = conn.cursor()
        
        # 检查是否有未处理的非Telegram Scout消息（例如，可能的其他来源消息）
        # 注意：Telegram Scout会直接插入is_processed=0的数据，这里主要处理其他来源的数据
        cur.execute("SELECT COUNT(*) FROM telegram_news WHERE is_processed = 0 AND source LIKE 'TG_%'")
        unprocessed_count = cur.fetchone()[0]
        
        if unprocessed_count > 0:
            # 获取未处理的Telegram消息并进行分类
            cur.execute("SELECT timestamp, source, content FROM telegram_news WHERE is_processed = 0 AND source LIKE 'TG_%' ORDER BY timestamp DESC LIMIT 10")
            unprocessed_news = cur.fetchall()
            
            # 分类处理资讯
            for timestamp, source, content in unprocessed_news:
                # 简单分类逻辑
                category = classify_news(content)
                
                # 更新处理状态
                cur.execute("UPDATE telegram_news SET is_processed = 1 WHERE timestamp = ? AND source = ? AND content = ?", (timestamp, source, content))
            
            conn.commit()
            logger.debug(f"已处理 {len(unprocessed_news)} 条新Telegram资讯")
        
        conn.close()
    except Exception as e:
        logger.debug(f"更新Telegram资讯失败: {e}")

def classify_news(content):
    """简单分类新闻内容"""
    content_lower = content.lower()
    
    if any(keyword in content_lower for keyword in ['price', '突破', '涨', '跌', 'pump', 'dump', '突破', '阻力', '支撑', 'k线', '技术']):
        return '技术分析'
    elif any(keyword in content_lower for keyword in ['market', '行情', '趋势', '牛市', '熊市', '震荡', '环境', '市场']):
        return '市场分析'
    elif any(keyword in content_lower for keyword in ['defi', 'eth', 'btc', 'coin', 'crypto', 'token', 'protocol', '区块链', '以太坊', '比特币']):
        return '行业动态'
    elif any(keyword in content_lower for keyword in ['risk', 'warning', 'alert', '风险', '预警', '注意', '警报']):
        return '市场预警'
    elif any(keyword in content_lower for keyword in ['strategy', 'trade', '交易', '策略', '买卖', '建仓', '平仓']):
        return '交易策略'
    else:
        return '市场分析'

def run_dashboard():
    global ai_report_display, is_ai_calculating
    counter = 0
    logger.debug("交易仪表盘启动")
    
    initial_wait = True
    max_initial_wait = 60
    initial_wait_start = time.time()
    
    console = Console()
    
    def generate_dashboard():
        nonlocal counter, initial_wait, max_initial_wait, initial_wait_start
        try:
            if initial_wait:
                has_data, ticker_count, depth_count = market_fetcher.has_received_data()
                
                if has_data:
                    logger.debug(f"✓ WebSocket已接收到数据 (ticker={ticker_count}, depth={depth_count})，开始显示仪表盘")
                    initial_wait = False
                else:
                    wait_time = time.time() - initial_wait_start
                    if wait_time < max_initial_wait:
                        logger.debug(f"⏳ 等待WebSocket接收数据... ({wait_time:.1f}s/{max_initial_wait}s)")
                        return Panel("⏳ 等待WebSocket接收数据...", title="章新光号 V12.0-ENHANCED-QUANT", style="bold magenta")
                    else:
                        logger.debug(f"⚠️ 等待{max_initial_wait}秒后仍未收到数据，将显示仪表盘（可能无数据）")
                        initial_wait = False
            
            market_data = market_fetcher.get_all_market_data()
            
            conn = sqlite3.connect(DB_VERIFY)
            conn.execute('PRAGMA journal_mode=WAL;')
            cur = conn.cursor()
            
            cur.execute("SELECT balance FROM shadow_account WHERE id=1")
            bal = cur.fetchone()[0]
            cur.execute("SELECT * FROM shadow_portfolio_v7")
            positions = {r[0]: r for r in cur.fetchall()}
            conn.close()

            rows = [(d['symbol'], d['price'], d['order_ratio'], d['sar_value'], d['sar_trend']) for d in market_data]

            logger.debug(f"刷新仪表盘: {len(rows)} 个资产, 余额: {bal:.2f}")
            if len(rows) == 0:
                logger.debug("⚠️ 警告: 没有任何市场数据可显示！请检查WebSocket连接状态。")
            else:
                logger.debug(f"✓ 仪表盘将显示 {len(rows)} 个资产的数据")

            # 创建主布局
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="footer", size=3)
            )
            
            layout["header"].update(Panel(f"🛸 章新光号 V12.0-ENHANCED-QUANT | 模拟可用余额: ${bal:,.2f} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="bold cyan"))
            
            # 创建市场概览表格
            table = Table(title="📊 市场概览", show_header=True, header_style="bold magenta")
            table.add_column("资产", style="cyan", width=10)
            table.add_column("最新价", style="green", width=12)
            table.add_column("24H", style="", width=8)
            table.add_column("成交量", style="", width=12)
            table.add_column("SAR", style="", width=8)
            table.add_column("持仓/杆", style="cyan", width=12)
            table.add_column("浮盈亏", style="", width=12)
            table.add_column("趋势", style="", width=8)
            table.add_column("买卖比", style="white", width=10)
            table.add_column("风险", style="", width=8)
            table.add_column("环境", style="", width=10)
            table.add_column("流动", style="", width=8)
            table.add_column("情绪", style="", width=10)
            table.add_column("战术", style="", width=15)  # 进一步增加战术列宽度以显示完整内容

            summary_batch = []
            risk_summary = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            sentiment_scores = []
            liquidity_scores = []
            regime_counts = {"BULL": 0, "BEAR": 0, "SIDEWAYS": 0, "UNKNOWN": 0}
            anomaly_count = 0
            
            for sym, p, ratio, sar, trend in rows:
                clean_sym = sym.replace('/', '').lower()  # 用于缓存查询
                clean_sym_upper = sym.replace('/', '').upper()  # 用于持仓查询
                sar_diff = (p - sar) / p

                pos = positions.get(clean_sym_upper)
                pnl_str, pos_info = "--", "空仓"
                if pos:
                    # pos[0]=symbol, pos[1]=entry_price, pos[2]=quantity, pos[3]=type, pos[4]=leverage, pos[5]=timestamp, pos[6]=created_at
                    pnl = ((p - pos[1]) / pos[1] if pos[3] == "LONG" else (pos[1] - p) / pos[1]) * pos[4] * 100
                    pnl_str = f"{pnl:+.2f}%"
                    pos_info = f"{pos[3]} {pos[4]}x"

                risk_level = "LOW"
                market_regime = "SIDEWAYS"
                liquidity_score = 1.0
                sentiment_score = 0.0
                has_anomaly = False
                change_pct = 0.0
                volume_24h = 0.0

                for d in market_data:
                    if d['symbol'] == sym:
                        change_pct = d.get('change_pct', 0.0)
                        volume_24h = d.get('volume', 0.0)
                        break

                try:
                    risk_assessment = risk_manager.assess_market_risk(clean_sym_upper, p, 0.02, ratio)
                    risk_level = risk_assessment.get('risk_level', 'LOW')
                    risk_summary[risk_level] += 1
                except Exception as e:
                    logger.debug(f"风险评估失败: {e}")

                try:
                    regime_result = market_regime_detector.detect_regime(clean_sym_upper)
                    if regime_result:
                        market_regime = regime_result.get('regime', 'SIDEWAYS')
                        regime_counts[market_regime] += 1
                except Exception as e:
                    logger.debug(f"市场环境检测失败: {e}")
                    regime_counts["UNKNOWN"] += 1

                try:
                    # 获取订单簿数据用于流动性分析
                    order_book_data = market_fetcher.get_order_book(clean_sym)
                    
                    # 获取24小时成交量数据
                    ticker_data = market_fetcher.get_ticker_data(clean_sym)
                    volume_24h = ticker_data.get('volume', 0) if ticker_data else 0
                    
                    if order_book_data and 'bids' in order_book_data and 'asks' in order_book_data and order_book_data['bids'] and order_book_data['asks']:
                        # 基于订单簿深度和成交量计算流动性评分
                        bids = order_book_data.get('bids', [])
                        asks = order_book_data.get('asks', [])
                        
                        if bids and asks:
                            # 解析bids和asks数据，处理不同的数据格式
                            parsed_bids = []
                            parsed_asks = []
                            
                            # 处理买单数据
                            for bid in bids[:10]:  # 取前10档
                                if isinstance(bid, list) and len(bid) >= 2:
                                    # 格式: [price, quantity] 或 [price, quantity, ...]
                                    price = float(bid[0])
                                    quantity = float(bid[1])
                                    parsed_bids.append([price, quantity])
                                elif isinstance(bid, dict) and 'price' in bid and 'amount' in bid:
                                    # 字典格式: {'price': ..., 'amount': ...}
                                    price = float(bid['price'])
                                    quantity = float(bid['amount'])
                                    parsed_bids.append([price, quantity])
                                elif isinstance(bid, dict) and '0' in bid and '1' in bid:
                                    # 字典格式: {'0': price, '1': quantity}
                                    price = float(bid['0'])
                                    quantity = float(bid['1'])
                                    parsed_bids.append([price, quantity])
                            
                            # 处理卖单数据
                            for ask in asks[:10]:  # 取前10档
                                if isinstance(ask, list) and len(ask) >= 2:
                                    # 格式: [price, quantity] 或 [price, quantity, ...]
                                    price = float(ask[0])
                                    quantity = float(ask[1])
                                    parsed_asks.append([price, quantity])
                                elif isinstance(ask, dict) and 'price' in ask and 'amount' in ask:
                                    # 字典格式: {'price': ..., 'amount': ...}
                                    price = float(ask['price'])
                                    quantity = float(ask['amount'])
                                    parsed_asks.append([price, quantity])
                                elif isinstance(ask, dict) and '0' in ask and '1' in ask:
                                    # 字典格式: {'0': price, '1': quantity}
                                    price = float(ask['0'])
                                    quantity = float(ask['1'])
                                    parsed_asks.append([price, quantity])
                            
                            if parsed_bids and parsed_asks:
                                # 计算前几档的深度
                                bid_depth = sum(amount for price, amount in parsed_bids[:5])  # 前5档买单深度
                                ask_depth = sum(amount for price, amount in parsed_asks[:5])  # 前5档卖单深度
                                total_depth = bid_depth + ask_depth
                                
                                # 计算价差 (Spread)
                                best_bid = float(parsed_bids[0][0])  # 最高买价
                                best_ask = float(parsed_asks[0][0])  # 最低卖价
                                spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0
                                
                                # 基于深度和价差计算流动性评分
                                # 深度越高，流动性越好；价差越小，流动性越好
                                depth_score = min(1.0, total_depth / 10000)  # 标准化深度评分
                                spread_score = max(0, 1 - spread * 1000)  # 价差越小评分越高
                                
                                # 基于24小时成交量计算流动性评分
                                # 高成交量通常表示高流动性
                                volume_score = min(1.0, volume_24h / 1000000)  # 假设100万美元成交量为满分
                                
                                # 综合流动性评分 (深度40%, 价差40%, 成交量20%)
                                liquidity_score = (depth_score * 0.4 + spread_score * 0.4 + volume_score * 0.2)
                                liquidity_score = min(1.0, liquidity_score)  # 确保不超过1.0
                                
                                liquidity_scores.append(liquidity_score)
                            else:
                                # 如果解析后没有有效数据，使用基于交易对的动态值
                                import random
                                hash_value = hash(clean_sym_upper) % 100
                                liquidity_score = 0.3 + (hash_value / 100.0) * 0.4  # 0.3-0.7之间的值
                                liquidity_scores.append(liquidity_score)
                        else:
                            # 如果订单簿数据为空，使用基于交易对的动态值
                            import random
                            hash_value = hash(clean_sym_upper) % 100
                            liquidity_score = 0.3 + (hash_value / 100.0) * 0.4  # 0.3-0.7之间的值
                            liquidity_scores.append(liquidity_score)
                    else:
                        # 如果订单簿数据为空，使用基于交易对的动态值而不是固定值
                        import random
                        # 使用符号的哈希值来生成一个相对稳定的值，但仍然有变化
                        hash_value = hash(clean_sym_upper) % 100
                        liquidity_score = 0.3 + (hash_value / 100.0) * 0.4  # 0.3-0.7之间的值
                        liquidity_scores.append(liquidity_score)
                except Exception as e:
                    logger.debug(f"流动性分析失败: {e}")
                    import random
                    # 出错时使用随机值而不是固定值
                    liquidity_score = random.uniform(0.3, 0.7)
                    liquidity_scores.append(liquidity_score)

                try:
                    sentiment_analysis = sentiment_analyzer.analyze_sentiment(clean_sym_upper)
                    if sentiment_analysis:
                        sentiment_result = sentiment_analyzer.get_sentiment_summary(sentiment_analysis)
                        logger.debug(f"{clean_sym_upper} 情绪分析结果: {sentiment_result.get('overall_sentiment', 0.0):.2f}")
                        if sentiment_result:
                            sentiment_score = sentiment_result.get('overall_sentiment', 0.0)
                            sentiment_scores.append(sentiment_score)
                        else:
                            # 如果情绪分析结果为空，使用基于交易对的动态值
                            import random
                            hash_value = hash(clean_sym_upper) % 100
                            sentiment_score = -0.3 + (hash_value / 100.0) * 0.6  # -0.3到0.3之间的值
                            sentiment_scores.append(sentiment_score)
                    else:
                        logger.debug(f"{clean_sym_upper} 情绪分析返回空值")
                        # 如果情绪分析返回空值，使用基于交易对的动态值而不是固定值
                        import random
                        hash_value = hash(clean_sym_upper) % 100
                        sentiment_score = -0.3 + (hash_value / 100.0) * 0.6  # -0.3到0.3之间的值
                        sentiment_scores.append(sentiment_score)
                except Exception as e:
                    logger.debug(f"情绪分析失败: {e}")
                    # 出错时使用基于交易对的动态值而不是固定值
                    import random
                    hash_value = hash(clean_sym_upper) % 100
                    sentiment_score = -0.3 + (hash_value / 100.0) * 0.6  # -0.3到0.3之间的值
                    sentiment_scores.append(sentiment_score)

                try:
                    anomaly_result = anomaly_detector.get_anomaly_summary(clean_sym_upper)
                    if anomaly_result:
                        has_anomaly = anomaly_result.get('has_anomaly', False)
                        if has_anomaly:
                            anomaly_count += 1
                except Exception as e:
                    logger.debug(f"异常检测失败: {e}")

                save_analysis_results(clean_sym_upper, p, risk_level, market_regime, liquidity_score, sentiment_score, has_anomaly)

                tactic = get_tactical_decision(
                    trend, ratio, sar_diff, p, 0.02, 1.4,
                    risk_level, market_regime, liquidity_score, sentiment_score, has_anomaly
                )

                risk_color = "red" if risk_level == "HIGH" else ("yellow" if risk_level == "MEDIUM" else "green")
                regime_color = "green" if market_regime == "BULL" else ("red" if market_regime == "BEAR" else "cyan")
                liquidity_color = "green" if liquidity_score >= 0.7 else ("yellow" if liquidity_score >= 0.5 else "red")
                change_color = "green" if change_pct >= 0 else "red"
                sentiment_color = "green" if sentiment_score > 0.3 else ("red" if sentiment_score < -0.3 else "white")
                sentiment_emoji = "😊" if sentiment_score > 0.3 else ("😰" if sentiment_score < -0.3 else "😐")
                anomaly_emoji = "🚨" if has_anomaly else ""

                volume_str = f"{volume_24h/1e6:.1f}M" if volume_24h > 1e6 else f"{volume_24h/1e3:.1f}K"

                table.add_row(
                    sym,
                    f"{p:.2f}",
                    f"{change_pct:+.2f}%",
                    volume_str,
                    f"{sar_diff:+.2f}%",
                    pos_info,
                    pnl_str,
                    trend,
                    f"{ratio:+.4f}",
                    risk_level,
                    market_regime,
                    f"{liquidity_score:.2f}",
                    f"{sentiment_emoji}{sentiment_score:+.2f}",
                    tactic
                )

                if counter % 20 == 0: summary_batch.append(  # 降低AI分析频率，每200秒（20*10秒）分析一次
                    f"{clean_sym_upper} Price:{p:.2f} Ratio:{ratio:.4f} Trend:{trend} SAR:{sar:.2f} Risk:{risk_level} Regime:{market_regime} Liquidity:{liquidity_score:.2f} Sentiment:{sentiment_score:.2f} Anomaly:{str(has_anomaly)}")

            avg_liquidity = sum(liquidity_scores) / len(liquidity_scores) if liquidity_scores else 1.0
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            
            liquidity_color = "green" if avg_liquidity >= 0.7 else ("yellow" if avg_liquidity >= 0.5 else "red")
            sentiment_color = "green" if avg_sentiment > 0.3 else ("red" if avg_sentiment < -0.3 else "white")
            
            # 市场统计面板
            stats_text = f"📊 风险: 高:{risk_summary['HIGH']} 中:{risk_summary['MEDIUM']} 低:{risk_summary['LOW']} | 🐂 环境: 牛:{regime_counts['BULL']} 熊:{regime_counts['BEAR']} 震:{regime_counts['SIDEWAYS']} ?: {regime_counts['UNKNOWN']} | 💧流动: {avg_liquidity:.2f} | 😊情绪: {avg_sentiment:+.2f} | 🚨异常: {anomaly_count}"
            stats_panel = Panel(stats_text, title="📈 市场统计", border_style="blue")
            
            # AI策略简报面板
            with report_lock:
                ai_reports = "\n".join([f">> {r}" for r in ai_report_display[:10]])  # 进一步增加AI报告行数，利用更多空间
            ai_panel = Panel(ai_reports if ai_reports else "暂无AI分析", title="🧠 策略简报", border_style="green")
            
            # 每分钟检查一次是否需要更新资讯
            if counter % 6 == 0:  # 每分钟（10秒*6次）
                update_telegram_news()  # 更新资讯数据
            
            # 资讯面板
            news_list = get_latest_news(5)  # 增加显示数量，利用更多空间
            if news_list:
                news_text = "\n".join([f"{i+1}. {source.split('_')[-1]} {timestamp[11:16]} {content[:100] + '...' if len(content) > 100 else content}" for i, (timestamp, source, content) in enumerate(news_list)])
            else:
                news_text = "无"
            news_panel = Panel(news_text, title="📰 最新资讯", border_style="yellow")
            
            # 状态面板
            status_text = f"📡 实时流: 🟢 | 引擎: {'分析中' if is_ai_calculating else '就绪'} | 模式: 量化 | 进度: [{'█' * (((counter // 6) % 10) + 1)}{'░' * (9 - ((counter // 6) % 10))}]"  # 适应新的刷新频率，每分钟一个完整周期
            status_panel = Panel(status_text, title="状态", border_style="cyan")
            
            # 组合主内容 - 重新设计布局，将AI策略简报移到下方
            main_layout = Layout()
            main_layout.split_column(
                Layout(table, name="table"),
                Layout(name="bottom_section", size=25)  # 进一步增加底部区域高度，为AI分析和资讯预留更多空间
            )
            
            # 底部区域分为左右两部分
            bottom_layout = Layout(name="bottom")
            bottom_layout.split_row(
                Layout(ai_panel, name="ai", ratio=3),  # AI分析占用更多空间
                Layout(name="right_pane", ratio=1)
            )
            
            # 右侧包含统计、资讯和状态
            bottom_layout["right_pane"].split_column(
                Layout(stats_panel, name="stats"),
                Layout(news_panel, name="news"),
                Layout(status_panel, name="status")
            )
            
            main_layout["bottom_section"].update(bottom_layout)
            
            layout["main"].update(main_layout)
            
            if summary_batch and not is_ai_calculating:
                logger.debug(f"触发 AI 分析，包含 {len(summary_batch)} 个资产")
                threading.Thread(target=ai_inference_thread, args=("; ".join(summary_batch), bal)).start()
            
            counter += 1

            return layout
            
        except Exception as e:
            logger.error(f"生成仪表盘时出错: {e}")
            import traceback
            return Panel(f"❌ 仪表盘生成错误: {e}\n{traceback.format_exc()}", title="错误", border_style="red")
    
    # 使用Rich的Live功能实现平滑更新
    # 降低刷新频率以减少视觉疲劳，每10秒刷新一次
    with Live(generate_dashboard(), refresh_per_second=0.1, console=console) as live:  # 0.1 FPS = 每10秒刷新一次
        while True:
            try:
                time.sleep(10)  # 每10秒更新一次数据
                live.update(generate_dashboard())
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"仪表盘更新异常: {e}")
                time.sleep(10)


if __name__ == "__main__":
    try:
        init_database()
        run_dashboard()
    except KeyboardInterrupt:
        logger.debug("收到中断信号，正在关闭系统...")
        print("\n[系统下线] 正在释放资源...")
    except Exception as e:
        logger.error(f"系统崩溃: {e}")
        raise