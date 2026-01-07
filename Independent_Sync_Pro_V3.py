import ccxt, sqlite3, time, pandas as pd, pandas_ta as ta
from datetime import datetime

# === 📁 核心对齐配置 ===
DB_VERIFY = "/mnt/data/quant_storage/sqlite/verification_pro.db"
PROXY_URL = 'socks5h://127.0.0.1:1080'
# 💡 确保这里包含驾驶舱显示的所有币种
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'DOGE/USDT', 'APT/USDT']

class MarketSyncV3:
    def __init__(self):
        self.exchange = ccxt.binance({
            'proxies': {'http': PROXY_URL, 'https': PROXY_URL},
            'enableRateLimit': True
        })
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(DB_VERIFY)
        conn.execute('''CREATE TABLE IF NOT EXISTS verify_pro_ticker (
            symbol TEXT PRIMARY KEY, price REAL, change_24h REAL, 
            volume_24h_usd REAL, order_ratio REAL, sar_value REAL, 
            sar_trend TEXT, last_update TEXT)''')
        conn.close()

    def get_sar(self, symbol):
        try:
            bars = self.exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
            df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            psar_df = ta.psar(df['high'], df['low'], df['close'], af=0.02, max_af=0.2)
            last = psar_df.iloc[-1]
            if not pd.isna(last.iloc[0]): return float(last.iloc[0]), "BULL"
            return float(last.iloc[1]), "BEAR"
        except: return None, "ERROR"

    def sync(self):
        tickers = self.exchange.fetch_tickers(SYMBOLS)
        conn = sqlite3.connect(DB_VERIFY)
        for sym in SYMBOLS:
            if sym not in tickers: continue
            # 5档盘口计算买卖力度
            ob = self.exchange.fetch_order_book(sym, limit=5)
            ratio = sum(b[1] for b in ob['bids']) / sum(a[1] for a in ob['asks']) if sum(a[1] for a in ob['asks']) > 0 else 1.0
            sar_val, trend = self.get_sar(sym)
            
            conn.execute("""INSERT OR REPLACE INTO verify_pro_ticker VALUES (?,?,?,?,?,?,?,?)""",
                (sym, float(tickers[sym]['last']), float(tickers[sym]['percentage']), 
                 float(tickers[sym]['quoteVolume']), round(ratio, 4), sar_val, trend, datetime.now().strftime('%H:%M:%S')))
        conn.commit()
        conn.close()

if __name__ == "__main__":
    sync_engine = MarketSyncV3()
    while True:
        try:
            sync_engine.sync()
            print(f"✅ {datetime.now().strftime('%H:%M:%S')} 同步成功")
            time.sleep(5)
        except Exception as e:
            print(f"❌ 同步异常: {e}"); time.sleep(5)