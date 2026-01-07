#!/usr/bin/env python3
"""
诊断脚本：检查主程序的数据流
"""
import sys
import time
import os
from config import PROXY_URL, CRYPTO_LIST

def test_imports():
    """测试所有依赖导入"""
    print("=" * 80)
    print("步骤 1: 测试依赖导入")
    print("=" * 80)
    
    try:
        import sqlite3
        print("✅ sqlite3")
    except ImportError as e:
        print(f"❌ sqlite3: {e}")
        return False
    
    try:
        import asyncio
        print("✅ asyncio")
    except ImportError as e:
        print(f"❌ asyncio: {e}")
        return False
    
    try:
        import websockets
        print("✅ websockets")
    except ImportError as e:
        print(f"❌ websockets: {e}")
        return False
    
    try:
        import socks
        print("✅ socks (PySocks)")
    except ImportError as e:
        print(f"❌ socks (PySocks): {e}")
        return False
    
    try:
        import requests
        print("✅ requests")
    except ImportError as e:
        print(f"❌ requests: {e}")
        return False
    
    try:
        from openai import OpenAI
        print("✅ openai")
    except ImportError as e:
        print(f"❌ openai: {e}")
        return False
    
    try:
        from colorama import Fore, Style
        print("✅ colorama")
    except ImportError as e:
        print(f"❌ colorama: {e}")
        return False
    
    try:
        from risk_manager import RiskManager
        print("✅ risk_manager")
    except ImportError as e:
        print(f"❌ risk_manager: {e}")
        return False
    
    try:
        from technical_indicators import TechnicalIndicators
        print("✅ technical_indicators")
    except ImportError as e:
        print(f"❌ technical_indicators: {e}")
        return False
    
    try:
        from market_regime_detector import MarketRegimeDetector
        print("✅ market_regime_detector")
    except ImportError as e:
        print(f"❌ market_regime_detector: {e}")
        return False
    
    try:
        from liquidity_manager import LiquidityManager
        print("✅ liquidity_manager")
    except ImportError as e:
        print(f"❌ liquidity_manager: {e}")
        return False
    
    try:
        from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer
        print("✅ advanced_sentiment_analyzer")
    except ImportError as e:
        print(f"❌ advanced_sentiment_analyzer: {e}")
        return False
    
    try:
        from anomaly_detector import AnomalyDetector
        print("✅ anomaly_detector")
    except ImportError as e:
        print(f"❌ anomaly_detector: {e}")
        return False
    
    print("\n✅ 所有依赖导入成功\n")
    return True

def test_config():
    """测试配置"""
    print("=" * 80)
    print("步骤 2: 测试配置")
    print("=" * 80)
    
    print(f"代理URL: {PROXY_URL}")
    print(f"加密货币数量: {len(CRYPTO_LIST)}")
    print(f"加密货币列表: {CRYPTO_LIST[:5]}...")
    
    if PROXY_URL and PROXY_URL != 'None':
        print("✅ 代理已配置")
    else:
        print("⚠️  代理未配置")
    
    print()

def test_database():
    """测试数据库初始化"""
    print("=" * 80)
    print("步骤 3: 测试数据库初始化")
    print("=" * 80)
    
    try:
        import sqlite3
        from config import DB_VERIFY, DB_MEMORY, DATA_DIR
        import os
        
        print(f"验证数据库路径: {DB_VERIFY}")
        print(f"内存数据库路径: {DB_MEMORY}")
        print(f"数据目录: {DATA_DIR}")
        
        # 检查数据库文件是否存在
        if os.path.exists(DB_VERIFY):
            print(f"✅ 验证数据库文件存在")
            conn = sqlite3.connect(DB_VERIFY)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cur.fetchall()
            print(f"   表: {[t[0] for t in tables]}")
            conn.close()
        else:
            print(f"⚠️  验证数据库文件不存在")
        
        if os.path.exists(DB_MEMORY):
            print(f"✅ 内存数据库文件存在")
            conn = sqlite3.connect(DB_MEMORY)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cur.fetchall()
            print(f"   表: {[t[0] for t in tables]}")
            conn.close()
        else:
            print(f"⚠️  内存数据库文件不存在")
        
        print()
        return True
    except Exception as e:
        print(f"❌ 数据库测试失败: {e}\n")
        return False

def test_market_data_fetcher():
    """测试市场数据获取器"""
    print("=" * 80)
    print("步骤 4: 测试市场数据获取器")
    print("=" * 80)
    
    try:
        import importlib.util
        import sys
        
        # 动态导入主程序
        spec = importlib.util.spec_from_file_location("main_module", "ZhangXingguang_Flight_Dash_V10.6_MAX.py")
        main_module = importlib.util.module_from_spec(spec)
        sys.modules["main_module"] = main_module
        spec.loader.exec_module(main_module)
        
        MarketDataFetcher = main_module.MarketDataFetcher
        
        print("正在初始化 MarketDataFetcher...")
        fetcher = MarketDataFetcher()
        print("✅ MarketDataFetcher 初始化成功")
        
        print(f"WebSocket 运行状态: {fetcher.is_running}")
        print(f"WebSocket 线程ID: {fetcher.ws_thread.ident if fetcher.ws_thread else 'None'}")
        
        # 等待 WebSocket 连接建立
        print("\n等待 10 秒让 WebSocket 接收数据...")
        time.sleep(10)
        
        # 检查缓存
        print("\n检查缓存状态:")
        with fetcher.lock:
            print(f"  ticker_24h_cache 大小: {len(fetcher.ticker_24h_cache)}")
            print(f"  order_book_cache 大小: {len(fetcher.order_book_cache)}")
            
            if fetcher.ticker_24h_cache:
                print(f"\n  ticker_24h_cache 内容:")
                for symbol, data in list(fetcher.ticker_24h_cache.items())[:3]:
                    print(f"    {symbol}: 价格={data['price']}, 涨跌={data['change_pct']:.2f}%")
            
            if fetcher.order_book_cache:
                print(f"\n  order_book_cache 内容:")
                for symbol, data in list(fetcher.order_book_cache.items())[:3]:
                    print(f"    {symbol}: 买卖盘比={data['order_ratio']:.4f}")
        
        # 测试获取所有市场数据
        print("\n测试 get_all_market_data()...")
        market_data = fetcher.get_all_market_data()
        print(f"✅ 获取到 {len(market_data)} 个资产的市场数据")
        
        if market_data:
            print("\n  前 3 个资产的数据:")
            for data in market_data[:3]:
                print(f"    {data['symbol']}: 价格={data['price']:.2f}, 涨跌={data['change_pct']:.2f}%, 买卖盘比={data['order_ratio']:.4f}")
        
        fetcher.close()
        print("\n✅ 市场数据获取器测试完成\n")
        return len(market_data) > 0
    except Exception as e:
        print(f"❌ 市场数据获取器测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 80)
    print("主程序诊断工具")
    print("=" * 80 + "\n")
    
    results = []
    
    # 步骤 1: 测试依赖导入
    results.append(("依赖导入", test_imports()))
    
    if not results[-1][1]:
        print("❌ 依赖导入失败，请先安装缺失的依赖")
        return False
    
    # 步骤 2: 测试配置
    test_config()
    
    # 步骤 3: 测试数据库
    results.append(("数据库", test_database()))
    
    # 步骤 4: 测试市场数据获取器
    results.append(("市场数据获取器", test_market_data_fetcher()))
    
    # 总结
    print("=" * 80)
    print("诊断总结")
    print("=" * 80)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✅ 所有测试通过！主程序应该可以正常运行。")
    else:
        print("\n❌ 部分测试失败，请检查上述错误信息。")
    
    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  诊断被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 诊断过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
