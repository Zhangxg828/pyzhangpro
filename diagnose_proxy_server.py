#!/usr/bin/env python3
"""
服务器端代理配置诊断脚本
用于诊断和测试代理连接配置
"""
import os
import sys
import socket
import time
import socks
import asyncio
import websockets
import json
from urllib.parse import urlparse

def test_direct_connection():
    """测试直接连接到Binance WebSocket"""
    print("=" * 80)
    print("测试 1: 直接连接到 Binance WebSocket（不使用代理）")
    print("=" * 80)
    
    uri = "wss://stream.binance.com:9443/ws/btcusdt@ticker"
    
    try:
        print(f"正在连接: {uri}")
        start_time = time.time()
        
        async def connect():
            try:
                async with websockets.connect(
                    uri,
                    ssl=True,
                    close_timeout=10,
                    ping_interval=30,
                    ping_timeout=20,
                    open_timeout=15
                ) as ws:
                    connect_time = time.time() - start_time
                    print(f"✓ 连接成功！耗时: {connect_time:.2f}秒")
                    
                    print("等待接收消息...")
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        data = json.loads(message)
                        print(f"✓ 收到消息: {json.dumps(data, indent=2)[:200]}")
                        return True
                    except asyncio.TimeoutError:
                        print("✗ 接收消息超时（10秒）")
                        return False
            except Exception as e:
                print(f"✗ 连接失败: {e}")
                return False
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(connect())
        loop.close()
        
        return result
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_proxy_connection(proxy_url):
    """测试代理连接"""
    print("\n" + "=" * 80)
    print(f"测试 2: 代理连接测试 - {proxy_url}")
    print("=" * 80)
    
    try:
        if proxy_url.startswith('socks5h://'):
            url = proxy_url.replace('socks5h://', '')
            proxy_type = 'socks5'
        elif proxy_url.startswith('socks5://'):
            url = proxy_url.replace('socks5://', '')
            proxy_type = 'socks5'
        elif proxy_url.startswith('http://'):
            url = proxy_url.replace('http://', '')
            proxy_type = 'http'
        elif proxy_url.startswith('https://'):
            url = proxy_url.replace('https://', '')
            proxy_type = 'http'
        else:
            url = proxy_url
            proxy_type = 'http'
        
        if ':' in url:
            host, port = url.split(':')
            port = int(port)
        else:
            print(f"✗ 代理地址格式错误: {proxy_url}")
            return False
        
        print(f"代理类型: {proxy_type}")
        print(f"代理地址: {host}:{port}")
        
        print(f"\n正在测试代理连接...")
        start_time = time.time()
        
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.settimeout(10)
        try:
            test_sock.connect((host, port))
            connect_time = time.time() - start_time
            print(f"✓ 代理连接成功！耗时: {connect_time:.2f}秒")
            test_sock.close()
            return True
        except socket.timeout:
            print(f"✗ 代理连接超时（10秒）")
            test_sock.close()
            return False
        except Exception as e:
            print(f"✗ 代理连接失败: {e}")
            test_sock.close()
            return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_websocket_with_proxy(proxy_url):
    """测试通过代理连接到Binance WebSocket"""
    print("\n" + "=" * 80)
    print(f"测试 3: 通过代理连接 Binance WebSocket - {proxy_url}")
    print("=" * 80)
    
    uri = "wss://stream.binance.com:9443/ws/btcusdt@ticker"
    
    try:
        if proxy_url.startswith('socks5h://'):
            url = proxy_url.replace('socks5h://', '')
        elif proxy_url.startswith('socks5://'):
            url = proxy_url.replace('socks5://', '')
        else:
            url = proxy_url
        
        if ':' in url:
            proxy_host, proxy_port = url.split(':')
            proxy_port = int(proxy_port)
        else:
            print(f"✗ 代理地址格式错误: {proxy_url}")
            return False
        
        print(f"代理地址: {proxy_host}:{proxy_port}")
        print(f"WebSocket URL: {uri}")
        
        print(f"\n正在创建 SOCKS5 代理 socket...")
        start_time = time.time()
        
        async def connect():
            try:
                parsed = urlparse(uri)
                print(f"目标服务器: {parsed.hostname}:{443}")
                
                sock = socks.socksocket()
                sock.set_proxy(
                    proxy_type=socks.SOCKS5,
                    addr=proxy_host,
                    port=proxy_port,
                    rdns=True
                )
                sock.settimeout(10)
                
                print("正在通过代理连接...")
                sock.connect((parsed.hostname, 443))
                
                connect_time = time.time() - start_time
                print(f"✓ 代理 socket 连接成功！耗时: {connect_time:.2f}秒")
                
                print("正在建立 WebSocket 连接...")
                async with websockets.connect(
                    uri,
                    sock=sock,
                    ssl=True,
                    close_timeout=10,
                    ping_interval=30,
                    ping_timeout=20,
                    open_timeout=15
                ) as ws:
                    ws_connect_time = time.time() - start_time
                    print(f"✓ WebSocket 连接成功！总耗时: {ws_connect_time:.2f}秒")
                    
                    print("等待接收消息...")
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        data = json.loads(message)
                        print(f"✓ 收到消息: {json.dumps(data, indent=2)[:200]}")
                        return True
                    except asyncio.TimeoutError:
                        print("✗ 接收消息超时（10秒）")
                        return False
            except Exception as e:
                print(f"✗ 连接失败: {e}")
                return False
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(connect())
        loop.close()
        
        return result
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def main():
    print("\n" + "=" * 80)
    print("🔍 服务器端代理配置诊断工具")
    print("=" * 80)
    
    proxy_url = os.getenv('PROXY_URL', '')
    print(f"\n当前代理配置:")
    print(f"  PROXY_URL 环境变量: {proxy_url if proxy_url else '未设置'}")
    
    if not proxy_url:
        print("\n⚠️  警告: 未设置 PROXY_URL 环境变量")
        print("\n💡 解决方案:")
        print("  方法 1: 在运行命令前设置环境变量")
        print("    export PROXY_URL='socks5h://your-proxy-host:port'")
        print("    python3 ZhangXingguang_Flight_Dash_V10.6_MAX.py")
        print("\n  方法 2: 在 .bashrc 或 .zshrc 中永久设置")
        print("    echo 'export PROXY_URL=\"socks5h://your-proxy-host:port\"' >> ~/.bashrc")
        print("    source ~/.bashrc")
        print("\n  方法 3: 在运行时直接设置")
        print("    PROXY_URL='socks5h://your-proxy-host:port' python3 ZhangXingguang_Flight_Dash_V10.6_MAX.py")
    
    results = {}
    
    results['direct'] = test_direct_connection()
    
    if proxy_url:
        results['proxy'] = test_proxy_connection(proxy_url)
        if results['proxy']:
            results['websocket_proxy'] = test_websocket_with_proxy(proxy_url)
    
    print("\n" + "=" * 80)
    print("📊 测试结果汇总")
    print("=" * 80)
    print(f"直接连接: {'✓ 成功' if results.get('direct') else '✗ 失败'}")
    if proxy_url:
        print(f"代理连接: {'✓ 成功' if results.get('proxy') else '✗ 失败'}")
        print(f"WebSocket+代理: {'✓ 成功' if results.get('websocket_proxy') else '✗ 失败'}")
    
    print("\n" + "=" * 80)
    print("💡 建议")
    print("=" * 80)
    
    if results.get('direct'):
        print("✓ 直接连接成功，可以不使用代理")
        print("  运行命令: python3 ZhangXingguang_Flight_Dash_V10.6_MAX.py")
    elif proxy_url and results.get('websocket_proxy'):
        print("✓ 通过代理连接成功")
        print(f"  运行命令: PROXY_URL='{proxy_url}' python3 ZhangXingguang_Flight_Dash_V10.6_MAX.py")
    else:
        print("✗ 所有连接方式均失败")
        print("\n可能的原因:")
        print("  1. 服务器网络无法访问 Binance")
        print("  2. 代理配置错误或代理不可用")
        print("  3. 防火墙阻止了连接")
        print("\n建议:")
        print("  1. 检查服务器网络连接")
        print("  2. 确认代理地址和端口是否正确")
        print("  3. 联系网络管理员检查防火墙设置")

if __name__ == "__main__":
    main()
