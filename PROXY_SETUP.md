# 服务器端代理配置说明

## 问题原因

主程序现在会使用 config.py 中的 PROXY_URL 配置。config.py 中的默认值是：
```python
PROXY_URL = os.getenv('PROXY_URL', 'socks5h://127.0.0.1:1080')
```

这个默认值是本地代理地址，在服务器端需要设置正确的代理地址。

## 解决方案

### 方法 1: 在运行前设置环境变量（推荐）

```bash
export PROXY_URL='socks5h://your-proxy-host:port'
python3 ZhangXingguang_Flight_Dash_V10.6_MAX.py
```

### 方法 2: 在一行命令中设置

```bash
PROXY_URL='socks5h://your-proxy-host:port' python3 ZhangXingguang_Flight_Dash_V10.6_MAX.py
```

### 方法 3: 在 .bashrc 中永久设置

```bash
echo 'export PROXY_URL="socks5h://your-proxy-host:port"' >> ~/.bashrc
source ~/.bashrc
python3 ZhangXingguang_Flight_Dash_V10.6_MAX.py
```

### 方法 4: 修改 config.py（不推荐，但可以临时使用）

编辑 config.py 文件，修改 PROXY_URL 的默认值：

```python
# 原来的默认值
# PROXY_URL = os.getenv('PROXY_URL', 'socks5h://127.0.0.1:1080')

# 修改为您的服务器代理地址
PROXY_URL = os.getenv('PROXY_URL', 'socks5h://your-proxy-host:port')
```

## 代理地址格式

支持以下格式：
- SOCKS5 代理: `socks5h://host:port` 或 `socks5://host:port`
- HTTP 代理: `http://host:port` 或 `https://host:port`

## 测试代理连接

运行诊断脚本测试代理配置：

```bash
python3 diagnose_proxy_server.py
```

## 常见问题

### Q: 如何知道服务器的代理地址？

A: 请联系您的服务器管理员或查看服务器配置文件。常见的代理地址包括：
- 本地代理: `socks5h://127.0.0.1:1080`
- 隧道代理: `socks5h://your-tunnel-host:port`
- 公司代理: `http://proxy.company.com:8080`

### Q: 为什么需要代理？

A: 服务器可能无法直接访问 Binance WebSocket，需要通过代理转发请求。

### Q: 如何验证代理是否可用？

A: 运行诊断脚本 `python3 diagnose_proxy_server.py`，它会测试：
1. 直接连接（不使用代理）
2. 代理连接测试
3. 通过代理连接 WebSocket

### Q: 日志显示"未配置代理"怎么办？

A: 这意味着 PROXY_URL 环境变量未设置或为空。请使用上述方法之一设置代理。

## 示例

假设您的服务器代理地址是 `socks5h://192.168.1.100:1080`，运行命令：

```bash
export PROXY_URL='socks5h://192.168.1.100:1080'
python3 ZhangXingguang_Flight_Dash_V10.6_MAX.py
```

或者一行命令：

```bash
PROXY_URL='socks5h://192.168.1.100:1080' python3 ZhangXingguang_Flight_Dash_V10.6_MAX.py
```

## 预期结果

配置正确后，日志应该显示：

```
代理地址解析成功: your-proxy-host:port (类型: socks5)
开始测试代理连接: your-proxy-host:port
代理连接测试成功: your-proxy-host:port
✓ 代理连接成功，将使用代理连接WebSocket
正在连接 WebSocket: wss://stream.binance.com:9443/stream?streams=... (代理: 是)
✓ WebSocket连接成功，订阅 10 个交易对
等待接收市场数据...
✓ 收到第一条ticker消息！
✓ 收到第一条depth消息！
```
