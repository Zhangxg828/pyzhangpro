# WebSocket连接测试说明

## 测试目的
测试Binance WebSocket连接，验证是否能够通过SOCKS5代理获取实时行情数据。

## 测试环境要求

### 1. 服务器端需要安装的依赖
```bash
pip install websocket-client PySocks
```

### 2. 代理配置
确保服务器端有可用的SOCKS5代理，并设置环境变量：
```bash
export PROXY_URL=socks5h://127.0.0.1:1080
```

## 测试步骤

### 方法1: 交互式测试
```bash
python3 test_websocket_connection.py
```

运行后会显示菜单：
```
请选择测试模式:
1. 直接连接（不使用代理）
2. 通过SOCKS5代理连接
3. 依次测试两种模式
0. 退出

请输入选项 (0-3):
```

### 方法2: 直接测试代理连接
```bash
# 设置代理环境变量
export PROXY_URL=socks5h://127.0.0.1:1080

# 运行测试脚本并选择选项2
python3 test_websocket_connection.py
```

### 方法3: 测试直接连接（不使用代理）
```bash
# 不设置代理环境变量或设置为空
export PROXY_URL=

# 运行测试脚本并选择选项1
python3 test_websocket_connection.py
```

## 预期结果

### 成功的输出示例：
```
============================================================================
测试2: 通过SOCKS5代理连接
代理地址: socks5h://127.0.0.1:1080
============================================================================
WebSocket URL: wss://stream.binance.com:9443/ws/btcusdt@ticker/ethusdt@ticker/bnbusdt@ticker
正在配置代理...
代理主机: 127.0.0.1
代理端口: 1080
✅ 代理配置成功
正在连接...
✅ WebSocket连接已建立
等待数据...
✅ [2026-01-05 13:45:23] BTCUSDT: $95000.50
✅ [2026-01-05 13:45:23] ETHUSDT: $3200.25
✅ [2026-01-05 13:45:24] BNBUSDT: $650.80
```

### 失败的输出示例：
```
❌ WebSocket错误: Python Socks is needed for SOCKS proxying but is not available
```
或
```
❌ WebSocket错误: [Errno 60] Operation timed out
```

## 故障排查

### 问题1: "Python Socks is needed for SOCKS proxying but is not available"
**原因**: PySocks库未正确安装或导入
**解决**: 
```bash
pip install --upgrade PySocks
```

### 问题2: "[Errno 60] Operation timed out"
**原因**: 
1. 代理服务器不可用
2. 代理地址或端口配置错误
3. 防火墙阻止了连接

**解决**:
1. 检查代理服务器是否正常运行
2. 验证代理地址和端口是否正确
3. 检查防火墙设置

### 问题3: 直接连接失败
**原因**: 
1. 网络环境无法直接访问Binance
2. 需要通过代理才能访问

**解决**: 使用代理连接模式

## 测试完成后的操作

如果测试成功，说明WebSocket连接配置正确，可以运行主程序：
```bash
python3 ZhangXingguang_Flight_Dash_V10.6_MAX.py
```

如果测试失败，需要先解决连接问题，然后再运行主程序。

## 常用命令

### 查看已安装的包
```bash
pip list | grep -i socks
pip list | grep -i websocket
```

### 检查代理连接
```bash
# 如果使用proxychains
proxychains curl https://api.binance.com/api/v3/ping

# 如果使用其他代理工具
curl -x socks5://127.0.0.1:1080 https://api.binance.com/api/v3/ping
```

### 查看测试日志
```bash
# 测试脚本的输出会直接显示在终端
# 如果需要保存日志，可以重定向输出
python3 test_websocket_connection.py > test_log.txt 2>&1
```