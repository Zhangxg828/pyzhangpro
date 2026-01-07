import sqlite3
import requests
from datetime import datetime
from openai import OpenAI

# === 📁 配置 ===
DB_VERIFY = "/mnt/data/quant_storage/sqlite/verification_pro.db"
VLLM_API = "http://localhost:8000/v1"
MODEL_NAME = "/models"
DINGTALK_WEBHOOK = "https://oapi.dingtalk.com/robot/send?access_token=c04fdac4d9b62a470904ddc0b4cbba8182b3e6c2546bc1c7fac333950b719136"

client = OpenAI(api_key="EMPTY", base_url=VLLM_API)


class XContentCreatorPro:
    def fetch_market_state(self):
        """获取验证源的硬核盘口数据"""
        try:
            conn = sqlite3.connect(DB_VERIFY)
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, price, order_ratio, volume_24h_usd FROM verify_pro_ticker")
            rows = cursor.fetchall()
            conn.close()

            summary = []
            for r in rows:
                sym = r[0].split('/')[0]
                ratio = r[2]
                # 自动判别盘口情绪
                sentiment = "买方扫货" if ratio > 1.5 else ("空头压制" if ratio < 0.5 else "震荡整理")
                summary.append(f"{sym}: ${r[1]} | 盘口比: {ratio} ({sentiment}) | 成交额: ${r[3] / 1e6:.1f}M")
            return "\n".join(summary)
        except:
            return "数据链路同步中..."

    def generate_post(self):
        market_data = self.fetch_market_state()
        ts = datetime.now().strftime('%H:%M')

        # 🎭 给 Qwen3 的深度指令
        system_role = (
            "你是一个在 X 拥有百万粉丝的资深量化交易员，专门捕捉盘口大单。语气要犀利、带点不屑，显得你洞察一切。"
            "必须使用提供的‘盘口比’数据。如果比值极低（如<0.5），说明上方全是抛压挂单，是虚假繁荣。"
        )

        prompt = (
            f"当前 {ts} 实时盘口透视：\n{market_data}\n\n"
            "任务：写一篇推文。要求：\n"
            "1. 重点点评 BTC 和 ETH 极低的盘口比（0.15和0.04），拆穿市场的虚假反弹。\n"
            "2. 提到这种‘上方抛压如山’的统计学含义。不要用‘祝大家愉快’等废话。\n"
            "3. 结尾问粉丝：这种盘口谁敢去接飞刀？"
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": system_role}, {"role": "user", "content": prompt}],
                temperature=0.8
            )
            content = response.choices[0].message.content
            self.send_dingtalk(f"深夜盘口透视 ({ts})", content)
        except Exception as e:
            print(f"生成失败: {e}")

    def send_dingtalk(self, title, text):
        payload = {
            "msgtype": "markdown",
            "markdown": {"title": title, "text": f"### 📊 {title}\n\n---\n{text}\n\n---\n**[数据源: 独立验证库]**"}
        }
        requests.post(DINGTALK_WEBHOOK, json=payload)
        print("✅ 深度样稿已发送至钉钉，请审阅。")


if __name__ == "__main__":
    XContentCreatorPro().generate_post()