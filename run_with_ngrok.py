import os
from pyngrok import ngrok

# 关闭所有已存在的隧道
ngrok.kill()

# 我们只需要 HTTP 隧道 (ngrok 会自动提供 https 的公网入口)
# 注意：国内网络环境如果 ngrok 不稳定，可能会报错。
try:
    public_url = ngrok.connect(8000, "http").public_url
    print(f"==================================================")
    print(f"🎉 内网穿透成功！手机请访问【HTTPS】开头的地址（支持录音）：")
    print(f"👉 {public_url.replace('http://', 'https://')}")
    print(f"==================================================")
except Exception as e:
    print(f"Ngrok 穿透失败: {e}")

# 启动普通 HTTP 的 uvicorn 即可（公网的 HTTPS 到本地的 HTTP 会被 ngrok 自动转换）
os.system("uvicorn backend.main:app --host 0.0.0.0 --port 8000")
