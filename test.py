from openai import OpenAI
import sys

# ── 連線設定（使用課程提供的伺服器） ──
client = OpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="vllm-token",          # 若無需驗證，可改成 "" 或 "sk-no-key-required"
)

# 使用的模型（請確認伺服器實際載入的 Qwen 模型名稱）
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"   # 或 "google/gemma-3-27b-it" / "Qwen/Qwen3-VL-8B-Instruct" 等

# 系統提示（可自行修改）
SYSTEM_PROMPT = "You are a helpful and friendly AI assistant. 用繁體中文回覆。"

# 對話歷史（保持上下文）
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("=== Qwen 聊天模式 ===")
print(f"模型：{MODEL_NAME}")
print("輸入訊息後按 Enter，輸入 exit 或 quit 結束\n")

while True:
    try:
        user_input = input("你： ").strip()
        
        if user_input.lower() in ["exit", "quit", "結束", "bye"]:
            print("再見！")
            break
        
        if not user_input:
            continue

        # 加入使用者訊息
        messages.append({"role": "user", "content": user_input})

        print("Qwen：", end="", flush=True)

        # 呼叫 API，使用 streaming
        stream_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stream=True
        )

        # 即時輸出
        assistant_reply = ""
        for chunk in stream_response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                assistant_reply += content

        print()  # 換行

        # 把Agent回覆加進歷史(Context)
        messages.append({"role": "assistant", "content": assistant_reply})

    except KeyboardInterrupt:
        print("\n\n已中斷，再見！")
        break
    except Exception as e:
        print(f"\n發生錯誤：{e}")
        print("請檢查網路、模型名稱或伺服器狀態")
        continue