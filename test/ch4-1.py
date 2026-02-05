from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

# 後續的 llm 初始化程式碼...

# 1. 初始化模型 (設定 API 代理位址與模型名稱)
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="your_api_key_here",  # 請替換為您的 API Key
    model="google/gemma-3-27b-it",
    temperature=0
)

# 2. 定義工具 (定義 LLM 需要提取的欄位與描述)
@tool
def extract_order_data(name: str, phone: str, product: str, quantity: int, address: str):
    """
    資料提取專用工具。
    專門用於從非結構化文本中提取訂單相關資訊（姓名、電話、商品、數量、地址）。
    """
    return {
        "name": name,
        "phone": phone,
        "product": product,
        "quantity": quantity,
        "address": address
    }

# 3. 綁定工具到模型
llm_with_tools = llm.bind_tools([extract_order_data])

# 4. 設定提示詞範本
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個精準的訂單管理員，請從對話中提取訂單資訊。"),
    ("user", "{user_input}")
])

# 5. 定義解析函數 (從 AI 回傳訊息中取出 Tool Call 的參數)
def extract_tool_args(ai_message):
    if ai_message.tool_calls:
        # 取得第一個工具呼叫的引數內容
        return ai_message.tool_calls[0]['args']
    return None

# 6. 建立處理鏈 (Chain)
chain = prompt | llm_with_tools | extract_tool_args

# 7. 執行測試
user_text = "你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週五送到台中市北區。"
result = chain.invoke({"user_input": user_text})

# 8. 輸出結果
if result:
    print("✅ 提取成功：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    print("❌ 提取失敗")