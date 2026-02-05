import json
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser  # 1. 必須匯入這個

# 初始化模型
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="your_api_key_here", 
    model="google/gemma-3-27b-it",
    temperature=0
)

# 2. 修正工具定義
@tool
def generate_tech_summary(article_content: str):
    """
    這是一個科技文章摘要工具。當使用者提供的內容涉及科技、AI、硬體或軟體新聞時，請使用此工具進行摘要。
    """
    # 修正：Prompt 列表遺漏逗號，且 text 應對應 invoke 的 key
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個科技筆記助手，請用簡短的三個重點摘要以下內容。"),
        ("user", "{text}")
    ])
    
    # 修正：工具內部也要能存取到外層的 llm
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": article_content})
    return result

# 3. 綁定工具
llm_with_tools = llm.bind_tools([generate_tech_summary])

# 4. 路由提示詞
router_prompt = ChatPromptTemplate.from_messages([
    ("user", "{user_input}")
])

while True:
    user_input = input("User: ")
    
    if user_input.lower() in ["exit", "q"]:
        print("Bye!")
        break
    
    # 5. 修正：這裡的 invoke 參數必須跟 router_prompt 的變數名 {user_input} 一致
    ai_msg = (router_prompt | llm_with_tools).invoke({"user_input": user_input})
    
    # 判斷邏輯
    if ai_msg.tool_calls:
        print(f"✅ [決策] 判斷為科技文章")
        # 取得工具參數
        tool_args = ai_msg.tool_calls[0]['args']
        
        # 執行工具
        final_result = generate_tech_summary.invoke(tool_args)
        print(f"📄 [執行結果]:\n{final_result}")
    else:
        print(f"❌ [決策] 判斷為閒聊")
        print(f"💬 [AI 回應]: {ai_msg.content}")