import json
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 初始化模型
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="your_api_key_here", # 記得填入正確的 Key
    model="google/gemma-3-27b-it",
    temperature=0
)

# 2. 定義工具 (重點：註解必須寫得非常明確)
@tool
def generate_tech_summary(article_content: str):
    """
    這是一個科技文章摘要工具。
    只要使用者的輸入內容包含：AI、人工智慧、輝達(NVIDIA)、台積電、新的技術發佈、或是長篇的科技新聞，
    請『務必』呼叫此工具進行摘要。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個科技專家，請將以下內容濃縮成三個核心重點。"),
        ("user", "{text}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": article_content})

# 3. 綁定工具
llm_with_tools = llm.bind_tools([generate_tech_summary])

# 4. 路由提示詞 (重點：給 AI 明確的判斷準則)
router_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個分類助手。如果使用者的輸入看起來像是一篇科技新聞或技術描述，請呼叫 generate_tech_summary。如果是打招呼(如：你好)或日常對話，請直接回答。"),
    ("user", "{user_input}")
])

# 5. 互動循環
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "q"]: break
    
    # 注意這裡的 key 必須是 user_input
    ai_msg = (router_prompt | llm_with_tools).invoke({"user_input": user_input})
    
    if ai_msg.tool_calls:
        print(f"✅ [決策] 偵測到科技內容，正在處理...")
        tool_args = ai_msg.tool_calls[0]['args']
        final_result = generate_tech_summary.invoke(tool_args)
        print(f"📄 [摘要結果]:\n{final_result}")
    else:
        print(f"❌ [決策] 判斷為一般對話")
        print(f"💬 [AI 回應]: {ai_msg.content}")