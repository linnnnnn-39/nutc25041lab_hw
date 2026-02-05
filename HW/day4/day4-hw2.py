import os
import requests
import operator
from typing import Annotated, List, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

# --- 1. 配置 ---
SEARXNG_URL = "https://puli-8080.huannago.com/search"

# 建立 LLM 連線
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="YOUR_API_KEY", 
    model="google/gemma-3-27b-it",
    temperature=0.3,
    timeout=20.0  # 避免無限等待
)

class AgentState(TypedDict):
    input: str
    knowledge_base: Annotated[List[str], operator.add]
    search_count: int
    next_step: str 
    current_plan: str
    final_response: str

# --- 2. 節點功能定義 ---

def router_node(state: AgentState):
    """判斷該走快速回覆還是搜尋"""
    print("--- [Node] 路由判斷 ---")
    text = state["input"].strip().lower()
    # 關鍵字攔截：包含這些字眼直接走搜尋
    realtime_keywords = ["股價", "現在", "新聞", "報價", "大跌", "為何", "為什麼"]
    if any(kw in text for kw in realtime_keywords):
        return {"next_step": "search_path"}
    
    try:
        prompt = f"判斷問題是否需要即時資訊或事實查詢：'{text}'。若是閒聊回傳 FAST，否則回傳 SEARCH。只准回傳單字。"
        res = llm.invoke(prompt).content.upper()
        return {"next_step": "fast_path" if "FAST" in res else "search_path"}
    except:
        return {"next_step": "search_path"}

def fast_answer_node(state: AgentState):
    """快速回覆閒聊"""
    print("--- [Node] 快速通道 ---")
    try:
        res = llm.invoke(state["input"]).content
        return {"final_response": res}
    except:
        return {"final_response": "你好！目前我有點連線困難，但很高興見到你。"}

def query_gen_node(state: AgentState):
    """生成或優化搜尋關鍵字"""
    print(f"--- [Node] 生成搜尋計畫 (第 {state['search_count']+1} 次) ---")
    query = state["input"]
    try:
        # 如果是第二次搜尋，嘗試變換關鍵字
        prompt_text = f"優化搜尋關鍵字：'{query}'" if state['search_count'] == 0 else f"換個方式搜：'{query}' 的原因與分析"
        res = llm.invoke(f"{prompt_text}。只回傳關鍵字，不要廢話。").content.strip()
        if res and len(res) > 1: query = res
    except:
        if "台機電" in query: query = "台積電 2330 股價"
    return {"current_plan": query}

def search_tool_node(state: AgentState):
    """執行實際聯網搜尋"""
    print(f"--- [Node] 執行搜尋: {state['current_plan']} ---")
    try:
        r = requests.get(SEARXNG_URL, params={"q": state["current_plan"], "format": "json"}, timeout=15)
        results = r.json().get('results', [])
        if not results:
            return {"knowledge_base": ["(未找到搜尋結果)"], "search_count": state["search_count"] + 1}
        
        info = "\n".join([f"來源: {res['url']}\n內容: {res.get('content')}" for res in results[:2]])
        return {"knowledge_base": [info], "search_count": state["search_count"] + 1}
    except Exception as e:
        return {"knowledge_base": [f"搜尋失敗: {e}"], "search_count": state["search_count"] + 1}

def planner_node(state: AgentState):
    """判斷資料是否足夠或達到次數上限"""
    print(f"--- [Node] 審查數據 ---")
    if state["search_count"] >= 2 or len("".join(state["knowledge_base"])) > 100:
        return {"next_step": "complete"}
    return {"next_step": "continue"}

def final_answer_node(state: AgentState):
    """整合資料給出最終答案"""
    print("--- [Node] 生成最終回覆 ---")
    context = "".join(state["knowledge_base"])
    prompt = f"根據資料回答問題：{state['input']}\n資料：\n{context}\n請用繁體中文回答。"
    try:
        res = llm.invoke(prompt).content
        if not res or len(res) < 5: raise ValueError("回覆異常")
        return {"final_response": res}
    except:
        # 救援機制：LLM 壞掉時直接給搜尋摘要
        return {"final_response": f"（即時摘要）：\n{context[:500]}..."}

# --- 3. 建構整合圖形結構 ---

workflow = StateGraph(AgentState)

# 新增節點
workflow.add_node("router", router_node)
workflow.add_node("fast_answer", fast_answer_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_tool_node)
workflow.add_node("planner", planner_node)
workflow.add_node("final_answer", final_answer_node)

# 設定連線
workflow.add_edge(START, "router")

# 條件路由
workflow.add_conditional_edges(
    "router",
    lambda x: x["next_step"],
    {"fast_path": "fast_answer", "search_path": "planner"}
)

workflow.add_conditional_edges(
    "planner",
    lambda x: x["next_step"],
    {"continue": "query_gen", "complete": "final_answer"}
)

workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "planner")
workflow.add_edge("fast_answer", END)
workflow.add_edge("final_answer", END)

app = workflow.compile()

# --- 4. 測試運行 ---
if __name__ == "__main__":
    # 列印結構圖 (ASCII)
    try:
        app.get_graph().print_ascii()
    except:
        print("無法列印 ASCII 圖，但流程已就緒。")

    print("\n--- 系統啟動 (輸入 'exit' 退出) ---")
    while True:
        user_input = input("👤 提問: ")
        if user_input.lower() == 'exit': break
        
        for output in app.stream({"input": user_input, "knowledge_base": [], "search_count": 0}):
            for node, data in output.items():
                if "final_response" in data:
                    print(f"\n🎯 AI：\n{data['final_response']}\n")