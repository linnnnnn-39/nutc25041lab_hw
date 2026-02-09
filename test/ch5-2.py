import json
from typing import Annotated, TypedDict, Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 1. 初始化模型
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="your_api_key_here", # ⬅️ 請填入您的 API Key
    model="google/gemma-3-27b-it",
    temperature=0
)

@tool
def get_weather(city: str):
    """查詢指定城市的天氣。輸入參數 city 必須是城市名稱。"""
    if "台北" in city:
        return "台北下大雨，氣溫 18 度"
    elif "台中" in city:
        return "台中晴天，氣溫 26 度"
    elif "高雄" in city:
        return "高雄多雲，氣溫 30 度"
    else:
        return "資料庫沒有這個城市的資料"

tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatbot_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node_executor = ToolNode(tools)

def router(state: AgentState) -> Literal["tools", "end"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return "end"

workflow = StateGraph(AgentState)
workflow.add_node("agent", chatbot_node)
workflow.add_node("tools", tool_node_executor)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", router, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

app = workflow.compile()

# 先印出圖結構
print("=== Graph Structure ===")
print(app.get_graph().draw_ascii())
print("=======================\n")

# 互動對話
while True:
    user_input = input("User (輸入 exit 結束): ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # 執行 Graph
    state = {"messages": [HumanMessage(content=user_input)]}
    
    # 使用 stream 獲取最後一個節點的輸出
    final_content = ""
    for output in app.stream(state):
        for node_name, value in output.items():
            # 獲取最新的一則訊息
            last_msg = value["messages"][-1]
            if node_name == "agent" and last_msg.content:
                final_content = last_msg.content
    
    print(f"AI: {final_content}\n")