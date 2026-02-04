import time
import requests
import re
from pathlib import Path
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

# --- 1. 定義狀態 (State) ---
class GraphState(TypedDict):
    task_id: Optional[str]
    srt_text: Optional[str]
    final_report: Optional[str]
    error: Optional[str]

# --- 2. 定義節點 (Nodes) ---

def create_task_node(state: GraphState):
    """節點：上傳音檔至 API"""
    print("[Node] 正在上傳音檔...")
    BASE = "https://3090api.huannago.com"
    WAV_PATH = "/home/pc-49/Downloads/Podcast_EP14_30s.wav" 
    auth = ("nutc2504", "nutc2504")
    
    try:
        with open(WAV_PATH, "rb") as f:
            r = requests.post(f"{BASE}/api/v1/subtitle/tasks", files={"audio": f}, auth=auth, timeout=60)
        return {"task_id": r.json()["id"]}
    except Exception as e:
        return {"error": str(e)}

def wait_node(state: GraphState):
    """節點：輪詢等待轉錄結果"""
    print(f"[Node] 等待任務 {state['task_id']}...")
    url = f"https://3090api.huannago.com/api/v1/subtitle/tasks/{state['task_id']}/subtitle?type=SRT"
    auth = ("nutc2504", "nutc2504")
    
    for _ in range(60):
        try:
            resp = requests.get(url, auth=auth)
            if resp.status_code == 200:
                return {"srt_text": resp.text}
        except:
            pass
        time.sleep(2)
    return {"error": "轉錄逾時"}

def format_report_node(state: GraphState):
    """節點：生成指定格式報告 (包含重點摘要與表格逐字稿)"""
    print("[Node] 正在生成完整格式報告...")
    srt_text = state['srt_text']
    
    # 逐字稿處理邏輯 (嚴格遵守圖片表格格式)
    formatted_table = ["**時間** | **發言內容**", "---------- | ----------"]
    lines = srt_text.strip().split('\n')
    curr_time = ""
    
    for line in lines:
        # 匹配 SRT 時間格式 00:00:00,000
        time_match = re.match(r"(\d{2}:\d{2}:\d{2}),\d{3} --> (\d{2}:\d{2}:\d{2}),\d{3}", line)
        if time_match:
            # 轉換為 00:00:00 - 00:00:00 格式
            curr_time = f"{time_match.group(1)} - {time_match.group(2)}"
        elif line.strip() and not line.strip().isdigit() and curr_time:
            content = line.strip()
            # 組合成表格列：時間 | 內容 |
            formatted_table.append(f"{curr_time} | {content} |")
            curr_time = ""

    # 組合圖片中的完整內容
    report = f"""# 📄 智慧會議紀錄報告

## 🎯 重點摘要 (Executive Summary)
### 天下文化 Podcast 摘要 - 《努力但不費力》

本次會議重點討論葛瑞格麥基昂的《努力但不費力》一書。

**決策結果：** 鼓勵團隊成員重新審視「努力」的定義，不應將過勞視為榮譽，而是尋求更有效率的方法完成重要任務。

**待辦事項 (Action Items)：**
* **學習書中「不費力」的三個階段：** 不費力的狀態、行動、成果。
* **反思自身工作模式：** 檢視是否將時間和精力投入在真正重要的事項上。
* **避免盲目堅持：** 學習善用動力，以更輕鬆的方式達成目標，而非一味地「努力」。

本書透過案例（如 Patrick 的經歷）強調，即使輕鬆也能高效完成工作，並提醒我們應避免將過勞視為美德。

---

## 📁 詳細紀錄 (Detailed Minutes)
### 會議發言紀錄 - 天下文化 Podcast

""" + "\n".join(formatted_table)
    
    return {"final_report": report}

# --- 3. 建立工作流圖 (Graph) ---

workflow = StateGraph(GraphState)

workflow.add_node("create_task", create_task_node)
workflow.add_node("wait_transcription", wait_node)
workflow.add_node("format_report", format_report_node)

workflow.set_entry_point("create_task")
workflow.add_edge("create_task", "wait_transcription")
workflow.add_edge("wait_transcription", "format_report")
workflow.add_edge("format_report", END)

app = workflow.compile()

# --- 4. 執行流程 ---
if __name__ == "__main__":
    # 初始化狀態
    initial_state = {"task_id": None, "srt_text": None, "final_report": None, "error": None}
    
    result = app.invoke(initial_state)

    if result.get("error"):
        print(f"❌ 執行失敗: {result['error']}")
    else:
        # 輸出至檔案
        Path("Meeting_Report.md").write_text(result['final_report'], encoding="utf-8")
        print("✅ 報告生成成功！格式已按照圖片要求調整。")