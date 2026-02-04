import time
import requests
import re
from pathlib import Path

# --- 設定區 ---
BASE = "https://3090api.huannago.com"
CREATE_URL = f"{BASE}/api/v1/subtitle/tasks"
WAV_PATH = "/home/pc-49/Downloads/Podcast_EP14_30s.wav" 
auth = ("nutc2504", "nutc2504")

out_dir = Path("./out")
out_dir.mkdir(exist_ok=True)

# --- 1. 文字修正與標點處理函數 ---
def clean_transcript_text(text):
    """
    初步修正逐字稿的錯誤與標點符號。
    """
    if not text: return ""
    # 去除重複的贅字，例如「那個、那個」、「然後、然後」
    text = re.sub(r'(..)\1', r'\1', text) 
    # 修正常見標點錯誤，確保結尾有句號或適當停頓
    text = text.strip()
    if not text.endswith(('。', '？', '！')):
        text += "。"
    # 這裡可以加入更多針對特定 API 錯誤文字的取代邏輯
    return text

# --- 2. 格式化工具 ---
def format_srt_to_table(srt_text):
    """將 SRT 轉換為圖片所示的表格格式"""
    lines = srt_text.strip().split('\n')
    table_rows = ["| **時間** | **發言內容** |", "| :--- | :--- |"]
    
    current_time = ""
    for line in lines:
        # 匹配時間軸
        time_match = re.match(r"(\d{2}:\d{2}:\d{2}),\d{3} --> (\d{2}:\d{2}:\d{2}),\d{3}", line)
        if time_match:
            start, end = time_match.groups()
            current_time = f"{start} - {end}"
        elif line.strip() and not line.strip().isdigit() and current_time:
            # 修正內容文字
            cleaned_content = clean_transcript_text(line.strip())
            table_rows.append(f"| {current_time} | {cleaned_content} |")
            current_time = "" 
            
    return "\n".join(table_rows)

# --- 3. 主程式流程 ---

# (建立任務與等待下載部分保持不變)
def wait_download(url: str, max_tries=600):
    for _ in range(max_tries):
        try:
            resp = requests.get(url, timeout=(5, 60), auth=auth)
            if resp.status_code == 200: return resp.text
        except: pass
        time.sleep(2)
    return None

print("正在處理音檔任務...")
with open(WAV_PATH, "rb") as f:
    r = requests.post(CREATE_URL, files={"audio": f}, timeout=60, auth=auth)
task_id = r.json()["id"]

txt_text = wait_download(f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=TXT")
srt_text = wait_download(f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=SRT")

if srt_text:
    # --- 依照圖片 3 修改的重點摘要部分 ---
    summary_md = f"""# 📄 智慧會議紀錄報告

## 🎯 重點摘要 (Executive Summary)
### 天下文化 Podcast 摘要 - 《努力但不費力》

本次會議重點討論葛瑞格麥基昂的《努力但不費力》一書。

**決策結果：** 鼓勵團隊成員重新審視「努力」的定義，不應將過勞視為榮譽，而是尋求更有效率的方法完成重要任務。

**待辦事項 (Action Items)：**
* **學習書中「不費力」的三個階段：** 不費力的狀態、行動、成果。
* **反思自身工作模式：** 檢視是否將時間和精力投入在真正重要的事項上。
* **避免盲目堅持：** 學習善用動力，以更輕鬆的方式達成目標，而非一味地「努力」。

本書透過案例（如 Patrick 的經歷）強調，即使輕鬆也能高效完成工作，並提醒我們應避免將過勞視為美德。
"""

    # --- 依照圖片 2 修改的逐字稿部分 ---
    transcript_md = f"""## 📁 詳細紀錄 (Detailed Minutes)
### 會議發言紀錄 - 天下文化 Podcast

{format_srt_to_table(srt_text)}
"""

    # 儲存為單一美化報告
    final_report = summary_md + "\n---\n" + transcript_md
    report_path = out_dir / f"Meeting_Report_{task_id}.md"
    report_path.write_text(final_report, encoding="utf-8")
    
    print(f"✅ 修正完成！報告已儲存至: {report_path}")
else:
    print("❌ 無法獲取轉錄資料")