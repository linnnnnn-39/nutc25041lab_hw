import os
import glob
import pandas as pd
import uuid
import time
import requests
from qdrant_client import QdrantClient, models

# --- 1. 基本設定 (請確保路徑正確) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

EMBED_API_URL = "https://ws-04.wade0426.me/embed"
LLM_API_URL = "https://ws-05.huannago.com/v1/chat/completions" # 使用您之前的 URL
LLM_MODEL = "google/gemma-3-27b-it" # 使用 Gemma-3
API_KEY = "YOUR_API_KEY" # ⚠️ 請填入您的 API Key

client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "gemma_multi_turn_rag"

# --- 2. 工具函數 ---

def get_embedding(texts: list):
    """取得向量"""
    try:
        res = requests.post(EMBED_API_URL, json={
            "texts": texts, "normalize": True, "task_description": "檢索技術文件"
        }, timeout=60)
        return res.json()["embeddings"]
    except Exception as e:
        print(f"❌ Embedding 錯誤: {e}")
        return None

def call_llm(prompt: str):
    """呼叫 LLM (Gemma-3)"""
    try:
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
        headers = {"Authorization": f"Bearer {API_KEY}"}
        res = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=120)
        return res.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"LLM 呼叫出錯: {e}"

# --- 3. 知識庫初始化 (Step 1/2) ---

def initialize_db():
    print(f"📡 正在初始化 Qdrant: {COLLECTION_NAME}...")
    
    # 偵測檔案 (支援有無 (1) 的情況)
    file_paths = sorted(glob.glob("data_0*(1).txt") or glob.glob("data_0*.txt"))
    
    if not file_paths:
        print("⚠️ 找不到 data_0*.txt，跳過初始化。")
        return

    # 取得向量維度
    sample_vec = get_embedding(["test"])[0]
    dim = len(sample_vec)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
    )

    all_points = []
    for path in file_paths:
        file_name = os.path.basename(path)
        print(f"📖 讀取檔案: {file_name}")
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 簡單切分 (每 400 字一段)
            chunks = [content[i:i+400] for i in range(0, len(content), 350)]
            vectors = get_embedding(chunks)
            for chunk, vec in zip(chunks, vectors):
                all_points.append(models.PointStruct(
                    id=str(uuid.uuid4()), vector=vec,
                    payload={"text": chunk, "source": file_name}
                ))
    
    client.upsert(collection_name=COLLECTION_NAME, points=all_points)
    print(f"✅ 知識庫匯入完成，共 {len(all_points)} 筆。")

# --- 4. 執行任務 (Step 2/2) ---

def run_task():
    input_file = "Re_Write_questions.csv"
    if not os.path.exists(input_file):
        print(f"❌ 找不到 {input_file}")
        return

    df = pd.read_csv(input_file)
    df['answer'] = ""
    df['source'] = ""
    
    # 核心：對話記憶字典
    session_history = {} 

    print("\n🚀 開始處理多輪 RAG 任務...")

    for i, row in df.iterrows():
        cid = str(row['conversation_id'])
        q = str(row['questions'])
        
        # 取得該對話的歷史
        history = session_history.get(cid, "尚未開始對話")

        print(f"\n👉 [Q{i+1}] CID:{cid} | {q}")

        # 關鍵步驟 1：Query Rewrite (查詢重寫)
        rewrite_prompt = (
            f"你是一個搜尋語句優化專家。請參考對話歷史，將「最新問題」改寫成一個語意完整、適合搜尋的獨立句子。\n"
            f"【歷史】：\n{history}\n"
            f"【最新問題】：{q}\n"
            f"請直接輸出改寫後的句子："
        )
        rewritten_q = call_llm(rewrite_prompt)
        print(f"   🔍 改寫後: {rewritten_q}")

        # 關鍵步驟 2：檢索 (使用改寫後的問題)
        q_vec = get_embedding([rewritten_q])[0]
        search_res = client.query_points(
            collection_name=COLLECTION_NAME, query=q_vec, limit=3
        ).points
        
        context = "\n".join([p.payload['text'] for p in search_res])
        source = search_res[0].payload['source'] if search_res else "未知"

        # 關鍵步驟 3：根據檢索結果回答
        final_prompt = (
            f"請根據以下資訊回答問題。若資訊不足請誠實回答無法回答。\n"
            f"【資訊】：\n{context}\n"
            f"【問題】：{q}\n"
            f"回答："
        )
        answer = call_llm(final_prompt)
        print(f"   💡 回答: {answer[:30]}...")

        # 更新該 CID 的歷史
        session_history[cid] = history + f"\n問：{q}\n答：{answer}\n"
        
        # 寫入 DataFrame
        df.at[i, 'answer'] = answer
        df.at[i, 'source'] = source
        time.sleep(0.5)

    df.to_csv("Re_Write_questions_final.csv", index=False, encoding="utf-8-sig")
    print("\n🎉 任務完成！結果儲存至: Re_Write_questions_final.csv")

if __name__ == "__main__":
    initialize_db()
    run_task()