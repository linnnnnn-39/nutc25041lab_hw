import os
import pandas as pd
import requests
import re
import time  # 新增：用於重試等待
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- 基礎設定 ---
EMBED_API_URL = "https://ws-04.wade0426.me/embed"
QDRANT_URL = "http://localhost:6333"
SUBMIT_URL = "https://hw-01.wade0426.me/submit_answer"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILES = [os.path.join(BASE_DIR, f"data_{i:02d}.txt") for i in range(1, 6)]
QUESTIONS_FILE = os.path.join(BASE_DIR, "questions.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "1111232039_RAG_HW_Final.csv")

client = QdrantClient(url=QDRANT_URL)

def get_embedding(texts):
    """
    整合小批量處理與重試機制，解決 API 超時導致資料缺失的問題
    """
    if not texts:
        return None
    
    text_list = texts if isinstance(texts, list) else [texts]
    all_embeddings = []
    batch_size = 10  # 縮小批量，避免伺服器端超時
    
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        payload = {
            "texts": batch,
            "task_description": "檢索技術文件",
            "normalize": True
        }
        
        # 實作重試機制 (最多 5 次)
        success = False
        for attempt in range(5):
            try:
                # 增加 timeout 到 60 秒
                response = requests.post(EMBED_API_URL, json=payload, timeout=60)
                if response.status_code == 200:
                    all_embeddings.extend(response.json()["embeddings"])
                    success = True
                    break
                else:
                    print(f"API 狀態碼異常: {response.status_code}，嘗試重試...")
            except Exception as e:
                print(f"Embedding 批次 {i//batch_size} 嘗試第 {attempt+1} 次失敗: {e}")
                time.sleep(2) # 等待 2 秒後重試
        
        if not success:
            print(f"!!! 批次 {i//batch_size} 最終處理失敗，這將導致部分資料缺失 !!!")
            
    return all_embeddings if len(all_embeddings) > 0 else None

def get_chunks(text, method):
    if method == "固定大小_500":
        return [text[i:i+500] for i in range(0, len(text), 500)]
    elif method == "滑動視窗_400_100":
        size, overlap = 400, 100
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start+size])
            if start + size >= len(text): break
            start += (size - overlap)
        return chunks
    elif method == "語意切塊_進階":
        # 語意切塊邏輯
        sentences = re.split(r'(?<=[。？！\n])', text)
        refined_chunks = []
        current_chunk = ""
        for s in sentences:
            if len(current_chunk) + len(s) <= 550:
                current_chunk += s
            else:
                if current_chunk: refined_chunks.append(current_chunk)
                current_chunk = s
        if current_chunk: refined_chunks.append(current_chunk)
        return refined_chunks
    return []

def vector_retrieve(question, collection_name):
    # 檢索單一問題時通常不批次，但共用同個 Embedding 函數
    emb_res = get_embedding(question)
    if not emb_res: return {"text": "Error"}
    
    search_result = client.query_points(
        collection_name=collection_name,
        query=emb_res[0],
        limit=1
    ).points
    return {"text": search_result[0].payload["text"]} if search_result else {"text": "None"}

def submit_homework(q_id, answer):
    clean_answer = " ".join(answer.split())
    payload = {"q_id": q_id, "student_answer": clean_answer}
    try:
        response = requests.post(SUBMIT_URL, json=payload, timeout=10)
        return response.json().get("score", 0)
    except: return 0

def main():
    # 1. 準備資料
    docs = []
    for file_path in DATA_FILES:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                docs.append({"source": os.path.basename(file_path), "content": f.read()})
    
    questions_df = pd.read_csv(QUESTIONS_FILE)
    results = []

    # 2. 定義測試組合
    methods = ["固定大小_500", "滑動視窗_400_100", "語意切塊_進階"]
    metrics = {
        "Cosine": Distance.COSINE,
        "Euclid": Distance.EUCLID,
        "Dot": Distance.DOT
    }

    # 先取得一個範例維度
    sample_emb = get_embedding("測試")
    v_size = len(sample_emb[0]) if sample_emb else 4096
    print(f"確認向量維度: {v_size}")

    # 3. 雙層迴圈開始測試
    for method in methods:
        print(f"\n>>> 正在處理切塊方法: {method}")
        all_data_points = []
        
        # 針對當前切塊方法，處理所有文件的向量
        for d in docs:
            chunks = get_chunks(d['content'], method)
            vectors = get_embedding(chunks) # 內部已實作批量處理
            
            if vectors and len(vectors) == len(chunks):
                for i in range(len(chunks)):
                    all_data_points.append({
                        "vector": vectors[i],
                        "text": chunks[i]
                    })
                print(f"   - {d['source']} 處理完成")
            else:
                print(f"   - 錯誤: {d['source']} 資料向量化不完整")

        # 套用到三種距離度量
        for metric_name, dist_type in metrics.items():
            print(f"   --- 測試度量方式: {metric_name} ---")
            col_name = f"col_{re.sub(r'[^a-zA-Z0-9]', '_', method)}_{metric_name.lower()}"
            
            if client.collection_exists(col_name): client.delete_collection(col_name)
            client.create_collection(col_name, vectors_config=VectorParams(size=v_size, distance=dist_type))

            # 批次寫入 Qdrant
            points = [
                PointStruct(id=idx, vector=p["vector"], payload={"text": p["text"]})
                for idx, p in enumerate(all_data_points)
            ]
            client.upsert(collection_name=col_name, points=points)

            # 檢索並評分
            for _, row in questions_df.iterrows():
                q_id, question = row['q_id'], row['questions']
                retrieved = vector_retrieve(question, col_name)
                score = submit_homework(q_id, retrieved['text'])
                results.append({
                    "method": method,
                    "metric": metric_name,
                    "q_id": q_id,
                    "score": score
                })

    # 4. 統計與輸出
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*40)
    print("最終測試報告 (平均分)")
    summary = df.groupby(['method', 'metric'])['score'].mean().unstack()
    print(summary)
    print("="*40)

if __name__ == "__main__":
    main()