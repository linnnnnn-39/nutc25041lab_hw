import os
import pandas as pd
import requests
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- 根據你的最新資訊設定 ---
EMBED_API_URL = "https://ws-04.wade0426.me/embed"
QDRANT_URL = "http://localhost:6333"
SUBMIT_URL = "https://hw-01.wade0426.me/submit_answer"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILES = [os.path.join(BASE_DIR, f"data_{i:02d}.txt") for i in range(1, 6)]
QUESTIONS_FILE = os.path.join(BASE_DIR, "questions.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "1111232039_RAG_HW_01.csv")

client = QdrantClient(url=QDRANT_URL)

def get_embedding(texts):
    """依照你提供的格式發送 POST 請求"""
    payload = {
        "texts": texts if isinstance(texts, list) else [texts],
        "task_description": "檢索技術文件",
        "normalize": True
    }
    try:
        response = requests.post(EMBED_API_URL, json=payload, timeout=20)
        return response.json()["embeddings"]
    except Exception as e:
        print(f"Embedding API 錯誤: {e}")
        return None

def get_chunks(text, method):
    if method == "固定大小_500":
        return [text[i:i+500] for i in range(0, len(text), 500)]
    
    elif method == "滑動視窗_高分版":
        # 你的高分參數：Size 400, Overlap 100
        size, overlap = 400, 100
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start+size])
            if start + size >= len(text): break
            start += (size - overlap)
        return chunks
    
    elif method == "語意切塊_進階":
        # 避免過於破碎，確保每一塊約 500 字左右
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
    # 取得問題向量
    emb_res = get_embedding(question)
    if not emb_res: return {"text": "Error", "source": "error"}
    
    # 執行檢索，嚴格遵守 limit=1
    search_result = client.query_points(
        collection_name=collection_name,
        query=emb_res[0],
        limit=1
    ).points
    
    if search_result:
        return {"text": search_result[0].payload["text"], "source": search_result[0].payload["source"]}
    return {"text": "找不到相關內容", "source": "unknown"}

def submit_homework(q_id, answer):
    clean_answer = " ".join(answer.split())
    payload = {"q_id": q_id, "student_answer": clean_answer}
    try:
        response = requests.post(SUBMIT_URL, json=payload, timeout=10)
        return response.json().get("score", 0)
    except: return 0

def main():
    # 讀取資料
    docs = []
    for file_path in DATA_FILES:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                docs.append({"source": os.path.basename(file_path), "content": f.read()})
    
    questions_df = pd.read_csv(QUESTIONS_FILE)
    results = []
    methods = ["固定大小_500", "滑動視窗_高分版", "語意切塊_進階"]

    # 自動偵測維度
    sample = get_embedding("測試")
    v_size = len(sample[0]) if sample else 384
    print(f"偵測到向量維度: {v_size}")

    for method in methods:
        print(f"\n>>> 執行方法：[{method}] (Limit=1)")
        col_name = f"col_{re.sub(r'[^a-zA-Z0-9]', '_', method)}"
        
        if client.collection_exists(col_name): client.delete_collection(col_name)
        client.create_collection(col_name, vectors_config=VectorParams(size=v_size, distance=Distance.COSINE))

        # 寫入資料
        for d in docs:
            chunks = get_chunks(d['content'], method)
            vectors = get_embedding(chunks)
            if not vectors: continue
            
            points = [
                PointStruct(id=i + hash(chunks[i]) % 10**12, vector=vectors[i], 
                            payload={"text": chunks[i], "source": d['source']})
                for i in range(len(chunks))
            ]
            client.upsert(collection_name=col_name, points=points)

        # 檢索並提交
        for _, row in questions_df.iterrows():
            q_id, question = row['q_id'], row['questions']
            retrieved = vector_retrieve(question, col_name)
            score = submit_homework(q_id, retrieved['text'])
            results.append({"q_id": q_id, "method": method, "score": score})
            print(f"Q{q_id}: {score:.4f}")

    # 存檔與統計
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print("\n--- 測試完成，平均分統計 ---")
    print(df.groupby('method')['score'].mean())

if __name__ == "__main__":
    main()