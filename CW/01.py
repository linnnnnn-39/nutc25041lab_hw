import requests
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# --- 參數設定 ---
QDRANT_URL = "http://localhost:6333"
EMBED_API_URL = "https://ws-04.wade0426.me/embed"

# --- 函式 0：從 API 取得向量並動態計算維度 ---
def get_embeddings_and_dimension(texts):
    data = {
        "texts": texts,
        "normalize": True,
        "batch_size": 32
    }
    response = requests.post(EMBED_API_URL, json=data)
    
    if response.status_code == 200:
        result = response.json()
        embeddings = result['embeddings']
        detected_dim = len(embeddings[0])
        print(f"✅ API 狀態碼: {response.status_code}")
        print(f"✅ 動態偵測維度: {detected_dim}")
        return embeddings, detected_dim
    else:
        raise Exception(f"❌ API 請求失敗")

# --- 函式 1：初始化環境 ---
def init_qdrant_environment(client, dimension):
    collections_config = {
        "euclidean_collection": Distance.EUCLID,
        "inner_product_collection": Distance.DOT,
        "cosine_collection": Distance.COSINE
    }
    print(f"--- 正在初始化 Qdrant 環境 (維度: {dimension}) ---")
    for name, dist in collections_config.items():
        if client.collection_exists(name):
            client.delete_collection(name)
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dimension, distance=dist),
        )
        print(f"✅ 庫 [{name}] 建立成功")
    return collections_config

# --- 函式 2：資料插入 ---
def insert_data(client, collections, embeddings, texts):
    points = [
        PointStruct(id=i, vector=embeddings[i], payload={"text": texts[i]})
        for i in range(len(embeddings))
    ]
    for name in collections.keys():
        client.upsert(collection_name=name, points=points)
    print(f"\n✅ 已將 {len(embeddings)} 筆向量插入至庫中")

# --- 函式 3：全量排名搜尋 (5筆全部排序) ---
def search_and_rank_all(client, collections, query_vector, query_text):
    print("\n" + "="*70)
    print(f"📥 【查詢基準】: {query_text}")
    print(f"📊 【排名邏輯】: 計算 5 筆資料的相似度並分出排名 (從最接近到最遠)")
    print("="*70)
    
    for name in collections.keys():
        print(f"\n🔍 庫名稱: {name}")
        print(f"{'排名':<6} | {'ID':<4} | {'相似度得分':<12} | {'對應文本'}")
        print("-" * 65)
        
        # 將 limit 設為 5，確保 5 筆都出來排名
        search_result = client.query_points(
            collection_name=name,
            query=query_vector,
            limit=5 
        ).points
        
        for i, hit in enumerate(search_result, 1):
            text = hit.payload.get("text", "未知")
            print(f"No.{i:<4} | {hit.id:<4} | {hit.score:<12.4f} | {text}")

# --- 主程式執行區塊 ---
if __name__ == "__main__":
    client = QdrantClient(url=QDRANT_URL)
    
    # 你的五個評分對象
    my_texts = [
        "人工智慧很有趣", 
        "深度學習的應用", 
        "機器學習初探", 
        "今天天氣真好", 
        "gta6延期幾次"
    ]
    
    try:
        # 1. 取得向量與維度
        embeddings, current_dim = get_embeddings_and_dimension(my_texts)
        
        # 2. 初始化
        colls = init_qdrant_environment(client, current_dim)
        
        # 3. 插入資料 (帶上文本標籤方便閱讀排名)
        insert_data(client, colls, embeddings, my_texts)
        
        # 4. 比較 5 筆資料的排名 (以第 0 筆為查詢基準)
        search_and_rank_all(client, colls, embeddings[0], my_texts[0])
        
        print(f"\n🚀 5 筆資料的比較與排名已完成！")
        
    except Exception as e:
        print(f"❌ 發生錯誤：{e}")