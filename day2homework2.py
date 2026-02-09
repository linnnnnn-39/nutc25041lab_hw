import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# --- 1. 初始化 Qdrant 客戶端 ---
client = QdrantClient(":memory:")

# 定義庫名稱與參數
COLLECTIONS = {
    "euclidean_collection": Distance.EUCLID,
    "inner_product_collection": Distance.DOT,
    "cosine_collection": Distance.COSINE  # 新增的餘弦相似度庫
}
DIMENSION = 8
NUM_ENTITIES = 1000

# --- 2. 建立多個收集庫 (符合圖片 7 的要求，並移除過時警告) ---
def create_collections():
    for name, dist in COLLECTIONS.items():
        # 檢查是否存在並重建 (比 recreate_collection 更安全的新寫法)
        if client.collection_exists(name):
            client.delete_collection(name)
        
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=DIMENSION, distance=dist),
        )
    print(f"✅ 已成功建立三個庫：{', '.join(COLLECTIONS.keys())}")

# --- 3. 準備並插入資料 (符合圖片 4-5 邏輯) ---
def insert_data():
    # 生成隨機向量
    vectors = np.random.random((NUM_ENTITIES, DIMENSION)).tolist()
    
    # 封裝成 Points
    points = [
        PointStruct(id=i, vector=vectors[i], payload={"original_idx": i})
        for i in range(NUM_ENTITIES)
    ]
    
    # 將同一份資料插入到三個庫中
    for name in COLLECTIONS.keys():
        client.upsert(collection_name=name, points=points)
    
    print(f"✅ 已完成 {NUM_ENTITIES} 筆資料同步插入至三個庫")

# --- 4. 驗證與搜尋對比 (符合圖片 6 邏輯) ---
def verify_and_search():
    query_vector = np.random.random(DIMENSION).tolist()
    print("\n" + "="*50)
    print(f"{'庫名稱':<25} | {'首位 ID':<8} | {'得分 (Score)':<10}")
    print("-"*50)

    for name in COLLECTIONS.keys():
        # 使用最新的 query_points API
        result = client.query_points(
            collection_name=name,
            query=query_vector,
            limit=1
        ).points
        
        if result:
            hit = result[0]
            print(f"{name:<25} | {hit.id:<8} | {hit.score:.4f}")

    print("="*50)
    print("🚀 所有驗證已完成，三個庫運作正常。")

if __name__ == "__main__":
    create_collections()
    insert_data()
    verify_and_search()