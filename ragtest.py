import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range

# 1. 建立 Qdrant 連接
client = QdrantClient(url="http://localhost:6333")

# 定義 Embedding API 函式 (對應作業要求 3)
def get_embedding(texts):
    embeds = []
    for text in texts:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "llama3", "prompt": text}
        )
        embeds.append(response.json()['embedding'])
    return embeds

# 2. 建立 Collection (對應作業要求 1)
collection_name = "test_collection"
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
)

# 3. 建立五個 Point 並嵌入 VDB (對應作業要求 2 & 4)
data_list = [
    {"id": 1, "text": "人工智能很有趣", "year": 5},
    {"id": 2, "text": "機器學習是未來", "year": 2},
    {"id": 3, "text": "深度學習模擬大腦", "year": 8},
    {"id": 4, "text": "向量資料庫很強大", "year": 10},
    {"id": 5, "text": "Python 是 AI 的首選", "year": 4}
]

# 取得向量並上傳
for item in data_list:
    vector = get_embedding([item["text"]])[0]
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=item["id"],
                vector=vector,
                payload={"text": item["text"], "year": item["year"]}
            )
        ]
    )

# 4. 召回內容 - 相似度搜尋 (對應作業要求 5 & 圖片 772078)
print("--- 執行相似度搜尋 ---")
query_text = ["AI 有什麼好處？"]
query_vector = get_embedding(query_text)[0]

search_result = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=3
)

for point in search_result.points:
    print(f"ID: {point.id}")
    print(f"相似度分數 (Score): {point.score}")
    print(f"內容: {point.payload['text']}")
    print("---")

# 5. 召回內容 - 帶過濾條件的搜尋 (對應圖片 772001)
print("\n--- 執行帶 Filter 的範圍查詢 (year 3-10) ---")
results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key='year',
                range=Range(
                    gte=3,  # 大於等於 3
                    lte=10  # 小於等於 10
                )
            )
        ]
    ),
    limit=5
)

for point in results.points:
    print(f"ID: {point.id}")
    print(f"Payload: {point.payload}")
    print("-" * 30)