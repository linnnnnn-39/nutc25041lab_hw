from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 1. 初始化
client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer('all-MiniLM-L6-v2')
collections = ["rag_cosine", "rag_euclidean", "rag_dot"]

def run_comprehensive_comparison(query):
    query_vector = model.encode(query).tolist()
    print(f"\n" + "🚀" * 30)
    print(f"🔍 測試問題：【{query}】")
    print("🚀" * 30)

    # 用於統計誰表現更好
    stats = {"Fixed-0": 0, "Sliding-100": 0}

    for col in collections:
        print(f"\n📊 [資料庫度量標準: {col.upper()}]")
        try:
            # 使用 query_points 取代 search 以相容新版本
            response = client.query_points(
                collection_name=col,
                query=query_vector,
                limit=3
            )
            results = response.points
        except Exception as e:
            # 如果還是失敗，嘗試舊版 search 方法
            results = client.search(
                collection_name=col,
                query_vector=query_vector,
                limit=3
            )

        if not results:
            print("  ⚠️ 找不到任何結果")
            continue

        # 紀錄該度量下的第一名
        top_strategy = results[0].payload.get("strategy")
        stats[top_strategy] += 1

        for i, hit in enumerate(results):
            strategy = hit.payload.get("strategy")
            score = hit.score
            content = hit.payload.get("content").replace('\n', ' ')[:70]
            
            # 標註第一名
            medal = "🥇" if i == 0 else f"{i+1}."
            print(f"  {medal} [{strategy}] 分數: {score:.4f} | 內容: {content}...")

    # 最終勝負判定
    print("\n" + "="*50)
    print("🏆 最終切塊策略評比結果排名")
    print("-" * 50)
    for s, count in stats.items():
        print(f"📍 {s} 策略：在三種度量中奪冠 {count} 次")
    
    # 邏輯判斷
    if stats["Sliding-100"] > stats["Fixed-0"]:
        winner = "滑動切塊 (Sliding-100)"
        reason = "它在不同數學度量下都能更準確地捕捉語意，建議用於表格與長文案。"
    elif stats["Sliding-100"] < stats["Fixed-0"]:
        winner = "固定切塊 (Fixed-0)"
        reason = "此問題的關鍵字剛好完整出現在固定區塊中，沒有被切斷。"
    else:
        winner = "兩者平手"
        reason = "在這個問題下，重疊與否不影響模型的判斷。"

    print(f"\n👑 表現較優：{winner}")
    print(f"💡 分析：{reason}")
    print("="*50)

if __name__ == "__main__":
    # 執行兩個不同性質的問題進行比較
    run_comprehensive_comparison("台中科大與中科管理局簽署的合作內容是什麼？")
    run_comprehensive_comparison("台積電 A14 製程預計什麼時候試產？")