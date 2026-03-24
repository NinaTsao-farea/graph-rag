import os
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. 配置 Gemini API
# 請至 https://aistudio.google.com/ 申請免費 Key
GEMINI_API_KEY = "你的_GEMINI_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)

# 2. 初始化 Qdrant 方案三 (本地資料夾模式)
DB_PATH = "./my_rag_poc/qdrant_data"
client = QdrantClient(path=DB_PATH)

COLLECTION_NAME = "retail_promotion"

# 3. 準備 Collection (Gemini Embedding 維度通常為 768)
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print(f"✅ 建立新集合: {COLLECTION_NAME}")

# 4. 模擬門市 DM 文字內容 (這部分未來會由 Docling 自動產出)
dm_content = """
【門市春季限時優惠】
產品：高效能 AI 筆電 Z-Series
活動期間：2026/04/01 - 2026/04/30
促銷內容：滿額現折 $2,000，加贈無線滑鼠乙只。
適用門市：全台直營門市。
"""

# 5. 呼叫 Gemini Embedding API 將文字轉為向量
print("🚀 正在呼叫 Gemini API 生成向量...")
result = genai.embed_content(
    model="models/text-embedding-004", # 2026 年最新穩定版
    content=dm_content,
    task_type="retrieval_document",
    title="Spring_Promo_DM"
)

embedding = result['embedding']
print(f"💡 向量生成成功，維度大小: {len(embedding)}")

# 6. 存入本地 Qdrant
client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        PointStruct(
            id=1,
            vector=embedding,
            payload={
                "category": "retail",
                "doc_type": "promotion_dm",
                "content": dm_content,
                "date": "2026-03-24"
            }
        )
    ]
)

print(f"🎉 資料已成功存入本地路徑: {DB_PATH}")

# 7. 驗證檢索 (搜尋「折價券」相關內容)
print("\n🔍 測試語義檢索: '請問現在有什麼筆電折扣？'")
query_result = genai.embed_content(
    model="models/text-embedding-004",
    content="請問現在有什麼筆電折扣？",
    task_type="retrieval_query"
)

search_results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_result['embedding'],
    limit=1
)

for res in search_results:
    print(f"找到最相關內容 (相似度 {res.score:.4f}):")
    print(res.payload['content'])
