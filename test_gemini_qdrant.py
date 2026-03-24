# 檔案三：門市 DM 向量測試
import os
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 配置
genai.configure(api_key="你的_GEMINI_API_KEY")
client = QdrantClient(path="./qdrant_data") # 方案三：本地路徑
COLLECTION = "retail_dm_test"

if not client.collection_exists(COLLECTION):
    client.create_collection(
        COLLECTION, 
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

def test_retail_ingest():
    dm_text = "門市 2026 促銷：AI 筆電現折 2000 元，適用全台店面。"
    
    # 使用 2026 最新 embedding 模型
    emb = genai.embed_content(
        model="models/text-embedding-004",
        content=dm_text,
        task_type="retrieval_document"
    )['embedding']
    
    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=101, vector=emb, payload={"text": dm_text})]
    )
    print("✅ 門市測試數據已存入 Qdrant 本地路徑。")

if __name__ == "__main__":
    test_retail_ingest()
