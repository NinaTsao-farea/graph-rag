# step 2 
import os
import pandas as pd
from dotenv import load_dotenv
from graphrag.query.context_builder.entity_extraction import EntityExtraction
from graphrag.query.llm.text_utils import chat_completion
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.global_search.search import GlobalSearch

# 1. 載入環境變數與資料
load_dotenv()
INPUT_DIR = "./ragtest/output/artifacts" # 指向索引產出的資料夾

# 讀取圖譜節點與社群摘要 (GraphRAG 3.x 格式)
nodes = pd.read_parquet(os.path.join(INPUT_DIR, "create_final_nodes.parquet"))
entities = pd.read_parquet(os.path.join(INPUT_DIR, "create_final_entities.parquet"))
communities = pd.read_parquet(os.path.join(INPUT_DIR, "create_final_communities.parquet"))

def run_local_search(query):
    """
    【局部搜索】：適合詢問標案的特定細節。
    例如：「本標案的罰則條款為何？」或「UPS 的施工要求是什麼？」
    """
    print(f"\n🔍 局部搜索中: {query}")
    # 這裡會結合向量檢索與圖譜關聯，找出最相關的實體與關係
    # [註：此處簡化了 Context Builder 的建置過程，實務上會調用 graphrag.query 模組]
    context = f"從標案文件中找到的實體數量: {len(entities)}"
    
    # 調用 Gemini 3.0 進行回答
    response = "這部分會根據圖譜中的具體實體點位，精準回答標案第 X 條第 Y 項的內容。"
    return response

def run_global_search(query):
    """
    【全域搜索】：適合詢問跨文件的趨勢或彙整。
    例如：「彙整所有門市的春季促銷重點」或「分析標案中關於資安的所有潛在要求」。
    """
    print(f"\n🌐 全域搜索中: {query}")
    # 全域搜索會讀取 Community Reports (社群摘要)，適合做「總結」
    response = "Gemini 3.0 會掃描所有社群摘要，給出一個高層次的策略建議。"
    return response

if __name__ == "__main__":
    # POC 測試情境一：標案細節
    ans1 = run_local_search("請列出標案中關於『逾期違約金』的具體計算方式。")
    print(f"結果: {ans1}")

    # POC 測試情境二：門市策略
    ans2 = run_global_search("彙整目前所有門市 DM 中提到的 AI 產品促銷方案。")
    print(f"結果: {ans2}")
