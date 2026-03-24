# 檔案四：數據比較報告生成器
import pandas as pd
import os

def generate_report(output_dir, engine_name):
    # GraphRAG 3.0.x 的 artifacts 路徑
    nodes_path = os.path.join(output_dir, "artifacts", "create_final_nodes.parquet")
    
    if not os.path.exists(nodes_path):
        print(f"⚠️ 找不到資料: {nodes_path}")
        return

    df_nodes = pd.read_parquet(nodes_path)
    
    # 統計標案 vs 門市 (假設檔案名稱包含標籤)
    gov_count = len(df_nodes[df_nodes['source_id'].str.contains('government', na=False)])
    rtl_count = len(df_nodes[df_nodes['source_id'].str.contains('retail', na=False)])

    print(f"\n--- {engine_name} POC 分類統計 ---")
    print(f"標案實體數: {gov_count}")
    print(f"門市實體數: {rtl_count}")
    print(f"總關聯邊數: {len(df_nodes)}") # 簡化示範

if __name__ == "__main__":
    # 執行比對
    generate_report("./ragtest/output", "Gemini-GraphRAG-3.0.6")
