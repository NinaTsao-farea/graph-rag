# 檔案四：數據比較報告生成器
import pandas as pd
import os

def generate_comparison_report(gemini_dir, azure_dir):
    def get_stats(path):
        # 讀取 GraphRAG 核心 Parquet 檔案
        n = pd.read_parquet(os.path.join(path, "artifacts", "create_final_nodes.parquet"))
        e = pd.read_parquet(os.path.join(path, "artifacts", "create_final_relationships.parquet"))
        return len(n), len(e)

    g_n, g_e = get_stats(gemini_dir)
    a_n, a_e = get_stats(azure_dir)

    report = f"""
    # POC 評測結果 (Gemini 3.0 vs Azure GPT-4o)
    | 指標 | Gemini (Google) | Azure (Microsoft) |
    | :--- | :--- | :--- |
    | 實體總數 (Nodes) | {g_n} | {a_n} |
    | 關係連線 (Edges) | {g_e} | {a_e} |
    | 關係密度 (E/N) | {round(g_e/g_n, 2)} | {round(a_e/a_n, 2)} |
    """
    with open("Comparison_Report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("✅ 比較報告已生成：Comparison_Report.md")

if __name__ == "__main__":
    generate_comparison_report("./output_gemini", "./output_azure")
