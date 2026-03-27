import subprocess
import time
import pandas as pd
from datetime import datetime

# ================= 配置區域 =================
PROJECT_ROOT = "./projects/Tender_A"  # 您的專案路徑
SEARCH_METHOD = "global"              # 使用 global 或 local
# 模型 ID 必須對應您 settings.yaml 裡的定義
MODELS = {
    "Flagship": "heavy_model",        # GPT-5.4
    "Mini": "fast_model"              # GPT-5.4 mini
}

# 10 個標案專業問題 (可自行修改)
QUESTIONS = [
    "本標案的主要工程範圍為何？",
    "逾期竣工的每日罰則金額與上限是多少？",
    "投標廠商必須具備的特定資格（實績與財力）為何？",
    "驗收流程中，關於複驗次數的限制與期限？",
    "保固期是幾年？保固金比例是多少？",
    "本案是否有規定特定品牌或產地的設備限制？",
    "合約中關於不可抗力因素（如天災）的延期處理邏輯？",
    "付款條件（訂金、期中款、尾款）的比例分配？",
    "資安防護方面有哪些特定的合規要求？",
    "總結本標案對投標者而言最大的風險點為何？"
]
# ===========================================

def run_graphrag_query(question, model_id):
    """執行 GraphRAG 查詢指令並回傳答案與耗時"""
    start_time = time.time()
    cmd = [
        "python", "-m", "graphrag", "query",
        "--root", PROJECT_ROOT,
        "--method", SEARCH_METHOD,
        "--model_id", model_id,
        question
    ]
    
    try:
        # 執行指令並捕捉輸出
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=300)
        duration = round(time.time() - start_time, 2)
        
        if result.returncode == 0:
            # 通常答案會在 stdout 的最後部分
            return result.stdout.strip(), duration
        else:
            return f"❌ 錯誤: {result.stderr}", duration
    except subprocess.TimeoutExpired:
        return "⏰ 錯誤: 查詢超時", 0
    except Exception as e:
        return f"🔥 異常: {str(e)}", 0

def main():
    print(f"🚀 開始自動評測... (模式: {SEARCH_METHOD})")
    results = []

    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/10] 正在提問: {q}")
        
        # 跑旗艦模型
        print(f"   - 正在調用旗艦模型 ({MODELS['Flagship']})...")
        ans_heavy, time_heavy = run_graphrag_query(q, MODELS["Flagship"])
        
        # 跑 Mini 模型
        print(f"   - 正在調用 Mini 模型 ({MODELS['Mini']})...")
        ans_mini, time_mini = run_graphrag_query(q, MODELS["Mini"])
        
        results.append({
            "問題": q,
            "GPT-5.4 (旗艦) 回答": ans_heavy,
            "旗艦耗時(s)": time_heavy,
            "GPT-5.4 mini 回答": ans_mini,
            "Mini耗時(s)": time_mini,
            "速度提升": f"{round(time_heavy/max(time_mini,1), 1)}x"
        })

    # 2. 轉換為 DataFrame 並產出 Markdown 報告
    df = pd.DataFrame(results)
    report_name = f"Evaluation_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    
    with open(report_name, "w", encoding="utf-8") as f:
        f.write(f"# GraphRAG 模型對比報告\n\n")
        f.write(f"- **日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- **專案**: {PROJECT_ROOT}\n")
        f.write(f"- **搜索方式**: {SEARCH_METHOD}\n\n")
        f.write(df.to_markdown(index=False))

    print(f"\n🎉 評測完成！報告已儲存至: {report_name}")

if __name__ == "__main__":
    main()
