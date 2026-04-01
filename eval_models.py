import asyncio
import pandas as pd
from datetime import datetime
from graphrag.query.factory import get_local_search_engine, get_global_search_engine
from graphrag.config import load_config_from_yaml

# ================= 配置區域 =================
ROOT_DIR = "./ragtest/stip"
METHOD = "local" # "local" or "global"

# 對應 settings.yaml 中的模型名稱
MODELS = {
    "Flagship": "heavy_model",
    "Mini": "fast_model"
}

QUESTIONS = [
    "亞太客戶如果已經辦理 INP 續約，後續是否還能改走『客戶搬遷』？",
    "請列出「有送市話免費分鐘數」的促銷方案，並說明其「適用對象與限制條件」。",
    # ... 其他問題 ...
]
# ===========================================

async def run_query(engine, question):
    start = asyncio.get_event_loop().time()
    result = await engine.asearch(question)
    duration = round(asyncio.get_event_loop().time() - start, 2)
    return result.response, duration

async def main():
    config = load_config_from_yaml(f"{ROOT_DIR}/settings.yaml")
    results = []

    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/{len(QUESTIONS)}] 正在評測: {q}")
        
        # 建立兩個引擎 (使用不同的 Model ID)
        if METHOD == "global":
            engine_heavy = get_global_search_engine(config, model_id=MODELS["Flagship"])
            engine_mini = get_global_search_engine(config, model_id=MODELS["Mini"])
        else:
            engine_heavy = get_local_search_engine(config, model_id=MODELS["Flagship"])
            engine_mini = get_local_search_engine(config, model_id=MODELS["Mini"])

        # 執行查詢
        ans_heavy, time_heavy = await run_query(engine_heavy, q)
        ans_mini, time_mini = await run_query(engine_mini, q)

        results.append({
            "問題": q,
            "GPT-5.4 (旗艦)": ans_heavy,
            "旗艦耗時(s)": time_heavy,
            "GPT-5.4 mini (輕量)": ans_mini,
            "Mini耗時(s)": time_mini
        })

    # 儲存報告
    df = pd.DataFrame(results)
    df.to_markdown(f"Comparison_Report_{datetime.now().strftime('%m%d_%H%M')}.md", index=False)
    print("🎉 評測報告已產生！")

if __name__ == "__main__":
    asyncio.run(main())
