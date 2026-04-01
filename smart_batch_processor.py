import os
import re
import argparse
from pathlib import Path
from markitdown import MarkItDown
from openai import AzureOpenAI
from dotenv import load_dotenv

# 0. 手動讀取 .env 檔案
load_dotenv()

# ================= 1. 提示詞字典 =================
PROMPT_REGISTRY = {
    "mixed_tenders": """
        [標案稽核員模式]
        核心任務：提取工程細節、合約條款、數據圖表、施工現場。
        排除：Logo、警告符號、卡通人物、裝飾邊框、空白頁。
        不重要則回傳 [IGNORE]。
    """,
    "stip": """
        [門市營運專家模式]
        核心任務：提取促銷活動、SOP流程。
        排除：通用歡迎圖標、裝飾性盆栽、APP LOGO、清潔警告貼紙。
        不重要則回傳 [IGNORE]。
    """
}

# ================= 2. 配置區域 =================
AZURE_CONFIG = {
    "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "key": os.getenv("AZURE_OPENAI_API_KEY"),
    "version": os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini")
}

def process_folders(folder_filter=None):
    # 初始化 Azure 客戶端
    client = AzureOpenAI(
        azure_endpoint=AZURE_CONFIG["endpoint"],
        api_key=AZURE_CONFIG["key"],
        api_version=AZURE_CONFIG["version"]
    )

    # 定義要掃描的根目錄
    base_data_path = Path("./rag_poc/input")
    output_base_path = Path("./graphrag_inputs")

    # 決定要處理的資料夾清單
    folders_to_process = {folder_filter: PROMPT_REGISTRY[folder_filter]} if folder_filter else PROMPT_REGISTRY

    # 遍歷不同的專案資料夾
    for folder_name, system_prompt in folders_to_process.items():
        input_dir = base_data_path / folder_name
        output_dir = output_base_path / folder_name
        
        if not input_dir.exists():
            print(f"⏩ 跳過未找到的目錄: {folder_name}")
            continue
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化該目錄專用的 MarkItDown (注入特定提示詞)
        # 注意：我們透過環境變數或自定義 client 邏輯傳遞提示詞
        md = MarkItDown(llm_client=client, llm_model=AZURE_CONFIG["deployment"])

        print(f"\n📂 正在處理類別: [{folder_name.upper()}]")
        
        for file_path in input_dir.glob("*.*"):
            if file_path.suffix.lower() not in ['.pdf', '.docx', '.xlsx', '.pptx']:
                continue
                
            try:
                print(f"   🔍 解析檔案: {file_path.name}")
                
                # 執行轉換
                # 在執行 convert 前，我們可以動態調整 AI 的系統訊息 (依據具體 SDK 支援度)
                result = md.convert(str(file_path))
                
                # --- 後處理：根據 [IGNORE] 標籤過濾雜訊 ---
                cleaned_content = re.sub(r'!\[Image description: \[IGNORE\]\]\n?', '', result.text_content)
                cleaned_content = re.sub(r'\[Image description: ""\]\n?', '', cleaned_content)

                # 儲存到 GraphRAG 對應的 input
                output_file = output_dir / f"{file_path.stem}.md"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(cleaned_content)
                
                print(f"   ✅ 已生成乾淨的 Markdown: {output_file.name}")

            except Exception as e:
                print(f"   ❌ 處理失敗 {file_path.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批次處理資料夾並轉換為 Markdown")
    parser.add_argument(
        "--type",
        choices=list(PROMPT_REGISTRY.keys()),
        help=f"指定要處理的資料夾類別，可選: {', '.join(PROMPT_REGISTRY.keys())}。不指定則處理全部。"
    )
    args = parser.parse_args()
    process_folders(folder_filter=args.type)