"""
doc_tasks.py — 文件類型任務設定（所有 parser 的共用權威來源）

新增文件類型：在 DOC_TASKS 列表新增一筆即可，
parse_tenders_azure.py / parse_tenders_local.py / web_app.py 無需修改。
"""

# ─────────────────────────────────────────────────────────────
# 文件類型任務清單
# 每筆欄位說明：
#   src          — 原始上傳目錄（rag_poc/input/）
#   dest         — 解析輸出目錄（ragtest/{type}/input/）
#   doc_type     — 類型識別碼（用於 .doc_type 檔 / API / UI 下拉）
#   description  — 中文說明（UI 顯示用）
#   vision_prompt — 傳給 Vision 模型的圖表描述指令
# ─────────────────────────────────────────────────────────────
DOC_TASKS: list[dict] = [
    {
        "src":          "./rag_poc/input/government",
        "doc_type":     "government",
        "description":  "政府標案",
        "vision_prompt": (
            "這是一張政府標案文件中的圖表。"
            "請詳細描述內容（如設備規格、工程場地願圖、機房佈線機制、驗收程序或金額表），"
            "並專注標案投標資格、違約條款或年限要求等重要標案內容，以繁體中文回答。"
        ),
    },
    {
        "src":          "./rag_poc/input/mixed_tenders",
        "doc_type":     "mixed_tenders",
        "description":  "軍服標案",
        "vision_prompt": (
            "這是一張軍服標案文件中的圖表。"
            "請詳細描述內容（提取服裝結構、迷彩規格、布料技術數據、徽章配戴位置、尺碼標準。），"
            "排除：Logo、通用洗衣符號、情境裝飾圖、文件浮水印。"
            "並專注標案投標資格、違約條款或年限要求等重要標案內容，以繁體中文回答。"
        ),
    },
    {
        "src":          "./rag_poc/input/stip",
        "doc_type":     "stip",
        "description":  "門市銷售指南",
        "vision_prompt": (
            "這是一張門市銷售指南中的圖表。"
            "請判斷此圖是否屬於以下任一類型："
            "『告警、警示、禁止符號、裝飾圖案、卡通人物、動畫角色、應用程式 logo、流程示意 icon、空白圖片、裝飾性插圖、點數回饋示意插圖、檢測報告、認證文件、保險證明、保單文件』。"
            "若是，請只回覆 [SKIP]，不要有任何其他文字。"
            "若否（例如商品照片、方案表格、促銷條件、操作步驟），"
            "請用繁體中文簡短描述圖片內容，不超過150字，不需要零售建議或總結。"
        ),
    },
    # ── 新增文件類型往此列表新增一筆，其他程式碼不需修改 ──────────
]


def get_doc_type_map() -> dict[str, dict]:
    """回傳 doc_type → task 字典，方便快速查詢。"""
    return {t["doc_type"]: t for t in DOC_TASKS}


def get_valid_doc_types() -> list[str]:
    """回傳所有有效的 doc_type 字串清單。"""
    return [t["doc_type"] for t in DOC_TASKS]


# ─────────────────────────────────────────────────────────────
# 陣營目錄路徑輔助函式
# ─────────────────────────────────────────────────────────────

RAGTEST_ROOT = "./ragtest"
CAMP_NAMES   = ("azure", "gemini", "local")   # 定義三大陣營


def get_dest(doc_type: str, camp: str) -> str:
    """解析輸出目錄：./ragtest/{camp}/{doc_type}/input/"""
    return f"{RAGTEST_ROOT}/{camp}/{doc_type}/input"


def get_workspace_root(doc_type: str, camp: str) -> str:
    """GraphRAG 工作區根目錄：./ragtest/{camp}/{doc_type}
    即 graphrag index --root 的參數值。
    settings.yaml / input/ / output/ / cache/ / logs/ 均位於此目錄下。
    """
    return f"{RAGTEST_ROOT}/{camp}/{doc_type}"


def get_index_root(doc_type: str, camp: str) -> str:
    """GraphRAG 索引輸出目錄：./ragtest/{camp}/{doc_type}/output/"""
    return f"{RAGTEST_ROOT}/{camp}/{doc_type}/output"


def model_id_to_camp(model_id: str | None) -> str:
    """
    依 model_id 前綴推導陣營名稱。
      ollama_* → local
      gemini   → gemini
      azure_*  → azure（預設）
    """
    if model_id and model_id.startswith("ollama"):
        return "local"
    if model_id == "gemini":
        return "gemini"
    return "azure"
