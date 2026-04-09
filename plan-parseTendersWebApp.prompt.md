# Plan & Implementation: 標案文件解析 Web App

> **最後更新：2026-04-08**（依實際程式碼同步）

## TL;DR

以 **FastAPI + 原生 HTML/JS** 建立四 Tab 單頁應用，整合：
- `parse_tenders_azure.py` — Azure Document Intelligence + Vision 文件解析
- `index_tenders.py` — GraphRAG 知識圖譜索引建立
- `query_tenders.py` — GraphRAG 知識圖譜查詢（含場景式 System Prompt）

提供上傳、解析、建立索引、查詢四個核心操作，並實作衝突備份、AI 模型選擇（解析 / 索引 / 查詢三層獨立）、SSE 即時串流進度。

---

## 決策（Decisions）

| 項目 | 決策 |
|---|---|
| Framework | FastAPI + 原生 HTML/JS（無前端 build 步驟） |
| Doc type 選擇 | 上傳時指定，對應 `DOC_TASKS` 的 `vision_prompt` |
| Vision AI 模型 | 解析時手動選擇；從 `.env` 自動偵測可用模型 |
| 索引 AI 模型 | 索引時手動選擇；settings 模板檔決定可用性 |
| 查詢 AI 模型 | 查詢時手動選擇；`QUERY_MODELS` 從 `.env` 自動掃描 |
| 備份範圍 | 整個衝突資料夾移至 backup（完整版本保留） |
| 進度顯示 | SSE 即時 Log 串流（`redirect_stdout` + `queue.Queue`） |
| Index 設定 | 無 `settings.yaml` 時自動從對應模板複製 |
| 場景式查詢 | 每種 `doc_type` 有獨立 `SYSTEM_PROMPT`（顧問角色 × 2 + 業務教練 × 1 + 通用） |

---

## 架構（Architecture）

```
web_app.py                ← FastAPI 主程式
parse_tenders_azure.py    ← Azure DI + Vision 解析核心
index_tenders.py          ← GraphRAG 索引包裝器
query_tenders.py          ← GraphRAG 查詢包裝器
static/
  index.html              ← 單頁 UI（四個 Tab）
  app.js                  ← 前端邏輯
  style.css               ← 深色系樣式
```

### 路徑規則

| 用途 | 路徑 |
|---|---|
| 上傳目錄 | `./rag_poc/input/{folder_name}/` |
| 解析輸出 | `./ragtest/{folder_name}/input/` |
| 備份格式 | `{資料夾名稱}_old_{yymmdd}_vXX`（同層） |
| 文件類型標記 | `rag_poc/input/{folder_name}/.doc_type`（純文字） |

---

## API 端點總覽

| Method | Path | 說明 |
|---|---|---|
| GET | `/api/folders` | 列出 `rag_poc/input/` 下非備份子目錄 |
| GET | `/api/folders/{name}/files` | 列出指定上傳資料夾的檔案 |
| POST | `/api/upload` | 上傳文件（含衝突備份邏輯） |
| GET | `/api/models` | 列出可用 Vision AI 模型（解析用） |
| GET | `/api/parse-folders` | 列出含 `.doc_type` 的可解析資料夾 |
| GET | `/api/parse-folders/{name}/files` | 列出可選擇的單一解析檔案 |
| POST | `/api/parse/stream` | 文件解析 SSE 串流 |
| GET | `/api/index-folders` | 列出含 `.md` 的可建立索引資料夾 |
| GET | `/api/index-models` | 列出可用的 GraphRAG 索引模型開隢 |
| POST | `/api/index/stream` | GraphRAG 索引建立 SSE 串流 |
| GET | `/api/query-types` | 列出已建立索引的可查詢資料類型 |
| GET | `/api/query-models` | 列出可用的 GraphRAG 查詢模型（`QUERY_MODELS`） |
| POST | `/api/query` | GraphRAG 查詢 SSE 串流（含 `model_id`） |

---

## 實作細節

### 1. 備份輔助函式（`web_app.py`）

```python
_BACKUP_PATTERN = re.compile(r"^(.+)_old_\d{6}_v(\d+)$")

def _get_backup_path(base: Path, folder_name: str) -> Path:
    today = _today_str()                    # yymmdd，例如 260401
    prefix = f"{folder_name}_old_{today}_v"
    # 掃描同層已存在的版本號，取最大值 +1
    ...
    return base / f"{prefix}{next_ver}"

def _backup_if_exists(base: Path, folder_name: str) -> str | None:
    # 若目標存在 → shutil.move 至備份路徑，回傳備份路徑字串
    ...
```

觸發時機：
- **上傳**：新上傳的檔名與 `rag_poc/input/{folder_name}/` 內既有檔名衝突時
- **解析**：`ragtest/{folder_name}/` 已存在時（整個資料夾備份）

---

### 2. Vision AI 模型動態載入（`parse_tenders_azure.py`）

`.env` 使用 **帶編號後綴** 的方式擴充部署，程式碼自動掃描，不需額外修改：

```ini
# .env — 無後綴為第 1 個，_2 為第 2 個，依此類推
AZURE_OPENAI_DEPLOYMENT=gpt-5.4
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/

AZURE_OPENAI_DEPLOYMENT_2=gpt-5.4-mini
# 未填時自動共用第 1 個的 KEY / ENDPOINT
AZURE_OPENAI_API_KEY_2=...         # 可選
AZURE_OPENAI_ENDPOINT_2=...        # 可選
AZURE_OPENAI_API_VERSION_2=...     # 可選

# 各部署的 Vision 定價（可選，未設則使用預設值）
AZURE_OPENAI_PRICE_INPUT=0.75      # 輸入 token / 百萬 USD（預設）
AZURE_OPENAI_PRICE_OUTPUT=4.50     # 輸出 token / 百萬 USD（預設）
AZURE_OPENAI_PRICE_INPUT_2=...     # 部署 _2 的輸入定價（可選）
AZURE_OPENAI_PRICE_OUTPUT_2=...    # 部署 _2 的輸出定價（可選）

GEMINI_API_KEY=...                  # 填入後 Gemini 即自動可用
GEMINI_MODEL=gemini-3.0-flash
GEMINI_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_PRICE_INPUT=0.10            # Gemini 輸入 token 定價（預設）
GEMINI_PRICE_OUTPUT=0.40           # Gemini 輸出 token 定價（預設）
```

掃描邏輯（`_load_azure_models`）：
- 無後綴開始掃描，遇到缺口（例如沒有 `_3`）即停止
- 每個部署的 Key/Endpoint/Version 若未設 `_N` 版本，退回無後綴預設值
- 每個部署可獨立設定 `AZURE_OPENAI_PRICE_INPUT[_N]` / `AZURE_OPENAI_PRICE_OUTPUT[_N]` 定價
- 掃描完成後附加 Gemini 項目（`openai_compat` 類型，使用 OpenAI SDK 相容介面）
- `GET /api/models` 自動回傳可用清單，`api_key` 含 placeholder 文字時標示 `available: false`

新增模型方式（**不需修改程式碼**）：
```ini
# 僅需在 .env 新增一組 _N 後綴
AZURE_OPENAI_DEPLOYMENT_3=my-new-model
AZURE_OPENAI_API_KEY_3=...
```

啟動時 console 顯示：
```
👉 已載入 Vision 模型: azure_gpt-5.4, azure_gpt-5.4-mini, gemini（預設: azure_gpt-5.4）
```

---

### 3. 文件解析流程（`POST /api/parse/stream`）

```
前端選擇 folder_name + file_name? + model_id
  ↓
web_app._do_parse()
  ├─ 讀取 .doc_type → 取得 vision_prompt
  ├─ build_vision_client(model_id) → (client, deployment, label)
  ├─ 若 ragtest/{folder_name}/ 已存在 → _backup_if_exists()
  ├─ DocumentIntelligenceClient 初始化
  └─ parse_tenders_azure.process_file() × N 個檔案
       ├─ Azure DI prebuilt-layout → Markdown
       ├─ 圖表裁切 (PyMuPDF) → Vision 模型描述
       └─ 輸出 .md 至 ragtest/{folder_name}/input/
  → _print_stats(model_label) 輸出費用摘要
```

SSE 機制：
- `ThreadPoolExecutor(max_workers=4)` 負責執行（支援 parse + index 並行）
- `contextlib.redirect_stdout` + `queue.Queue` 捕捉所有 `print`
- `_parsing_lock: set[str]` 防止同資料夾重複觸發（回傳 409）
- 心跳 `: heartbeat` 每 200ms 一次，避免 SSE 連線逾時

---

### 4. GraphRAG 索引建立流程（`POST /api/index/stream`）

#### 4a. 模型開隢設定（`web_app.py` — `INDEX_MODEL_PROFILES`）

```python
def _load_index_model_profiles() -> dict[str, dict]:
    # 自動掃描 .env 中 AZURE_OPENAI_DEPLOYMENT[_N]，每個部署獨立建立一筆 azure_{dep}
    # settings 模板選擇順序：AZURE_INDEX_SETTINGS[_N] > ragtest/settings{_N}.yaml > ragtest/settings.yaml
    ...
    profiles["gemini"] = {
        "label":    "Google Gemini",
        "template": Path("./settings_gemini.yaml"),
    }
```

- `GET /api/index-models` 回傳可用清單：`template` 檔存在 → `available: true`
- 建立索引前依選擇自動將對應模板複製至 `ragtest/{folder_name}/settings.yaml`
- 新增索引模型：在 `.env` 新增 `AZURE_OPENAI_DEPLOYMENT_N`（自動產生） 或新建 settings 模板檔

**Settings 模板對照：**

| model_id | 模板來源 | LLM | Embedding |
|---|---|---|---|
| `azure_{dep}` | `ragtest/settings.yaml` | Azure `gpt-5.4-mini` | Azure `text-embedding-3-large` |
| `gemini` | `settings_gemini.yaml` | Gemini `gemini-3-flash` (OpenAI compat) | Gemini `text-embedding-004` |

> `ragtest/settings.yaml` 已設定 Azure rate-limit 保護：`concurrent_requests: 4`、`tokens_per_minute: 40000`、`max_retries: 10`。

#### 4b. 執行流程

```
前端選擇 folder_name + model_id
  ↓
web_app._run_index()
  ├─ 依 model_id 從 INDEX_MODEL_PROFILES 取得模板
  ├─ shutil.copy(template → ragtest/{folder_name}/settings.yaml)
  ├─ subprocess.Popen(["python", "-m", "graphrag", "index", "--root", root])
  ├─ 「鶓默債測器」執行緒（daemon）：30 秒無輸出 → 推送進度提示
  └─ 完成後 index_tenders._print_index_stats() 輸出費用摘要
       ├─ 讀取 output/stats.json 取得 workflow 耗時 / 文件數
       ├─ 解析最新 logs/*.log 取得 LLM + Embedding token 累積統計
       └─ 依 _load_index_pricing() 計算各部署定價
```

**定價環境變數（`index_tenders.py`）：**

```ini
AZURE_OPENAI_PRICE_INPUT=2.50     # Azure LLM 輸入 token / 百萬（index 用，可獨立設定）
AZURE_OPENAI_PRICE_OUTPUT=15.00   # Azure LLM 輸出 token / 百萬
AZURE_EMBED_PRICE=0.13            # Azure Embedding / 百萬（所有部署共用）
GEMINI_EMBED_PRICE=0.00           # Gemini Embedding（依官方定價調整）
```

> 注意：`AZURE_OPENAI_PRICE_INPUT[_N]` 同一鍵值在解析與索引模組中共用，調整一次即生效。

`GET /api/index-folders` 回傳各資料夾狀態：

| 欄位 | 說明 |
|---|---|
| `md_count` | `ragtest/{name}/input/*.md` 檔案數 |
| `has_settings` | 是否有 `settings.yaml` |
| `indexed` | `ragtest/{name}/output/lancedb/` 是否非空 |
| `indexing` | 是否正在建立索引中 |

`_indexing_lock: set[str]` 防止同資料夾重複觸發（回傳 409）

**SSE sentinel 規格（索引 vs. 解析）：**

| 模組 | sentinel | `_sse_*_generator` 處理 |
|---|---|---|
| `_run_index` | `None` → `event: done` / `("__done__", rc)` tuple → done 或 error | 兩種格式皆相容 |
| `_run_parse` | `("__done__", returncode)` tuple（⚠️ `returncode` 於 except 時設定，正常完成為 `UnboundLocalError`，sentinel 改以 `None` 實際觸發） | 只處理 `None` |

> **已知問題**：`_run_parse` 的 `finally` 區塊嘗試讀取未初始化的 `returncode`，若解析成功則 `NameError` 使 sentinel 永不放入佇列；`_sse_generator` 在 stream 正常結束後仍會輸出 `✅ 解析完成！`（靠 `done` 判讀迴圈結束），實際可用。後續可在 `_run_parse` 加 `returncode = 0` 初始化修正。

#### 4c. 鶓默債測器（解決長時間無輸出問題）

GraphRAG 在 Embedding / Community Report 階段可能長达數分鐘無任何輸出。從兩層保護：

| 層次 | 機制 | 說明 |
|---|---|---|
| SSE 連線 | `heartbeat` 每 200ms | `_sse_index_generator` 的 `queue.Empty` 分支推送 `: heartbeat` |
| 輸出內容 | 鶓默債測器執行緒 | `_silence_ticker()`：超過 30 秒無輸出即主動推送進度字串 |

```python
# _run_index 內的鶓默債測器
_last_output_time = [time.time()]
_proc_done = threading.Event()

def _silence_ticker():
    SILENCE_THRESHOLD = 30   # 秒
    TICK_INTERVAL = 10       # 秒
    while not _proc_done.wait(timeout=TICK_INTERVAL):
        idle = time.time() - _last_output_time[0]
        if idle >= SILENCE_THRESHOLD:
            elapsed = int(time.time() - t0)
            log_queue.put(f"⏳ GraphRAG 遠行中… ({elapsed//60}m{elapsed%60:02d}s，{idle:.0f}s 無輸出)")

threading.Thread(target=_silence_ticker, daemon=True).start()
```

---

### 5. GraphRAG 查詢流程（`POST /api/query`）

#### 5a. 查詢 AI 模型動態載入（`query_tenders.py` — `QUERY_MODELS`）

`query_tenders.py` 同樣掃描 `.env` 中的 `AZURE_OPENAI_DEPLOYMENT[_N]`，建立 `QUERY_MODELS` 字典。每個模型包含：
- `chat` — `ModelConfig`（`graphrag_llm`）
- `embed` — `ModelConfig`（共用同一 Azure Embedding 部署）
- `input_price_per_1m` / `output_price_per_1m` — 查詢定價

`GET /api/query-models` 回傳清單；前端查詢 Tab 有獨立 AI 模型下拉，`POST /api/query` 接受 `model_id`。已設定有效 Gemini Key 時 `gemini` 自動加入。

#### 5b. 場景式 System Prompt（`SYSTEM_PROMPTS`）

`build_engines(input_dir, doc_type, model_id)` 依 `doc_type` 選擇場景 prompt：

| doc_type | 角色 | 特色 |
|---|---|---|
| `government` | 資深標案法律顧問 | 嚴謹、必標條款數、禁虛構 |
| `stip` | 遠傳門市銷售教練 | 三情境（操作 / KPI / 代客詢問），禁內部術語對外輸出 |
| `mixed_tenders` | 軍事採購合約顧問 | 必表格化期限 / 金額，引用條款節次 |
| `default` | 文件分析助理 | 通用，嚴禁虛構 |

#### 5c. GraphRAG 3.x Parquet 結構

GraphRAG 3.x 輸出目錄多了 `communities.parquet`（社群層次），`build_engines` 現讀取：

```python
entity_df       = pd.read_parquet(f"{input_dir}/entities.parquet")
community_df    = pd.read_parquet(f"{input_dir}/communities.parquet")       # NEW
report_df       = pd.read_parquet(f"{input_dir}/community_reports.parquet")
relationship_df = pd.read_parquet(f"{input_dir}/relationships.parquet")
text_unit_df    = pd.read_parquet(f"{input_dir}/text_units.parquet")
```

#### 5d. 查詢執行流程

```
前端選擇 doc_type + mode + query + show_context + model_id
  ↓
web_app._stream()
  │  驗證 lancedb/ 非空—後不存在回傳 event: error
  ├─ run_in_executor → query_tenders.build_engines(input_dir, doc_type, model_id)
  │    載入 entities / communities / community_reports / relationships / text_units parquet
  │    依 doc_type 套用 SYSTEM_PROMPTS[doc_type]
  ├─ run_local_search() 或 run_global_search()
  ├─ 按行推送查詢結果
  └─ redirect_stdout → _print_query_stats() 輸出 Token 費用（含模型定價）
```

`GET /api/query-types` 回傳清單：

| 欄位 | 說明 |
|---|---|
| `id` | `DOC_INDEX_ROOTS` 鍵値（government / stip / mixed_tenders / default）|
| `label` | 中文顯示名稱 |
| `available` | `{output_dir}/lancedb/` 非空則 true |

#### 搜索模式區分

| 模式 | 適用情境 |
|---|---|
| `local` | 標案特定細節（罰則條款、規格、金額表）|
| `global` | 跨文件趨勢彙整（所有門市春季促銷重點）|

---

### 6. 前端四 Tab 結構（`static/index.html` + `app.js`）

#### Tab 1：☁️ 上傳文件
- 資料夾名稱輸入（正則驗證：字母/數字/底線/連字號/中文）
- 文件類型下拉（對應 `DOC_TASKS`：government / mixed_tenders / stip）
- 拖曳 / 點擊多檔上傳，已選檔案可個別移除（`Map<filename, File>` 維護唯一清單）
- 上傳後展示資料夾內容；衝突時顯示備份路徑提示

#### Tab 2：⚙️ 解析文件
- 資料夾下拉（呼叫 `GET /api/parse-folders`）
- 檔案下拉：「全部（批次解析）」+ 單一檔案
- AI 模型下拉（呼叫 `GET /api/models`）：不可用的模型 disabled + ⚠️ 提示
- 解析按鈕 → `POST /api/parse/stream` SSE 連線
- 即時 Log 串流（auto-scroll）+ 中止按鈕

#### Tab 3：🗂️ 建立索引
- 資料夾下拉（呼叫 `GET /api/index-folders`）
  - ✅ 已索引 / 🆕 尚未索引 視覺區分
  - ⚠️ 無 settings.yaml 警示（系統會自動從模板複製）
- **AI 模型下拉**（呼叫 `GET /api/index-models`）：不可用模型（settings 模板檔不存在）disabled + ⚠️ 提示
- 開始建立索引 → `POST /api/index/stream` SSE 連線
- 即時 Log 串流（auto-scroll）+ 中止按鈕
- 完成後自動重整資料夾清單（更新 ✅ 狀態）
- **結果狀態**：`event: done` → ✅ 成功；`event: error` → ❌ 失敗

#### Tab 4：🔎 查詢
- **資料類型下拉**（呼叫 `GET /api/query-types`）：`lancedb/` 非空才可選，尚未索引者 disabled + ⚠️
- **搜索模式**：局部搜索（適合標案細節）/ 全域搜索（跨文件彙整）
- **AI 模型下拉**（呼叫 `GET /api/query-models`）：不可用模型 disabled + ⚠️ 提示
- **查詢問題輸入框**（`<textarea>`）— 任一欄位空值時查詢按鈕自動 disabled
- **顯示 context_data 選項**：勾選後 Log 內容包含參考內容段落
- 查詢按鈕 → `POST /api/query` SSE 連線（`model_id` 隨請求傳送）
- 即時結果串流（auto-scroll）+ 中止按鈕
- 查詢完成後輸出 Token 費用摘要（含模型定價）

#### 共用 SSE 解析邏輯（`fetch` + `ReadableStream`）

> 原生 `EventSource` 僅支援 GET，故改用 `fetch` 手動解析 SSE stream：

```js
const res = await fetch('/api/parse/stream', {
  method:  'POST',
  headers: { 'Content-Type': 'application/json' },
  body:    JSON.stringify({ folder_name, file_name, model_id }),
  signal:  abortController.signal,
});
const reader  = res.body.getReader();
const decoder = new TextDecoder();
let   buffer  = '';

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  for (const line of buffer.split('\n')) {
    if (line.startsWith('data: ')) appendLog(line.slice(6));
  }
}
```

---

## 相依套件

```txt
# requirements.txt 核心依賴（實際版本）
graphrag>=3.0.6
graphrag-vectors>=0.1.0       # GraphRAG 3.x LanceDB 向量資料庫介面
graphrag-llm                  # ModelConfig / create_completion / create_embedding

fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9

openai>=1.30.0
python-dotenv>=1.0.0
pandas>=2.1.0
pyarrow>=14.0.0

pymupdf>=1.24.0               # Azure DI 版本：PDF 頁面裁切圖表
azure-ai-documentintelligence
azure-core

google-generativeai>=0.4.0    # Gemini SDK（備用）
```

---

## 啟動方式

```bash
# 進入虛擬環境
.venv\Scripts\Activate.ps1

# 啟動（開發模式，含 hot-reload）
uvicorn web_app:app --reload --port 8000

# 瀏覽器開啟
http://localhost:8000
```

---

## 驗證步驟

1. 啟動後瀏覽 `http://localhost:8000`，確認四個 Tab 正常顯示
2. **上傳**：上傳 1 個 PDF 至新資料夾 `test_folder`，確認 `rag_poc/input/test_folder/` 建立
3. **備份**：重複上傳同名檔 → 確認 `test_folder_old_{yymmdd}_v1` 出現
4. **解析**：切換至「解析文件」Tab，選 `test_folder`，選擇 Vision AI 模型，點解析 → Log 串流即時出現；完成後 `ragtest/test_folder/input/` 有 `.md` 輸出
5. **再次解析**：確認舊輸出備份至 `ragtest/test_folder_old_{yymmdd}_v1`
6. **索引**：切換至「建立索引」Tab，選 `test_folder`，選擇索引 AI 模型，點開始 → Log 串流即時出現；完成後資料夾下拉的 ✅ 狀態更新
7. **查詢**：切換至「🔎 查詢」Tab，選擇已索引的類型（`stip`）、選擇查詢 AI 模型、選局部搜索，輸入問題，點查詢 → Log 為即時結果串流，最後顯示 Token 費用摘要
8. **多模型（Vision）**：在 `.env` 新增 `AZURE_OPENAI_DEPLOYMENT_2`，重啟後解析 Tab 的模型下拉應自動出現新選項
9. **多模型（查詢）**：同上，查詢 Tab 的模型下拉也應同步出現

---

## 注意事項

1. **SSE 斷線**：前端使用 `AbortController` 中止連線；中止後後端執行緒會繼續跑完當前檔案（無法強制中斷 Azure DI 呼叫），待完成後自動釋放 lock
2. **索引時間**：GraphRAG index 有兩層保護：（1）SSE `heartbeat` 每 200ms 保持連線；（2）`_silence_ticker` daemon 執行緒視子行程鶓默時間，超過 30 秒自動往 SSE 推送進度提示
3. **費用估算（三層）**：
   - **解析**：`parse_tenders_azure.py` —— Azure DI 頁數費用 + Vision token 費用，定價常數於模組頂部或 `.env`
   - **索引**：`index_tenders.py` —— 解析 log 的 LLM + Embedding 累積 token，定價由 `_load_index_pricing()` 讀取 `.env`
   - **查詢**：`query_tenders.py` —— LLM 呼叫 token，定價由 `QUERY_MODELS` 各項設定
   - 所有費用為估算值，實際請查閱 Azure Portal 帳單
4. **settings.yaml 模板**：`ragtest/settings.yaml`（Azure）與 `settings_gemini.yaml`（Gemini）為索引用模板；索引啟動前自動複製至 `ragtest/{folder_name}/settings.yaml`
5. **並行限制**：`ThreadPoolExecutor(max_workers=4)` 支援同時解析與索引，但同一資料夾不允許重複觸發（lock 保護）
6. **`_run_parse` 已知問題**：`finally` 區塊的 `returncode` 在非例外路徑時是未定義（`UnboundLocalError`），不影響解析結果正常推送，但 sentinel 改由 `_sse_generator` 的迴圈自然結束觸發。修正方式：在 `try` 區塊前加 `returncode = 0`
7. **GraphRAG 3.x**：`query_tenders.py` 讀取 `communities.parquet`（GraphRAG 3.x 新增），舊版索引輸出目錄若缺此檔將報錯，需重新執行索引

---

## POC 多陣營比較評估規劃

### 背景與約束

本專案為 POC 階段，核心目的是**系統性比較三陣營的優缺點**，作為正式專案架構選型依據。
評估過程受以下環境約束限制：

| 陣營 | 可用機器 | 限制原因 |
|---|---|---|
| **Azure** | 公司環境筆電 | Azure 訂閱 / VPN / 公司憑證 |
| **Gemini** | 公司環境筆電 ✅ 或個人機器 ✅ | 只需 API Key，無 VPN 需求 |
| **Local/Ollama** | 個人機器（有 GPU） | Ollama 須本機安裝，與公司環境隔離 |

---

### 決策：**同一 Git Repo，不同機器跑不同陣營**

**不分專案的理由：**

「分開」的需求來自**執行環境不同**，不是**程式碼不同**。若分成兩個專案：
- 共用的備份邏輯、SSE 串流、評測問題集、`compare_results.py` 需同步維護兩份
- 評估結果格式若不一致，比較報告難以合併
- 後續正式專案選型後，整合工作量反而更大

**正確的分離層次是 `.env` 檔，不是 repo：**

```
graph-rag/                 ← 同一個 git repo
├── .env                   ← gitignore，每台機器各自維護
├── .env.example           ← 文件化所有可用變數（commit 進 repo）
└── （所有 .py / .yaml 完全共用）
```

---

### 兩台機器的 `.env` 差異

**公司環境筆電（跑 Azure + Gemini）：**

```ini
# ── Azure 陣營 ──────────────────────────────
AZURE_COGNITIVE_SERVICES_ENDPOINT=https://...
AZURE_COGNITIVE_SERVICES_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-5.4
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...

# ── Gemini 陣營 ──────────────────────────────
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.0-flash

# ── Local 陣營：不設定，UI 不顯示 ─────────
# OLLAMA_BASE_URL 未設定 → ollama 不出現在 model 下拉
```

**個人機器（跑 Local + 可選 Gemini）：**

```ini
# ── Azure 陣營：不設定，UI 不顯示 ─────────
# AZURE_OPENAI_DEPLOYMENT 未設定 → azure 不出現

# ── Gemini 陣營（可選，用來對照）────────────
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.0-flash

# ── Local/Ollama 陣營 ────────────────────────
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_QUERY_MODEL=qwen2.5:14b
OLLAMA_EMBED_MODEL=nomic-embed-text
PARSE_BACKEND=local        # 解析後端：azure_di | local（docling）
```

> **Gemini 是「橋梁陣營」**：API Key 在兩台機器都能跑，可作為跨機器結果校驗基準。

---

### POC 比較實驗流程

```
Phase 1  公司筆電                Phase 2  個人機器
─────────────────────            ─────────────────────
選同批測試 PDF（stip 2份）        clone 同一 repo
↓                                ↓
Azure 陣營：解析→索引             Local 陣營：解析→索引
Gemini 陣營：解析→索引            Gemini 陣營：解析→索引（可選，交叉驗證）
↓                                ↓
跑 10 題標準問答，記錄答案        跑同樣 10 題標準問答
↓                                ↓
commit ragtest/{camp}/output/    commit ragtest/{camp}/output/
到共用 branch 或 USB 同步         到共用 branch 或 USB 同步
              ↓
        Phase 3  任一機器
     ─────────────────────
     python compare_results.py
     → 輸出三陣營對比報告
       Comparison_Report_{date}.md
```

---

### Repo 目錄結構（評估期間）

```
ragtest/
├── stip_azure/           ← Azure 陣營跑出的索引結果
│   ├── input/            （解析後 .md）
│   └── output/           （GraphRAG parquet）
├── stip_gemini/          ← Gemini 陣營
│   ├── input/
│   └── output/
├── stip_local/           ← Local/Ollama 陣營
│   ├── input/
│   └── output/
└── settings.yaml         ← Azure 模板
```

`DOC_INDEX_ROOTS`（`query_tenders.py`）新增對應鍵值，`compare_results.py` 橫向讀取三套 parquet 比較實體數 / 關係數 / 查詢品質。

---

### 評估維度與評分方式

| 維度 | 量化指標 | 評分方式 |
|---|---|---|
| **解析品質** | 表格欄位完整率、圖表描述字數 | 人工抽查 5 頁（同一 PDF） |
| **索引品質** | 實體數、關係數、社群報告數（parquet 統計） | `compare_results.py` 自動輸出 |
| **查詢準確度** | 10 題回答的事實正確率、條款引用率 | LLM-as-Judge（GPT 評分 1–5） |
| **中文流利度** | 語言自然度、術語使用 | 人工評分 1–5 |
| **速度** | 解析/索引/查詢各自耗時（秒） | `_print_stats` / `_print_index_stats` 已輸出 |
| **費用** | Azure / Gemini 實際 token 費用；Ollama 以 0 計 | `_print_stats` 已輸出 USD 估算 |
| **維運複雜度** | 環境建置步驟數、依賴套件數 | 人工計數 |

**10 題標準問題建議組成（`eval_models.py` 維護）：**
- 5 題局部搜索（特定金額/條款/步驟）
- 3 題全域搜索（跨文件彙整）
- 2 題陷阱題（文件未提及，測試幻覺率）

---

### 待辦事項（實作順序）

1. **新增 `settings_ollama.yaml`**（複製 `settings_gemini.yaml`，改 `api_base` 為 `localhost:11434/v1`）
2. **`.env` 掃描加入 `OLLAMA_BASE_URL`**：`_load_index_model_profiles()` + `_load_query_models()` 尾部各加 5 行
3. **`parse_tenders.py` 補 `AI_MODE=ollama`**：Vision 改呼叫 Ollama `/v1/chat/completions`（傳 base64 圖）
4. **`web_app.py` 加 `PARSE_BACKEND`**：`_do_parse()` 依前綴分派 `_do_parse_azure_di()` vs `_do_parse_local()`
5. **更新 `eval_models.py`**：改呼叫現行 `query_tenders.build_engines()` + `run_local_search()`（現行版本用舊版 graphrag API）
6. **擴充 `compare_results.py`**：橫向讀取三套 `output/` parquet，輸出對比表格
7. **新增 `.env.example`**：文件化所有環境變數，commit 進 repo
