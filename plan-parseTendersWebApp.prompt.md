# Plan & Implementation: 標案文件解析 Web App

## TL;DR

以 **FastAPI + 原生 HTML/JS** 建立三 Tab 單頁應用，整合：
- `parse_tenders_azure.py` — Azure Document Intelligence + Vision 文件解析
- `index_tenders.py` — GraphRAG 知識圖譜索引建立

提供上傳、解析、建立索引三個核心操作，並實作衝突備份、AI 模型選擇、SSE 即時串流進度。

---

## 決策（Decisions）

| 項目 | 決策 |
|---|---|
| Framework | FastAPI + 原生 HTML/JS（無前端 build 步驟） |
| Doc type 選擇 | 上傳時指定，對應 `DOC_TASKS` 的 `vision_prompt` |
| Vision AI 模型 | 解析時手動選擇；從 `.env` 自動偵測可用模型 |
| 備份範圍 | 整個衝突資料夾移至 backup（完整版本保留） |
| 進度顯示 | SSE 即時 Log 串流（`redirect_stdout` + `queue.Queue`） |
| Index 設定 | 無 `settings.yaml` 時自動從 `ragtest/` 根目錄複製 |

---

## 架構（Architecture）

```
web_app.py                ← FastAPI 主程式
parse_tenders_azure.py    ← Azure DI + Vision 解析核心
index_tenders.py          ← GraphRAG 索引包裝器
static/
  index.html              ← 單頁 UI（三個 Tab）
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
| GET | `/api/models` | 列出可用 Vision AI 模型 |
| GET | `/api/parse-folders` | 列出含 `.doc_type` 的可解析資料夾 |
| GET | `/api/parse-folders/{name}/files` | 列出可選擇的單一解析檔案 |
| POST | `/api/parse/stream` | 文件解析 SSE 串流 |
| GET | `/api/index-folders` | 列出含 `.md` 的可建立索引資料夾 |
| POST | `/api/index/stream` | GraphRAG 索引建立 SSE 串流 |

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

GEMINI_API_KEY=...                  # 填入後 Gemini 即自動可用
GEMINI_MODEL=gemini-3.0-flash
GEMINI_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai/
```

掃描邏輯（`_load_azure_models`）：
- 無後綴開始掃描，遇到缺口（例如沒有 `_3`）即停止
- 每個部署的 Key/Endpoint/Version 若未設 `_N` 版本，退回無後綴預設值
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

```
前端選擇 folder_name
  ↓
web_app._run_index()
  ├─ 若 ragtest/{folder_name}/settings.yaml 不存在
  │    → 自動從 ragtest/settings.yaml 複製
  ├─ subprocess.Popen(["python", "-m", "graphrag", "index", "--root", root])
  │    stdout/stderr 即時送入 log_queue → SSE
  └─ 完成後呼叫 index_tenders._print_index_stats()
       輸出 Token 統計 + 費用摘要（同樣透過 redirect_stdout 送入 SSE）
```

`GET /api/index-folders` 回傳各資料夾狀態：

| 欄位 | 說明 |
|---|---|
| `md_count` | `ragtest/{name}/input/*.md` 檔案數 |
| `has_settings` | 是否有 `settings.yaml` |
| `indexed` | `ragtest/{name}/output/lancedb/` 是否非空 |
| `indexing` | 是否正在建立索引中 |

`_indexing_lock: set[str]` 防止同資料夾重複觸發（回傳 409）

---

### 5. 前端三 Tab 結構（`static/index.html` + `app.js`）

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
  - ⚠️ 無 settings.yaml 警示（系統會自動複製，按鈕仍可按）
- 開始建立索引 → `POST /api/index/stream` SSE 連線
- 即時 Log 串流（auto-scroll）+ 中止按鈕
- 完成後自動重整資料夾清單（更新 ✅ 狀態）

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
# requirements.txt 需包含
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
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

1. 啟動後瀏覽 `http://localhost:8000`，確認三個 Tab 正常顯示
2. **上傳**：上傳 1 個 PDF 至新資料夾 `test_folder`，確認 `rag_poc/input/test_folder/` 建立
3. **備份**：重複上傳同名檔 → 確認 `test_folder_old_{yymmdd}_v1` 出現
4. **解析**：切換至「解析文件」Tab，選 `test_folder`，選擇 AI 模型，點解析 → Log 串流即時出現；完成後 `ragtest/test_folder/input/` 有 `.md` 輸出
5. **再次解析**：確認舊輸出備份至 `ragtest/test_folder_old_{yymmdd}_v1`
6. **索引**：切換至「建立索引」Tab，選 `test_folder`，點開始 → Log 串流即時出現；完成後下拉選單的 ✅ 狀態更新
7. **多模型**：在 `.env` 新增 `AZURE_OPENAI_DEPLOYMENT_2`，重啟後解析 Tab 的模型下拉應自動出現新選項

---

## 注意事項

1. **SSE 斷線**：前端使用 `AbortController` 中止連線；中止後後端執行緒會繼續跑完當前檔案（無法強制中斷 Azure DI 呼叫），待完成後自動釋放 lock
2. **索引時間**：GraphRAG index 依文件量可能需數分鐘至數十分鐘，SSE 心跳確保連線不斷開
3. **費用估算**：Log 最後的費用摘要為估算值，實際請查閱 Azure Portal 帳單；定價常數位於 `index_tenders.py` 與 `parse_tenders_azure.py` 頂部
4. **settings.yaml**：各 `ragtest/{folder_name}/` 共用 `ragtest/settings.yaml` 為預設模板；若需個別設定（不同嵌入模型等），可在子資料夾放置專屬 `settings.yaml`
5. **並行限制**：`ThreadPoolExecutor(max_workers=4)` 支援同時解析與索引，但同一資料夾不允許重複觸發（lock 保護）
