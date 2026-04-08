"""
標案文件解析 Web App — FastAPI 後端
"""
from __future__ import annotations

import asyncio
import io
import os
import queue
import re
import shutil
import sys
import threading
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────
# 延遲載入 parse_tenders_azure，避免 app 啟動時就觸發 Azure client 初始化
# ─────────────────────────────────────────────────────────────
import parse_tenders_azure as _pta
import index_tenders as _it
import query_tenders as _qt

# ─────────────────────────────────────────────────────────────
# GraphRAG 索引模型開隢（指向 settings.yaml 模板）
# 自動扫描 .env 中 AZURE_OPENAI_DEPLOYMENT[_N]，每個部署獨立建立一筆↳azure_{{dep}}
# settings 模板路徑順序：AZURE_INDEX_SETTINGS[_N] > ragtest/settings[_N].yaml > ragtest/settings.yaml
# ─────────────────────────────────────────────────────────────

def _load_index_model_profiles() -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    suffixes = [""] + [f"_{i}" for i in range(2, 10)]
    for sfx in suffixes:
        dep = os.getenv(f"AZURE_OPENAI_DEPLOYMENT{sfx}")
        if not dep:
            if sfx:
                break
            continue
        key = f"azure_{dep}"
        # 模板選擇順序：自訂環境變數 > ragtest/settings{sfx}.yaml > ragtest/settings.yaml
        default_tpl = Path(f"./ragtest/settings{sfx}.yaml") if sfx else Path("./ragtest/settings.yaml")
        custom_tpl  = os.getenv(f"AZURE_INDEX_SETTINGS{sfx}")
        tpl_path    = Path(custom_tpl) if custom_tpl else default_tpl
        if sfx and not tpl_path.exists():
            tpl_path = Path("./ragtest/settings.yaml")  # 共用預設
        profiles[key] = {
            "label":    f"Azure {dep}",
            "template": tpl_path,
        }
    profiles["gemini"] = {
        "label":    "Google Gemini",
        "template": Path("./settings_gemini.yaml"),
    }
    return profiles


INDEX_MODEL_PROFILES: dict[str, dict] = _load_index_model_profiles()
_FIRST_AZURE_KEY = next((k for k in INDEX_MODEL_PROFILES if k.startswith("azure_")), "gemini")

# ─────────────────────────────────────────────────────────────
# 常數
# ─────────────────────────────────────────────────────────────
UPLOAD_BASE = Path("./rag_poc/input")
RAGTEST_BASE = Path("./ragtest")
DOC_TYPE_FILE = ".doc_type"
ALLOWED_SUFFIXES = {
    ".pdf", ".doc", ".docx",
    ".xls", ".xlsx",
    ".ppt", ".pptx",
}
# 備份目錄名稱 pattern：{name}_old_{yymmdd}_vXX
_BACKUP_PATTERN = re.compile(r"^(.+)_old_\d{6}_v(\d+)$")

# 解析 / 索引進行中的資料夾名稱集合（防止重複觸發）
_parsing_lock: set[str] = set()
_indexing_lock: set[str] = set()

_executor = ThreadPoolExecutor(max_workers=4)

# ─────────────────────────────────────────────────────────────
# 輔助函式
# ─────────────────────────────────────────────────────────────

def _today_str() -> str:
    """回傳今天的 yymmdd 字串（例如 260331）。"""
    return datetime.now().strftime("%y%m%d")


def _get_backup_path(base: Path, folder_name: str) -> Path:
    """
    在 base 目錄下，為 folder_name 找下一個可用的備份路徑。
    格式：{folder_name}_old_{yymmdd}_vXX，XX 從 1 開始遞增。
    """
    today = _today_str()
    prefix = f"{folder_name}_old_{today}_v"
    existing_versions = []
    for item in base.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            suffix = item.name[len(prefix):]
            if suffix.isdigit():
                existing_versions.append(int(suffix))
    next_ver = max(existing_versions, default=0) + 1
    return base / f"{prefix}{next_ver}"


def _is_backup_folder(name: str) -> bool:
    """判斷資料夾名稱是否為備份格式。"""
    return bool(_BACKUP_PATTERN.match(name))


def _backup_if_exists(base: Path, folder_name: str) -> str | None:
    """
    若 base/folder_name 存在，將整個資料夾移至備份路徑。
    回傳備份路徑字串；若不存在則回傳 None。
    """
    target = base / folder_name
    if not target.exists():
        return None
    backup_path = _get_backup_path(base, folder_name)
    shutil.move(str(target), str(backup_path))
    return str(backup_path)


def _get_doc_type_map() -> dict[str, dict]:
    """從 DOC_TASKS 建立 doc_type → task 的查詢字典。"""
    return {t["doc_type"]: t for t in _pta.DOC_TASKS}


# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="標案文件解析 Web App")

# 靜態檔案（HTML/JS/CSS）
static_dir = Path("./static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index = static_dir / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>static/index.html 尚未建立</h1>")


# ─────────────────────────────────────────────────────────────
# Step 3 — GET /api/folders
# ─────────────────────────────────────────────────────────────

@app.get("/api/folders")
async def list_folders():
    """
    列出 rag_poc/input/ 下的非備份子目錄，
    並附帶 doc_type 與檔案數資訊。
    """
    UPLOAD_BASE.mkdir(parents=True, exist_ok=True)
    result = []
    for item in sorted(UPLOAD_BASE.iterdir()):
        if not item.is_dir():
            continue
        if _is_backup_folder(item.name):
            continue
        doc_type_path = item / DOC_TYPE_FILE
        doc_type = doc_type_path.read_text(encoding="utf-8").strip() if doc_type_path.exists() else ""
        files = [
            f.name for f in item.iterdir()
            if f.is_file() and f.suffix.lower() in ALLOWED_SUFFIXES
        ]
        result.append({
            "name": item.name,
            "doc_type": doc_type,
            "file_count": len(files),
        })
    return result


# ─────────────────────────────────────────────────────────────
# Step 4 — GET /api/folders/{name}/files
# ─────────────────────────────────────────────────────────────

@app.get("/api/folders/{name}/files")
async def list_folder_files(name: str):
    """列出指定上傳資料夾內的所有文件檔案。"""
    folder = UPLOAD_BASE / name
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"資料夾 '{name}' 不存在")
    files = [
        f.name for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() in ALLOWED_SUFFIXES
    ]
    doc_type_path = folder / DOC_TYPE_FILE
    doc_type = doc_type_path.read_text(encoding="utf-8").strip() if doc_type_path.exists() else ""
    return {"folder": name, "doc_type": doc_type, "files": files}


# ─────────────────────────────────────────────────────────────
# Step 5 — POST /api/upload
# ─────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_files(
    folder_name: str = Form(...),
    doc_type: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """
    上傳文件至 rag_poc/input/{folder_name}/。
    若已存在同名資料夾且有衝突檔名，整個資料夾先備份再重建。
    """
    # 驗證 doc_type
    valid_types = [t["doc_type"] for t in _pta.DOC_TASKS]
    if doc_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"無效的 doc_type，可用: {valid_types}")

    # 驗證資料夾名稱（避免路徑穿越）
    if not re.match(r"^[\w\-\u4e00-\u9fff]+$", folder_name):
        raise HTTPException(status_code=400, detail="資料夾名稱只允許字母、數字、底線、連字號與中文")

    # 驗證所有檔案類型
    for f in files:
        suffix = Path(f.filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"不支援的檔案類型: {f.filename}（允許: {', '.join(ALLOWED_SUFFIXES)}）"
            )

    target_dir = UPLOAD_BASE / folder_name
    backup_path: str | None = None

    # 若目錄已存在，檢查是否有衝突檔名
    if target_dir.exists():
        incoming_names = {Path(f.filename).name for f in files}
        existing_names = {
            x.name for x in target_dir.iterdir()
            if x.is_file() and x.suffix.lower() in ALLOWED_SUFFIXES
        }
        if incoming_names & existing_names:
            # 有衝突 → 備份整個資料夾
            backup_path = _backup_if_exists(UPLOAD_BASE, folder_name)

    target_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for upload in files:
        dest = target_dir / Path(upload.filename).name
        content = await upload.read()
        dest.write_bytes(content)
        saved.append(Path(upload.filename).name)

    # 寫入 .doc_type
    (target_dir / DOC_TYPE_FILE).write_text(doc_type, encoding="utf-8")

    return {
        "folder": folder_name,
        "doc_type": doc_type,
        "saved_files": saved,
        "backup": backup_path,
    }


# ─────────────────────────────────────────────────────────────
# Step 3b — GET /api/models
# ─────────────────────────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    """
    回傳可選用的 Vision AI 模型清單，含可用狀態。
    api_key 為空或含有 placeholder 文字（在此輸入）時標示為不可用。
    """
    result = []
    for model_id, cfg in _pta.VISION_MODELS.items():
        api_key = cfg.get("api_key") or ""
        available = bool(api_key) and "在此輸入" not in api_key
        result.append({
            "id": model_id,
            "label": cfg["label"],
            "available": available,
        })
    return result


# ─────────────────────────────────────────────────────────────
# Step 6 — GET /api/parse-folders
# ─────────────────────────────────────────────────────────────

@app.get("/api/parse-folders")
async def list_parse_folders():
    """列出 rag_poc/input/ 下含有 .doc_type 檔的非備份資料夾（可解析）。"""
    UPLOAD_BASE.mkdir(parents=True, exist_ok=True)
    result = []
    for item in sorted(UPLOAD_BASE.iterdir()):
        if not item.is_dir():
            continue
        if _is_backup_folder(item.name):
            continue
        doc_type_path = item / DOC_TYPE_FILE
        if not doc_type_path.exists():
            continue
        doc_type = doc_type_path.read_text(encoding="utf-8").strip()
        files = [
            f.name for f in item.iterdir()
            if f.is_file() and f.suffix.lower() in ALLOWED_SUFFIXES
        ]
        parsing_now = item.name in _parsing_lock
        result.append({
            "name": item.name,
            "doc_type": doc_type,
            "file_count": len(files),
            "parsing": parsing_now,
        })
    return result


# ─────────────────────────────────────────────────────────────
# Step 7 — GET /api/parse-folders/{name}/files
# ─────────────────────────────────────────────────────────────

@app.get("/api/parse-folders/{name}/files")
async def list_parse_folder_files(name: str):
    """列出指定資料夾內可選擇的單一解析檔案。"""
    folder = UPLOAD_BASE / name
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"資料夾 '{name}' 不存在")
    doc_type_path = folder / DOC_TYPE_FILE
    if not doc_type_path.exists():
        raise HTTPException(status_code=400, detail=f"資料夾 '{name}' 尚未設定 doc_type")
    doc_type = doc_type_path.read_text(encoding="utf-8").strip()
    files = [
        f.name for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() in {".pdf", ".doc", ".docx"}
    ]
    return {"folder": name, "doc_type": doc_type, "files": files}


# ─────────────────────────────────────────────────────────────
# Step 8 — POST /api/parse/stream  (SSE)
# ─────────────────────────────────────────────────────────────

class ParseRequest(BaseModel):
    folder_name: str
    file_name: str | None = None
    model_id: str | None = None


def _run_parse(folder_name: str, file_name: str | None, log_queue: queue.Queue, model_id: str | None = None) -> None:
    """
    在背景執行緒中執行文件解析，將所有 stdout print 轉入 log_queue。
    解析完成後放入 sentinel None。
    """
    class QueueWriter(io.TextIOBase):
        def write(self, s: str) -> int:
            if s and s != "\n":
                log_queue.put(s.rstrip("\n"))
            return len(s)

    writer = QueueWriter()
    try:
        with redirect_stdout(writer):
            _do_parse(folder_name, file_name, log_queue, model_id)
    except Exception as e:
        log_queue.put(f"❌ 未預期錯誤: {e}")
        returncode = -1
    finally:
        log_queue.put(("__done__", returncode))  # sentinel 帶返回碼


def _do_parse(folder_name: str, file_name: str | None, log_queue: queue.Queue, model_id: str | None = None) -> None:
    """核心解析邏輯（在 redirect_stdout 環境內執行）。"""
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    import parse_tenders_azure as pta

    folder = UPLOAD_BASE / folder_name
    doc_type_path = folder / DOC_TYPE_FILE
    if not doc_type_path.exists():
        print(f"❌ 找不到 .doc_type 設定: {folder}")
        return

    doc_type = doc_type_path.read_text(encoding="utf-8").strip()
    task_map = _get_doc_type_map()
    task = task_map.get(doc_type)
    if not task:
        print(f"❌ 無效的 doc_type: {doc_type}")
        return

    vision_prompt = task["vision_prompt"]

    # 建立 Vision 客戶端
    vision_client, vision_deployment, model_label = _pta.build_vision_client(model_id)
    print(f"🤖 Vision 模型: {model_label}")

    # 輸出目錄：ragtest/{folder_name}/input/
    output_dir = RAGTEST_BASE / folder_name / "input"

    # 若輸出目錄已存在 → 備份
    ragtest_folder = RAGTEST_BASE / folder_name
    if output_dir.exists():
        backup_path = _backup_if_exists(RAGTEST_BASE, folder_name)
        print(f"📦 已備份既有解析輸出至: {backup_path}")
        output_dir = RAGTEST_BASE / folder_name / "input"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 Azure DI 客戶端
    di_client = DocumentIntelligenceClient(
        pta.AZURE_DI_ENDPOINT,
        AzureKeyCredential(pta.AZURE_DI_KEY),
    )

    # 重置統計
    pta._stats = pta._RunStats()

    # 決定要處理的檔案清單
    if file_name:
        candidate = folder / file_name
        if not candidate.exists():
            print(f"❌ 找不到指定檔案: {file_name}")
            return
        files_to_process = [candidate]
    else:
        files_to_process = sorted(
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in {".pdf", ".doc", ".docx"}
        )

    if not files_to_process:
        print(f"⚠️ 資料夾 '{folder_name}' 內沒有可解析的 PDF/DOCX 檔案")
        return

    print(f"\n📂 開始解析資料夾 [{folder_name}] (類型: {doc_type})")
    print(f"   共 {len(files_to_process)} 個檔案，輸出至: {output_dir}")

    for file_path in files_to_process:
        try:
            pta.process_file(
                di_client=di_client,
                file_path=str(file_path),
                output_dir=str(output_dir),
                vision_prompt=vision_prompt,
                doc_type=doc_type,
                vision_client=vision_client,
                vision_deployment=vision_deployment,
            )
            pta._stats.files_processed += 1
        except Exception as e:
            print(f"❌ 處理 {file_path.name} 失敗: {e}")
            pta._stats.files_failed += 1

    pta._print_stats(model_label, model_id)
    print("\n✅ 解析任務完成！")


async def _sse_generator(folder_name: str, file_name: str | None, model_id: str | None = None) -> AsyncGenerator[str, None]:
    """從 log_queue 讀取訊息並以 SSE 格式推送。"""
    log_queue: queue.Queue = queue.Queue()
    loop = asyncio.get_event_loop()

    # 啟動背景執行緒
    future = loop.run_in_executor(
        _executor, _run_parse, folder_name, file_name, log_queue, model_id
    )

    try:
        while True:
            try:
                msg = await loop.run_in_executor(None, lambda: log_queue.get(timeout=0.2))
            except queue.Empty:
                # 心跳，避免 SSE 連線逾時
                yield ": heartbeat\n\n"
                continue

            if msg is None:
                # sentinel：解析結束
                yield "event: done\ndata: done\n\n"
                break

            # 跳脫換行字元以符合 SSE 格式
            safe = msg.replace("\n", " | ")
            yield f"data: {safe}\n\n"
    finally:
        _parsing_lock.discard(folder_name)
        await future


@app.post("/api/parse/stream")
async def parse_stream(req: ParseRequest):
    """
    啟動文件解析並以 SSE 串流回傳 log 訊息。
    同一資料夾若已在解析中，回傳 409。
    """
    folder_name = req.folder_name

    # 驗證資料夾存在
    folder = UPLOAD_BASE / folder_name
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"資料夾 '{folder_name}' 不存在")

    # 防重複觸發
    if folder_name in _parsing_lock:
        raise HTTPException(status_code=409, detail=f"資料夾 '{folder_name}' 正在解析中，請稍候")

    _parsing_lock.add(folder_name)

    return StreamingResponse(
        _sse_generator(folder_name, req.file_name, req.model_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────
# Index — GET /api/index-models
# ─────────────────────────────────────────────────────────────

@app.get("/api/index-models")
async def list_index_models():
    """回傳可用的 GraphRAG 索引模型開隢清單（依 settings 模板檔存在與否設定 availability）。"""
    result = []
    for model_id, cfg in INDEX_MODEL_PROFILES.items():
        result.append({
            "id": model_id,
            "label": cfg["label"],
            "available": cfg["template"].exists(),
        })
    return result


# ─────────────────────────────────────────────────────────────
# Index — GET /api/index-folders
# ─────────────────────────────────────────────────────────────

@app.get("/api/index-folders")
async def list_index_folders():
    """列出 ragtest/ 下含有 input/*.md 的非備份子目錄（可建立索引）。"""
    RAGTEST_BASE.mkdir(parents=True, exist_ok=True)
    result = []
    for item in sorted(RAGTEST_BASE.iterdir()):
        if not item.is_dir():
            continue
        if _is_backup_folder(item.name):
            continue
        input_dir = item / "input"
        if not input_dir.exists():
            continue
        md_files = list(input_dir.glob("*.md"))
        if not md_files:
            continue
        has_settings = (item / "settings.yaml").exists()
        lancedb_dir = item / "output" / "lancedb"
        indexed = lancedb_dir.exists() and any(True for _ in lancedb_dir.iterdir()) if lancedb_dir.exists() else False
        result.append({
            "name": item.name,
            "md_count": len(md_files),
            "has_settings": has_settings,
            "indexed": indexed,
            "indexing": item.name in _indexing_lock,
        })
    return result


# ─────────────────────────────────────────────────────────────
# Index — POST /api/index/stream  (SSE)
# ─────────────────────────────────────────────────────────────

class IndexRequest(BaseModel):
    folder_name: str
    model_id: str | None = None


def _run_index(folder_name: str, log_queue: queue.Queue, model_id: str | None = None) -> None:
    """
    在背景執行緒中執行 graphrag index，將 stdout/stderr 轉入 log_queue。
    內建「鶓默債測器」：超過 30 秒無輸出時自動推送進度提示。
    完成後呼叫 _print_index_stats 輸出費用摘要，最後放入 sentinel None。
    """
    import sys
    import subprocess
    import time

    class QueueWriter(io.TextIOBase):
        def write(self, s: str) -> int:
            if s and s != "\n":
                log_queue.put(s.rstrip("\n"))
            return len(s)

    root = RAGTEST_BASE / folder_name
    settings_path = root / "settings.yaml"

    # 依選擇的模型開隢決定 settings.yaml 來源
    profile = INDEX_MODEL_PROFILES.get(model_id or _FIRST_AZURE_KEY) or INDEX_MODEL_PROFILES[_FIRST_AZURE_KEY]
    template = profile["template"]
    label = profile["label"]

    if template.exists():
        shutil.copy(str(template), str(settings_path))
        log_queue.put(f"🤖 索引模型: {label}  （settings 經複製自 {template}）")
    else:
        log_queue.put(f"⚠️ 找不到 settings 模板 ({template})，嘗試使用現有 settings")
        if not settings_path.exists():
            log_queue.put("❌ 無可用的 settings.yaml，索引將失敗")
            log_queue.put(None)
            return

    cmd = [sys.executable, "-m", "graphrag", "index", "--root", str(root)]
    log_queue.put(f"🚀 執行: {' '.join(cmd)}")

    t0 = time.time()
    returncode = -1
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # 「鶓默債測器」：子執行緒定期檢查最後輸出時間，超過 30 秒就推送進度提示
        _last_output_time = [time.time()]   # list 讓內層函式可更新
        _proc_done = threading.Event()

        def _silence_ticker():
            SILENCE_THRESHOLD = 30  # 秒
            TICK_INTERVAL = 10      # 秒
            while not _proc_done.wait(timeout=TICK_INTERVAL):
                idle = time.time() - _last_output_time[0]
                if idle >= SILENCE_THRESHOLD:
                    elapsed = int(time.time() - t0)
                    log_queue.put(
                        f"⏳ GraphRAG 遠行中… (已者過 {elapsed // 60}m{elapsed % 60:02d}s，{idle:.0f}s 無輸出)"
                    )

        ticker = threading.Thread(target=_silence_ticker, daemon=True)
        ticker.start()

        for line in iter(proc.stdout.readline, ""):
            stripped = line.rstrip("\n\r")
            if stripped:
                log_queue.put(stripped)
                _last_output_time[0] = time.time()

        proc.stdout.close()
        returncode = proc.wait()
        _proc_done.set()
        wall = time.time() - t0

        # 輸出索引統計摘要（重導 stdout 至 queue）
        with redirect_stdout(QueueWriter()):
            _it._print_index_stats(root, wall, returncode, model_id)

    except Exception as e:
        log_queue.put(f"❌ 未預期錯誤: {e}")
    finally:
        log_queue.put(None)  # sentinel


async def _sse_index_generator(folder_name: str, model_id: str | None = None) -> AsyncGenerator[str, None]:
    """從 log_queue 讀取索引訊息並以 SSE 格式推送。"""
    log_queue: queue.Queue = queue.Queue()
    loop = asyncio.get_event_loop()

    future = loop.run_in_executor(_executor, _run_index, folder_name, log_queue, model_id)

    try:
        while True:
            try:
                msg = await loop.run_in_executor(None, lambda: log_queue.get(timeout=0.2))
            except queue.Empty:
                yield ": heartbeat\n\n"
                continue

            if msg is None:
                # 舊格式相容
                yield "event: done\ndata: done\n\n"
                break

            if isinstance(msg, tuple) and msg[0] == "__done__":
                rc = msg[1]
                if rc == 0:
                    yield "event: done\ndata: done\n\n"
                else:
                    yield f"event: error\ndata: exit {rc}\n\n"
                break

            safe = msg.replace("\n", " | ")
            yield f"data: {safe}\n\n"
    finally:
        _indexing_lock.discard(folder_name)
        await future


@app.post("/api/index/stream")
async def index_stream(req: IndexRequest):
    """
    啟動 GraphRAG 索引建立並以 SSE 串流回傳 log 訊息。
    同一資料夾若已在建立索引中，回傳 409。
    """
    folder_name = req.folder_name
    root = RAGTEST_BASE / folder_name
    input_dir = root / "input"

    if not input_dir.exists() or not list(input_dir.glob("*.md")):
        raise HTTPException(status_code=404, detail=f"ragtest/{folder_name}/input/ 尚無 .md 檔案，請先執行解析")

    if folder_name in _indexing_lock:
        raise HTTPException(status_code=409, detail=f"資料夾 '{folder_name}' 正在建立索引中，請稍候")

    _indexing_lock.add(folder_name)

    return StreamingResponse(
        _sse_index_generator(folder_name, req.model_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────
# 查詢 API
# ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    doc_type: str = "default"
    mode: str = "local"
    query: str
    show_context: bool = False
    model_id: str | None = None


@app.get("/api/query-models")
async def list_query_models():
    """列出可用 Query AI 模型（比照 /api/models）。"""
    result = []
    for mid, cfg in _qt.QUERY_MODELS.items():
        has_key = bool(cfg.get("chat") and getattr(cfg["chat"], "api_key", None)
                       and cfg["chat"].api_key != "your-api-key-here")
        result.append({
            "id":        mid,
            "label":     cfg["label"],
            "available": has_key,
        })
    return result


@app.get("/api/query-types")
async def list_query_types():
    """列出可查詢的索引目錄（只回傳 lancedb 非空的）。"""
    result = []
    for type_name, output_dir in _qt.DOC_INDEX_ROOTS.items():
        lancedb = Path(output_dir) / "lancedb"
        result.append({
            "id":        type_name,
            "label":     {"government":    "government（政府標案）",
                          "stip":          "stip（門市銷售指南）",
                          "mixed_tenders": "mixed_tenders（軍服標案）",
                          "default":       "default（通用）"}.get(type_name, type_name),
            "available": lancedb.exists() and any(lancedb.iterdir()),
        })
    return result


@app.post("/api/query")
async def run_query(req: QueryRequest):
    """
    執行 GraphRAG 查詢並以 SSE 串流回傳結果。
    使用 AsyncGenerator 推送：進度提示 → 回答正文 → done。
    """
    async def _stream() -> AsyncGenerator[str, None]:
        try:
            input_dir = _qt.DOC_INDEX_ROOTS.get(req.doc_type)
            if not input_dir:
                yield f"data: ❌ 不支援的 doc_type: {req.doc_type}\n\n"
                yield "event: error\ndata: invalid type\n\n"
                return

            lancedb = Path(input_dir) / "lancedb"
            if not lancedb.exists() or not any(lancedb.iterdir()):
                yield f"data: ❌ {req.doc_type} 尚未建立索引，請先執行索引建立\n\n"
                yield "event: error\ndata: not indexed\n\n"
                return

            mode_label = "🔍 局部搜索" if req.mode == "local" else "🌐 全域搜索"
            yield f"data: {mode_label}：{req.query}\n\n"
            yield f"data: 📂 索引目錄：{input_dir}\n\n"
            yield ": heartbeat\n\n"

            # 建立搜索引擎（CPU 密集，放入執行緒）
            loop = asyncio.get_event_loop()
            local_engine, global_engine = await loop.run_in_executor(
                _executor, _qt.build_engines, input_dir, req.doc_type, req.model_id
            )
            yield f"data: ✅ 索引載入完成，開始查詢…\n\n"
            yield ": heartbeat\n\n"

            # 執行查詢（async）
            if req.mode == "local":
                result = await _qt.run_local_search(local_engine, req.query, req.show_context, req.model_id)
            else:
                result = await _qt.run_global_search(global_engine, req.query, req.show_context, req.model_id)

            # 推送答案（按行拆分以利前端逐行顯示）
            yield f"data: \n\n"
            yield f"data: ━━━ 查詢結果 ━━━\n\n"
            for line in result.splitlines():
                safe = line.replace("\n", " ")
                yield f"data: {safe}\n\n"

            # 費用摘要（重導 _print_query_stats stdout）
            yield f"data: \n\n"
            buf = io.StringIO()
            with redirect_stdout(buf):
                _qt._print_query_stats()
            for line in buf.getvalue().splitlines():
                if line.strip():
                    yield f"data: {line}\n\n"

            yield "event: done\ndata: done\n\n"

        except Exception as e:
            yield f"data: ❌ 查詢失敗：{e}\n\n"
            yield "event: error\ndata: exception\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=True)
