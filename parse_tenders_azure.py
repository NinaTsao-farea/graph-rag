import os
import io
import re
import base64
import argparse
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from openai import AzureOpenAI, OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
from dotenv import load_dotenv

# 0. 手動讀取 .env 檔案
load_dotenv()

# ─────────────────────────────────────────────────────────────
# 1. Azure Document Intelligence 憑證
# ─────────────────────────────────────────────────────────────
AZURE_DI_ENDPOINT = os.getenv("AZURE_COGNITIVE_SERVICES_ENDPOINT")
AZURE_DI_KEY = os.getenv("AZURE_COGNITIVE_SERVICES_KEY")

# ─────────────────────────────────────────────────────────────
# 2. Vision 模型設定（可在解析時手動選擇）
# ─────────────────────────────────────────────────────────────

def _load_azure_models() -> dict[str, dict]:
    """
    自動掃描 .env 中的 Azure 部署設定：
      - 無後綴：AZURE_OPENAI_DEPLOYMENT
      - 編號後綴：AZURE_OPENAI_DEPLOYMENT_2, _3, _4...（發現空白即停止）
    每個部署可獨立設定 API_KEY / ENDPOINT / API_VERSION（未設時退回無後綴版本）。
    """
    models: dict[str, dict] = {}
    suffixes = [""] + [f"_{i}" for i in range(2, 10)]
    for sfx in suffixes:
        dep = os.getenv(f"AZURE_OPENAI_DEPLOYMENT{sfx}")
        if not dep:
            if sfx:  # 遇到缺口就停止（_2 缺就不繼續找 _3）
                break
            continue
        key = f"azure_{dep}"
        models[key] = {
            "label":      f"Azure {dep}",
            "type":       "azure",
            "api_key":    os.getenv(f"AZURE_OPENAI_API_KEY{sfx}",    os.getenv("AZURE_OPENAI_API_KEY")),
            "endpoint":   os.getenv(f"AZURE_OPENAI_ENDPOINT{sfx}",   os.getenv("AZURE_OPENAI_ENDPOINT")),
            "api_version":os.getenv(f"AZURE_OPENAI_API_VERSION{sfx}", os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")),
            "deployment": dep,
        }
    return models


VISION_MODELS: dict[str, dict] = _load_azure_models()
VISION_MODELS["gemini"] = {
    "label":      f"Gemini {os.getenv('GEMINI_MODEL', 'gemini-3.0-flash')}",
    "type":       "openai_compat",
    "api_key":    os.getenv("GEMINI_API_KEY", ""),
    "base_url":   os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    "deployment": os.getenv("GEMINI_MODEL", "gemini-3.0-flash"),
}

# 第一個 Azure 部署為預設
DEFAULT_VISION_MODEL_ID = next(
    (k for k, v in VISION_MODELS.items() if v["type"] == "azure"),
    next(iter(VISION_MODELS)),
)

# 向下相容 — CLI / batch_process_folders 使用
AZURE_DEPLOYMENT = VISION_MODELS.get(DEFAULT_VISION_MODEL_ID, {}).get("deployment", "gpt-5.4")
azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
)
_discovered = ", ".join(VISION_MODELS.keys())
print(f"👉 已載入 Vision 模型: {_discovered}（預設: {DEFAULT_VISION_MODEL_ID}）")


def build_vision_client(model_id: str | None = None) -> tuple:
    """
    依 model_id 建立 Vision 客戶端。
    回傳 (client, deployment, label) 元組。
    model_id 為 None 或無效時，退回預設 azure 模型。
    """
    cfg = VISION_MODELS.get(model_id or DEFAULT_VISION_MODEL_ID)
    if cfg is None:
        cfg = VISION_MODELS[DEFAULT_VISION_MODEL_ID]

    if cfg["type"] == "azure":
        client = AzureOpenAI(
            api_key=cfg["api_key"],
            azure_endpoint=cfg["endpoint"],
            api_version=cfg["api_version"],
        )
    else:  # openai_compat（Gemini 等）
        client = OpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
        )
    return client, cfg["deployment"], cfg["label"]

# ─────────────────────────────────────────────────────────────
# 3. 文件類型任務設定（與 parse_tenders.py 一致）
#    來源資料夾 → (輸出資料夾, 描述, 視覺詢問 Prompt)
# ─────────────────────────────────────────────────────────────
DOC_TASKS = [
    {
        "src":          "./rag_poc/input/government",
        "dest":         "./ragtest/government/input",
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
        "dest":         "./ragtest/mixed_tenders/input",
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
        "dest":         "./ragtest/stip/input",
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
    # 新增類型往此列表新增一筆即可，不需修改其他程式碼
]

# Azure DI 座標系統為英吋；PyMuPDF 使用點 (1 點 = 1/72 英吋)
_INCH_TO_PT = 72.0
# 圖片渲染縮放比例（與 parse_tenders.py 一致）
_RENDER_SCALE = 4.16
# 過濾尺寸門檻 (點)
_MIN_SIZE_PT = 50
# 過濾頁首頁尾的邊界 (英吋)
_MARGIN_INCH = 0.7
# 超過此頁數自動拆分為多個區塊，避免 Azure DI 請求超時
_MAX_PAGES_PER_CHUNK = 50

# ─────────────────────────────────────────────────────────────
# 費用估算常數 (USD) — 依 Azure 定價頁更新
# ─────────────────────────────────────────────────────────────
_DI_PRICE_PER_1K_PAGES      = 1.50   # Document Intelligence prebuilt-layout / 千頁
_VISION_INPUT_PRICE_PER_1M  = 0.75   # Vision 模型輸入 token / 百萬（依實際部署定價調整）
_VISION_OUTPUT_PRICE_PER_1M = 4.50   # Vision 模型輸出 token / 百萬（依實際部署定價調整）


@dataclass
class _RunStats:
    files_processed: int = 0
    files_failed: int = 0
    di_pages: int = 0
    vision_calls: int = 0        # 實際呼叫 Vision API 的次數（含後來 SKIP 的）
    vision_skipped: int = 0      # Vision 回傳 [SKIP] 的次數
    vision_input_tokens: int = 0
    vision_output_tokens: int = 0
    output_total_bytes: int = 0
    start_time: float = field(default_factory=time.time)


_stats = _RunStats()


# ─────────────────────────────────────────────────────────────
# 4. 工具函式
# ─────────────────────────────────────────────────────────────

def _html_table_to_markdown(table_tag) -> str:
    """
    將 BeautifulSoup <table> 節點轉為 Markdown 表格字串。
    處理 colspan（欄位合併），rowspan 以重複內容填補。
    """
    # 第一步：將所有 tr/th/td 展開到二維 grid（處理 rowspan/colspan）
    grid: list[list[str]] = []
    # rowspan 待填補佇列：{col_index: (剩餘列數, 內容)}
    pending: dict[int, tuple[int, str]] = {}

    for tr in table_tag.find_all("tr"):
        row: list[str] = []
        col_idx = 0

        for cell in tr.find_all(["th", "td"]):
            # 跳過被 rowspan 佔用的欄位
            while col_idx in pending and pending[col_idx][0] > 0:
                row.append(pending[col_idx][1])
                pending[col_idx] = (pending[col_idx][0] - 1, pending[col_idx][1])
                if pending[col_idx][0] == 0:
                    del pending[col_idx]
                col_idx += 1

            text = cell.get_text(separator=" ", strip=True)
            # 移除 Markdown 表格中不允許的換行與 pipe
            text = text.replace("\n", " ").replace("|", "｜")

            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            for _ in range(colspan):
                row.append(text)
                if rowspan > 1:
                    pending[col_idx] = (rowspan - 1, text)
                col_idx += 1

        # 填補行尾殘餘的 rowspan
        while col_idx in pending and pending[col_idx][0] > 0:
            row.append(pending[col_idx][1])
            pending[col_idx] = (pending[col_idx][0] - 1, pending[col_idx][1])
            if pending[col_idx][0] == 0:
                del pending[col_idx]
            col_idx += 1

        if row:
            grid.append(row)

    if not grid:
        return ""

    # 統一欄數
    max_cols = max(len(r) for r in grid)
    for r in grid:
        while len(r) < max_cols:
            r.append("")

    # 計算每欄最大寬度（至少 3 字元）
    col_widths = [
        max(3, max(len(grid[row][col]) for row in range(len(grid))))
        for col in range(max_cols)
    ]

    def fmt_row(cells):
        return "| " + " | ".join(
            cell.ljust(col_widths[i]) for i, cell in enumerate(cells)
        ) + " |"

    lines = []
    for i, row in enumerate(grid):
        lines.append(fmt_row(row))
        # 在第一列後插入分隔線
        if i == 0:
            sep = "| " + " | ".join("-" * col_widths[j] for j in range(max_cols)) + " |"
            lines.append(sep)

    return "\n".join(lines)


def _convert_html_tables(markdown: str) -> str:
    """
    掃描 markdown 字串中所有 <table>…</table> 區塊，
    逐一替換為 Markdown 表格格式。
    """
    def replace_table(match):
        soup = BeautifulSoup(match.group(0), "html.parser")
        table_tag = soup.find("table")
        if not table_tag:
            return match.group(0)
        md_table = _html_table_to_markdown(table_tag)
        return md_table if md_table else match.group(0)

    return re.sub(
        r"<table[\s\S]*?</table>",
        replace_table,
        markdown,
        flags=re.IGNORECASE,
    )


def _bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _extract_docx_images(file_path: Path) -> list[bytes]:
    """
    DOCX 是 ZIP 檔案，直接從 word/media/ 按檔名順序提取所有嵌入圖片。
    回傳图片 bytes 清單，順序與文件中圖片出現順序對應。
    """
    supported = {"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff"}
    images: list[bytes] = []
    try:
        with zipfile.ZipFile(file_path, "r") as z:
            media = sorted(
                n for n in z.namelist()
                if n.startswith("word/media/")
                and n.rsplit(".", 1)[-1].lower() in supported
            )
            for name in media:
                images.append(z.read(name))
    except Exception as e:
        print(f"  ⚠️  無法讀取 DOCX 內嵌圖片: {e}")
    return images


def describe_image(
    image_bytes: bytes,
    vision_prompt: str,
    client=None,
    deployment: str | None = None,
) -> str:
    """呼叫 Vision 模型對圖表進行文字描述。client/deployment 為 None 時使用預設 Azure 客戶端。"""
    _stats.vision_calls += 1
    _client = client if client is not None else azure_client
    _deployment = deployment if deployment is not None else AZURE_DEPLOYMENT
    try:
        b64 = _bytes_to_base64(image_bytes)
        response = _client.chat.completions.create(
            model=_deployment,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        if response.usage:
            _stats.vision_input_tokens  += response.usage.prompt_tokens
            _stats.vision_output_tokens += response.usage.completion_tokens
        return response.choices[0].message.content
    except Exception as e:
        return f"[Azure 圖片解析失敗: {str(e)}]"


def _extract_figure_image(pdf_doc: fitz.Document, bounding_region) -> bytes | None:
    """
    依據 Azure DI 回傳的 bounding_region，從 PDF 頁面裁切並渲染圖表區域。
    回傳 PNG bytes；若因尺寸/位置過濾而跳過則回傳 None。
    """
    page_num = bounding_region.page_number - 1  # Azure DI 頁碼從 1 開始
    polygon = bounding_region.polygon           # [x0,y0, x1,y1, …] 英吋

    x_coords = polygon[0::2]
    y_coords = polygon[1::2]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    width_inch  = x_max - x_min
    height_inch = y_max - y_min

    # ── 策略一：尺寸過濾（英吋換算至點後比較）
    if (width_inch * _INCH_TO_PT) < _MIN_SIZE_PT or (height_inch * _INCH_TO_PT) < _MIN_SIZE_PT:
        print(f"  ⏭️ 忽略小型圖片 (Size: {width_inch:.2f}\" x {height_inch:.2f}\")")
        return None

    # ── 策略二：位置過濾（過濾頁首頁尾）
    page = pdf_doc[page_num]
    page_height_inch = page.rect.height / _INCH_TO_PT
    if y_min < _MARGIN_INCH or y_max > (page_height_inch - _MARGIN_INCH):
        print(f"  ⏭️ 忽略頁首/頁尾圖片 (top={y_min:.2f}\", bottom={y_max:.2f}\")")
        return None

    # ── 裁切並渲染（PyMuPDF 座標單位為點）
    clip = fitz.Rect(
        x_min * _INCH_TO_PT, y_min * _INCH_TO_PT,
        x_max * _INCH_TO_PT, y_max * _INCH_TO_PT,
    )
    mat = fitz.Matrix(_RENDER_SCALE, _RENDER_SCALE)
    pix = page.get_pixmap(matrix=mat, clip=clip)

    if pix.width < _MIN_SIZE_PT or pix.height < _MIN_SIZE_PT:
        print(f"  ⏭️ 渲染後圖片過小 ({pix.width}x{pix.height} px)，略過")
        return None

    return pix.tobytes("png")


def _split_pdf_to_chunks(file_path: Path, tmp_dir: Path) -> list[tuple[Path, int]]:
    """
    若 PDF 超過 _MAX_PAGES_PER_CHUNK 頁，拆分至 tmp_dir 並回傳區塊清單。
    每個元素為 (chunk_path, page_offset)，page_offset 為該區塊在原始 PDF 中的起始頁（0-indexed）。
    不需拆分時直接回傳 [(file_path, 0)]。
    """
    pdf_doc = fitz.open(str(file_path))
    total_pages = len(pdf_doc)

    if total_pages <= _MAX_PAGES_PER_CHUNK:
        pdf_doc.close()
        return [(file_path, 0)]

    print(f"  ✂️  共 {total_pages} 頁，超過上限 {_MAX_PAGES_PER_CHUNK} 頁，自動拆分...")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[Path, int]] = []

    for start in range(0, total_pages, _MAX_PAGES_PER_CHUNK):
        end = min(start + _MAX_PAGES_PER_CHUNK, total_pages) - 1
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(pdf_doc, from_page=start, to_page=end)
        chunk_path = tmp_dir / f"{file_path.stem}_chunk_{start+1:04d}-{end+1:04d}.pdf"
        chunk_doc.save(str(chunk_path))
        chunk_doc.close()
        chunks.append((chunk_path, start))
        print(f"     → 區塊 {len(chunks)}: 第 {start+1}–{end+1} 頁 → {chunk_path.name}")

    pdf_doc.close()
    return chunks


def _process_pdf_chunk(
    di_client: DocumentIntelligenceClient,
    chunk_path: Path,
    chunk_fitz_doc: fitz.Document,
    image_save_dir: Path,
    image_counter: int,
    vision_prompt: str,
    page_offset: int,
    vision_client=None,
    vision_deployment: str | None = None,
) -> tuple[str, int]:
    """
    對單一 PDF 區塊呼叫 Azure DI，提取圖表並嵌入描述。
    chunk_fitz_doc: 已開啟的 fitz.Document（呼叫端負責 open/close）。
    page_offset: 此區塊在原始 PDF 中的起始頁碼偏移（僅用於日誌顯示）。
    回傳 (markdown_content, updated_image_counter)。
    """
    with open(chunk_path, "rb") as f:
        poller = di_client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=f.read()),
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
    result = poller.result()
    _stats.di_pages += len(result.pages or [])
    markdown_content: str = result.content

    figures = result.figures or []
    if not figures:
        return markdown_content, image_counter

    figures_info = []
    for i, figure in enumerate(figures):
        if not figure.bounding_regions or not figure.spans:
            continue

        region = figure.bounding_regions[0]
        span   = figure.spans[0]
        orig_page = region.page_number + page_offset  # 顯示用：相對原始 PDF 的頁碼

        print(f"  🖼️ 偵測到圖表 #{i+1} (原始第 {orig_page} 頁)，正在處理...")
        image_bytes = _extract_figure_image(chunk_fitz_doc, region)
        if image_bytes is None:
            continue

        image_counter += 1
        image_filename = f"image_{image_counter}.png"
        (image_save_dir / image_filename).write_bytes(image_bytes)

        print(f"  🔍 正在辨識圖片 {image_filename} ...")
        vision_desc = describe_image(image_bytes, vision_prompt, vision_client, vision_deployment)

        # 告警/裝飾類圖片跳過
        if vision_desc.strip().startswith("[SKIP]"):
            print(f"  ⏭️ 跳過告警/裝飾圖片: {image_filename}")
            (image_save_dir / image_filename).unlink(missing_ok=True)
            image_counter -= 1
            _stats.vision_skipped += 1
            continue

        figures_info.append({
            "offset":      span.offset,
            "length":      span.length,
            "filename":    image_filename,
            "description": vision_desc,
        })

    # 逆序替換，保持偏移量正確
    figures_info.sort(key=lambda x: x["offset"], reverse=True)
    for fig in figures_info:
        replacement = (
            f"\n> ### 🖼️ 圖表解析: {fig['filename']}\n"
            f"> {fig['description']}\n"
        )
        s = fig["offset"]
        e = s + fig["length"]
        markdown_content = markdown_content[:s] + replacement + markdown_content[e:]

    return markdown_content, image_counter


# ─────────────────────────────────────────────────────────────
# 5. 主要文件處理函式
# ─────────────────────────────────────────────────────────────

def process_file(
    di_client: DocumentIntelligenceClient,
    file_path: str,
    output_dir: str,
    vision_prompt: str,
    doc_type: str = "",
    vision_client=None,
    vision_deployment: str | None = None,
) -> None:
    """
    使用 Azure Document Intelligence 解析單一 PDF/DOCX：
      - PDF  → 超過 _MAX_PAGES_PER_CHUNK 頁時自動拆分，分批送出後合併
      - DOCX → 直接送出（通常頁數較少；fitz 不支援 DOCX 圖表裁切）
      - 文字、表格 → 由 Azure DI 直接輸出 Markdown
      - 圖表 → 以 PyMuPDF 裁切後交由 Azure OpenAI Vision 描述，
               並將描述嵌回 Markdown 對應位置
    """
    file_path = Path(file_path)
    file_id   = file_path.stem
    is_pdf    = file_path.suffix.lower() == ".pdf"

    print(f"\n☁️  正在上傳並解析 [{doc_type}]: {file_path.name} ...")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    image_save_dir = output_dir_path / "images" / file_id
    image_save_dir.mkdir(parents=True, exist_ok=True)

    all_markdowns: list[str] = []
    image_counter = 0

    if is_pdf:
        # ── 拆分（必要時）並逐區塊處理 ─────────────────────────
        tmp_dir = output_dir_path / "_tmp" / file_id
        chunks = _split_pdf_to_chunks(file_path, tmp_dir)
        is_split = len(chunks) > 1

        for idx, (chunk_path, page_offset) in enumerate(chunks, start=1):
            if is_split:
                print(f"\n  📄 處理區塊 {idx}/{len(chunks)}: {chunk_path.name} ...")

            chunk_fitz_doc = fitz.open(str(chunk_path))
            try:
                md, image_counter = _process_pdf_chunk(
                    di_client, chunk_path, chunk_fitz_doc,
                    image_save_dir, image_counter, vision_prompt, page_offset,
                    vision_client, vision_deployment,
                )
            finally:
                chunk_fitz_doc.close()

            all_markdowns.append(md)

        # 清除暫存拆分檔案
        if is_split:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"\n  🗑️  已清除暫存區塊檔案")

    else:
        # ── DOCX：直接送出，圖片從 ZIP 提取後交 Vision 描述 ──────
        with open(file_path, "rb") as f:
            poller = di_client.begin_analyze_document(
                "prebuilt-layout",
                AnalyzeDocumentRequest(bytes_source=f.read()),
                output_content_format=DocumentContentFormat.MARKDOWN,
            )
        result = poller.result()
        _stats.di_pages += len(result.pages or [])
        markdown_content: str = result.content

        figures = result.figures or []
        docx_images = _extract_docx_images(file_path)

        if figures and docx_images:
            print(f"  🖼️  Azure DI 偵測到 {len(figures)} 圖表 / DOCX 內嵌圖片 {len(docx_images)} 張")
            figures_info = []
            for i, figure in enumerate(figures):
                if not figure.spans:
                    continue
                if i >= len(docx_images):
                    print(f"  ⚠️  圖表 #{i+1} 超出內嵌圖片數量，跳過")
                    break

                span = figure.spans[0]
                image_bytes = docx_images[i]

                image_counter += 1
                image_filename = f"image_{image_counter}.png"
                (image_save_dir / image_filename).write_bytes(image_bytes)

                print(f"  🔍 正在辨識圖片 {image_filename} ...")
                vision_desc = describe_image(image_bytes, vision_prompt, vision_client, vision_deployment)

                if vision_desc.strip().startswith("[SKIP]"):
                    print(f"  ⏭️ 跳過告警/裝飾圖片: {image_filename}")
                    (image_save_dir / image_filename).unlink(missing_ok=True)
                    image_counter -= 1
                    _stats.vision_skipped += 1
                    continue

                figures_info.append({
                    "offset":      span.offset,
                    "length":      span.length,
                    "filename":    image_filename,
                    "description": vision_desc,
                })

            # 逆序替換，保持偏移量正確
            figures_info.sort(key=lambda x: x["offset"], reverse=True)
            for fig in figures_info:
                replacement = (
                    f"\n> ### 🖼️ 圖表解析: {fig['filename']}\n"
                    f"> {fig['description']}\n"
                )
                s = fig["offset"]
                e = s + fig["length"]
                markdown_content = markdown_content[:s] + replacement + markdown_content[e:]

        elif docx_images:
            # Azure DI 未偵測到圖表，但 DOCX 確實有嵌入圖片 → 直接從 ZIP 提取描述，附加至文末
            print(f"  ⚠️  Azure DI 未回傳圖表資訊，改直接從 DOCX 提取 {len(docx_images)} 張內嵌圖片...")
            appended_sections: list[str] = []
            for i, image_bytes in enumerate(docx_images):
                image_counter += 1
                image_filename = f"image_{image_counter}.png"
                (image_save_dir / image_filename).write_bytes(image_bytes)

                print(f"  🔍 正在辨識圖片 {image_filename} ...")
                vision_desc = describe_image(image_bytes, vision_prompt, vision_client, vision_deployment)

                if vision_desc.strip().startswith("[SKIP]"):
                    print(f"  ⏭️ 跳過告警/裝飾圖片: {image_filename}")
                    (image_save_dir / image_filename).unlink(missing_ok=True)
                    image_counter -= 1
                    _stats.vision_skipped += 1
                    continue

                appended_sections.append(
                    f"\n> ### 🖼️ 圖表解析: {image_filename}\n"
                    f"> {vision_desc}\n"
                )

            if appended_sections:
                markdown_content += "\n\n## 附件圖表\n" + "\n".join(appended_sections)

        all_markdowns.append(markdown_content)

    # ── 合併所有區塊、轉換 HTML 表格並儲存 ──────────────────────
    markdown_content = "\n\n".join(all_markdowns)
    # markdown_content = _convert_html_tables(markdown_content)
    final_md_path = output_dir_path / f"{file_id}.md"

    with open(final_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    _stats.output_total_bytes += len(markdown_content.encode("utf-8"))

    print(f"✅ 解析完成！輸出路徑: {final_md_path}")


# ─────────────────────────────────────────────────────────────
# 6. 批次處理（與 parse_tenders.py 一致）
# ─────────────────────────────────────────────────────────────

def _print_stats(model_label: str | None = None) -> None:
    """印出本次執行的費用與資源使用統計摘要。"""
    _label = model_label or AZURE_DEPLOYMENT
    elapsed     = time.time() - _stats.start_time
    di_cost     = (_stats.di_pages / 1000) * _DI_PRICE_PER_1K_PAGES
    vision_cost = (
        (_stats.vision_input_tokens  / 1_000_000) * _VISION_INPUT_PRICE_PER_1M +
        (_stats.vision_output_tokens / 1_000_000) * _VISION_OUTPUT_PRICE_PER_1M
    )
    total_cost   = di_cost + vision_cost
    vision_kept  = _stats.vision_calls - _stats.vision_skipped

    print("\n" + "═" * 58)
    print("📊  本次執行統計摘要")
    print("═" * 58)
    print(f"  ⏱️  總執行時間     : {elapsed:.1f} 秒  ({elapsed / 60:.1f} 分鐘)")
    print(f"  📄 成功 / 失敗    : {_stats.files_processed} 個 / {_stats.files_failed} 個")
    print(f"  📦 輸出總大小     : {_stats.output_total_bytes / 1024:.1f} KB")
    print()
    print("  ── Azure Document Intelligence ──────────────────")
    print(f"  📑 分析頁數       : {_stats.di_pages:,} 頁")
    print(f"  💰 預估費用       : ${di_cost:.4f} USD")
    print(f"     （{_stats.di_pages:,} 頁 × ${_DI_PRICE_PER_1K_PAGES}/千頁）")
    print()
    print(f"  ── Vision 模型 ({_label}) ───────────────")
    print(f"  🖼️  呼叫次數       : {_stats.vision_calls} 次  (保留 {vision_kept} / 跳過 {_stats.vision_skipped})")
    print(f"  🔤 Token 使用     : 輸入 {_stats.vision_input_tokens:,}  / 輸出 {_stats.vision_output_tokens:,}")
    print(f"  💰 預估費用       : ${vision_cost:.4f} USD")
    print()
    print(f"  💵 本次總預估費用 : ${total_cost:.4f} USD")
    print("═" * 58)
    print("  ⚠️  費用為估算，實際請查閱 Azure Portal 帳單")
    print("═" * 58)


def batch_process_folders(only_type: str = None, only_file: str = None) -> None:
    """
    依照 DOC_TASKS 設定，按文件類型分別掃描來源資料夾，輸出到獨立目錄。
    每個 dest 目錄將來對應一個獨立的 GraphRAG 索引工作區。

    only_type: 若指定，只處理該 doc_type；None 表示處理全部。
    only_file: 若指定，只處理該檔名（含副檔名，例如 foo.pdf）；None 表示處理全部。
    """
    global _stats
    _stats = _RunStats()
    tasks_to_run = [t for t in DOC_TASKS if only_type is None or t["doc_type"] == only_type]

    if not tasks_to_run:
        print(f"⚠️  找不到 doc_type='{only_type}' 的任務，可用類型: {[t['doc_type'] for t in DOC_TASKS]}")
        return

    di_client = DocumentIntelligenceClient(AZURE_DI_ENDPOINT, AzureKeyCredential(AZURE_DI_KEY))

    for task in tasks_to_run:
        src           = task["src"]
        dest          = task["dest"]
        doc_type      = task["doc_type"]
        description   = task["description"]
        vision_prompt = task["vision_prompt"]

        if not os.path.exists(src):
            print(f"⚠️  來源資料夾不存在，跳過: {src}")
            continue

        print(f"\n📂 開始處理類型 [{doc_type}] {description}，來源: {src} → 輸出: {dest}")
        os.makedirs(dest, exist_ok=True)

        all_files = [f for f in os.listdir(src) if f.lower().endswith((".pdf", ".docx"))]

        if only_file:
            files = [f for f in all_files if f == only_file]
            if not files:
                print(f"  ⚠️ 在 {src} 中找不到指定檔案: {only_file}")
                print(f"     可用檔案: {all_files}")
                continue
        else:
            files = all_files

        if not files:
            print(f"  (該資料夾內沒有 PDF/DOCX 檔案)")
            continue

        for file in files:
            try:
                process_file(
                    di_client=di_client,
                    file_path=os.path.join(src, file),
                    output_dir=dest,
                    vision_prompt=vision_prompt,
                    doc_type=doc_type,
                )
                _stats.files_processed += 1
            except Exception as e:
                print(f"❌ 處理 {file} 時發生錯誤: {e}")
                _stats.files_failed += 1

    print("\n" + "-" * 40)
    print("🎉 批次任務完成！現在可以執行 GraphRAG Index 了。")
    _print_stats()


# ─────────────────────────────────────────────────────────────
# 7. CLI 入口
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="標案文件轉 Markdown 工具 (Azure Document Intelligence 版)"
    )
    parser.add_argument(
        "--type",
        dest="doc_type",
        default=None,
        choices=[t["doc_type"] for t in DOC_TASKS],
        help="只處理指定類型的資料夾，例如: --type government",
    )
    parser.add_argument(
        "--file",
        dest="only_file",
        default=None,
        help="只處理指定檔案（含副檔名），需搭配 --type 使用，例如: --type stip --file foo.pdf",
    )
    args = parser.parse_args()

    if args.only_file and not args.doc_type:
        parser.error("--file 需要搭配 --type 一起使用")

    batch_process_folders(only_type=args.doc_type, only_file=args.only_file)