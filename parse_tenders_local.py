"""
parse_tenders_local.py — Local 陣營文件解析模組

架構：
  文字 / 表格  → markitdown（內部使用 pdfminer + pdfplumber，本地執行）
  圖片擷取     → PyMuPDF page.get_images()（PDF）/ zipfile word/media/（DOCX）
  圖片描述     → Ollama Vision（OpenAI-compat API，本地推論）

介面契約（與 parse_tenders_azure.py 對齊）：
  公開 process_file(file_path, output_dir, vision_prompt, doc_type,
                    vision_client, vision_deployment)  ← 相同呼叫簽名（無 di_client）
  公開 LOCAL_MODELS, build_local_client(), _print_stats()

新增文件類型：修改 doc_tasks.py 即可，本檔案不需修改。

─────────────────────────────────────────────────────────────────────────────
MarkItDown PDF 圖片限制說明
─────────────────────────────────────────────────────────────────────────────
  markitdown 的 PDF converter 使用 pdfminer + pdfplumber 擷取文字與表格，
  llm_caption 模組只被 DOCX / Image converter 呼叫，**PDF 沒有圖片描述功能**。

本地版補充方案（PyMuPDF）：
  ┌──────────────────────┬─────────────────────────────────────────┐
  │ 項目                 │ 說明                                    │
  ├──────────────────────┼─────────────────────────────────────────┤
  │ 擷取方式             │ page.get_images() → extract_image(xref) │
  │ 圖片類型             │ 點陣圖（JPEG/PNG/BMP…）                  │
  │ 向量圖（SVG）        │ ❌ 無法擷取，需改用頁面截圖（page render）│
  │ 圖片位置             │ 僅知道所在頁碼，無 Bounding Box         │
  │ 合併方式             │ 附加至文末「## 文件圖表」區塊            │
  │ 對比 Azure DI        │ DI 有 figure bounding box，可插入文中   │
  └──────────────────────┴─────────────────────────────────────────┘

向量圖補充（可選）：若 PDF 含大量向量圖（組織架構、系統架構圖），
  可改用 page.get_pixmap().tobytes() 將整頁渲染後送 Vision 描述（費時）。
  本模組預設關閉，可透過環境變數 OLLAMA_RENDER_FULL_PAGE=1 開啟。
"""
from __future__ import annotations

import hashlib
import os
import io
import re
import uuid
import base64
import argparse
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import fitz  # PyMuPDF — 圖片擷取（PDF）
from openai import OpenAI
from dotenv import load_dotenv

from doc_tasks import DOC_TASKS, get_dest

from markitdown import MarkItDown

load_dotenv()

# ─────────────────────────────────────────────────────────────
# 1. 本地 Vision 模型設定（Ollama）
# ─────────────────────────────────────────────────────────────

# 費用估算：本地推論設為 0（依實際 GPU 電費可另行設定）
_LOCAL_INPUT_PRICE_PER_1M  = 0.0
_LOCAL_OUTPUT_PRICE_PER_1M = 0.0


def _load_local_models() -> dict[str, dict]:
    """
    掃描 .env 中的 Ollama 設定，建立本地 Vision 模型清單。
    OLLAMA_BASE_URL 未設定時回傳空字典（UI 不顯示 ollama 選項）。

    支援多個 Vision 模型（透過 OLLAMA_VISION_MODEL_2、_3 擴充）：
      OLLAMA_BASE_URL=http://localhost:11434/v1
      OLLAMA_VISION_MODEL=qwen2.5-vl:7b       ← 第 1 個（預設）
      OLLAMA_VISION_MODEL_2=llava:13b          ← 第 2 個（可選）
    """
    models: dict[str, dict] = {}
    base_url = os.getenv("OLLAMA_BASE_URL")
    if not base_url:
        return models

    suffixes = [""] + [f"_{i}" for i in range(2, 6)]
    for sfx in suffixes:
        vision_model = os.getenv(f"OLLAMA_VISION_MODEL{sfx}")
        if not vision_model:
            if sfx:
                break
            vision_model = "qwen2.5-vl:7b"  # 無後綴且未設定時使用預設值

        safe_key = vision_model.replace(":", "_").replace(".", "_")
        model_key = f"ollama_{safe_key}"
        models[model_key] = {
            "label":               f"Ollama {vision_model}",
            "type":                "ollama",
            "api_key":             "ollama",
            "base_url":            base_url,
            "deployment":          vision_model,
            "input_price_per_1m":  _LOCAL_INPUT_PRICE_PER_1M,
            "output_price_per_1m": _LOCAL_OUTPUT_PRICE_PER_1M,
        }

    return models


LOCAL_MODELS: dict[str, dict] = _load_local_models()
DEFAULT_LOCAL_MODEL_ID: str | None = next(iter(LOCAL_MODELS), None)

_discovered = ", ".join(LOCAL_MODELS.keys()) if LOCAL_MODELS else "（未設定 OLLAMA_BASE_URL）"
print(f"🏠 已載入 Local Vision 模型: {_discovered}（預設: {DEFAULT_LOCAL_MODEL_ID}）")

# 是否將整頁渲染後送 Vision（補充向量圖）—— 預設關閉
_RENDER_FULL_PAGE = os.getenv("OLLAMA_RENDER_FULL_PAGE", "0") == "1"
_PAGE_RENDER_SCALE = 2.0   # 渲染解析度倍率（過高會很慢）


def build_local_client(model_id: str | None = None) -> tuple:
    """
    依 model_id 建立 Ollama OpenAI-compat 客戶端。
    回傳 (client, deployment, label) 元組，與 parse_tenders_azure.build_vision_client 介面相同。
    """
    cfg = LOCAL_MODELS.get(model_id or DEFAULT_LOCAL_MODEL_ID or "")
    if cfg is None:
        if not LOCAL_MODELS:
            raise RuntimeError(
                "未設定 OLLAMA_BASE_URL，無法使用 Local 解析模式。\n"
                "請在 .env 加入：OLLAMA_BASE_URL=http://localhost:11434/v1"
            )
        cfg = next(iter(LOCAL_MODELS.values()))

    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
    )
    return client, cfg["deployment"], cfg["label"]


# ─────────────────────────────────────────────────────────────
# 2. 圖片過濾常數
# ─────────────────────────────────────────────────────────────

_MIN_IMAGE_WIDTH_PX  = 100   # 最小有效圖片寬度（像素）
_MIN_IMAGE_HEIGHT_PX = 100   # 最小有效圖片高度（像素）
# PyMuPDF 擷取困難的格式，轉為 Pixmap 時可能報錯
_SKIP_IMG_EXTENSIONS = {"jbig2", "jpx"}


# ─────────────────────────────────────────────────────────────
# 3. 執行統計
# ─────────────────────────────────────────────────────────────

@dataclass
class _RunStats:
    files_processed:    int   = 0
    files_failed:       int   = 0
    pages_processed:    int   = 0
    images_extracted:   int   = 0   # 過濾前擷取到的圖片數
    vision_calls:       int   = 0
    vision_skipped:     int   = 0
    vision_input_tokens:  int = 0
    vision_output_tokens: int = 0
    output_total_bytes: int   = 0
    start_time: float = field(default_factory=time.time)


_stats = _RunStats()


# ─────────────────────────────────────────────────────────────
# 4. 圖片擷取
# ─────────────────────────────────────────────────────────────

def _extract_pdf_images(pdf_path: Path) -> list[tuple[int, bytes]]:
    """
    使用 PyMuPDF 從 PDF 擷取嵌入的點陣圖。

    回傳 [(page_num_1based, png_bytes), ...]

    注意：
      - 已按 xref 去重，相同物件跨頁重複出現（如頁眉 Logo）只描述一次
      - 向量圖（Form XObject / SVG）無法用此方式擷取
      - OLLAMA_RENDER_FULL_PAGE=1 時，另外對每頁做整頁截圖補充向量圖
    """
    result:     list[tuple[int, bytes]] = []
    seen_xrefs: set[int] = set()

    doc = fitz.open(str(pdf_path))
    _stats.pages_processed += len(doc)

    for page_idx in range(len(doc)):
        page     = doc[page_idx]
        page_num = page_idx + 1

        # ── 點陣圖（XObject 圖片）────────────────────────────────
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue  # 已處理過（跨頁重複），略過
            seen_xrefs.add(xref)

            try:
                base_image = doc.extract_image(xref)
                ext    = base_image.get("ext", "").lower()
                width  = base_image.get("width",  0)
                height = base_image.get("height", 0)

                if ext in _SKIP_IMG_EXTENSIONS:
                    continue
                if width < _MIN_IMAGE_WIDTH_PX or height < _MIN_IMAGE_HEIGHT_PX:
                    continue

                _stats.images_extracted += 1

                # 統一轉為 PNG（CMYK → RGB 先轉）
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:                  # CMYK / 其他多通道
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                png_bytes = pix.tobytes("png")
                pix = None

                result.append((page_num, png_bytes))

            except Exception as e:
                print(f"  ⚠️  第 {page_num} 頁圖片擷取失敗 (xref={xref}): {e}")

        # ── 整頁渲染（補充向量圖，預設關閉）──────────────────────
        if _RENDER_FULL_PAGE:
            mat = fitz.Matrix(_PAGE_RENDER_SCALE, _PAGE_RENDER_SCALE)
            pix = page.get_pixmap(matrix=mat)
            result.append((page_num, pix.tobytes("png")))
            pix = None

    doc.close()
    return result


def _extract_docx_images(file_path: Path) -> list[bytes]:
    """
    從 DOCX（ZIP 格式）的 word/media/ 取出所有嵌入點陣圖 bytes。
    與 parse_tenders_azure.py 中同名函式邏輯相同。
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


# ─────────────────────────────────────────────────────────────
# 5. 圖片描述（Ollama Vision）
# ─────────────────────────────────────────────────────────────

def describe_image(
    image_bytes:       bytes,
    vision_prompt:     str,
    client             = None,
    deployment:        str | None = None,
) -> str:
    """
    呼叫 Ollama Vision（OpenAI-compat）對圖片進行描述。
    client 為 None 時回傳提示字串（不影響主流程）。
    """
    if client is None:
        return "[未提供 Vision 客戶端，圖片略過]"

    _stats.vision_calls += 1
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        response = client.chat.completions.create(
            model=deployment or "qwen2.5-vl:7b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": vision_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
        )
        if response.usage:
            _stats.vision_input_tokens  += response.usage.prompt_tokens
            _stats.vision_output_tokens += response.usage.completion_tokens
        return response.choices[0].message.content

    except Exception as e:
        return f"[Ollama Vision 解析失敗: {e}]"


# ─────────────────────────────────────────────────────────────
# 5.5 Markdown 二次加工
# ─────────────────────────────────────────────────────────────

def _enrich_markdown(raw_text: str) -> tuple[str, int, str]:
    """
    Markdown 二次加工：偵測頁碼、清理雜訊、產生內容 hash。

    步驟：
      A. 依 Form Feed (\\f) 或明確分頁符號拆頁，插入可見頁碼標記
      B. 清理殘留的 [SKIP] 旗標（Vision 模型跳過標記）
      C. 生成 MD5 hash 供版本追蹤

    回傳 (enriched_text, total_pages, content_hash)。
    """
    # A. 依分頁符號拆頁並插入頁碼標記
    pages = re.split(r'\f|---pagebreak---|<!--\s*PageBreak\s*-->', raw_text)
    enriched_pages = []
    for i, page_content in enumerate(pages):
        page_marker = f"\n\n**[第 {i + 1} 頁]**\n"
        enriched_pages.append(page_marker + page_content.strip())
    content = "\n".join(enriched_pages)

    # B. 清理殘留的 [SKIP] 標記
    content = re.sub(r'\[SKIP\]\n?', '', content)

    # C. 生成內容 hash（MD5，用於版本比對）
    content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

    return content, len(pages), content_hash


# ─────────────────────────────────────────────────────────────
# 6. 主要文件處理函式（介面契約與 parse_tenders_azure.py 對齊）
# ─────────────────────────────────────────────────────────────

def process_file(
    file_path:          str,
    output_dir:         str,
    vision_prompt:      str,
    doc_type:           str        = "",
    vision_client                  = None,
    vision_deployment:  str | None = None,
    file_id:            str | None = None,   # 自訂文件 ID；None 時自動生成 {stem}_{uuid8}
    # ── 注意：無 di_client 參數（Local 版無需 Azure DI）──────
) -> None:
    """
    本地解析單一 PDF / DOCX：

      PDF：
        文字 / 表格  → markitdown（pdfminer + pdfplumber）
        點陣圖       → PyMuPDF extract_image() → Ollama Vision
        向量圖       → 需開啟 OLLAMA_RENDER_FULL_PAGE=1（整頁截圖）

      DOCX：
        文字 / 表格  → markitdown
        圖片         → zipfile word/media/ → Ollama Vision

      圖片描述結果附加於文末「## 文件圖表」區塊。
      vision_client=None 時以純文字模式輸出（不呼叫 Ollama）。
      file_id 指定時寫入 YAML；None 時自動生成 "{stem}_{uuid8}"（輸出檔名不受影響）。
    """    
    file_path_obj  = Path(file_path)
    _output_stem   = file_path_obj.stem                        # 輸出 .md 的檔名（固定用原始檔名）
    file_id        = file_id or f"{_output_stem}_{uuid.uuid4().hex[:8]}"  # YAML 用的唯一 ID
    is_pdf         = file_path_obj.suffix.lower() == ".pdf"

    print(f"\n🏠 本地解析 [{doc_type}]: {file_path_obj.name} ...")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    image_save_dir = output_dir_path / "images" / _output_stem
    image_save_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 文字 / 表格擷取（markitdown）────────────────────────
    #   注意：不傳 llm_client，圖片由下方 PyMuPDF 流程處理
    md_converter = MarkItDown()
    try:
        conversion    = md_converter.convert(str(file_path_obj))
        markdown_text = conversion.text_content or ""
        print(f"  📝 markitdown 擷取文字：{len(markdown_text):,} 字元")
    except Exception as e:
        print(f"  ❌ markitdown 轉換失敗: {e}")
        markdown_text = f"<!-- markitdown 轉換失敗: {e} -->"

    # ── 2. 圖片擷取 ─────────────────────────────────────────────
    images_with_meta: list[tuple[str, bytes]] = []   # (label, png_bytes)

    if is_pdf:
        page_images = _extract_pdf_images(file_path_obj)
        print(f"  🖼️  PyMuPDF 擷取到 {len(page_images)} 張點陣圖（過濾後）")
        for page_num, img_bytes in page_images:
            images_with_meta.append((f"第 {page_num} 頁", img_bytes))
    else:
        docx_images = _extract_docx_images(file_path_obj)
        print(f"  🖼️  DOCX 內嵌圖片 {len(docx_images)} 張")
        for i, img_bytes in enumerate(docx_images, start=1):
            images_with_meta.append((f"圖片 {i}", img_bytes))

    # ── 3. Vision 描述 ──────────────────────────────────────────
    image_sections: list[str] = []
    image_counter = 0

    if vision_client is not None:
        for label, img_bytes in images_with_meta:
            image_counter += 1
            image_filename = f"image_{image_counter}.png"
            (image_save_dir / image_filename).write_bytes(img_bytes)

            print(f"  🔍 辨識 {image_filename}（{label}）...")
            vision_desc = describe_image(img_bytes, vision_prompt, vision_client, vision_deployment)

            if vision_desc.strip().startswith("[SKIP]"):
                print(f"  ⏭️  跳過裝飾性圖片: {image_filename}")
                (image_save_dir / image_filename).unlink(missing_ok=True)
                image_counter -= 1
                _stats.vision_skipped += 1
                continue

            image_sections.append(
                f"\n> ### 🖼️ 圖表解析: {image_filename}（{label}）\n"
                f"> {vision_desc}\n"
            )
    else:
        print("  ℹ️  vision_client=None，略過圖片描述（純文字模式）")

    # ── 4. 合併文字與圖片描述 ────────────────────────────────────
    if image_sections:
        markdown_text += "\n\n## 文件圖表\n" + "\n".join(image_sections)

    # ── 4.5 Markdown 二次加工（頁碼標記、雜訊清理、hash）────────
    enriched_text, total_pages, content_hash = _enrich_markdown(markdown_text)

    # ── 5. 插入文件元數據 ────────────────────────────────────────
    _doc_desc   = next((t["description"] for t in DOC_TASKS if t["doc_type"] == doc_type), doc_type)
    _parse_date = datetime.now().strftime("%Y-%m-%d")
    _file_mtime = datetime.fromtimestamp(file_path_obj.stat().st_mtime).strftime("%Y-%m-%d")
    _metadata_header = (
        f"---\n"
        f"來源檔案: {file_path_obj.name}\n"
        f"文件類型: {doc_type} ({_doc_desc})\n"
        f"解析引擎: markitdown + PyMuPDF (Local)\n"
        f"Vision  : {vision_deployment or 'N/A'}\n"
        f"file_id : {file_id}\n"
        f"原始修改日期: {_file_mtime}\n"
        f"解析日期: {_parse_date}\n"
        f"總頁數: {total_pages}\n"
        f"hash: {content_hash}\n"
        f"---\n\n"
    )
    _footer = (
        f"\n\n---\n"
        f"> **來源溯源**：{file_path_obj.name}"
        f" | 類型：{doc_type} ({_doc_desc})"
        f" | 解析日期：{_parse_date}"
    )
    final_content = _metadata_header + enriched_text + _footer

    # ── 6. 寫出 .md ──────────────────────────────────────────────
    final_md_path = output_dir_path / f"{_output_stem}.md"
    with open(final_md_path, "w", encoding="utf-8") as f:
        f.write(final_content)
    _stats.output_total_bytes += len(final_content.encode("utf-8"))

    print(f"  ✅ 解析完成！輸出: {final_md_path}")


# ─────────────────────────────────────────────────────────────
# 7. 統計輸出
# ─────────────────────────────────────────────────────────────

def _print_stats(model_label: str | None = None, model_id: str | None = None) -> None:
    """印出本次執行統計摘要（Local 版，費用顯示為 $0）。"""
    elapsed      = time.time() - _stats.start_time
    vision_kept  = _stats.vision_calls - _stats.vision_skipped
    bar          = "═" * 58

    print(f"\n{bar}")
    print("📊  本次執行統計摘要（Local 模式）")
    print(bar)
    print(f"  ⏱️  總執行時間     : {elapsed:.1f} 秒  ({elapsed / 60:.1f} 分鐘)")
    print(f"  📄 成功 / 失敗    : {_stats.files_processed} 個 / {_stats.files_failed} 個")
    print(f"  📑 處理頁數       : {_stats.pages_processed} 頁")
    print(f"  📦 輸出總大小     : {_stats.output_total_bytes / 1024:.1f} KB")
    print()
    print("  ── 解析引擎 ─────────────────────────────────────")
    print("  文字 / 表格       : markitdown（pdfminer + pdfplumber，本地）")
    print("  圖片擷取（PDF）   : PyMuPDF page.get_images()（本地）")
    print(f"  整頁渲染模式      : {'開啟' if _RENDER_FULL_PAGE else '關閉（OLLAMA_RENDER_FULL_PAGE=1 可開啟）'}")
    print()
    print(f"  ── Vision 模型 ({model_label or 'N/A'}) ─────────")
    print(f"  🖼️  擷取圖片       : {_stats.images_extracted} 張（原始）")
    print(f"  🤖 呼叫次數       : {_stats.vision_calls} 次  (保留 {vision_kept} / 跳過 {_stats.vision_skipped})")
    if _stats.vision_input_tokens:
        print(f"  🔤 Token 使用    : 輸入 {_stats.vision_input_tokens:,} / 輸出 {_stats.vision_output_tokens:,}")
    print(f"  💰 預估費用       : $0.00 USD（本地推論）")
    print()
    print(f"  💵 本次總費用     : $0.00 USD（本地模式，不計 GPU 電費）")
    print(bar)


# ─────────────────────────────────────────────────────────────
# 8. 批次處理
# ─────────────────────────────────────────────────────────────

def batch_process_folders(only_type: str = None, only_file: str = None) -> None:
    """依照 DOC_TASKS 設定批次解析（Local 版）。"""
    global _stats
    _stats = _RunStats()

    vision_client, vision_deployment, model_label = build_local_client()

    tasks_to_run = [t for t in DOC_TASKS if only_type is None or t["doc_type"] == only_type]
    if not tasks_to_run:
        print(f"⚠️  找不到 doc_type='{only_type}'，可用: {[t['doc_type'] for t in DOC_TASKS]}")
        return

    for task in tasks_to_run:
        src           = task["src"]
        doc_type      = task["doc_type"]
        dest          = get_dest(doc_type, "local")
        description   = task["description"]
        vision_prompt = task["vision_prompt"]

        if not os.path.exists(src):
            print(f"⚠️  來源資料夾不存在，跳過: {src}")
            continue

        print(f"\n📂 開始處理 [{doc_type}] {description}：{src} → {dest}")
        os.makedirs(dest, exist_ok=True)

        all_files = [f for f in os.listdir(src) if f.lower().endswith((".pdf", ".docx"))]
        files     = [f for f in all_files if f == only_file] if only_file else all_files

        if not files:
            print("  (沒有可解析的 PDF/DOCX 檔案)")
            continue

        for file in files:
            try:
                process_file(
                    file_path         = os.path.join(src, file),
                    output_dir        = dest,
                    vision_prompt     = vision_prompt,
                    doc_type          = doc_type,
                    vision_client     = vision_client,
                    vision_deployment = vision_deployment,
                )
                _stats.files_processed += 1
            except Exception as e:
                print(f"❌ 處理 {file} 失敗: {e}")
                _stats.files_failed += 1

    print("\n" + "-" * 40)
    print("🎉 本地批次解析完成！")
    _print_stats(model_label)


# ─────────────────────────────────────────────────────────────
# 9. CLI 入口
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="標案文件轉 Markdown — Local / Ollama 版（markitdown + PyMuPDF）"
    )
    parser.add_argument(
        "--type",
        dest="doc_type",
        default=None,
        choices=[t["doc_type"] for t in DOC_TASKS],
        help="只處理指定類型，例如: --type stip",
    )
    parser.add_argument(
        "--file",
        dest="only_file",
        default=None,
        help="只處理指定檔案（含副檔名），需搭配 --type 使用",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="純文字模式：略過圖片描述（不呼叫 Ollama）",
    )
    args = parser.parse_args()

    if args.only_file and not args.doc_type:
        parser.error("--file 需要搭配 --type 一起使用")

    if args.no_vision:
        # 純文字模式：不初始化 Ollama 客戶端
        _stats = _RunStats()
        tasks_to_run = [t for t in DOC_TASKS if args.doc_type is None or t["doc_type"] == args.doc_type]
        for task in tasks_to_run:
            src, dest = task["src"], task["dest"]
            if not os.path.exists(src):
                continue
            os.makedirs(dest, exist_ok=True)
            files = [f for f in os.listdir(src) if f.lower().endswith((".pdf", ".docx"))]
            if args.only_file:
                files = [f for f in files if f == args.only_file]
            for file in files:
                try:
                    process_file(
                        file_path=os.path.join(src, file),
                        output_dir=dest,
                        vision_prompt=task["vision_prompt"],
                        doc_type=task["doc_type"],
                        vision_client=None,
                    )
                    _stats.files_processed += 1
                except Exception as e:
                    print(f"❌ {file}: {e}")
                    _stats.files_failed += 1
        _print_stats("N/A（純文字模式）")
    else:
        batch_process_folders(only_type=args.doc_type, only_file=args.only_file)
