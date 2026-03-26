# 檔案一：標案解析與圖片提取
import argparse
import os
import base64
import io
from openai import AzureOpenAI
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import DocItemLabel
from dotenv import load_dotenv
from pathlib import Path

# 0. 手動讀取 .env 檔案
load_dotenv()

# 1. 讀取 AI 模式設定 (預設為 gemini)
AI_MODE = os.getenv("AI_MODE", "gemini").lower()
print(f"🤖 AI 模式: {AI_MODE.upper()}")

# 1. 指定您的本地模型路徑
LOCAL_ARTIFACTS_PATH = Path("D:/docling_artifacts/docling-models")

# ─────────────────────────────────────────────────────────────
# 2. 文件類型任務設定：來源資料夾 → (輸出資料夾, 描述, 視覺依詢 Prompt)
# 每個類型獨立輸出，未來 GraphRAG 索引將分別執行
# ─────────────────────────────────────────────────────────────
DOC_TASKS = [
    {
        "src":         "./rag_poc/input/government",
        "dest":        "./ragtest/government/input",   # 獨立索引目錄
        "doc_type":    "government",
        "description": "政府標案",
        "vision_prompt": (
            "這是一張政府標案文件中的圖表。"
            "請詳細描述內容（如設備規格、工程場地願圖、機房佈線機制、驗收程序或金額表），"
            "並專注標案投標資格、違約條款或年限要求等重要標案內容，以繁體中文回答。"
        ),
    },
    {
        "src":         "./rag_poc/input/stip",
        "dest":        "./ragtest/stip/input",          # 獨立索引目錄
        "doc_type":    "stip",
        "description": "門市銷售指南",
        "vision_prompt": (
            "這是一張門市銷售指南中的圖表。"
            "請判斷此圖是否屬於以下任一類型："
            "『告警、警示、禁止符號、裝飾圖案、卡通人物、動畫角色、應用程式 logo、流程示意 icon、空白圖片、簣明裝飾』。"
            "若是，請只回覆 [SKIP]，不要有任何其他文字。"
            "若否（例如商品照片、方案表格、促銷條件、操作步驟），"
            "請用繁體中文簡短描述圖片內容，不超過150字，不需要零售建議或總結。"
        ),
    },
    # 新增類型往此列表新增一筆即可，不需修改其他程式碼
]

# ─────────────────────────────────────────────────────────────
# 3. 依模式初始化對應的 AI 客戶端
# ─────────────────────────────────────────────────────────────
if AI_MODE == "azure":
    azure_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )
    AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    vision_model = None
    print(f"👉 Azure 部署名稱: {AZURE_DEPLOYMENT}")

else:
    raise ValueError(f"不支援的 AI_MODE：'{AI_MODE}'，請設定為 'gemini' 或 'azure'")


def _pil_to_base64(pil_image) -> str:
    """將 PIL 圖片轉換為 base64 字串 (Azure 用)"""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def describe_image(pil_image, vision_prompt: str) -> str:
    """統一入口：依 AI_MODE 選擇 Gemini 或 Azure OpenAI 進行圖片描述
    vision_prompt 來自各文件類型的設定，描述重點會不同
    """
    if AI_MODE == "gemini":
        try:
            response = vision_model.generate_content([vision_prompt, pil_image])
            return response.text
        except Exception as e:
            return f"[Gemini 圖片解析失敗: {str(e)}]"

    elif AI_MODE == "azure":
        try:
            b64 = _pil_to_base64(pil_image)
            response = azure_client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
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
                # max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Azure 圖片解析失敗: {str(e)}]"

def process_tender_with_images_v2(file_path, output_dir, vision_prompt: str, doc_type: str = ""):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.artifacts_path = LOCAL_ARTIFACTS_PATH  # 👈 強制指向本地
    pipeline_options.do_ocr = False          # 數位 PDF 不需要 OCR，略過 RapidOCR 模型
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST  # 省記憶體模式（複雜表格用 ACCURATE）
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 4.16
    format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    converter = DocumentConverter(format_options=format_options)
    result = converter.convert(file_path)
    doc = result.document
    
    file_id = os.path.splitext(os.path.basename(file_path))[0]
    image_save_dir = os.path.join(output_dir, "images", file_id)
    os.makedirs(image_save_dir, exist_ok=True)

    md_output = []
    image_counter = 0

    print(f"開始處理 [{doc_type}]: {file_id} ...")

    for item, level in doc.iterate_items():
        # 1. 優先判斷是否為表格
        if item.label == DocItemLabel.TABLE:
            print("  📊 偵測到表格，正在轉換結構...")
            # 傳入 doc 讓 RichTableCell 能正確解析嵌套內容，否則會輸出 <!-- rich cell -->
            table_md = item.export_to_markdown(doc=doc)
            md_output.append(f"\n{table_md}\n")
        # 2. 判斷是否為圖片     
        elif item.label == DocItemLabel.PICTURE:
            # 1. 取得座標資訊 (Bounding Box)
            # prov 是一個列表，通常取第一個元素
            bbox = item.prov[0].bbox if item.prov else None
            if bbox:
                width = bbox.r - bbox.l
                height = bbox.b - bbox.t

                # 策略一：尺寸過濾 (過濾掉太小的圖)
                # 標案中的 Logo 或 Emoji 通常寬高小於 40-50 像素
                if width < 50 or height < 50:
                    print(f"  ⏭️ 忽略小型圖片 (Size: {width:.1f}x{height:.1f})")
                    continue

                # 策略二：位置過濾 (過濾掉頁首頁尾)
                # 假設頁面高度為 842 (A4 點數)，頂部 50 與底部 50 通常是頁首頁尾
                # bbox.t 是頂部座標, bbox.b 是底部座標
                if bbox.t < 50 or bbox.b > 790:
                    print(f"  ⏭️ 忽略邊界圖片 (Location: top={bbox.t:.1f}, bottom={bbox.b:.1f})")
                    continue

            # B. 處理圖片 (增加 NoneType 安全檢查)
            if hasattr(item, "image") and item.image is not None:
                print(f"  🖼️ 偵測到有效圖片，正在處理...")
                image_counter += 1
                image_filename = f"image_{image_counter}.png"
                image_path = os.path.join(image_save_dir, image_filename)
                
                # 安全存取 pil_image
                try:
                    pil_img = item.image.pil_image
                    if pil_img:
                        # 1. 實體儲存圖片檔案 (用於備份)
                        pil_img.save(image_path)
                        
                        # 2. 呼叫 Vision AI 獲取文字描述
                        print(f"  🔍 [{AI_MODE.upper()}] 正在辨識圖片 {image_filename}...")
                        vision_description = describe_image(pil_img, vision_prompt)
                        
                        # 告警/裝飾類圖片跳過
                        if vision_description.strip().startswith("[SKIP]"):
                            print(f"  ⏭️ 跳過告警/裝飾圖片: {image_filename}")
                            continue
                        
                        # 3. 將描述直接嵌入 Markdown 位置
                        md_output.append(f"\n> ### 🖼️ 圖表解析: {image_filename}")
                        md_output.append(f"> {vision_description}\n")
                    else:
                        md_output.append("\n> [圖片內容為空，無法解析]\n")
                except Exception as e:
                    print(f"  ⚠️ 圖片處理異常: {str(e)}")
                    md_output.append(f"\n> [圖片處理失敗: {image_filename}]\n")        
        # elif hasattr(item, "text") and not hasattr(item, "image"):
        #         md_output.append(item.text)
        # C. 文字處理 (TEXT / TITLE / SECTION_HEADER / LIST_ITEM 等)
        elif hasattr(item, "text") and not hasattr(item, "image"):
            text_content = getattr(item, "text", "").strip()
            if not text_content:
                continue

            # --- 標題層級邏輯開始 ---
            # 判斷是否為標題類型的標籤
            header_labels = [DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER]
            
            if item.label in header_labels:
                # docling 的 level 通常從 0 或 1 開始
                # Markdown 標題限制為 1-6 層
                h_level = min(max(level + 1, 1), 6) 
                md_prefix = "#" * h_level
                md_output.append(f"\n{md_prefix} {text_content}\n")
            
            # 判斷是否為列表項目 (List Item)
            elif item.label == DocItemLabel.LIST_ITEM:
                md_output.append(f"* {text_content}")
            
            # 一般段落
            else:
                md_output.append(text_content)
            # --- 標題層級邏輯結束 ---
        elif hasattr(item, "export_to_markdown") and not hasattr(item, "image"):
                md_output.append(item.export_to_markdown())            

    # 儲存最終 Markdown
    final_md_path = os.path.join(output_dir, f"{file_id}.md")
    with open(final_md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(md_output))
    
    print(f"✨ POC 文件解析完成！輸出路徑: {final_md_path}")

def batch_process_folders(only_type: str = None, only_file: str = None):
    """
    依照 DOC_TASKS 設定，按文件類型分別掃描來源資料夾，輸出到獨立目錄。
    每個 dest 目錄將來對應一個獨立的 GraphRAG 索引工作區。

    only_type: 若指定，只處理該 doc_type；None 表示處理全部。
    only_file: 若指定，只處理該檔名（含副檔名，例如 foo.pdf）；None 表示處理全部。
    """
    tasks_to_run = [t for t in DOC_TASKS if only_type is None or t["doc_type"] == only_type]

    if not tasks_to_run:
        print(f"⚠️  找不到 doc_type='{only_type}' 的任務，可用類型: {[t['doc_type'] for t in DOC_TASKS]}")
        return
    for task in tasks_to_run:
        src          = task["src"]
        dest         = task["dest"]
        doc_type     = task["doc_type"]
        description  = task["description"]
        vision_prompt = task["vision_prompt"]

        if not os.path.exists(src):
            print(f"⚠️  來源資料夾不存在，跳過: {src}")
            continue

        print(f"\n📂 開始處理類型 [{doc_type}] {description}，來源: {src} → 輸出: {dest}")
        os.makedirs(dest, exist_ok=True)

        all_files = [f for f in os.listdir(src) if f.lower().endswith(('.pdf', '.docx'))]

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
            full_path = os.path.join(src, file)
            process_tender_with_images_v2(
                file_path=full_path,
                output_dir=dest,
                vision_prompt=vision_prompt,
                doc_type=doc_type,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解析文件並轉換為 Markdown")
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
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="僅下載 Docling HuggingFace 模型並退出（建議首次使用前先執行）",
    )
    args = parser.parse_args()

    if args.only_file and not args.doc_type:
        parser.error("--file 需要搭配 --type 一起使用")

    batch_process_folders(only_type=args.doc_type, only_file=args.only_file)

