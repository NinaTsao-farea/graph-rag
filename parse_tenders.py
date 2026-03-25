# 檔案一：標案解析與圖片提取
import os
import base64
import io
import google.generativeai as genai
from openai import AzureOpenAI
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from dotenv import load_dotenv

# 0. 手動讀取 .env 檔案
load_dotenv()

# 1. 讀取 AI 模式設定 (預設為 gemini)
AI_MODE = os.getenv("AI_MODE", "gemini").lower()
print(f"🤖 AI 模式: {AI_MODE.upper()}")

VISION_PROMPT = "這是一張標案或產品 DM 中的圖表。請詳細描述其內容（如產品規格、金額、日期或機房佈線邏輯），並以繁體中文回答。"

# 2. 依模式初始化對應的 AI 客戶端
if AI_MODE == "gemini":
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"👉 模型名稱: {m.name}")
    except Exception as e:
        print(f"❌ 查詢失敗: {e}")
    vision_model = genai.GenerativeModel('gemini-2.0-flash')
    azure_client = None

elif AI_MODE == "azure":
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


def describe_image(pil_image) -> str:
    """統一入口：依 AI_MODE 選擇 Gemini 或 Azure OpenAI 進行圖片描述"""
    if AI_MODE == "gemini":
        try:
            response = vision_model.generate_content([VISION_PROMPT, pil_image])
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
                            {"type": "text", "text": VISION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Azure 圖片解析失敗: {str(e)}]"

def process_tender_with_images_v2(file_path, output_dir):
    pipeline_options = PdfPipelineOptions()
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

    print(f"開始處理: {file_id} ...")

    for item, level in doc.iterate_items():
        # A. 處理文字與表格 (維持不變)
        if hasattr(item, "text") and not hasattr(item, "image"):
            md_output.append(item.text)
        elif hasattr(item, "export_to_markdown") and not hasattr(item, "image"):
            md_output.append(item.export_to_markdown())

        # B. 處理圖片 (增加 NoneType 安全檢查)
        elif hasattr(item, "image") and item.image is not None:
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
                    vision_description = describe_image(pil_img)
                    
                    # 3. 將描述直接嵌入 Markdown 位置
                    md_output.append(f"\n> ### 🖼️ 圖表解析: {image_filename}")
                    md_output.append(f"> {vision_description}\n")
                else:
                    md_output.append("\n> [圖片內容為空，無法解析]\n")
            except Exception as e:
                print(f"  ⚠️ 圖片處理異常: {str(e)}")
                md_output.append(f"\n> [圖片處理失敗: {image_filename}]\n")

    # 儲存最終 Markdown
    final_md_path = os.path.join(output_dir, f"{file_id}.md")
    with open(final_md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(md_output))
    
    print(f"✨ POC 文件解析完成！輸出路徑: {final_md_path}")

def batch_process_folders():
    """
    自動掃描標案與門市資料夾中的所有 PDF 與 Word
    """
    # 定義對應關係：(來源資料夾, 輸出資料夾)
    tasks = [
        ("./rag_poc/input/government", "./ragtest/input"),
        ("./rag_poc/input/retail", "./ragtest/input")
    ]
    
    for src, dest in tasks:
        if not os.path.exists(src):
            continue
            
        for file in os.listdir(src):
            # 同時支援 pdf 與 docx
            if file.lower().endswith(('.pdf', '.docx')):
                full_path = os.path.join(src, file)
                process_tender_with_images_v2(full_path, dest)

if __name__ == "__main__":
    batch_process_folders()
