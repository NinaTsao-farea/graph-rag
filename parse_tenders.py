# 檔案一：標案解析與圖片提取
import os
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

def process_tender_with_images(file_path, output_dir):
    # 1. 啟動圖片提取與 300 DPI 強化
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True # 這是關鍵：真正把圖切出來
    pipeline_options.images_scale = 4.16 
    
    format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    converter = DocumentConverter(format_options=format_options)
    
    result = converter.convert(file_path)
    doc = result.document
    
    # 建立圖片存放資料夾
    file_id = os.path.splitext(os.path.basename(file_path))[0]
    image_save_dir = os.path.join(output_dir, "images", file_id)
    os.makedirs(image_save_dir, exist_ok=True)

    # 2. 遍歷文件模型，重建 Markdown
    md_output = []
    image_counter = 0

    for item, level in doc.iterate_items():
        # 如果是文字內容
        if hasattr(item, "text") and not hasattr(item, "image"):
            md_output.append(item.text)
        
        # 如果是表格內容
        elif hasattr(item, "export_to_markdown") and not hasattr(item, "image"):
            md_output.append(item.export_to_markdown())

        # 如果是圖片元素 (PictureItem)
        elif hasattr(item, "image"):
            image_counter += 1
            image_filename = f"image_{image_counter}.png"
            image_path = os.path.join(image_save_dir, image_filename)
            
            # 儲存圖片檔案
            item.image.pil_image.save(image_path)
            
            # --- 這裡就是你的 POC 加分項：插入 Gemini Vision 描述 ---
            # 這裡我們先放一個具備「位置資訊」的標籤
            # 未來這裡會由 describe_image_with_gemini(image_path) 替換
            img_placeholder = f"\n> **[圖表編號：{image_filename} | 位置：此處緊跟在上述文字之後]**\n"
            img_placeholder += f"> *[提示：請將此圖片送往 Gemini Vision 獲取施工圖/DM 描述]*\n"
            
            md_output.append(img_placeholder)

    # 3. 儲存最終整合後的 Markdown
    final_md_path = os.path.join(output_dir, f"{file_id}.md")
    with open(final_md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(md_output))
    
    print(f"✅ 定位完成！Markdown 與圖片已分離並關聯。")
    print(f"🖼️ 圖片存儲於: {image_save_dir}")

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
                process_tender_with_images(full_path, dest)

if __name__ == "__main__":
    batch_process_folders()
