# 檔案一：標案解析與圖片提取
import os
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

def process_document(file_path, output_dir):
    """
    支援 PDF 與 Word (.docx) 自動識別與解析
    """
    # 針對 PDF 的影像強化設定 (Word 檔案會自動跳過此設定，不影響運行)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 4.16  # 確保標案圖表達到 300 DPI
    
    # 初始化轉換器
    converter = DocumentConverter(pipeline_options=pipeline_options)
    
    print(f"🚀 正在解析文件: {file_path} ...")
    
    try:
        # Docling 會根據副檔名自動判斷 InputFormat.PDF 或 InputFormat.DOCX
        result = converter.convert(file_path)
        
        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # 產生輸出檔名 (.md)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        md_path = os.path.join(output_dir, f"{file_name}.md")
        
        # 導出為 Markdown 格式
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result.document.export_to_markdown())
            
        print(f"✅ 解析成功！Markdown 已儲存至: {md_path}")
        return md_path

    except Exception as e:
        print(f"❌ 解析失敗 ({file_path}): {str(e)}")
        return None

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
                process_document(full_path, dest)

if __name__ == "__main__":
    batch_process_folders()
