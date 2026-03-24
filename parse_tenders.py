# 檔案一：標案解析與圖片提取
import os
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from PIL import Image

def process_tender_file(file_path, output_dir):
    # 1. 設定解析參數：強化圖片解析度 (Scale=4.16 趨近 300 DPI)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 4.16 
    
    converter = DocumentConverter(pipeline_options=pipeline_options)
    
    # 2. 執行轉換
    print(f"正在解析文件: {file_path} ...")
    result = converter.convert(file_path)
    
    # 3. 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 4. 導出 Markdown 檔案
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    md_path = os.path.join(output_dir, f"{file_name}.md")
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result.document.export_to_markdown())
        
    print(f"✅ Markdown 已生成: {md_path}")
    return md_path

# 執行範例
if __name__ == "__main__":
    process_tender_file("tender_proposal.pdf", "./rag_poc/input")
