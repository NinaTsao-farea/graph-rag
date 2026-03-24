# 檔案一：標案解析與圖片提取
import os
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions

def process_tender_file(file_path, output_dir):
    # 設定 300 DPI 影像強化 (Scale=4.16)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 4.16 
    
    converter = DocumentConverter(pipeline_options=pipeline_options)
    
    print(f"正在解析標案/輔銷文件: {file_path} ...")
    result = converter.convert(file_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    md_path = os.path.join(output_dir, f"{file_name}.md")
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result.document.export_to_markdown())
        
    print(f"✅ Markdown 已導出至: {md_path}")
    return md_path

if __name__ == "__main__":
    # 測試解析
    process_tender_file("./rag_poc/input/government/tender_01.pdf", "./ragtest/input")
