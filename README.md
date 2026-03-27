
### python install
https://www.python.org/downloads/windows/

## step 1 parse
# 處理所有類型的所有檔案
python .\parse_tenders_azure.py

# 只處理 stip 類型的所有檔案
python .\parse_tenders_azure.py --type stip

# 只處理 stip 類型中的特定檔案
python .\parse_tenders_azure.py --type stip --file "門市銷售指南.pdf"

## step 2 graphrag index
# 使用包裝器執行（自動計時 + Token 費用統計）
python .\index_tenders.py --root ./ragtest/stip

python .\index_tenders.py --root ./ragtest/government

# 或直接使用原生指令（無統計）
python -m graphrag index --root ./ragtest/stip

## step 3 graphrag query
# 查詢 stip 索引（局部搜索，預設）
python .\query_tenders.py --type stip --query "請列出 iPhone 16 的促銷方案"

python .\query_tenders.py --type stip --mode local --query "請列出 有送市話免費分鐘數 的促銷方案"

# 查詢 stip 索引（全域搜索）
python .\query_tenders.py --type stip --mode global --query "彙整所有門市的春季促銷重點"

# 查詢政府標案索引
python .\query_tenders.py --type government --query "請列出逾期違約金的計算方式"

# 使用預設索引（向下相容舊的 ./ragtest/output）
python .\query_tenders.py --query "請摘要主要內容"

# 一般查詢（不顯示參考內容）
python .\query_tenders.py --type stip --mode local --query "請列出有送市話免費分鐘數的促銷方案"

# 加上 --context 顯示完整參考段落
python .\query_tenders.py --type stip --mode local --query "請列出有送市話免費分鐘數的促銷方案" --context

## 直接使用指令
python -m graphrag query --root ./projects/Tender_A --method global --model_id heavy_model "測試問題"
