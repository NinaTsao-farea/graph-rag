# 多模態 GraphRAG 知識中台 POC 專案系統需求規劃書 (SRS)

## 1. 專案概述 (Project Overview)
### 1.1 背景說明
本專案旨在解決企業內部兩大非結構化資料處理難點：
* **政府標案 (Government Tenders)**：文件長且規章複雜，人工比對 RFP（需求說明書）與 Proposal（建議書）耗時且易出錯。
* **門市輔銷 (Retail Enablement)**：商品型錄、活動 DM 包含大量圖片與表格，店員難以即時檢索精準銷售資訊。

### 1.2 POC 核心目標
1.  **架構驗證**：驗證 GraphRAG（結合知識圖譜與向量檢索）處理長文本與多模態資料的精準度。
2.  **引擎對比**：評測 **Google Gemini 3.0** 與 **Azure OpenAI (GPT-4o)** 在繁體中文與專業領域的效能差異。
3.  **流程自動化**：建立從 PDF/Word 解析、300 DPI 影像強化到自動化問答的完整管線。

---

## 2. 系統架構與技術棧 (System Architecture & Tech Stack)
### 2.1 技術組件
| 組件分類 | 推薦技術 | 說明 |
| :--- | :--- | :--- |
| **文件解析層** | **IBM Docling** | 地端執行，支援 PDF/Word/圖片，強化 300 DPI 輸出。 |
| **向量儲存層** | **Qdrant (Local)** | 儲存語義向量，支援標籤過濾 (Metadata Filtering)。 |
| **圖譜儲存層** | **Neo4j / Parquet** | 儲存實體關聯（如：產品-規格-法規）。 |
| **AI 推理引擎** | **Gemini 3.0 / GPT-4o** | 負責實體提取、多模態圖表描述與自然語言生成。 |

### 2.2 部署模式
* **數據預處理**：完全地端運行，確保原始機敏文件不外流。
* **API 調用**：僅傳送去識別化後的結構化文字至雲端 LLM 進行推理。

---

## 3. 功能需求 (Functional Requirements)
### 3.1 多模態資料接入 (Multimodal Ingestion)
* **格式轉化**：支援將 `.pdf` 與 `.docx` 統一轉換為 Markdown 格式。
* **影像自動化 (VLM)**：自動辨識標案施工圖、機房佈置圖或產品 DM，並生成結構化文字描述。
* **解析強化**：確保表格數據（如預算表、規格表）轉換後維持邏輯對齊，不產生亂碼。

### 3.2 知識圖譜索引 (Knowledge Indexing)
* **自動提取**：識別實體（Entity）如金額、日期、型號、罰則及法律條文。
* **關係鏈結**：自動連結跨文件、跨章節的邏輯關係（如：RFP 要求與建議書承諾之對應）。

### 3.3 分類檢索問答 (Segmented Q&A)
* **多領域切換**：透過標籤區隔「政府標案」與「門市輔銷」資料夾。
* **可溯源回答**：系統回答時必須精準標註引用文件名稱與頁碼。

---

## 4. 非功能需求 (Non-Functional Requirements)
### 4.1 資安與合規性 (Security & Compliance)
* **工具排除**：嚴格禁用特定地區（如大陸地區）之開源庫與 API 服務。
* **弱點掃描**：系統組件（Docker Image）需通過 CVE 弱點掃描，符合企業資安基準。
* **資料保護**：雲端傳輸過程需加密，且不得將使用者資料用於模型訓練。

### 4.2 效能指標 (Performance KPI)
* **實體提取精準度**：抽樣檢查準確率需 > 85%。
* **檢索延遲**：單次問答反應時間（包含圖譜搜尋）需在 10-15 秒內。

---

## 5. POC 測試與驗證計畫 (Test Plan)
### 5.1 測試場景設計
1.  **合規比對測試**：比對建議書是否滿足 RFP 中的所有資安條款。
2.  **圖表理解測試**：根據機房佈置圖判斷 UPS 配置數量是否正確。
3.  **銷售問答測試**：查詢特定活動 DM 中的優惠期間與適用機型。

### 5.2 評估工具
* 使用 `compare_results_v2.py` 自動產出數據對照報告。
* 對比 Gemini 3.0 與 Azure GPT-4o 在 Node/Edge 提取數量的差異。

---

## 6. 專案交付物 (Deliverables)
1.  **POC 原始程式碼**（含地端解析、GraphRAG 腳本）。
2.  **雙引擎對比評估報告**（Markdown 格式）。
3.  **系統佈署與操作說明書**。

---

## 7.環境設定   
### 1.核心開發環境
| 軟體名稱 | 建議版本 | 說明 |
| :--- | :--- | :--- |
| Python | 3.10.x - 3.11.x | GraphRAG 對 3.12 的支援尚在完善中，3.10 最穩定。 |
| Node.js | 18.x 或 20.x (LTS) | 若後續需部署 GraphRAG 可視化介面時需要。 |
| Docker Desktop | 4.25.0+ | 用於運行本地向量資料庫 (Qdrant) 或 Neo4j。 |

### 2. 關鍵 Python 函式庫 (Python Packages)
| 套件名稱 | 建議版本 | 說明 |
| :--- | :--- | :--- |
| docling | 1.2.x+ | IBM 出品，負責 PDF/Word 解析與 300 DPI 圖片提取。 |
| graphrag | 0.3.x+ | Microsoft 核心框架，負責知識圖譜建立與檢索。 |
| pandas | 2.1.x+ | 用於 compare_results.py 數據處理與報告生成。 |
| pyarrow | 14.0.x+ | 處理 GraphRAG 輸出的 Parquet 檔案必備。 |
| Pillow (PIL) | 10.x+ | 用於檢查圖片 DPI 與影像預處理。 |
| python-dotenv | 1.0.x+ | 管理 Gemini 與 Azure 的 API Key 環境變數。 |
| openai | 1.30.x+ | 用於調用 Azure 與 Gemini (OpenAI 兼容接口)。 |

