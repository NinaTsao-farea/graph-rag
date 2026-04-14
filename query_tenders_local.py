# query_tenders_local.py — GraphRAG 查詢模組（Local / Ollama 陣營）
"""
環境變數（與 parse_tenders_local.py 的 OLLAMA_BASE_URL 共用，新增 LLM 專屬鍵值）：

  OLLAMA_BASE_URL     = http://localhost:11434/v1   （必填，與 Vision 共用）
  OLLAMA_LLM_MODEL    = gemma4                     （Query 用 LLM，可與 Vision 模型不同）
  OLLAMA_LLM_MODEL_2  = llama3.1                   （可選，支援多顆本地 LLM）
  OLLAMA_EMBED_MODEL  = qwen3-embedding:0.6b        （需與建立索引時相同；
                                                      未設定則僅支援 global search）
"""
import asyncio
import argparse
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from doc_tasks import get_index_root, get_valid_doc_types, SYSTEM_PROMPTS
from graphrag_llm.completion import create_completion
from graphrag_llm.config import ModelConfig
from graphrag_llm.embedding import create_embedding
from graphrag_vectors import IndexSchema, VectorStoreConfig, VectorStoreType, create_vector_store
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch

load_dotenv()

CAMP = "local"
COMMUNITY_LEVEL = 2  # Leiden 社群層級（與 azure 版保持一致）

# SYSTEM_PROMPTS 從 doc_tasks.py 匯入（所有陣營的單一來源）


# ─────────────────────────────────────────────────────────────
# 查詢統計
# ─────────────────────────────────────────────────────────────
@dataclass
class _QueryStats:
    mode: str = ""
    query: str = ""
    model_id: str = ""
    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    completion_time: float = 0.0
    wall_time: float = 0.0
    start_time: float = field(default_factory=time.time)

_stats = _QueryStats()


# ─────────────────────────────────────────────────────────────
# 1. 動態載入 Ollama LLM 模型清單
# ─────────────────────────────────────────────────────────────
def _load_query_models() -> dict[str, dict]:
    """
    掃描 .env 中的 Ollama 設定，建立 Local 查詢模型清單。
    OLLAMA_BASE_URL 未設定時回傳空字典（Web UI 不顯示 local query 選項）。
    """
    models: dict[str, dict] = {}
    # ollama_base_raw = os.getenv("OLLAMA_BASE_URL", "")
    # if not ollama_base_raw:
    #     return models
    ollama_base = os.getenv("OLLAMA_BASE_URL", "")
    if not ollama_base:
        return models

    # LiteLLM ollama provider 的 api_base 不帶 /v1 後綴
    # ollama_base = ollama_base_raw.rstrip("/").removesuffix("/v1")

    # 嵌入模型（局部搜索必需，全域搜索不需要）
    ollama_embed = os.getenv("OLLAMA_EMBED_MODEL", "")
    embed_cfg: ModelConfig | None = (
        ModelConfig(
            model_provider="openai",
            model=ollama_embed,
            api_base=ollama_base,
            api_key="ollama",   # Ollama 不驗證 key，但 ModelConfig 要求必填
        )
        if ollama_embed else None
    )

    # 支援多顆 LLM：OLLAMA_LLM_MODEL、OLLAMA_LLM_MODEL_2、...
    llm_suffixes = [""] + [f"_{i}" for i in range(2, 6)]
    for sfx in llm_suffixes:
        ollama_llm = os.getenv(f"OLLAMA_LLM_MODEL{sfx}", "")
        if not ollama_llm:
            if sfx:
                break
            continue
        safe_key  = ollama_llm.replace(":", "_").replace(".", "_")
        model_key = f"ollama_{safe_key}"
        models[model_key] = {
            "label": f"Ollama {ollama_llm}",
            "type":  "local",
            "camp":  "local",
            "chat": ModelConfig(
                model_provider="openai",
                model=ollama_llm,
                api_base=ollama_base,
                api_key="ollama",   # Ollama 不驗證 key，但 ModelConfig 要求必填
                call_args={"temperature": 0.0, "max_tokens": 2_000},
            ),
            "embed": embed_cfg,   # None 表示僅支援 global search
            "input_price_per_1m":  0.0,
            "output_price_per_1m": 0.0,
        }

    return models


QUERY_MODELS: dict[str, dict] = _load_query_models()
DEFAULT_QUERY_MODEL_ID: str = next(iter(QUERY_MODELS), "")

_discovered = ", ".join(QUERY_MODELS.keys()) if QUERY_MODELS else "（未設定 OLLAMA_LLM_MODEL）"
print(f"🏠 Local Query 可用模型: {_discovered}  (預設: {DEFAULT_QUERY_MODEL_ID or '無'})")


def build_query_client(model_id: str | None = None):
    """建立並回傳 (chat_model, text_embedder, tokenizer, cfg) 四元組。"""
    mid = model_id or DEFAULT_QUERY_MODEL_ID
    if not mid:
        raise ValueError(
            "未設定 OLLAMA_LLM_MODEL，無法使用 Local Query。"
            "請在 .env 中新增 OLLAMA_LLM_MODEL=<model_name>。"
        )
    cfg = QUERY_MODELS.get(mid)
    if cfg is None:
        raise ValueError(f"未知的 Local 查詢模型 ID：'{mid}'，可用: {list(QUERY_MODELS.keys())}")
    chat  = create_completion(cfg["chat"])
    embed = create_embedding(cfg["embed"]) if cfg.get("embed") else None
    tok   = chat.tokenizer
    return chat, embed, tok, cfg


# ─────────────────────────────────────────────────────────────
# 2. 依索引目錄建立搜索引擎
# ─────────────────────────────────────────────────────────────
def build_engines(input_dir: str, doc_type: str, model_id: str | None = None):
    lancedb_uri = f"{input_dir}/lancedb"

    chat_model, text_embedder, tokenizer, _cfg = build_query_client(model_id)
    print(f"🏠 查詢選用模型: {_cfg['label']}  （陣營: local）")

    if text_embedder is None:
        print(
            "⚠️  未設定 OLLAMA_EMBED_MODEL，LocalSearch 將無法使用向量查詢。"
            "若需局部搜索，請在 .env 設定 OLLAMA_EMBED_MODEL。"
        )

    _system_prompt = SYSTEM_PROMPTS.get(doc_type, SYSTEM_PROMPTS["default"])
    entity_df       = pd.read_parquet(f"{input_dir}/entities.parquet")
    community_df    = pd.read_parquet(f"{input_dir}/communities.parquet")
    report_df       = pd.read_parquet(f"{input_dir}/community_reports.parquet")
    relationship_df = pd.read_parquet(f"{input_dir}/relationships.parquet")
    text_unit_df    = pd.read_parquet(f"{input_dir}/text_units.parquet")

    entities      = read_indexer_entities(entity_df, community_df, COMMUNITY_LEVEL)
    communities   = read_indexer_communities(community_df, report_df)
    reports       = read_indexer_reports(report_df, community_df, COMMUNITY_LEVEL)
    relationships = read_indexer_relationships(relationship_df)
    text_units    = read_indexer_text_units(text_unit_df)

    try:
        covariate_df = pd.read_parquet(f"{input_dir}/covariates.parquet")
        covariates   = {"claims": read_indexer_covariates(covariate_df)}
    except FileNotFoundError:
        covariates = None

    print(f"✅ 實體: {len(entity_df)} 筆 | 關係: {len(relationship_df)} 筆 | 社群報告: {len(report_df)} 筆")

    description_embedding_store = create_vector_store(
        config=VectorStoreConfig(type=VectorStoreType.LanceDB, db_uri=lancedb_uri),
        index_schema=IndexSchema(index_name="entity_description"),
    )
    description_embedding_store.connect()

    _local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 12_000,
    }
    _model_params = {
        "max_tokens": 2_000,
        "temperature": 0.0,
    }

    local_engine = LocalSearch(
        model=chat_model,
        context_builder=LocalSearchMixedContext(
            community_reports=reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            covariates=covariates,
            entity_text_embeddings=description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            text_embedder=text_embedder,
            tokenizer=tokenizer,
        ),
        system_prompt=_system_prompt,
        tokenizer=tokenizer,
        model_params=_model_params,
        context_builder_params=_local_context_params,
        response_type="multiple paragraphs",
    )

    _global_context_params = {
        "use_community_summary": False,
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,
        "context_name": "Reports",
    }

    global_engine = GlobalSearch(
        model=chat_model,
        context_builder=GlobalCommunityContext(
            community_reports=reports,
            communities=communities,
            entities=entities,
            tokenizer=tokenizer,
        ),
        tokenizer=tokenizer,
        max_data_tokens=12_000,
        map_llm_params={"max_tokens": 1000, "temperature": 0.0, "response_format": {"type": "json_object"}},
        reduce_llm_params={"max_tokens": 2000, "temperature": 0.0},
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params=_global_context_params,
        concurrent_coroutines=4,   # Ollama 本地單執行緒，降低並發避免壅塞
        response_type="multiple paragraphs",
    )

    return local_engine, global_engine


# ─────────────────────────────────────────────────────────────# 2b. 頁碼提取與引用出處
# ─────────────────────────────────────────────────────────────

_RE_PAGE_MARKERS = [
    re.compile(r'\*\*\[第\s*(\d+)\s*頁\]\*\*'),           # **[第 N 頁]**
    re.compile(r'<!--\s*PageNumber\s*=\s*"(\d+)"\s*-->'), # <!-- PageNumber="N" -->
    re.compile(r'<!--\s*PageFooter\s*=\s*"第(\d+)頁'),     # <!-- PageFooter="第N頁，共M頁" -->
]


def _extract_page_numbers(text: str) -> list[int]:
    """從 markdown 文字片段中提取所有頁碼。"""
    pages: set[int] = set()
    for pattern in _RE_PAGE_MARKERS:
        for m in pattern.finditer(text):
            try:
                pages.add(int(m.group(1)))
            except ValueError:
                pass
    return sorted(pages)


def _build_citations(result, input_dir: str) -> str:
    """
    從 Local Search 結果的 context_data 擷取文字片段，
    提取頁碼並對應來源檔案，回傳格式化的引用出處字串。
    """
    ctx = getattr(result, "context_data", None)
    if not ctx:
        return ""

    sources_df = None
    for key in ("sources", "text_units"):
        val = ctx.get(key) if isinstance(ctx, dict) else None
        if val is not None and hasattr(val, "iterrows") and not val.empty:
            sources_df = val
            break
    if sources_df is None:
        return ""

    base = Path(input_dir)

    # 從 text_units.parquet 建立 「文字前 200 字 → document_id」 映射
    # （context_data sources id 是 human_readable_id 整數，不是 hash ID，只能從文字內容匹配）
    text_to_doc: dict[str, str] = {}
    try:
        tu_df = pd.read_parquet(str(base / "text_units.parquet"))
        for _, row in tu_df.iterrows():
            text_key = str(row.get("text", ""))[:200]
            text_to_doc[text_key] = str(row.get("document_id", ""))
    except Exception:
        pass

    doc_to_title: dict[str, str] = {}
    try:
        doc_df = pd.read_parquet(str(base / "documents.parquet"))
        for _, row in doc_df.iterrows():
            title = str(row.get("title", ""))
            if title.endswith(".md"):
                title = title[:-3]
            doc_to_title[str(row["id"])] = title
    except Exception:
        pass

    file_pages: dict[str, set[int]] = defaultdict(set)
    for _, row in sources_df.iterrows():
        text  = str(row.get("text", ""))
        pages = _extract_page_numbers(text)
        if not pages:
            continue

        doc_id   = text_to_doc.get(text[:200], "")
        filename = doc_to_title.get(doc_id, "")

        # fallback：從文字 front matter 抓「來源檔案」
        if not filename:
            m = re.search(r'來源檔案:\s*(.+)', text)
            filename = m.group(1).strip() if m else "（來源不明）"

        for p in pages:
            file_pages[filename].add(p)

    if not file_pages:
        return ""

    lines = ["\n📎 引用出處："]
    for idx, (filename, pages) in enumerate(sorted(file_pages.items()), start=1):
        page_str = "、".join(str(p) for p in sorted(pages)) if pages else "—"
        lines.append(f"  [{idx}] {filename}  第 {page_str} 頁")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────# 3. Context 印出工具（共用邏輯）
# ─────────────────────────────────────────────────────────────
def _print_context(result) -> None:
    """以條列方式印出 GraphRAG 本次查詢實際使用的參考內容"""
    ctx = result.context_data
    if not ctx:
        print("\n（無 context_data）")
        return

    items = ctx.items() if isinstance(ctx, dict) else [("context", ctx)]

    _SECTION_COLUMNS = {
        "entities":      [("title", "實體名稱"), ("type", "類型"), ("description", "描述"), ("rank", "重要性")],
        "relationships": [("source", "來源實體"), ("target", "目標實體"), ("description", "關係描述"), ("weight", "權重")],
        "sources":       [("id", "段落 ID"), ("text", "原文內容")],
        "text_units":    [("id", "段落 ID"), ("text", "原文內容")],
        "reports":       [("title", "報告標題"), ("summary", "摘要"), ("rank", "重要性")],
    }
    _SECTION_ICONS = {
        "entities":      "🔷 實體 (Entities)",
        "relationships": "🔗 關係 (Relationships)",
        "sources":       "📄 原文段落 (Sources)",
        "text_units":    "📄 原文段落 (Text Units)",
        "reports":       "📊 社群報告 (Reports)",
    }

    print("\n" + "═" * 70)
    print("📋 GraphRAG 參考內容 (context_data)")
    print("═" * 70)

    for key, value in items:
        icon = _SECTION_ICONS.get(key, f"📌 {key}")
        print(f"\n{'─' * 70}")
        print(f"  {icon}")
        print(f"{'─' * 70}")

        if not hasattr(value, "iterrows"):
            print(f"  {value}")
            continue
        if value.empty:
            print("  （無資料）")
            continue

        columns = _SECTION_COLUMNS.get(key, [(col, col) for col in value.columns])
        columns = [(col, label) for col, label in columns if col in value.columns]

        for idx, (_, row) in enumerate(value.iterrows(), start=1):
            print(f"\n  [{idx}]")
            for col, label in columns:
                raw = row[col]
                if pd.isna(raw) or raw == "" or raw is None:
                    continue
                text = str(raw).strip()
                if len(text) > 80:
                    lines = [text[i:i+80] for i in range(0, len(text), 80)]
                    print(f"      • {label}: {lines[0]}")
                    for line in lines[1:]:
                        print(f"               {line}")
                else:
                    print(f"      • {label}: {text}")

    print("\n" + "═" * 70)


# ─────────────────────────────────────────────────────────────
# 4. 搜索函式
# ─────────────────────────────────────────────────────────────
async def run_local_search(
    engine: LocalSearch,
    query: str,
    show_context: bool = False,
    model_id: str | None = None,
    input_dir: str | None = None,
) -> str:
    """【局部搜索】：需要 OLLAMA_EMBED_MODEL 才能使用向量查詢。
    提供 input_dir 時，自動從文字片段提取頁碼並附加引用出處。
    """
    global _stats
    print(f"\n🔍 局部搜索中: {query}")
    t0 = time.time()
    result = await engine.search(query)
    _stats.wall_time       = time.time() - t0
    _stats.mode            = "local"
    _stats.query           = query
    _stats.model_id        = model_id or DEFAULT_QUERY_MODEL_ID
    _stats.llm_calls       = getattr(result, "llm_calls",       1)
    _stats.prompt_tokens   = getattr(result, "prompt_tokens",   0)
    _stats.output_tokens   = getattr(result, "output_tokens",   0)
    _stats.completion_time = getattr(result, "completion_time", 0.0)
    if show_context:
        _print_context(result)
    response = result.response
    if input_dir:
        citations = _build_citations(result, input_dir)
        if citations:
            response = response + citations
    return response


async def run_global_search(engine: GlobalSearch, query: str, show_context: bool = False, model_id: str | None = None) -> str:
    """【全域搜索】：不需嵌入模型，Ollama 陣營推薦使用此模式。"""
    global _stats
    print(f"\n🌐 全域搜索中: {query}")
    t0 = time.time()
    result = await engine.search(query)
    _stats.wall_time       = time.time() - t0
    _stats.mode            = "global"
    _stats.query           = query
    _stats.model_id        = model_id or DEFAULT_QUERY_MODEL_ID
    _stats.llm_calls       = getattr(result, "llm_calls",       1)
    _stats.prompt_tokens   = getattr(result, "prompt_tokens",   0)
    _stats.output_tokens   = getattr(result, "output_tokens",   0)
    _stats.completion_time = getattr(result, "completion_time", 0.0)
    if show_context:
        _print_context(result)
    return result.response


# ─────────────────────────────────────────────────────────────
# 5. 費用統計（本地推論費用為 $0）
# ─────────────────────────────────────────────────────────────
def _print_query_stats() -> None:
    """印出本次查詢的 Token 使用量（本地推論，費用 $0）。"""
    s = _stats
    _mcfg  = QUERY_MODELS.get(s.model_id or DEFAULT_QUERY_MODEL_ID, {})
    _label = _mcfg.get("label", s.model_id or DEFAULT_QUERY_MODEL_ID)

    mode_label = "🔍 局部搜索" if s.mode == "local" else "🌐 全域搜索"
    bar = "═" * 58
    print(f"\n{bar}")
    print("📊  查詢統計摘要（Local Ollama）")
    print(bar)
    print(f"  模式             : {mode_label}")
    print(f"  🏠 模型          : {_label}")
    print(f"  ⏱️  總挂牆時間   : {s.wall_time:.2f} 秒")
    if s.completion_time:
        print(f"  ℹ️  LLM 純計時間 : {s.completion_time:.2f} 秒")
    print()
    print(f"  ── LLM ({_label}) ────────────")
    print(f"  LLM 呼叫次數      : {s.llm_calls}")
    print(f"  輸入 tokens        : {s.prompt_tokens:>10,}")
    if s.output_tokens:
        print(f"  輸出 tokens        : {s.output_tokens:>10,}")
    print(f"  💰 本地推論，費用: $0.00 USD")
    print(bar)


# ─────────────────────────────────────────────────────────────
# 6. 主程式（CLI 直接執行）
# ─────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="GraphRAG 查詢工具（Local Ollama 陣營）")
    parser.add_argument(
        "--type",
        dest="doc_type",
        default="government",
        choices=get_valid_doc_types(),
        help="查詢的索引類型（預設: government）",
    )
    parser.add_argument(
        "--model",
        dest="model_id",
        default=None,
        choices=list(QUERY_MODELS.keys()) or None,
        help=f"查詢 LLM 模型（預設: {DEFAULT_QUERY_MODEL_ID or '無'}）",
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        default="global",
        choices=["local", "global"],
        help="搜索模式：local 需設定 OLLAMA_EMBED_MODEL；global 不需嵌入（預設: global）",
    )
    parser.add_argument(
        "--query",
        dest="query",
        default=None,
        help="查詢字串，例如: --query \"請摘要文件主要內容\"",
    )
    parser.add_argument(
        "--context",
        dest="show_context",
        action="store_true",
        default=False,
        help="印出本次查詢實際參考的完整段落清單",
    )
    args = parser.parse_args()

    input_dir = get_index_root(args.doc_type, CAMP)
    model_id  = args.model_id or DEFAULT_QUERY_MODEL_ID
    print(f"📂 索引目錄: {input_dir}")

    local_engine, global_engine = build_engines(input_dir, args.doc_type, model_id)

    query = args.query or "請摘要這份文件的主要內容。"

    if args.mode == "local":
        result = await run_local_search(local_engine, query, show_context=args.show_context, model_id=model_id, input_dir=input_dir)
    else:
        result = await run_global_search(global_engine, query, show_context=args.show_context, model_id=model_id)

    print(f"\n結果:\n{result}")
    _print_query_stats()


if __name__ == "__main__":
    asyncio.run(main())
