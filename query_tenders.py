# step 2: GraphRAG 查詢模組 (比照官方 GraphRAG 3.x API)
import asyncio
import argparse
import os
import time
from dataclasses import dataclass, field

import pandas as pd
from dotenv import load_dotenv
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

# ─────────────────────────────────────────────────────────────
# 費用定價常數 (USD / 百萬 tokens) — 依實際模型定價調整
# ─────────────────────────────────────────────────────────────
_LLM_INPUT_PRICE_PER_1M  = 0.75   # GPT-5.4 mini 輸入 token
_LLM_OUTPUT_PRICE_PER_1M = 4.50   # GPT-5.4 mini 輸出 token


@dataclass
class _QueryStats:
    mode: str = ""
    query: str = ""
    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    completion_time: float = 0.0   # GraphRAG 回傳的 LLM 純計時間
    wall_time: float = 0.0         # 包含建構 context 的總挂牆時間
    start_time: float = field(default_factory=time.time)


_stats = _QueryStats()

# ─────────────────────────────────────────────────────────────
# 1. 文件類型 → 索引輸出目錄對應表
#    與 parse_tenders*.py 的 DOC_TASKS 保持一致
# ─────────────────────────────────────────────────────────────
DOC_INDEX_ROOTS = {
    "government": "./ragtest/government/output",
    "stip":       "./ragtest/stip/output",
    "default":    "./ragtest/output",   # 向下相容：未指定 --type 時使用
}

COMMUNITY_LEVEL = 2  # Leiden 社群層級 (數值越高越細緻)

# ─────────────────────────────────────────────────────────────
# 2. AI 模式選擇 (AI_MODE=gemini 或 azure，預設 gemini)
# ─────────────────────────────────────────────────────────────
AI_MODE = os.getenv("AI_MODE", "gemini").lower()
print(f"🤖 Query AI 模式: {AI_MODE.upper()}")

if AI_MODE == "azure":
    _chat_config = ModelConfig(
        model_provider="azure",
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        call_args={"temperature": 0.0, "max_tokens": 2_000},
    )
    _embed_config = ModelConfig(
        model_provider="azure",
        model=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT", "https://mess-demo.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
    )
else:
    raise ValueError(f"不支援的 AI_MODE：'{AI_MODE}'，請設定為 'gemini' 或 'azure'")

chat_model    = create_completion(_chat_config)
text_embedder = create_embedding(_embed_config)
tokenizer     = chat_model.tokenizer


# ─────────────────────────────────────────────────────────────
# 3. 依索引目錄建立搜索引擎
# ─────────────────────────────────────────────────────────────
def build_engines(input_dir: str):
    lancedb_uri = f"{input_dir}/lancedb"

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
        concurrent_coroutines=32,
        response_type="multiple paragraphs",
    )

    return local_engine, global_engine


# ─────────────────────────────────────────────────────────────
# 4. 搜索函式
# ─────────────────────────────────────────────────────────────
def _print_context(result) -> None:
    """以條列方式印出 GraphRAG 本次查詢實際使用的參考內容"""
    ctx = result.context_data
    if not ctx:
        print("\n（無 context_data）")
        return

    items = ctx.items() if isinstance(ctx, dict) else [("context", ctx)]

    # 各區塊要顯示的欄位與標籤設定
    _SECTION_COLUMNS = {
        "entities": [
            ("title",       "實體名稱"),
            ("type",        "類型"),
            ("description", "描述"),
            ("rank",        "重要性"),
        ],
        "relationships": [
            ("source",      "來源實體"),
            ("target",      "目標實體"),
            ("description", "關係描述"),
            ("weight",      "權重"),
        ],
        "sources": [
            ("id",          "段落 ID"),
            ("text",        "原文內容"),
        ],
        "text_units": [
            ("id",          "段落 ID"),
            ("text",        "原文內容"),
        ],
        "reports": [
            ("title",       "報告標題"),
            ("summary",     "摘要"),
            ("rank",        "重要性"),
        ],
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
            # 非 DataFrame，直接印出
            print(f"  {value}")
            continue

        if value.empty:
            print("  （無資料）")
            continue

        columns = _SECTION_COLUMNS.get(key, [(col, col) for col in value.columns])
        # 只取存在於 DataFrame 的欄位
        columns = [(col, label) for col, label in columns if col in value.columns]

        for idx, (_, row) in enumerate(value.iterrows(), start=1):
            print(f"\n  [{idx}]")
            for col, label in columns:
                raw = row[col]
                if pd.isna(raw) or raw == "" or raw is None:
                    continue
                text = str(raw).strip()
                # 長文字換行縮排
                if len(text) > 80:
                    lines = [text[i:i+80] for i in range(0, len(text), 80)]
                    print(f"      • {label}: {lines[0]}")
                    for line in lines[1:]:
                        print(f"               {line}")
                else:
                    print(f"      • {label}: {text}")

    print("\n" + "═" * 70)


async def run_local_search(engine: LocalSearch, query: str, show_context: bool = False) -> str:
    """
    【局部搜索】：適合詢問標案的特定細節。
    例如：「本標案的罰則條款為何？」或「UPS 的施工要求是什麼？」
    """
    global _stats
    print(f"\n🔍 局部搜索中: {query}")
    t0 = time.time()
    result = await engine.search(query)
    _stats.wall_time       = time.time() - t0
    _stats.mode            = "local"
    _stats.query           = query
    _stats.llm_calls       = getattr(result, "llm_calls",       1)
    _stats.prompt_tokens   = getattr(result, "prompt_tokens",   0)
    _stats.output_tokens   = getattr(result, "output_tokens",   0)
    _stats.completion_time = getattr(result, "completion_time", 0.0)
    if show_context:
        _print_context(result)
    return result.response


async def run_global_search(engine: GlobalSearch, query: str, show_context: bool = False) -> str:
    """
    【全域搜索】：適合詢問跨文件的趨勢或彙整。
    例如：「彙整所有門市的春季促銷重點」或「分析標案中關於資安的所有潛在要求」。
    """
    global _stats
    print(f"\n🌐 全域搜索中: {query}")
    t0 = time.time()
    result = await engine.search(query)
    _stats.wall_time       = time.time() - t0
    _stats.mode            = "global"
    _stats.query           = query
    _stats.llm_calls       = getattr(result, "llm_calls",       1)
    _stats.prompt_tokens   = getattr(result, "prompt_tokens",   0)
    _stats.output_tokens   = getattr(result, "output_tokens",   0)
    _stats.completion_time = getattr(result, "completion_time", 0.0)
    if show_context:
        _print_context(result)
    return result.response


# ─────────────────────────────────────────────────────────────
# 5. 費用統計
# ─────────────────────────────────────────────────────────────

def _print_query_stats() -> None:
    """印出本次查詢的 Token 使用量與預估費用。"""
    s = _stats
    in_cost  = (s.prompt_tokens  / 1_000_000) * _LLM_INPUT_PRICE_PER_1M
    out_cost = (s.output_tokens  / 1_000_000) * _LLM_OUTPUT_PRICE_PER_1M
    total_cost = in_cost + out_cost

    mode_label = "🔍 局部搜索" if s.mode == "local" else "🌐 全域搜索"
    bar = "═" * 58
    print(f"\n{bar}")
    print("📊  查詢統計摘要")
    print(bar)
    print(f"  模式             : {mode_label}")
    print(f"  ⏱️  總挂牆時間      : {s.wall_time:.2f} 秒")
    if s.completion_time:
        print(f"  ℹ️  LLM 純計時間   : {s.completion_time:.2f} 秒")
    print()
    print("  ── Azure OpenAI LLM ────────────────────────────")
    print(f"  LLM 呼叫次數      : {s.llm_calls}")
    print(f"  輸入 tokens        : {s.prompt_tokens:>10,}")
    if s.output_tokens:
        print(f"  輸出 tokens        : {s.output_tokens:>10,}")
        print(f"  💰 預估費用        : ${total_cost:.6f} USD")
    else:
        print(f"  💰 預估輸入費用     : ${in_cost:.6f} USD")
        print(f"  （輸出 tokens 不可用，輸出費用未計入）")
    print(bar)
    print("  ⚠️  費用為估算唃，實際請查閱 Azure Portal 帳單")
    print(bar)


# ─────────────────────────────────────────────────────────────
# 6. 主程式
# ─────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="GraphRAG 查詢工具")
    parser.add_argument(
        "--type",
        dest="doc_type",
        default="default",
        choices=list(DOC_INDEX_ROOTS.keys()),
        help="查詢的索引類型，例如: --type stip（預設: default）",
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        default="local",
        choices=["local", "global"],
        help="搜索模式：local（局部，適合細節）或 global（全域，適合彙整），預設: local",
    )
    parser.add_argument(
        "--query",
        dest="query",
        default=None,
        help="查詢字串，例如: --query \"請列出逾期違約金的計算方式\"",
    )
    parser.add_argument(
        "--context",
        dest="show_context",
        action="store_true",
        default=False,
        help="印出本次查詢實際參考的完整段落清單",
    )
    args = parser.parse_args()

    input_dir = DOC_INDEX_ROOTS[args.doc_type]
    print(f"📂 索引目錄: {input_dir}")

    local_engine, global_engine = build_engines(input_dir)

    query = args.query or "請摘要這份文件的主要內容。"

    if args.mode == "local":
        result = await run_local_search(local_engine, query, show_context=args.show_context)
    else:
        result = await run_global_search(global_engine, query, show_context=args.show_context)

    print(f"\n結果:\n{result}")
    _print_query_stats()


if __name__ == "__main__":
    asyncio.run(main())
