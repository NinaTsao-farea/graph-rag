"""
GraphRAG Index 包裝器
自動計時、解析 log，並於完成後印出 Token 使用量與預估費用摘要。

用法：
    python index_tenders.py --root ./ragtest/stip
    python index_tenders.py --root ./ragtest/government

費用定價常數依 Azure 定價頁調整：
    https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 費用定價常數 (USD / 百萬 tokens) — 請依實際模型定價更新
# ─────────────────────────────────────────────────────────────
# _LLM_INPUT_PRICE_PER_1M   = 0.75    # GPT-5.4 mini 輸入 token
# _LLM_OUTPUT_PRICE_PER_1M  = 4.50    # GPT-5.4 mini 輸出 token
# _EMBED_PRICE_PER_1M       = 0.13    # text-embedding-3-large
_LLM_INPUT_PRICE_PER_1M   = 2.50    # GPT-5.4 輸入 token
_LLM_OUTPUT_PRICE_PER_1M  = 15.0    # GPT-5.4 輸出 token
_EMBED_PRICE_PER_1M       = 0.13    # text-embedding-3-large


def _parse_log_stats(log_path: Path) -> dict:
    """
    解析 GraphRAG indexing-engine.log，回傳各模型最終累積統計。

    GraphRAG 會在每次請求完成後將 running total 以 JSON blob 的形式寫入 log。
    blob 有 completion_tokens 的是 LLM 呼叫，只有 prompt_tokens 的是 Embedding 呼叫。
    取各自最大值（= 最後一次累積）作為最終數字。
    """
    if not log_path.exists():
        return {}

    text = log_path.read_text(encoding="utf-8", errors="ignore")

    blob_re = re.compile(r"\{[^{}]{50,2000}\}", re.DOTALL)

    llm_stats    = {"prompt_tokens": 0, "completion_tokens": 0, "attempted": 0, "failed": 0}
    embed_stats  = {"prompt_tokens": 0, "attempted": 0, "failed": 0}
    llm_cost_raw = 0.0

    for m in blob_re.finditer(text):
        raw = m.group(0)
        if '"prompt_tokens"' not in raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        pt = data.get("prompt_tokens", 0)
        ct = data.get("completion_tokens")        # None for embedding
        tc = data.get("total_cost", 0) or 0

        if ct is not None:
            # LLM — keep the largest (= final cumulative) values
            if pt > llm_stats["prompt_tokens"]:
                llm_stats["prompt_tokens"]    = pt
                llm_stats["completion_tokens"] = ct
                llm_stats["attempted"]        = data.get("attempted_request_count", 0)
                llm_stats["failed"]           = data.get("failed_response_count", 0)
                llm_cost_raw                  = tc
        else:
            # Embedding — same logic
            if pt > embed_stats["prompt_tokens"]:
                embed_stats["prompt_tokens"] = pt
                embed_stats["attempted"]     = data.get("attempted_request_count", 0)
                embed_stats["failed"]        = data.get("failed_response_count", 0)

    # first / last timestamps for wall-clock reference
    ts_list = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text)
    return {
        "llm":       llm_stats,
        "embedding": embed_stats,
        "first_ts":  ts_list[0]  if ts_list else None,
        "last_ts":   ts_list[-1] if ts_list else None,
    }


def _print_index_stats(root: Path, wall_seconds: float, returncode: int) -> None:
    """讀取 stats.json + log，印出完整摘要。"""

    # ── stats.json ────────────────────────────────────────────
    stats_path = root / "output" / "stats.json"
    workflow_times: dict[str, float] = {}
    num_documents = 0
    if stats_path.exists():
        try:
            s = json.loads(stats_path.read_text(encoding="utf-8"))
            num_documents = s.get("num_documents", 0)
            for wf_name, wf_data in s.get("workflows", {}).items():
                workflow_times[wf_name] = wf_data.get("overall", 0)
        except Exception:
            pass

    # ── log ───────────────────────────────────────────────────
    log_dir = root / "logs"
    log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime) if log_dir.exists() else []
    log_stats = _parse_log_stats(log_files[-1]) if log_files else {}

    llm   = log_stats.get("llm",       {})
    embed = log_stats.get("embedding", {})

    # ── cost calculation ──────────────────────────────────────
    llm_in  = llm.get("prompt_tokens",     0)
    llm_out = llm.get("completion_tokens", 0)
    emb_in  = embed.get("prompt_tokens",   0)

    llm_cost  = (llm_in / 1_000_000) * _LLM_INPUT_PRICE_PER_1M + \
                (llm_out / 1_000_000) * _LLM_OUTPUT_PRICE_PER_1M
    emb_cost  = (emb_in / 1_000_000) * _EMBED_PRICE_PER_1M
    total_cost = llm_cost + emb_cost

    # ── status ────────────────────────────────────────────────
    status = "✅ 成功" if returncode == 0 else f"❌ 失敗 (exit {returncode})"

    # ── print ─────────────────────────────────────────────────
    bar = "═" * 60
    print(f"\n{bar}")
    print("📊  GraphRAG Index 統計摘要")
    print(bar)
    print(f"  狀態             : {status}")
    print(f"  處理文件數       : {num_documents}")
    w = int(wall_seconds)
    print(f"  ⏱️  總掛牆時間      : {wall_seconds:.1f} 秒  "
          f"({w // 3600}h {(w % 3600) // 60}m {w % 60}s)")
    if log_stats.get("first_ts") and log_stats.get("last_ts"):
        print(f"  Log 時間範圍      : {log_stats['first_ts']}  →  {log_stats['last_ts']}")

    # workflow breakdown
    if workflow_times:
        print()
        print("  ── Workflow 耗時 ─────────────────────────────────────")
        for wf, sec in sorted(workflow_times.items(), key=lambda x: -x[1]):
            print(f"     {wf:<35s} {sec:>8.1f} 秒")

    print()
    print("  ── Azure OpenAI LLM ──────────────────────────────────")
    print(f"  呼叫次數          : {llm.get('attempted', 0):,}  (失敗 {llm.get('failed', 0):,})")
    print(f"  輸入 tokens       : {llm_in:>14,}")
    print(f"  輸出 tokens       : {llm_out:>14,}")
    print(f"  總計 tokens       : {llm_in + llm_out:>14,}")
    print(f"  💰 預估費用        : ${llm_cost:.4f} USD")

    print()
    print("  ── Azure OpenAI Embedding ────────────────────────────")
    print(f"  呼叫次數          : {embed.get('attempted', 0):,}  (失敗 {embed.get('failed', 0):,})")
    print(f"  輸入 tokens       : {emb_in:>14,}")
    print(f"  💰 預估費用        : ${emb_cost:.4f} USD")

    print()
    print(f"  💵 本次總預估費用  : ${total_cost:.4f} USD")
    print(bar)
    print("  ⚠️  費用為估算值，實際請查閱 Azure Portal 帳單")
    print("     定價常數位於 index_tenders.py 頂部，請依部署調整")
    print(bar)


def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG Index 包裝器 — 自動計時並顯示 Token 費用統計"
    )
    parser.add_argument(
        "--root",
        required=True,
        help="GraphRAG 根目錄，例如: ./ragtest/stip",
    )
    # 允許傳遞額外的 graphrag 參數（如 --resume）
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="額外傳給 graphrag index 的參數",
    )
    args = parser.parse_args()

    root = Path(args.root)
    cmd  = [sys.executable, "-m", "graphrag", "index", "--root", str(root)] + args.extra

    print(f"🚀 執行: {' '.join(cmd)}")
    print(f"   開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    t0 = time.time()
    result = subprocess.run(cmd)
    wall_seconds = time.time() - t0

    _print_index_stats(root, wall_seconds, result.returncode)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
