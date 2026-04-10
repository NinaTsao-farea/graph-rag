"""
GraphRAG Index 包裝器 (Local / Ollama 陣營)
自動計時、解析 log，並於完成後印出摘要。
本地 Ollama 模型執行，無 API 費用。

用法：
    python index_tenders_local.py --type stip
    python index_tenders_local.py --type government
    python index_tenders_local.py --type mixed_tenders

    # 允許傳遞額外的 graphrag 參數（如 --resume）：
    python index_tenders_local.py --type stip --resume

    # 直接指定目錄（向下相容）：
    python index_tenders_local.py --root ./ragtest/local/stip

備註：Azure / Gemini 陣營請使用 index_tenders_azure.py
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from doc_tasks import get_workspace_root, get_valid_doc_types  # noqa: E402

_CAMP = "local"


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

    llm_stats   = {"prompt_tokens": 0, "completion_tokens": 0, "attempted": 0, "failed": 0}
    embed_stats = {"prompt_tokens": 0, "attempted": 0, "failed": 0}

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

        if ct is not None:
            if pt > llm_stats["prompt_tokens"]:
                llm_stats["prompt_tokens"]     = pt
                llm_stats["completion_tokens"] = ct
                llm_stats["attempted"]         = data.get("attempted_request_count", 0)
                llm_stats["failed"]            = data.get("failed_response_count", 0)
        else:
            if pt > embed_stats["prompt_tokens"]:
                embed_stats["prompt_tokens"] = pt
                embed_stats["attempted"]     = data.get("attempted_request_count", 0)
                embed_stats["failed"]        = data.get("failed_response_count", 0)

    ts_list = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text)
    return {
        "llm":       llm_stats,
        "embedding": embed_stats,
        "first_ts":  ts_list[0]  if ts_list else None,
        "last_ts":   ts_list[-1] if ts_list else None,
    }


def _read_model_names_from_settings(root: Path) -> tuple[str, str]:
    """從 settings.yaml 讀取 completion / embedding 模型名稱，回傳 (llm_model, embed_model)。"""
    settings_path = root / "settings.yaml"
    llm_model = embed_model = "(未知)"
    if not settings_path.exists():
        return llm_model, embed_model
    try:
        raw = settings_path.read_text(encoding="utf-8")
        text = "\n".join(
            line for line in raw.splitlines()
            if not line.lstrip().startswith("#")
        )
        m = re.search(r"completion_models:.*?(?<!\w)model:\s*(\S+)", text, re.DOTALL)
        if m:
            llm_model = m.group(1)
        m = re.search(r"embedding_models:.*?(?<!\w)model:\s*(\S+)", text, re.DOTALL)
        if m:
            embed_model = m.group(1)
    except Exception:
        pass
    return llm_model, embed_model


def _print_index_stats(root: Path, wall_seconds: float, returncode: int) -> None:
    """讀取 stats.json + log，印出摘要（Local 陣營，無 API 費用）。"""

    llm_model, embed_model = _read_model_names_from_settings(root)

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

    # ── status ────────────────────────────────────────────────
    status = "✅ 成功" if returncode == 0 else f"❌ 失敗 (exit {returncode})"

    # ── print ─────────────────────────────────────────────────
    bar = "═" * 60
    print(f"\n{bar}")
    print("📊  GraphRAG Index 統計摘要（Local / Ollama 陣營）")
    print(bar)
    print(f"  狀態             : {status}")
    print(f"  處理文件數       : {num_documents}")
    print(f"  🤖 LLM 模型       : {llm_model}")
    print(f"  🔢 Embedding 模型 : {embed_model}")
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

    llm_in  = llm.get("prompt_tokens",     0)
    llm_out = llm.get("completion_tokens", 0)
    emb_in  = embed.get("prompt_tokens",   0)

    print()
    print(f"  ── Ollama LLM ({llm_model}) ──────────────────────────────")
    print(f"  呼叫次數          : {llm.get('attempted', 0):,}  (失敗 {llm.get('failed', 0):,})")
    print(f"  輸入 tokens       : {llm_in:>14,}")
    print(f"  輸出 tokens       : {llm_out:>14,}")
    print(f"  總計 tokens       : {llm_in + llm_out:>14,}")

    print()
    print(f"  ── Ollama Embedding ({embed_model}) ───────────────────────")
    print(f"  呼叫次數          : {embed.get('attempted', 0):,}  (失敗 {embed.get('failed', 0):,})")
    print(f"  輸入 tokens       : {emb_in:>14,}")

    print()
    print(f"  💡 本地執行，無 API 費用")
    print(bar)


def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG Index 包裝器 (Local / Ollama 陣營) — 自動計時並顯示統計"
    )
    parser.add_argument(
        "--type",
        dest="doc_type",
        default=None,
        choices=get_valid_doc_types(),
        help="文件類型，例如: --type stip",
    )
    parser.add_argument(
        "--root",
        dest="root",
        default=None,
        help="直接指定 GraphRAG 根目錄（向下相容，覆蓋 --type），例如: ./ragtest/local/stip",
    )
    # 允許傳遞額外的 graphrag 參數（如 --resume）
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="額外傳給 graphrag index 的參數",
    )
    args = parser.parse_args()

    if args.root:
        root = Path(args.root)
    elif args.doc_type:
        root = Path(get_workspace_root(args.doc_type, _CAMP))
    else:
        parser.error("請指定 --type（例如 --type stip）或 --root")
        return

    cmd = [sys.executable, "-m", "graphrag", "index", "--root", str(root)] + args.extra

    print(f"🚀 執行: {' '.join(cmd)}")
    print(f"   陣營: {_CAMP}  根目錄: {root}")
    print(f"   開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    t0 = time.time()
    result = subprocess.run(cmd)
    wall_seconds = time.time() - t0

    _print_index_stats(root, wall_seconds, result.returncode)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
