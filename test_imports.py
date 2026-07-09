# -*- coding: utf-8 -*-
"""SmartKB import and structure checks.

Default mode is offline and does not access external APIs or local databases.
Use ``--live`` to verify configured LLM, embedding, Milvus, and PostgreSQL
services on your own machine.
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))


def step(name: str, fn):
    print(f"[TEST] {name} ...", end=" ")
    try:
        result = fn()
        print("[OK]")
        if result:
            print(f"       {result}")
    except Exception as exc:
        print("[FAIL]")
        print(f"       {type(exc).__name__}: {exc}")
        raise


def check_config():
    import config

    assert config.ROOT_DIR.exists()
    assert config.DOCUMENTS_DIR.exists()
    assert config.DATA_DIR.exists()
    assert config.CHUNK_SIZE > 0
    assert config.TOP_K_RETRIEVAL > 0
    assert config.QUERY_CACHE_TTL_SECONDS > 0
    return f"root={config.ROOT_DIR}"


def check_core_imports():
    import cache_store  # noqa: F401
    import agent.graph  # noqa: F401
    import agent.tools  # noqa: F401
    import database.milvus_store  # noqa: F401
    import database.postgres_store  # noqa: F401
    import eval.dataset  # noqa: F401
    import eval.metrics  # noqa: F401
    import models.embedding  # noqa: F401
    import models.llm  # noqa: F401
    import rag.chain  # noqa: F401
    import rag.loader  # noqa: F401
    import rag.retriever  # noqa: F401
    import rag.splitter  # noqa: F401

    return "core modules imported without creating live clients"


def check_query_cache():
    from cache_store import InMemoryTTLCacheStore, make_query_cache_key

    cache = InMemoryTTLCacheStore("offline")
    key = make_query_cache_key("RAG 混合检索", top_k=3)
    cache.set_json(key, {"answer": "cached", "sources": ["rag.md"]}, ttl_seconds=60)
    hit = cache.get_json(key)
    assert hit and hit["answer"] == "cached"
    assert cache.stats()["keys"] == 1
    assert cache.delete_prefix("query:") == 1
    assert cache.get_json(key) is None
    return "query cache ok"


def check_document_pipeline():
    from rag.loader import Document
    from rag.splitter import TextSplitter

    doc = Document(
        "RAG 使用文档加载、文本分割、Embedding、Milvus 和混合检索完成问答。",
        {"source": "offline.md", "file_hash": "offline", "page": 0},
    )
    chunks = TextSplitter(chunk_size=30, chunk_overlap=5).split_documents([doc])
    assert chunks
    assert chunks[0].metadata["chunk_id"].startswith("offline_chunk_")
    return f"chunks={len(chunks)}"


def check_bm25_retrieval():
    from rag.retriever import HybridRetriever

    class FakeMilvus:
        pass

    docs = [
        {
            "chunk_id": "a",
            "content": "RAG 混合检索 使用 BM25 和 向量检索。",
            "file_name": "rag.md",
            "chunk_index": 0,
            "source_page": 0,
        },
        {
            "chunk_id": "b",
            "content": "部署需要 PostgreSQL、Milvus 和 API Key。",
            "file_name": "deploy.md",
            "chunk_index": 0,
            "source_page": 0,
        },
        {
            "chunk_id": "c",
            "content": "评测关注 Hit Rate、MRR 和延迟。",
            "file_name": "eval.md",
            "chunk_index": 0,
            "source_page": 0,
        },
    ]
    retriever = HybridRetriever(FakeMilvus())
    retriever.build_bm25_index(docs)
    results = retriever._bm25_search("BM25 混合检索", top_k=2)
    assert results
    assert results[0]["chunk_id"] == "a"
    return f"bm25_top={results[0]['file_name']}"


def live_checks():
    from config import DB_URL, MILVUS_HOST
    from database.milvus_store import get_milvus_store
    from database.postgres_store import get_postgres_store
    from models.embedding import get_embedding_model
    from models.llm import get_llm

    print(f"[LIVE] Milvus host: {MILVUS_HOST}")
    print(f"[LIVE] PostgreSQL configured: {bool(DB_URL)}")

    emb = get_embedding_model()
    emb_ok, emb_msg = emb.test_connection()
    print(f"[LIVE] Embedding: {emb_ok} {emb_msg}")

    llm_ok, llm_msg = get_llm(max_tokens=64).test_connection()
    print(f"[LIVE] LLM: {llm_ok} {llm_msg}")

    milvus = get_milvus_store()
    milvus_ok = milvus.connect()
    print(f"[LIVE] Milvus: {milvus_ok}")

    pg = get_postgres_store()
    pg.init_tables()
    print(f"[LIVE] PostgreSQL: documents={len(pg.get_all_documents())}")


def main():
    step("配置加载", check_config)
    step("核心模块导入", check_core_imports)
    step("热门查询缓存", check_query_cache)
    step("文档切分流水线", check_document_pipeline)
    step("BM25 离线检索", check_bm25_retrieval)

    if "--live" in sys.argv:
        step("外部服务连通", live_checks)

    print("\nSmartKB import checks passed.")


if __name__ == "__main__":
    main()
