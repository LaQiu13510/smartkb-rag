"""SmartKB FastAPI application."""

from __future__ import annotations

import hashlib
import html
import json
import time
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cache_store import get_cache_store, make_query_cache_key
from config import QUERY_CACHE_ENABLED, QUERY_CACHE_TTL_SECONDS, TOP_K_RETRIEVAL
from database.milvus_store import get_milvus_store
from database.postgres_store import get_postgres_store
from models.embedding import get_embedding_model
from models.llm import get_llm
from rag.chain import get_rag_chain
from rag.loader import Document
from rag.retriever import HybridRetriever
from rag.splitter import TextSplitter


app = FastAPI(
    title="SmartKB",
    description="FastAPI RAG knowledge-base application",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: str = "smartkb-demo"
    top_k: int = Field(default=TOP_K_RETRIEVAL, ge=1, le=12)


class TextIndexRequest(BaseModel):
    file_name: str = Field(default="manual-note.md", min_length=1, max_length=120)
    content: str = Field(..., min_length=1, max_length=200000)


def safe_call(label: str, fn) -> dict[str, Any]:
    try:
        detail = fn()
        return {"label": label, "ok": True, "detail": str(detail)}
    except Exception as exc:
        return {"label": label, "ok": False, "detail": str(exc)[:220]}


@lru_cache(maxsize=1)
def splitter() -> TextSplitter:
    return TextSplitter()


def build_retriever() -> HybridRetriever:
    milvus = get_milvus_store()
    retriever = HybridRetriever(milvus_store=milvus, top_k=TOP_K_RETRIEVAL)
    rebuild_bm25(retriever)
    return retriever


def rebuild_bm25(retriever: HybridRetriever) -> None:
    try:
        collection = retriever.milvus.get_collection()
        collection.load()
        rows = collection.query(
            expr="id != ''",
            output_fields=["id", "content", "file_name", "chunk_index", "source_page"],
            limit=10000,
        )
    except Exception:
        return

    docs = [
        {
            "chunk_id": row.get("id", ""),
            "content": row.get("content", ""),
            "file_name": row.get("file_name", ""),
            "chunk_index": row.get("chunk_index", 0),
            "source_page": row.get("source_page", 0),
        }
        for row in rows
    ]
    if docs:
        retriever.build_bm25_index(docs)


def list_documents() -> list[dict[str, Any]]:
    pg = get_postgres_store()
    return [
        {
            "file_name": doc.file_name,
            "file_type": doc.file_type,
            "file_size": doc.file_size,
            "chunk_count": doc.chunk_count,
            "status": doc.status,
            "updated_at": doc.updated_at.isoformat() if doc.updated_at else "",
        }
        for doc in pg.get_all_documents()
    ]


def cache_stats() -> dict[str, Any]:
    try:
        stats = get_cache_store().stats()
        return {
            **stats,
            "enabled": QUERY_CACHE_ENABLED,
            "ttl_seconds": QUERY_CACHE_TTL_SECONDS,
        }
    except Exception as exc:
        return {
            "backend": "unavailable",
            "enabled": QUERY_CACHE_ENABLED,
            "keys": 0,
            "error": str(exc)[:180],
        }


def persist_chat(
    session_id: str,
    query_text: str,
    answer: str,
    sources: list[str],
    latency_ms: float,
) -> None:
    try:
        get_postgres_store().add_chat(
            session_id=session_id,
            role="user",
            content=query_text,
        )
        get_postgres_store().add_chat(
            session_id=session_id,
            role="assistant",
            content=answer,
            sources=", ".join(sources),
            latency_ms=latency_ms,
        )
    except Exception:
        pass


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/api/health")
def health() -> dict[str, Any]:
    checks = [
        safe_call("Milvus", lambda: f"{get_milvus_store().count()} vectors"),
        safe_call("PostgreSQL", lambda: f"{len(list_documents())} documents"),
        safe_call("Embedding", lambda: get_embedding_model().test_connection()[1]),
        safe_call("LLM", lambda: get_llm(max_tokens=64).test_connection()[1]),
        safe_call("Cache", lambda: cache_stats()),
    ]
    return {"checks": checks, "cache": cache_stats()}


@app.get("/api/documents")
def documents() -> dict[str, Any]:
    try:
        docs = list_documents()
        return {"ok": True, "documents": docs}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:220], "documents": []}


@app.get("/api/cache")
def cache() -> dict[str, Any]:
    return {"cache": cache_stats()}


@app.post("/api/query")
def query(request: QueryRequest) -> dict[str, Any]:
    start = time.time()
    try:
        cache_key = make_query_cache_key(request.query, request.top_k)
        if QUERY_CACHE_ENABLED:
            try:
                cached = get_cache_store().get_json(cache_key)
            except Exception:
                cached = None
            if cached:
                elapsed = round((time.time() - start) * 1000, 1)
                cached_latency = cached.get("latency_ms", 0)
                result = {
                    **cached,
                    "latency_ms": elapsed,
                    "cached_latency_ms": cached_latency,
                    "cache_hit": True,
                    "cache": {"hit": True, "key": cache_key, **cache_stats()},
                }
                persist_chat(
                    request.session_id,
                    request.query,
                    result.get("answer", ""),
                    result.get("sources", []),
                    elapsed,
                )
                return result

        retriever = build_retriever()
        results = retriever.search(request.query, top_k=request.top_k)
        answer = get_rag_chain().answer(request.query, results)
        elapsed = round((time.time() - start) * 1000, 1)
        persist_chat(
            request.session_id,
            request.query,
            answer["answer"],
            answer.get("sources", []),
            elapsed,
        )
        result = {
            "ok": True,
            "answer": answer["answer"],
            "sources": answer.get("sources", []),
            "results": results,
            "latency_ms": elapsed,
            "cache_hit": False,
            "cache": {"hit": False, "key": cache_key, **cache_stats()},
        }
        if QUERY_CACHE_ENABLED:
            try:
                get_cache_store().set_json(cache_key, result, QUERY_CACHE_TTL_SECONDS)
            except Exception:
                pass
        return result
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:500], "answer": "", "sources": [], "results": [], "latency_ms": 0}


@app.get("/api/query/stream")
def query_stream(
    query: str,
    session_id: str = "smartkb-demo",
    top_k: int = TOP_K_RETRIEVAL,
) -> StreamingResponse:
    def generate():
        yield sse("stage", {"message": "正在检索知识库"})
        try:
            result = query_answer(query, session_id=session_id, top_k=top_k)
            if not result.get("ok"):
                yield sse("error", {"message": result.get("error", "查询失败")})
                return
            yield sse("stage", {"message": "正在生成回答"})
            for chunk in chunk_text(result.get("answer", "")):
                yield sse("delta", {"text": chunk})
            yield sse("final", result)
        except Exception as exc:
            yield sse("app_error", {"message": str(exc)[:500]})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def query_answer(query_text: str, session_id: str, top_k: int) -> dict[str, Any]:
    request = QueryRequest(query=query_text, session_id=session_id, top_k=top_k)
    return query(request)


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def chunk_text(text: str, size: int = 28):
    for index in range(0, len(text), size):
        yield text[index:index + size]


@app.post("/api/index-text")
def index_text(request: TextIndexRequest) -> dict[str, Any]:
    try:
        file_name = request.file_name.strip() or "manual-note.md"
        file_hash = hashlib.md5(request.content.encode("utf-8")).hexdigest()[:12]
        doc = Document(
            request.content,
            {
                "source": file_name,
                "file_type": file_name.rsplit(".", 1)[-1] if "." in file_name else "txt",
                "file_hash": file_hash,
                "page": 0,
            },
        )
        chunks = splitter().split_documents([doc])
        contents = [chunk.page_content for chunk in chunks]
        embeddings = get_embedding_model().embed_documents(contents)
        ids = [
            hashlib.md5(f"{file_hash}:{idx}".encode("utf-8")).hexdigest()[:24]
            for idx, _ in enumerate(chunks)
        ]
        get_milvus_store().insert(
            ids=ids,
            contents=contents,
            embeddings=embeddings,
            file_names=[file_name] * len(chunks),
            chunk_indices=[chunk.metadata.get("chunk_index", idx) for idx, chunk in enumerate(chunks)],
            source_pages=[0] * len(chunks),
        )
        get_postgres_store().add_document(
            file_name=file_name,
            file_type=doc.metadata.get("file_type", "txt"),
            file_size=len(request.content.encode("utf-8")),
            chunk_count=len(chunks),
            total_chars=len(request.content),
        )
        try:
            cache_invalidated = get_cache_store().delete_prefix("query:")
        except Exception:
            cache_invalidated = 0
        return {
            "ok": True,
            "file_name": file_name,
            "chunks": len(chunks),
            "cache_invalidated": cache_invalidated,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:500]}


INDEX_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SmartKB</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #fff;
      --line: #d9dee7;
      --muted: #667085;
      --text: #101828;
      --accent: #0f766e;
      --bad: #b42318;
      --ok: #087443;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    button, textarea, input { font: inherit; }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      border-bottom: 1px solid var(--line);
      background: rgba(246,247,249,.96);
      backdrop-filter: blur(10px);
      box-shadow: 0 8px 24px rgba(16,24,40,.04);
    }
    .top-inner {
      max-width: 1440px;
      margin: 0 auto;
      padding: 14px 20px;
      display: grid;
      grid-template-columns: minmax(240px, 1fr) 3fr;
      gap: 16px;
      align-items: center;
    }
    h1 { margin: 0; font-size: 22px; letter-spacing: 0; }
    .subtitle { color: var(--muted); font-size: 13px; margin-top: 3px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0,1fr));
      gap: 10px;
    }
    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(16,24,40,.04);
    }
    .metric { padding: 9px 10px; }
    .metric-label { color: var(--muted); font-size: 12px; }
    .metric-value { margin-top: 2px; font-weight: 650; overflow-wrap: anywhere; }
    .shell {
      max-width: 1440px;
      margin: 0 auto;
      padding: 18px 20px 32px;
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(360px, .9fr);
      gap: 18px;
      align-items: start;
    }
    .main { padding: 16px; }
    .side {
      position: sticky;
      top: 96px;
      max-height: calc(100vh - 116px);
      overflow: auto;
      padding: 16px;
    }
    .title-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      border-bottom: 1px solid var(--line);
      padding-bottom: 14px;
      margin-bottom: 14px;
    }
    .title-row h2 { font-size: 18px; margin: 0; }
    .title-row p { margin: 4px 0 0; color: var(--muted); font-size: 13px; }
    .section-title { font-weight: 700; margin: 0 0 10px; }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 11px;
      background: #fff;
      color: var(--text);
      outline: none;
    }
    input:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15,118,110,.12);
    }
    textarea { min-height: 120px; resize: vertical; }
    .row { display: grid; grid-template-columns: minmax(0,1fr) auto; gap: 10px; }
    .btn {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 9px 11px;
      cursor: pointer;
      text-align: center;
      white-space: nowrap;
    }
    .primary {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
      font-weight: 650;
    }
    .block { margin-top: 14px; }
    .msg {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #fff;
      margin-top: 10px;
    }
    .question { background: #f0fdfa; border-color: #99f6e4; }
    .source-chip {
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      color: #344054;
      background: #f9fafb;
      margin: 3px;
    }
    details {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 9px 10px;
      margin: 8px 0;
    }
    summary { cursor: pointer; font-weight: 650; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; }
    .ok { color: var(--ok); }
    .bad { color: var(--bad); }
    .muted { color: var(--muted); }
    .loading { opacity: .64; pointer-events: none; }
    @media (max-width: 980px) {
      .top-inner, .shell { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, minmax(0,1fr)); }
      .side { position: static; max-height: none; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="top-inner">
      <div>
        <h1>SmartKB</h1>
        <div class="subtitle">RAG Knowledge Base · Hybrid Retrieval · Source-grounded Answers</div>
      </div>
      <div class="metrics">
        <div class="metric"><div class="metric-label">文档</div><div class="metric-value" id="metric-docs">-</div></div>
        <div class="metric"><div class="metric-label">服务</div><div class="metric-value" id="metric-health">-</div></div>
        <div class="metric"><div class="metric-label">延迟</div><div class="metric-value" id="metric-latency">-</div></div>
        <div class="metric"><div class="metric-label">来源</div><div class="metric-value" id="metric-sources">0</div></div>
      </div>
    </div>
  </header>

  <main class="shell">
    <section class="panel main">
      <div class="title-row">
        <div>
          <h2>知识库问答</h2>
          <p>检索 Milvus + BM25，并用上下文管理器生成带来源的回答。</p>
        </div>
        <button class="btn" id="health-btn">检查服务</button>
      </div>
      <div class="row">
        <input id="query" placeholder="输入问题，例如：RAG 的核心组件有哪些？" />
        <button class="btn primary" id="ask-btn">提问</button>
      </div>
      <div id="answer"></div>

      <div class="block">
        <div class="section-title">快速添加文本到知识库</div>
        <input id="file-name" value="manual-note.md" />
        <textarea id="doc-content" placeholder="粘贴一段 Markdown/TXT 内容，用于快速构建演示知识库。"></textarea>
        <button class="btn primary" id="index-btn">写入知识库</button>
        <div id="index-result" class="muted"></div>
      </div>
    </section>

    <aside class="panel side">
      <div class="section-title">服务状态</div>
      <div id="health"></div>
      <div class="block">
        <div class="section-title">文档列表</div>
        <div id="documents"></div>
      </div>
      <div class="block">
        <div class="section-title">最近来源</div>
        <div id="sources" class="muted">提问后会展示来源。</div>
      </div>
      <div class="block">
        <div class="section-title">检索片段</div>
        <div id="chunks" class="muted">提问后会展示 Top-K 片段。</div>
      </div>
    </aside>
  </main>

  <script>
    const queryInput = document.getElementById("query");
    const askBtn = document.getElementById("ask-btn");
    const healthBtn = document.getElementById("health-btn");
    const answerBox = document.getElementById("answer");

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function setMetric(id, value) {
      document.getElementById(id).textContent = value;
    }

    async function loadDocuments() {
      const data = await fetch("/api/documents").then(r => r.json());
      const docs = data.documents || [];
      setMetric("metric-docs", docs.length);
      document.getElementById("documents").innerHTML = docs.map(doc => `
        <details>
          <summary>${escapeHtml(doc.file_name)}</summary>
          <div class="muted">${escapeHtml(doc.file_type)} · ${doc.chunk_count} chunks · ${escapeHtml(doc.status)}</div>
        </details>
      `).join("") || "<div class='muted'>暂无文档。</div>";
    }

    async function loadHealth() {
      const data = await fetch("/api/health").then(r => r.json());
      const checks = data.checks || [];
      const okCount = checks.filter(item => item.ok).length;
      setMetric("metric-health", `${okCount}/${checks.length}`);
      document.getElementById("health").innerHTML = checks.map(item => `
        <div class="${item.ok ? "ok" : "bad"}">${escapeHtml(item.label)} · ${item.ok ? "正常" : "异常"}</div>
        <div class="muted">${escapeHtml(item.detail)}</div>
      `).join("");
    }

    async function ask() {
      const query = queryInput.value.trim();
      if (!query) return;
      document.body.classList.add("loading");
      answerBox.innerHTML = `<div class="msg question">${escapeHtml(query)}</div><div class="msg muted" id="stream-state">正在连接流式接口...</div><div class="msg" id="stream-answer"></div>`;
      const stateBox = document.getElementById("stream-state");
      const streamAnswer = document.getElementById("stream-answer");
      const params = new URLSearchParams({query, session_id: "smartkb-web"});
      const events = new EventSource(`/api/query/stream?${params.toString()}`);
      let streamDone = false;
      events.addEventListener("stage", event => {
        const data = JSON.parse(event.data);
        stateBox.textContent = data.message || "处理中...";
      });
      events.addEventListener("delta", event => {
        const data = JSON.parse(event.data);
        streamAnswer.textContent += data.text || "";
      });
      events.addEventListener("final", event => {
        streamDone = true;
        const data = JSON.parse(event.data);
        setMetric("metric-latency", `${data.latency_ms} ms${data.cache_hit ? " · cache" : ""}`);
        setMetric("metric-sources", (data.sources || []).length);
        stateBox.textContent = data.cache_hit ? "完成 · 缓存命中" : "完成";
        document.getElementById("sources").innerHTML = (data.sources || []).map(src =>
          `<span class="source-chip">${escapeHtml(src)}</span>`
        ).join("") || "<div class='muted'>无来源。</div>";
        document.getElementById("chunks").innerHTML = (data.results || []).map((item, idx) => `
          <details>
            <summary>${idx + 1}. ${escapeHtml(item.file_name || "unknown")} · score=${escapeHtml(item.rerank_score || item.hybrid_score || item.score || "-")}</summary>
            <pre>${escapeHtml((item.content || "").slice(0, 900))}</pre>
          </details>
        `).join("");
        events.close();
        document.body.classList.remove("loading");
      });
      events.addEventListener("app_error", event => {
        let message = "流式连接失败";
        try { message = JSON.parse(event.data).message || message; } catch {}
        stateBox.className = "msg bad";
        stateBox.textContent = message;
        events.close();
        document.body.classList.remove("loading");
      });
      events.onerror = () => {
        if (streamDone) return;
        stateBox.className = "msg bad";
        stateBox.textContent = "流式连接失败";
        events.close();
        document.body.classList.remove("loading");
      };
    }

    async function indexText() {
      const fileName = document.getElementById("file-name").value.trim() || "manual-note.md";
      const content = document.getElementById("doc-content").value.trim();
      if (!content) return;
      document.body.classList.add("loading");
      const box = document.getElementById("index-result");
      box.textContent = "写入中...";
      try {
        const response = await fetch("/api/index-text", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({file_name: fileName, content})
        });
        const data = await response.json();
        box.textContent = data.ok ? `已写入 ${data.file_name}，chunks=${data.chunks}` : `写入失败: ${data.error}`;
        loadDocuments();
      } finally {
        document.body.classList.remove("loading");
      }
    }

    askBtn.addEventListener("click", ask);
    healthBtn.addEventListener("click", loadHealth);
    document.getElementById("index-btn").addEventListener("click", indexText);
    queryInput.addEventListener("keydown", event => {
      if (event.key === "Enter") ask();
    });

    loadDocuments();
  </script>
</body>
</html>
"""
