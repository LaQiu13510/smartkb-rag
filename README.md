# SmartKB 智能知识库问答系统

SmartKB 是一个本地 RAG（Retrieval-Augmented Generation，检索增强生成）知识库问答系统。项目支持文档解析、文本切分、Embedding、Milvus 向量检索、BM25 关键词检索、RRF 融合、上下文管理和带来源的答案生成，并提供 FastAPI Web 界面。

## 项目背景

在企业内部，制度文档、技术规范、项目资料、接口说明和运维手册通常分散在不同文件和系统中。随着文档数量增长，员工在查找信息时需要反复搜索和阅读多个文档，效率低，也容易遗漏关键信息。

SmartKB 面向这个问题构建企业内部知识库问答系统：将内部文档解析、切分、向量化并写入知识库，员工可以直接用自然语言提问，系统通过混合检索找到相关片段，并结合大模型生成带来源的回答，从而降低内部资料检索成本，提高知识复用效率。

## 功能特性

- 支持 Markdown、TXT、PDF、DOCX 文档处理。
- 支持语义感知分块，并保留递归字符切分作为 fallback。
- 支持智谱、DashScope、HuggingFace、Google 等 Embedding 后端。
- 使用 Milvus 存储向量。
- 使用 PostgreSQL 存储文档元数据、对话历史和评测记录。
- 支持向量检索、BM25 检索、RRF 融合、轻量 query rewrite 和 rerank。
- 支持热门查询结果缓存，可使用进程内缓存或 Redis，降低重复查询延迟。
- 使用 RAG 上下文管理器处理来源标注、去重、长度预算和敏感信息脱敏。
- 提供 LangGraph RAG Agent，支持 `retrieve`、`list`、`chat` 路由。
- 提供 FastAPI Web 界面，支持文本入库、知识库问答、来源展示和检索片段展示。
- 支持 SSE（Server-Sent Events）流式输出。

## 系统架构

```text
Documents
  -> loader
  -> splitter
  -> embedding model
  -> Milvus vectors + PostgreSQL metadata
  -> hybrid retriever
  -> query result cache
  -> RAG context manager
  -> generation chain
  -> FastAPI UI / LangGraph agent
```

## 目录结构

```text
smartkb-rag/
├── app.py
├── cache_store.py
├── config.py
├── test_imports.py
├── test_e2e.py
├── agent/
├── database/
├── docs/
├── documents/
├── eval/
├── models/
└── rag/
```

## 安装

```bash
git clone <your-repository-url>
cd smartkb-rag
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 配置

复制环境变量模板，并填写自己的模型、数据库和向量库配置。

```bash
cp .env.example .env
```

真实运行时通常需要：

- DeepSeek 兼容聊天模型 API
- 至少一个 Embedding 服务或本地 HuggingFace Embedding 模型
- Milvus
- PostgreSQL
- Redis（可选，用于热门查询结果缓存；未配置时自动使用进程内缓存）

常用缓存配置：

```env
REDIS_URL=redis://127.0.0.1:6379/0
CACHE_BACKEND=auto
QUERY_CACHE_ENABLED=true
QUERY_CACHE_TTL_SECONDS=600
```

## 运行

启动 FastAPI 应用：

```bash
uvicorn app:app --host 127.0.0.1 --port 8501
```

打开浏览器访问：

```text
http://127.0.0.1:8501
```

## 测试

默认测试为离线测试，不依赖外部服务。

```bash
python test_imports.py
python test_e2e.py
python eval/agent_eval.py
python eval/retrieval_eval.py
```

配置 `.env` 后可以运行 live 检查：

```bash
python test_imports.py --live
python test_e2e.py --live
```

## 文档

- `docs/ARCHITECTURE.md`
- `docs/DEPLOYMENT.md`
- `docs/EVALUATION.md`
- `docs/PROJECT_REPORT.md`
