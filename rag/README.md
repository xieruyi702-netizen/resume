# 美食汇 · 食谱 RAG（HowToCook）

知识库来源：[Anduin2017/HowToCook](https://github.com/Anduin2017/HowToCook)（约 369 道菜 Markdown，Unlicense）。

## 配置

```bash
cp rag/.env.example rag/.env   # 填入 ZHIPU_API_KEY（生成/Judge）
# 向量默认本地 BAAI/bge-small-zh-v1.5；MEISHI_RETRIEVAL=hybrid|bm25|dense
```

## 构建知识库 / 向量索引 / 测试集

```bash
# git clone --depth 1 https://github.com/Anduin2017/HowToCook.git rag/data/HowToCook
python rag/scripts/build_kb.py
python rag/scripts/build_vectors.py --backend local   # → kb/embeddings.npy
python rag/scripts/build_testset.py --n-kb 70 --n-missing 15 --n-ood 15
# → evals/testset.jsonl：100 条（库内70 + 库外菜15 + 非食谱15），均含 reference 标准答案
```

## 双路检索 + Agentic RAG（LangChain）

| 路 | 方法 |
|---|---|
| 关键词 | jieba + BM25（`keyword_search`） |
| 向量 | `BAAI/bge-small-zh-v1.5`（`vector_search`） |
| 融合 | RRF（`hybrid_search`） |

Agent：**LangChain `create_tool_calling_agent` + `AgentExecutor`（ReAct）**，以及 **Plan-and-Execute（`--mode pae`）**——先 JSON 计划再顺序执行工具再合成；底层检索/读菜谱仍用本地 `RecipeTools`。  
工具：`get_recipe` / `get_section` / `recall_memory` / `web_search`（Tavily，可 keyless）；入口 `rag/agent/agentic.py`。

```bash
pip install -r rag/requirements.txt
# 可选：export TAVILY_API_KEY=tvly-xxx
cd rag/agent && python -c "from web_search import tavily_search; import json; print(json.dumps(tavily_search('宫保鸡丁家常做法',3),ensure_ascii=False,indent=2))"
# 短期记忆 Redis（可选；未起则进程内兜底）
# docker compose -f docker/docker-compose.yml up -d redis-memory
python rag/agent/agent.py --user alice --session s1 --agentic "宫保鸡丁怎么做"
python rag/agent/agent.py --mode pae --no-llm "宫保鸡丁的原料有哪些"
# 短期：Redis mem:u:{user}:s:{session}:*（TTL + LTRIM + 会话摘要）
# 长期：Memdir rag/memory_store/memdir/{user}/（MEMORY.md + PROFILE.md + 主题笔记）
```

### 记忆与上下文组装

| 层 | 存储 | 淘汰 |
|---|---|---|
| 会话短期 | 独立 Redis `:6382` | 最近 N 轮 `LTRIM` + Key TTL（默认 24h）；轮次够多滚会话摘要 |
| 用户长期 | Memdir 文件笔记 | 索引行数上限；注入预算截断 |
| 注入 Agent | `ContextAssembler` | 长期 ≤800 / 短期 ≤1200 / 总 ≤2000 字符；PROFILE/忌口优先 |

组装顺序：系统策略 → **PROFILE / Memdir** → **近轮会话** → 当前问题；检索/工具结果走 tool 观察，不挤进记忆预算。

## 评测

```bash
# 检索全量 100；生成+Judge 分层抽样 30
HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 python rag/evals/run_eval.py \
  --judge-sample 30 --seed 42 \
  --out rag/evals/metrics_100.json --details rag/evals/details_100.jsonl
```
