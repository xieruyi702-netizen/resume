# Agentic-RAG 简历项目完整方案
## 知识库构建 · 记忆机制 · 工具调用 · 评测体系

> 目标仓库：https://github.com/leechenwei/Agentic-RAG （已核实，public，main 分支）
> 本方案基于 2026-09 对该仓库源码的实际核查编写，**已纠正此前流传信息中的多处不实描述**。

---

## 0. 先纠错：原仓库的真实情况（直接影响你的简历怎么写）

我把仓库源码拉下来逐文件核查过，你之前拿到的描述有几处**与事实相反**，如果照抄会踩坑：

| 之前说法 | 实际情况 |
|---|---|
| 评测集是 `evals/golden.toml`，字段有 `expect_tools / forbid_tools / expect_sources / expect_contains / must_refuse` | ❌ 实际是 `eval/golden_dataset.json`，共 **19 条** case，字段只有 `id / question / expected_source / expected_keywords / expect_refusal`。**没有任何工具维度标注** |
| "没有内置 Hit@k、MRR，需要自己补充实现" | ❌ **不成立**。`eval/runner.py` 已完整实现 Hit@k、MRR、keyword_hit、refusal_acc（还有 pytest 的 recall@5 回归门槛）。**"我自己补了 Hit@k/MRR" 这个简历亮点不能用了**，面试官 clone 代码一眼拆穿 |
| 核心目录叫 `agentic_rag/` | ❌ 实际是 `rag/` + `eval/`（README 结构图自己都写错了） |
| `docs/` 里是英文员工手册/退货政策/财报 | ❌ 实际是 `data/` 下 **5 份作者个人简历/项目介绍 txt**（about_chen_wei.txt 等）， golden 集也是问作者本人的 GPA、实习经历 |
| "prompt 全部抽离 py 文件，可以切换" | ❌ 只有一个 `SYSTEM_PROMPT` 常量硬编码在 `rag/agent.py` 里 |
| "会话状态管理" | ⚠️ 只有简单的 `history` 参数透传；ChromaDB 是**每 session 临时重建、不持久化** |
| —（未提及） | ⚠️ **无 LICENSE 文件**、0 star、0 fork；LLM 强绑定 **Gemini**（google-genai SDK，BYOK）；Streamlit UI 与核心逻辑耦合 |
| —（未提及） | ✅ 真实可借鉴的部分：hybrid 检索（ChromaDB dense + rank-bm25，RRF k=60 融合）、父子分块（parent 1024 / child 256）、BGE cross-encoder 重排、LangGraph 编排的 agent 工具循环、eval runner 的双层指标结构（检索层 + 生成层） |

**结论**：这个仓库的真实价值 = 一份 ~600 行的"混合检索 + agent 工具循环 + 规则评测"参考实现。它的数据集、评测维度（无工具评测、无 LLM-as-Judge）、记忆能力（无）、规划能力（无）、上下文管理（无）确实全是空白——**你要做的二开点依然成立，但叙事必须换成下面的版本**。

---

## 1. 总体决策：fork 改造 vs 参考架构自研

**推荐：参考其架构自研（把 fork 仓库留作 reference 对照），不直接在 fork 上改。** 理由：

1. **授权风险**：无 LICENSE = 默认保留所有权利。你公开一个整体衍生自它的项目，法律和学术诚信上都不干净。
2. **观感风险**：0 star 个人仓库 + 结构高度雷同的 fork，面试官点开原始仓库的瞬间，"基于开源项目二次开发"的含金量直接归零；而"参考业界模式自研的 Agent 框架"是加分项。
3. **成本其实不高**：核心代码就 `rag/` 下 6 个文件约 600 行，且模式非常标准（RRF 融合、父子分块、LangGraph 循环、eval runner）。照着它的成熟模式重写 + 扩展，比在别人的 Gemini 强绑定代码上做"换模型、删 Streamlit、改目录"的手术更快更干净。
4. **你真正要写的亮点（Plan-and-Execute、失败记忆、token 预算、多维评测）本来就全是新代码**，占比会远超移植部分。

折中做法：`git clone` 原仓库到本地 `reference/` 目录（不 fork、不进你的 git 历史），实现某个模块时对照它的写法，README 里注明 "retrieval 设计参考了 Agentic-RAG 的 RRF + rerank 模式" 即可。

---

## 2. 项目定位与简历叙事

**一句话**：自研的一个中文领域 Agentic-RAG 框架，支持 Plan-and-Execute 规划执行、失败轨迹长期记忆、上下文压缩与 token 预算，并配套一套覆盖「检索 / Agent 行为 / 答案质量」三层指标的自建评测体系，含多组消融实验。

简历亮点（修订版，全部可被代码验证）：

1. **知识库构建**：中文文档父子分块入库，BM25(jieba) + 向量(bge) 双路召回、RRF 融合、BGE cross-encoder 重排；
2. **Agent 双模式**：ReAct 工具循环为 baseline，改造为 **Plan-and-Execute**（规划 → 执行 → 失败重规划 → 自检）双模式可切换；
3. **长期失败记忆**：任务失败轨迹经 LLM 根因归因后写入向量库，新任务检索相似失败案例注入 planner 的 few-shot，**用消融实验证明收益**；
4. **上下文工程**：token 预算分配 + 历史轮次摘要压缩 + 检索块按重排分数裁剪，输出压缩前后 token 消耗对比；
5. **自建评测体系**：基于 CRUD-RAG 公开基准构建中文黄金集（单跳/多跳/拒答/工具约束四类 40–60 条，人工校对），指标覆盖 Hit@k、MRR、**Recall@k、nDCG@k、工具调用 F1、拒答准确率、faithfulness（LLM-as-Judge）**，eval-history 支持两次实验 diff。

> ⚠️ 简历红线：不要写"补充实现了 Hit@k/MRR"（参考仓库已有）；不要写"使用仓库自带数据集"（自带的是作者个人简历，且你本来就不用 fork 路线）。写"**自建**中文领域知识库与黄金评测集"。

---

## 3. 总体架构

```
                        ┌─────────────────────────────────────────────┐
                        │                  Agent Runtime               │
                        │                                             │
  用户问题 ──► 预算控制器 ──► memory_recall ──► planner ──► executor ◄──┐
              (token 预算)   (失败记忆检索)      │  (JSON plan) │          │
                                               │              ▼          │
                                               │        ┌─ 成功? ─┤      │
                                               │        ▼        ▼      │
                                               │   (下一步)   replanner ─┘
                                               │                 │
                                               ▼                 ▼
                                          answer ◄───────────────┘
                                               │
                                               ▼
                                          self_check ──失败──► 失败记忆写入
                                               │                (根因归因→向量库)
                                               ▼ 成功
                                            返回答案 + trace

  executor 可用工具：
    - retrieve(query, k)      混合检索（dense + BM25 → RRF → BGE rerank → 返回父块）
    - web_search(query)       （可选，用于「必须拒答/外部知识」对比 case）
    - calculator(expr)        （可选，体现多工具路由；成本极低）

  ─────────────────────────────────────────────────────────────
  支撑层：ingest(父子分块) │ ChromaDB(向量+失败记忆两个collection)
          │ BM25索引(jieba) │ eval runner(三层指标) │ LLM接入层(OpenAI兼容,可切换)
```

模式开关：`mode=react`（单循环 baseline）| `mode=plan_execute`（上图）——同一个 eval harness 跑两种模式，天然形成 E1 对比实验。

---

## 4. 目录结构（新建项目）

```
agentic-rag-zh/
├── configs/
│   ├── settings.yaml          # 模型、分块、检索、预算、模式开关
│   └── .env.example           # API keys（LLM_BASE_URL / LLM_API_KEY / LLM_MODEL）
├── knowledge/                 # 你的中文语料（markdown，50–80 篇）
├── evals/
│   ├── golden.jsonl           # 40–60 条中文黄金评测集
│   ├── runner.py              # 批量评测：检索层 + 行为层 + 答案层
│   ├── metrics.py             # hit@k / mrr / recall@k / ndcg@k / tool_f1
│   ├── judge.py               # LLM-as-Judge（faithfulness / answer_relevancy）
│   └── history/               # eval-history-*.jsonl，支持两次 run diff
├── src/
│   ├── llm/
│   │   └── provider.py        # OpenAI 兼容接入层（deepseek/qwen/glm 可切换）
│   ├── ingest/
│   │   ├── chunker.py         # 父子分块（中文分隔符）
│   │   └── indexer.py         # ChromaDB 入库 + BM25 索引构建
│   ├── retrieval/
│   │   ├── retriever.py       # 双路召回 + RRF 融合
│   │   └── reranker.py        # BGE cross-encoder
│   ├── agent/
│   │   ├── graph_react.py     # ReAct 循环（baseline）
│   │   ├── graph_pae.py       # Plan-and-Execute（核心二开）
│   │   ├── planner.py         # 规划器（结构化 JSON plan）
│   │   ├── executor.py        # 工具执行
│   │   ├── tools.py           # retrieve / web_search / calculator 注册表
│   │   └── prompts/           # 全部 prompt 独立 py 文件，可切换版本
│   ├── memory/
│   │   ├── failure_store.py   # 失败记忆向量库
│   │   └── attributor.py      # LLM 根因归因（轨迹 → root_cause + lesson）
│   ├── context/
│   │   ├── budget.py          # token 预算表
│   │   └── compressor.py      # 历史摘要压缩 + 检索块裁剪
│   └── trace.py               # AgentTrace（答案/工具调用/步数/token 消耗全程记录）
├── scripts/
│   ├── build_knowledge.py     # ingest 入口
│   ├── run_eval.py            # 评测入口（--mode react|pae --memory on|off ...）
│   └── gen_golden.py          # 辅助造 golden 集（LLM 生成候选 → 人工校对）
├── tests/                     # pytest：检索回归门槛、指标单测
├── reference/agentic-rag/     # 原仓库 clone，仅作对照，不进 git
├── requirements.txt
└── README.md                  # 架构图 + 实验表格 + 指标说明
```

---

## 5. 技术选型（中文场景适配）

| 组件 | 选型 | 说明 |
|---|---|---|
| LLM 接入 | **OpenAI 兼容协议自封装一层**（`src/llm/provider.py`） | DeepSeek-V3 / Qwen-max / GLM 任选，`.env` 三行切换。不用 LangChain 的 ChatModel 抽象也不必绑 Gemini；function calling 走 OpenAI tools 格式 |
| 编排 | LangGraph（沿用参考仓库的选型） | StateGraph 显式节点/条件边，面试常问，且 planner↔replanner 循环表达自然 |
| 向量模型 | `BAAI/bge-small-zh-v1.5`（默认，CPU 可跑）或 `bge-m3` | 中文检索效果好、本地推理零 API 成本 |
| 重排 | `BAAI/bge-reranker-base`（本地） | 与参考仓库同思路，换中文版 |
| BM25 | `rank-bm25` + **jieba 分词** | 原仓库用英文空格分词，中文必须换 jieba，否则 BM25 基本失效——这本身是个可以讲的细节 |
| 向量库 | ChromaDB（两个 collection：`kb_chunks` / `failure_memory`） | 轻量、持久化到本地目录（修正原仓库 ephemeral 的问题） |
| 分块 | `RecursiveCharacterTextSplitter`，中文分隔符 `["\n\n", "\n", "。", "；", "，"]` | parent 1024 / child 256（字符数，中文按 1 char≈1 token 量级估算） |
| token 计数 | tiktoken 估算或 `len(text)` 系数法 | 预算控制不需要精确到个位 |
| 评测增强 | Ragas（faithfulness / answer_relevancy），或自写 judge prompt | Ragas 对中文支持一般，自写 judge 更可控，二选一即可 |
| 测试 | pytest + 检索回归门槛（如 recall@5 ≥ 0.85 才算通过） | 参考仓库有此模式，保留 |

---

## 6. 模块设计

### 6.1 知识库构建（ingest）

1. 中文 markdown 语料 → 父子分块：parent（1024，保证上下文完整）→ child（256，保证检索精度）；**入库和 BM25 索引都建在 child 上，命中后返回其 parent**——面试必问"为什么要父子两块"：child 短、语义集中、向量/BM25 匹配准；parent 长、上下文全、喂给 LLM 答得对。
2. child 向量化入 ChromaDB（持久化），child 经 jieba 分词建 BM25 索引（pickle 持久化）。
3. 检索时双路各取 top-20 → RRF 融合（`score = Σ 1/(60 + rank_i)`，k=60 是原论文经验值，要能讲出公式）→ 取 top-20 → BGE reranker 精排 → top-5 parent 返回。

### 6.2 Agent 核心：ReAct → Plan-and-Execute

**ReAct baseline**（保留，作为对照）：LLM ↔ 工具循环，每次由模型即时决定下一步，max steps 封顶（参考实现就是这种，LangGraph 三节点：llm → tools → llm）。

**Plan-and-Execute 改造**（核心亮点）：

- **planner**：一次输出结构化 JSON 计划，不执行：

```json
{
  "understanding": "用户问 Pixel 手机退货期限",
  "steps": [
    {"id": 1, "tool": "retrieve", "query": "Pixel 退货政策 退货期限 天数", "expect": "找到退货窗口条款"},
    {"id": 2, "tool": "retrieve", "query": "激活日期对退货期的影响", "expect": "确认是否从激活日起算", "depends_on": [1]}
  ],
  "answer_strategy": "若两步结果一致，给出天数并引用条款出处"
}
```

- **executor**：按依赖顺序逐步执行，每步结果落盘到 state；
- **replanner**（触发条件：检索结果为空 / rerank 最高分低于阈值 / 步骤 expect 未满足）：把「原计划 + 已执行结果 + 失败原因」喂回 planner 局部重规划，不推倒重来；
- **answer**：汇总所有步骤结果生成带引用的最终答案；
- **self_check**：判定本轮是否"成功"（答案有引用、非拒答式糊弄、case 属于拒答类时确认拒答），失败则进入记忆写入流程。

**价值叙事**：多跳/跨文档问题上，planner 一次性显式分解比 ReAct 边走边看更稳（步数更少、更可控）；简单单跳问题上 ReAct 开销更低——这正是 E1 实验要量化的，避免"为了改而改"。

### 6.3 长期失败记忆模块（最大亮点，按此规格实现）

```python
# memory/failure_store.py 核心设计
@dataclass
class FailureRecord:
    question: str            # 原始任务
    plan_digest: str         # 当时的规划摘要
    outcome: str             # wrong_answer | wrong_source | missed_refusal | tool_misuse
    root_cause: str          # LLM 归因：如"检索 query 未覆盖同义词，BM25 与向量双路都没召回目标文档"
    lesson: str              # 可执行教训："涉及『退换/退货/7天无理由』类问题，query 需同时包含产品型号与政策关键词"
    embedding: list[float]   # question 的向量
```

- **写入**：self_check 判失败 → `attributor` 用 LLM 对轨迹做根因归因（结构化 JSON 输出 root_cause + lesson，限定 4 类枚举 outcome）→ 存入 `failure_memory` collection。**只存 lesson 和摘要，不存全轨迹**（省 token、防污染）。
- **读取**：新任务进入时，用 question 向量在 `failure_memory` 检 top-2，**相似度 ≥ 0.75 才注入**，渲染成 planner 的 few-shot 前缀：

```
[历史失败案例提醒]
· 相似问题「耳机七天内能退吗」曾失败：query 只写了"耳机 退货"漏掉政策关键词，导致检索未命中。
  本次注意：检索 query 需包含 具体产品 + 政策类关键词。
```

- **防污染三件套**（面试必追问）：① 相似度阈值卡入口；② 同一根因去重（归因时先查重）；③ lesson 只含"下次该怎么做"的可执行信息，不含错误答案本身。可再加人工抽检脚本 + 记录条数上限。
- **实验价值**：天然支持消融——E2：`--memory off` vs `--memory on`，重点观察"失败过的易错 case 二次运行"的通过率变化。建议 golden 集里留 8–10 条"陷阱题"（易错表述、易混淆文档），让记忆收益可测量。

### 6.4 上下文压缩与 token 预算

```python
# context/budget.py —— 预算表进入 trace，每步可观测
BUDGET = {
    "system_prompt": 1200,
    "failure_fewshot": 600,    # 记忆注入上限，超出截断最低相似度的
    "plan": 400,
    "history": 2000,           # 超出 → 旧轮次 LLM 摘要成 1 条
    "retrieval": 3000,         # rerank 分数降序贪心装入，装不下的丢
    "output_reserve": 1000,
}
```

- 历史压缩：超过 N 轮后，把最旧 k 轮用 LLM 压成一条 `summary` 消息（保留关键事实与结论，丢弃寒暄）；
- 检索裁剪：多步任务中各步检索块按 rerank 分数全局排序、去重（同 parent 只留最高分 child 的 parent）、贪心装入预算；
- trace 里记录每步 prompt tokens / completion tokens → README 输出「E3：压缩 off vs on」的 token 消耗与质量对照表。

### 6.5 评测体系（三层指标）

在参考仓库双层结构（检索层 + 生成层）上扩展为三层：

| 层 | 指标 | 来源 |
|---|---|---|
| L1 检索 | Hit@k、MRR（参考实现已有，照模式重写）、**Recall@k、nDCG@k（你新增，支撑多相关文档 case）** | 离线、确定性强、不花 LLM 钱 |
| L2 Agent 行为 | **工具调用 F1（expect_tools / forbid_tools，你新增）**、平均步数、token 消耗、重规划触发率 | 你新增的维度 |
| L3 答案 | 关键词命中率、拒答准确率（规则）+ **faithfulness、answer_relevancy（LLM-as-Judge，你新增）** | 规则 + judge |

**新增指标实现**（`evals/metrics.py`）：

```python
import math

def recall_at_k(chunks: list, expected_sources: list[str], k: int) -> float:
    """多相关文档场景：top-k 命中的相关文档数 / 总相关文档数"""
    if not expected_sources:
        return 0.0
    top = {c.source for c in chunks[:k]}
    return sum(1 for s in expected_sources if s in top) / len(expected_sources)

def ndcg_at_k(chunks: list, expected_sources: list[str], k: int) -> float:
    """排名质量：命中的越靠前分越高"""
    dcg = sum(1.0 / math.log2(i + 2)
              for i, c in enumerate(chunks[:k]) if c.source in expected_sources)
    ideal = min(len(expected_sources), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg else 0.0

def tool_metrics(called: list[str], expect: list[str], forbid: list[str]) -> dict:
    """工具维度：精确率×召回率合成 F1 + 禁用工具违例数"""
    called_set = set(called)
    tp = len(called_set & set(expect))
    precision = tp / len(called_set) if called_set else 1.0
    recall = tp / len(expect) if expect else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tool_f1": round(f1, 4),
            "forbid_violations": len(called_set & set(forbid))}
```

**LLM-as-Judge（`evals/judge.py`）**：faithfulness = 答案中的每个论断是否都能在检索到的 chunk 里找到支撑（逐论断打 0/1 取均值）；answer_relevancy = 答案与问题的相关度 1–5 分。judge 用与被测不同的模型（或至少不同 seed/温度），并随机抽 10% 人工校验 judge 的一致性——这句话面试能救场。

---

## 7. 中文黄金评测集

**格式**（jsonl，一行一条，字段即评测契约）：

```json
{
  "id": "news-2docs-07",
  "type": "multi_hop",
  "question": "某公司与供应商签订新协议后，其季度营收预期发生了什么变化？",
  "expected_sources": ["news/xinhua_2019_0417_3821.txt", "news/xinhua_2019_0512_0947.txt"],
  "expected_keywords": ["上调"],
  "expect_tools": ["retrieve"],
  "forbid_tools": ["web_search"],
  "expect_refusal": false,
  "note": "来自 questanswer_2docs 子集转换；协议事件与财报预期分布两篇文档，考多跳整合"
}
```

**配比建议（总量 40–60 条）**：单跳事实 45% · 跨文档/多跳 20% · 必须拒答（库中无答案）20% · 工具约束/陷阱题 15%。拒答类 `expected_sources: []` + `expect_refusal: true`。留 8–10 条"陷阱题"专门服务记忆消融实验。

**语料与测试集选型（GitHub 已逐仓核实，2026-09）**：

| 候选 | 知识库（语料） | 测试集 | License | 结论 |
|---|---|---|---|---|
| **CRUD-RAG**（`IAAR-Shanghai/CRUD_RAG`，主推荐） | ✅ 8万+ 篇新华新闻文档（`data/80000_docs/`），直接当检索库 | ✅ **有 QA，共约 2,400 条**（实测 `split_merged.json`：`questanswer_1doc` 800 / `questanswer_2docs` 797 / `questanswer_3docs` 797），**每条带 `news1..news_k` 来源文档全文 + `questions` + `answers` 标准答案**（2docs 起还有 `thoughts` 推理链）；另有 `hallu_modified` 反事实 1,268 条、`event_summary` 2,000 条 | ✅ Apache-2.0，数据直接在仓库里（split 文件 26MB） | **首选**，不用造语料、不用造多跳标注 |
| DuReader-robust（`baidu/DuReader`） | 百度搜索真实网页段落 | 鲁棒性 MRC，**含 deniable（不可答/应拒答）问题**，天然拒答 case | ✅ Apache-2.0，需经 luge.ai（千帆）渠道下载 | 拒答子集的补充来源 |
| CMRC 2018（`ymcui/cmrc2018`） | 中文维基百科段落 | 抽取式 QA 现成、带源段落 | 以仓库为准 | ⚠️ 维基常识内容大概率被 LLM 背过，会污染"必须检索"评测 |
| LawRefBook/Laws（1.8k star） | ✅ 真实法律条文 markdown（民法典/刑法/司法解释，一部法一个 md，天然章节结构） | ❌ 无配套 QA，需自构（LLM 辅助标注 + 人工校对） | ⚠️ 仓库无 license（法律官方文本本身不受著作权保护，可用，但别整库照搬声明版权） | 想要"领域知识库"叙事时的替代方案 |
| T2Ranking（`baidu/T2Ranking`） | 200万+ 中文网页段落 | 查询-段落相关性标注（纯检索） | 以仓库为准 | 可选：专门跑检索消融实验 |

**推荐组合：CRUD-RAG 为主 + 自标注补齐**（转换脚本即 `scripts/gen_golden.py`）：

1. 从 80k 文档库中**抽 200–500 篇做知识库**（控制规模和评测成本；抽子集本身要在 README 写清抽样规则，如按文档长度分桶分层抽样）；
2. 从 `questanswer_1doc/2docs/3docs` 抽取 30–40 条转成 golden jsonl（`n` 个来源文档 → `expected_sources: [...]`，正好覆盖单跳/跨文档/多跳三档，**Recall@k/nDCG 在多跳子集上才有区分度**）；
3. 自标注补齐 CRUD-RAG 没有的两类：**拒答 case 8–12 条**（问抽样子库里不存在的事件/实体，`expect_refusal: true`）+ **工具约束/陷阱 case 6–10 条**（服务记忆消融）；`hallu_modified` 子集可抽几条进 faithfulness 评测；
4. 每条转换后人工核对：答案关键词与语料一致、来源文档名映射正确——**用公开基准 ≠ 免校对**。

简历叙事相应调整："基于 CRUD-RAG 公开基准构建领域评测集（单跳/多跳/拒答/工具约束四类 40–60 条，人工校对），知识库为 8 万篇新闻文档的分层抽样子集"——比"全部自造"更可验证，比"直接用公开数据集"更有工程量。

**防"背题"（参数记忆污染评测）**：新闻语料时效性强、被预训练语料覆盖概率低，这是选 CRUD-RAG 而非维基类语料的核心理由；进阶做法——对 5 条 case 做实体替换对照实验（换掉人名/日期后模型若仍答原答案 = 靠记忆不是靠检索），这本身又是一段能写进 README 的分析。

---

## 8. 对比实验矩阵（README 的灵魂）

| 实验 | 对比组 | 核心指标 | 预期叙事 |
|---|---|---|---|
| E1 | `mode=react` vs `mode=plan_execute` | 多跳子集上的 answer_acc / 步数 / token | PAE 在多跳上更准或步数更少；单跳上两者持平（诚实呈现） |
| E2 | `--memory off` vs `--memory on` | 陷阱题子集通过率、整体 tool_f1 | 开记忆后曾失败的易错 case 通过率提升 |
| E3 | `--compress off` vs `--compress on` | token 消耗、answer_acc 保持率 | token 降 40%+，质量几乎不掉 |
| 附 | 检索消融：纯向量 / 纯 BM25 / hybrid / hybrid+rerank | Hit@5、MRR、nDCG@5 | 证明每层模块都有增益（这张表面试最好讲） |

运行方式：`python scripts/run_eval.py --mode pae --memory on --compress on --tag exp_e2_on`，输出 `evals/history/eval-history-*.jsonl`；`run_eval.py --diff exp_e2_off exp_e2_on` 直接生成 markdown 对照表贴 README。

---

## 9. 实施排期（约 10 个工作日）

| 天 | 交付 |
|---|---|
| D1 | 项目骨架、`llm/provider.py`（OpenAI 兼容 + function calling）、settings、trace |
| D2 | ingest：父子分块 + jieba BM25 + ChromaDB 入库 + 持久化 |
| D3 | 检索联调：双路召回 + RRF + BGE rerank；拉取 CRUD-RAG，按抽样规则切出知识库子集入库 |
| D4 | ReAct agent（LangGraph）跑通中文 QA，引用格式 [1][2] |
| D5 | eval runner（三层规则的 L1+L3）+ golden v1（从 questanswer 子集转换 40 条并人工校对）跑通基线 |
| D6 | Plan-and-Execute（planner/executor/replanner/answer/self_check 三节点循环） |
| D7 | 失败记忆：attributor + failure_store + planner 注入 |
| D8 | 上下文压缩 + token 预算 + trace 统计 |
| D9 | L2 工具指标 + Recall@k/nDCG + LLM-as-Judge；跑全 4 组实验 |
| D10 | README（架构图/实验表/复现命令）、pytest 回归门槛、收尾 |

每阶段结束立刻 commit 一个 tag，README 写清楚"如何复现每组实验"。

---

## 10. 面试防守要点（按新事实修订）

1. **Hit@k / MRR 必须会手推**（虽然不能声称是你补的，但一定会被问）：
   - Hit@k = 期望文档出现在 top-k 则记 1，按 case 求平均；
   - MRR = 每条 case 取第一个相关文档排名的倒数（1/rank），对全 case 求平均；
   - 加分补充：单相关文档时 Recall@k 退化为 Hit@k，所以引入多相关文档 case 后才需要 Recall@k / nDCG@k——这正好解释你为什么加这两个指标。
2. **RRF 公式**：`RRF(d) = Σ_i 1/(k + rank_i(d))`，k=60 抑制单路排名的极端权重；两路召回互补（向量管同义改写，BM25 管型号/数字等精确词）。
3. **父子分块为什么**：检索用 child（短、准），生成用 parent（全、稳）。
4. **失败记忆怎么防止越记越错**：阈值准入 + 根因去重 + 只存 lesson 不存错误答案 + 抽检。
5. **PAE vs ReAct 取舍**：显式分解适合多跳/跨文档；步数固定更可控；代价是简单问题上多一次 planner 调用——用 E1 数据说话。
6. **LLM-as-Judge 的可靠性**：judge 与被测模型不同源、固定 rubric、抽样人工对齐。
7. **千万别说**："我给某个仓库补了 Hit@k/MRR"、"用了仓库自带测试集"。
8. **为什么选 CRUD-RAG 当数据集**：Apache-2.0 协议干净；QA 按来源文档数分层（1/2/3 docs）正好支撑多跳评测与 Recall@k/nDCG；新闻语料时效性强、被 LLM 背过概率低，能证明答案真来自检索——"防参数记忆污染"这个意识本身就是加分项。

---

## 11. 已知风险

- **LLM API 成本**：全量 eval（60 case × 4 组实验 × 多步调用）建议用便宜模型（deepseek-chat / glm-4-flash 级别），E2/E3 各跑 2 次取均值即可；
- **本地嵌入/重排资源**：bge-small-zh + bge-reranker-base 在 MacBook CPU 上可跑（rerank top-20 单条 <1s），无需 GPU；
- **数据校对成本**：CRUD-RAG 转换后的 golden 条目必须逐条核对（答案关键词与语料一致、来源文档映射正确），拒答/陷阱类自标注 case 要有第二人（或隔天自查）复核一遍，否则规则评测失真；
- **参数记忆污染**：见 §7 防背题——新闻语料已大幅降低此风险，实体替换对照实验作为抽查手段；
- **Ragas 中文效果一般**：若引入后 faithfulness 波动大，果断换成自写 judge prompt（提前在 README 说明选择理由，反而加分）。
