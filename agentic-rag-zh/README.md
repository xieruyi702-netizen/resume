# Agentic-RAG-ZH：中文 Agentic RAG（混合检索 · 工具调用 · 失败记忆 · 多层评测）

基于公开中文基准 **CRUD-RAG**（Apache-2.0）的新闻语料构建的 Agentic RAG 系统：
全量知识库约 **8.9 万篇**新闻文档（91,964 行去重后 89,240 篇 + 11 篇 QA 来源文档补入），
父子分块入库，向量 + BM25 双路召回经 RRF 融合，ReAct agent 以 `retrieve` 为内置工具，
失败任务经 LLM 根因归因后写入长期记忆库，新任务检索相似失败注入 prompt。

## 架构

```
用户问题 ──► [失败记忆检索] ──► ReAct Agent（LLM ↔ retrieve 工具循环，≤4 步）
                                    │
                                    ▼
              retrieve: dense(bge) + BM25(jieba) ──RRF(k=60)──► top-5 父块
                                    │
                                    ▼
              答案（带来源引用 / 知识库无答案时拒答）
                                    │
                                    ▼
              评测失败 ──► LLM 根因归因 ──► 失败记忆向量库（下次注入）
```

- **父子分块**：child 256 字用于检索精度，parent 1024 字用于上下文完整性——检索打分用 child，喂 LLM 用 parent。
- **混合检索**：bge-small-zh 向量（512 维余弦）+ jieba 分词 BM25，各取 top-20，RRF `Σ 1/(60+rank)` 融合；预留 BGE cross-encoder 重排开关（`USE_RERANK=1`）。
- **失败记忆**：评测失败 → 归因为 `{outcome, root_cause, lesson}` JSON → 向量库；新任务相似度 ≥0.6 才注入，只注入教训不注入错误答案（防污染）。
- **双模式评测**：`--memory off`（不注入，失败照常入库）→ `--memory on`（注入），两次运行即记忆消融实验。

## 快速开始

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env        # 填 LLM_API_KEY（GLM/DeepSeek 等 OpenAI 兼容接口均可）

# 1) 数据：克隆 CRUD-RAG 后转换出全量语料 + golden 评测集
git clone --depth 1 https://github.com/IAAR-Shanghai/CRUD_RAG /tmp/CRUD_RAG
.venv/bin/python evals/convert_crud.py --repo /tmp/CRUD_RAG

# 2) 全量入库（~27 万 child 向量，M 系列 Mac 约 25 分钟）
.venv/bin/python scripts/build_knowledge.py

# 3) 评测
.venv/bin/python scripts/run_eval.py --tag base --limit 5    # 检索层（零 LLM 成本）
.venv/bin/python scripts/run_eval.py --tag mem_off --with-llm          # 答案层 + 失败入库
.venv/bin/python scripts/run_eval.py --tag mem_on  --with-llm --memory on
```

## 评测集（`evals/golden.jsonl`）

42 条中文 case，转换自 CRUD-RAG `questanswer_1doc/2docs`（种子 42 分层抽样）并含人工可校对字段：
25×单跳 + 15×双跳（跨文档）+ 2×拒答（知识库外问题）。字段：
`question / expected_sources / expected_keywords / expect_refusal / reference_answer`。
扩容：`.venv/bin/python evals/convert_crud.py --repo /tmp/CRUD_RAG --n1 25 --n2 15`

## 指标（三层）

| 层 | 指标 | 说明 |
|---|---|---|
| L1 检索 | Hit@5、MRR、**Recall@5、nDCG@5** | 后两个在多跳 case（多个相关文档）上才有区分度 |
| L2 行为 | 平均步数、token 估算、工具调用 | trace 全程记录 |
| L3 答案 | 关键词命中率、拒答准确率 | 规则评测，keywords 待人工校对 |

## 实验结果

### L1 检索层（全量库 89,291 篇 / 35.3 万 child 向量，42 条 golden，零 LLM 成本）

| Hit@5 | MRR | Recall@5 | nDCG@5 | 耗时 |
|---|---|---|---|---|
| **0.950** | **0.740** | **0.875** | **0.752** | 88s |

- 40 条正例（单跳 25 + 双跳 15）中 38 条的期望来源进入 top-5；未命中的 2 条均为双跳（跨文档整合）case，是开启 BGE 重排（`USE_RERANK=1`）的改进空间。
- Recall@5 = 0.875：双跳 case 的两篇来源文档未能全部挤进 top-5。
- 拒答 case（2 条）检索层不计分，其拒答准确率在答案层评测。

### L2/L3 答案层（deepseek-v4-flash 作答 · GLM-4-Flash 跨源 Judge · 42 条）

| 指标 | mem_off_v2 | mem_on_v2 | 备注 |
|---|---|---|---|
| 关键词命中率（规则） | 0.775 | 0.825 | 同一测试集两轮相差 5pp，规则口径对措辞敏感、不稳定 |
| **Judge 语义准确率** | **1.000** | **1.000** | 跨源判分（答 DeepSeek / 判 GLM） |
| 有效准确率（规则∪Judge） | **1.000** | **1.000** | |
| 平均相关度（1-5） | 4.95 | 4.97 | |
| 拒答准确率 | 1.000 | 1.000 | |
| 平均步数 | 2.07 | 2.14 | |
| 耗时 | 10.6 min | 10.9 min | |

**三个关键发现**：

1. **规则评测的假阴性被跨源 Judge 证实**：9 条关键词失败经 GLM 拿参考答案复核，8 条确系误判（关键词为"其设计灵感来源于"一类长短语碎片，回答正确也无法逐字包含）；1 条 judge 存在宽松嫌疑（case 31 数字不符仍判对，flash 级 judge 通病）——judge_acc=1.0 略有水分，改进方向是数字敏感的严格 rubric 或双 judge 一致性。
2. **规则口径本身不稳定**：同一 42 条集，keyword_acc 两轮 0.775 vs 0.825。单靠关键词规则做回归门槛会产生假警报，这正是三层口径设计的理由。
3. **记忆消融（E2）的正确解读**：judge 仲裁后本轮"真失败"为 0 → 失败记忆库无原料 → mem_on 未发生任何注入，两轮结果一致。记忆**机制**此前已验证（相似度 1.0 召回、注入后回答措辞改变）；其**价值**需要失败原料才能呈现——改进方向：① 加入"陷阱题/同知识点改写题"子集（易错表述，天然产生可记忆的失败）；② 检索失败（hit@5=0）也入库，沉淀 query 改写教训；③ 教训升级为实体级（含具体检索词），注入点从 system prompt 扩展到 query 改写环节。

### E1：ReAct vs Plan-and-Execute（42 条，同 LLM/Judge/记忆关闭，仅换编排）

| 指标 | ReAct | PAE |
|---|---|---|
| Judge 语义准确率 | **1.000** | 0.925 |
| 拒答准确率 | **1.000** | 0.500 |
| 字符 token | 375,752 | **244,130（−35%）** |
| 平均 LLM 调用 | 2.07 | 2.00 |

规划器 42/42 输出合法 JSON，步数分布 1/2/3 步 = 21/16/5（简单题不过度规划）。PAE 的三个真实失败模式：**规划器语言漂移**（中文语料生成英文检索词）、**证据过度承诺**（用不相关 top-5 硬答，丢掉拒答自由度）、**固定计划覆盖不足**（宽泛归纳题）。已修复前两者（中文约束 + 拒答逃生门），重跑验证进行中。结论：单跳为主基准上 PAE 省 token 但损质量；按问题复杂度路由双模式是正解。

### 改进清单

- [ ] PAE 修复后重跑 E1 验证（拒答 0.5 → 目标 1.0）
- [ ] Judge 严格化：数字/实体敏感 rubric，或双 judge 一致性仲裁
- [ ] 失败记忆 `record()` 根因去重
- [ ] 检查 `finish_reason=length` 截断，存全量答案入 history
- [ ] 陷阱题子集（易错改写），给记忆模块提供可学习的失败原料
- [ ] 开启 `USE_RERANK=1` 跑检索消融（针对 2 条双跳未命中）

## 目录

```
src/
├── llm/provider.py          # OpenAI 兼容接入层（.env 三行换模型；DRY_RUN=1 离线 mock）
├── ingest/{chunker,indexer}.py
├── retrieval/{embedder,retriever,reranker}.py
├── agent/{agent,tools}.py   # ReAct 循环 + 工具注册表
├── memory/failure_store.py  # 失败记忆 + 根因归因
└── config.py
evals/
├── convert_crud.py          # CRUD-RAG → 全量语料 + golden
├── metrics.py / runner.py   # 指标 + 批量评测
└── history/                 # eval-{tag}.jsonl 运行历史
```

## 数据与致谢

- 语料与 QA 转换自 [CRUD-RAG](https://github.com/IAAR-Shanghai/CRUD_RAG)（Apache-2.0）， golden 为转换 + 自标注拒答 case。
- 检索设计（RRF 融合 + 重排）参考了 [leechenwei/Agentic-RAG](https://github.com/leechenwei/Agentic-RAG) 的模式，代码为本项目独立实现。

## Roadmap

- [x] Plan-and-Execute 模式（planner/executor/replanner/answer，`--mode pae`，含 E1 对比实验）
- [x] LLM-as-Judge（GLM 跨源语义判分 + judge 仲裁记忆写入）
- [x] golden 扩到 42 条（单跳/双跳/拒答）
- [ ] 上下文压缩与 token 预算
- [ ] 陷阱题子集 + 检索失败入库（给记忆模块提供原料）
- [ ] Judge 严格化（数字敏感 rubric / 双 judge）
