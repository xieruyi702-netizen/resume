# 实验记录（Experiment Log）

> 项目：Agentic-RAG-ZH（中文新闻知识库 · ReAct Agent · 失败记忆 · 三层评测）
> 本文档记录 2026-09-05 全部评测运行的原始数据与结论，README 只保留最终版数字，细节以此为准。
> 所有运行历史逐 case 存于 `evals/history/eval-*.jsonl`。

---

## 1. 环境与配置

| 项 | 值 |
|---|---|
| 机器 | MacBook (Apple Silicon, MPS) |
| 知识库语料 | CRUD-RAG `80000_docs`：91,964 行 → 去重 89,240 篇 + QA 来源文档补入 51 篇 = **89,291 篇** |
| 索引规模 | **97,254 parents / 352,867 children**（ChromaDB cosine + rank-bm25/jieba） |
| 嵌入模型 | `BAAI/bge-small-zh-v1.5`（512 维，归一化；MPS fp16 batch512 实测 ~177 条/s） |
| 分块 | parent 1024 / child 256 / overlap 32，中文分隔符 |
| 检索 | dense top-20 + BM25 top-20 → RRF(k=60) → parent 去重 → top-5 |
| 被测 LLM | `deepseek-v4-flash`（temp 0.2，function calling，max 4 步） |
| Judge | `glm-4-flash`（跨源判分，结构化 JSON：semantic_correct + relevancy 1-5） |
| Python | 3.12 (uv venv)；chromadb / sentence-transformers / rank-bm25 / jieba / langchain-text-splitters / openai |

评测集：`evals/golden.jsonl` 42 条 = 25 单跳（questanswer_1doc 抽样）+ 15 双跳（questanswer_2docs）+ 2 拒答（自标注）。种子 42。
QA 来源文档中 43 篇不在原始去重语料中，已通过 `scripts/index_extras.py` 增量补入索引后再评测。

---

## 2. 检索层基线（零 LLM 成本）

### E-R1：13 条 golden（索引刚建成，parent 去重修复前）

| Hit@5 | MRR | Recall@5 | nDCG@5 | 耗时 |
|---|---|---|---|---|
| 1.000 | 0.750 | 0.909 | **1.035（bug）** | 34s |

> nDCG > 1 暴露 bug：同一文档多个 child 同时进 top-5 被重复计分。修复：检索器按 parent 去重（同时腾出 top-k 槽位）。该 bug 修复本身即为一条消融证据。

### E-R2：42 条 golden（`eval-base42.jsonl`，最终基线）

| Hit@5 | MRR | Recall@5 | nDCG@5 | 耗时 |
|---|---|---|---|---|
| **0.950** | **0.740** | **0.875** | **0.752** | 88s |

- 40 条正例命中 38；未命中 2 条均为双跳 case（含 `multi-64fa9b2f-40`）。
- 拒答 2 条检索层不计分。

---

## 3. 答案层：四轮完整运行（42 条 × 4）

### 聚合对比总表

| 指标 | E1-off `mem_off` | E1-on `mem_on` | E2-off `mem_off_v2`（+judge） | E2-on `mem_on_v2`（+judge） |
|---|---|---|---|---|
| Hit@5 | 0.950 | 0.950 | 0.950 | 0.950 |
| MRR | 0.740 | 0.740 | 0.740 | 0.740 |
| Recall@5 | 0.875 | 0.875 | 0.875 | 0.875 |
| nDCG@5 | 0.752 | 0.752 | 0.752 | 0.752 |
| 关键词命中率（规则） | **0.825** | **0.825** | **0.775** | **0.825** |
| Judge 语义准确率 | — | — | 1.000 | 1.000 |
| 有效准确率（规则∪Judge） | — | — | 1.000 | 1.000 |
| 平均相关度（1-5） | — | — | 4.95 | 4.97 |
| 拒答准确率 | 1.000 | 1.000 | 1.000 | 1.000 |
| 平均步数 | 2.19 | 2.17 | 2.07 | 2.14 |
| 字符 token 总量 | 484,816 | 464,589 | 375,752 | 411,880 |
| 失败 case 数 | 7 | 7（同批） | 9 | 7 |
| 耗时 | 13.6 min | 14.1 min | 10.6 min | 10.9 min |

### E1（无 Judge）：记忆注入首轮消融

- 两轮失败为**同一批 7 条**，零翻转。
- 但 7 条回答文本两轮之间**全部发生变化** → 记忆注入确实影响了生成（机制生效），只是未跨过规则判定线。
- 失败清单：`single-64fa9b32-01, single-64fa9b2d-05, single-64fa9b2b-10, single-64fa9b32-15, single-64fa9b27-17, multi-64fa9b2d-35, multi-64fa9b27-36`
- 失败记忆召回验证：对失败问题召回相似度 **1.0 / 0.981**，lesson 成功注入 system prompt。
- 记忆库污染问题：`record()` 未去重，两轮后库里 14 条（实为 7 条 × 2）。

### E2（GLM Judge 仲裁）：规则误判的跨源裁决

- E2-off 中 9 条规则失败被 judge **全部改判通过**（`eval_artifact=true`，不写入失败记忆库 → 真失败 0 条）。
- 逐条人工审计 9 条改判：**8 条改判正确**（关键词确为长短语碎片，如「其设计灵感来源于」「遭袭击身亡的联合」，回答内容正确）；**1 条 judge 宽松**（`multi-64fa9b2f-31`：judge 理由自述"罚款金额有所不同"，数字不符仍判对）。
- Judge 冒烟（`single-64fa9b32-01` 潮宏基）：semantic_correct=1, relevancy=5，与人工判断一致。
- E2-on：因真失败 0 条 → 记忆库为空 → 注入从未触发 → 与 E2-off 结果一致（符合预期，非故障）。

### 规则口径的运行间方差

同一 42 条测试集：keyword_acc = 0.825（E1）→ 0.775（E2-off）→ 0.825（E2-on）。
失败 case 集合随回答措辞漂移（±2 条）。**结论：关键词规则不能单独做回归门槛，必须配合语义 Judge。**

---

## 3b. E1：ReAct vs Plan-and-Execute（42 条，deepseek-v4-flash + GLM Judge + memory off）

PAE 实现：`src/agent/graph_pae.py`（planner → executor → replanner → answer，与 ReAct 同接口，`--mode` 切换）。

| 指标 | ReAct（`mem_off_v2`） | PAE（`pae_off`） | 差异 |
|---|---|---|---|
| Judge 语义准确率 | **1.000** | 0.925 | react +3 case |
| 有效准确率（规则∪Judge） | **1.000** | 0.975 | |
| 拒答准确率 | **1.000** | **0.500** | PAE 用不相关证据硬答 |
| 关键词命中率 | 0.775 | 0.775 | 持平 |
| 平均 LLM 调用数 | 2.07 | 2.00 | 持平 |
| 字符 token 总量 | 375,752 | **244,130** | **PAE −35%** |
| 耗时 | 10.6 min | 15.2 min | PAE 更慢（规划器串行开销） |
| Hit@5 / MRR / Recall / nDCG | 0.95 / 0.74 / 0.875 / 0.752 | 同左 | 检索器相同 |

**PAE 规划质量**：42/42 输出合法 JSON（0 次解析兜底）；步数分布 1 步 21 / 2 步 16 / 3 步 5——简单问题不过度规划；跨文档问题能拆出精确子查询（如莫斯科无人机袭击次数 + 乌克兰击落数量）。

**PAE 三个真实失败模式（judge 翻转 3:0）**：
1. **规划器语言漂移**：case 23 规划器生成英文检索词 `Forbach France German man arrested`（中文语料）→ 检索失效 → 误答"未找到"（judge=0）；
2. **证据过度承诺**：检索永远返回 top-5（可能不相关），PAE 作答阶段倾向用不相关证据拼回答——`refuse-01` 用 2023 年新闻里的"滑雪登山"硬答 2026 冬奥会问题，丢失了 ReAct"继续找/拒绝回答"的自由度，拒答准确率 1.0→0.5；
3. **固定计划覆盖不足**：宽泛归纳题（case 26 答 4 次应 5 次、case 31 漏具体银行）单次计划不如 ReAct 迭代补证。

**结论**：本基准以单跳为主，PAE 的收益是 **token −35%**，代价是拒答与语义质量；ReAct 的迭代自由度在"该不该答"的判断上更稳。两模式保留、按问题复杂度路由（简单题 PAE 省钱、疑难/拒答敏感题 ReAct）是后续方向。

**已实施的修复（待重跑验证）**：规划器强制中文检索词；作答阶段增加"资料与问题无关→拒答"逃生门。

---

## 4. 失败 Case 明细（E2-off 的 9 条规则失败）

| Case ID | 类型 | 关键词（自动提取） | Judge 改判 | 审计结论 |
|---|---|---|---|---|
| single-64fa9b32-01 | 单跳 | 其设计灵感来源于 | 1 (rel 5) | 误判：关键词为 7 字长短语，回答（MING DESIGN STUDIO + 祥云纹）正确 |
| single-64fa9b2d-05 | 单跳 | 遭袭击身亡的联合 | 1 (rel 5) | 误判：碎片关键词 |
| single-64fa9b2b-10 | 单跳 | 证监会已关注到上 | 1 (rel 5) | 误判：碎片关键词 |
| single-64fa9b32-15 | 单跳 | 在中国企业就职的 | 1 (rel 5) | 误判：碎片关键词 |
| single-64fa9b27-17 | 单跳 | 李丰当选为广东省 | 1 (rel 5) | 误判：回答"选举李丰为省侨联主席"断句不同 |
| multi-64fa9b2f-31 | 双跳 | 55, 30 | 1 (rel 4) | **judge 宽松**：理由自述"罚款金额有所不同"，数字不符应判 0 |
| multi-64fa9b2d-35 | 双跳 | 并且强调不会缺席 | 1 (rel 5) | 大体合理 |
| multi-64fa9b2d-27-36 | 双跳 | 干性皮肤应选择质 | 1 (rel 5) | 误判：碎片关键词 |
| multi-64fa9b2f-40 | 双跳 | 中国在当前全球背 | 1 (rel 5) | 误判：关键词取自题干尾部 |

**修正后真实水平估计**：有效准确率 ≈ 39/40（唯一定罪保留：case 31 数字不符），与 effective_acc=1.0 的差异来自 judge 宽松。

---

## 5. 工程问题记录（过程中修复）

| 问题 | 现象 | 修复 |
|---|---|---|
| ChromaDB 批量写入上限 | `Batch size of 20000 is greater than max batch size of 5461` | `get_max_batch_size()` 动态取值，add 按 4096 分批 |
| 编码维度不匹配 | 库内 bge 512 维 vs Chroma 默认查询编码器 MiniLM 384 维 | 自建 `embedder.py` 单例，add/query 全部显式编码 |
| 失败记忆中文相似度失真 | Chroma 默认英文 MiniLM 编码中文 | 失败记忆库同样走 bge 编码 |
| nDCG > 1 | 同文档多 child 重复计分 | 检索器按 parent 去重 |
| MPS 编码速度 | fp32 bs256 仅 111 条/s，全量需 55 min | fp16 + bs512 → 177 条/s；分块编码防内存膨胀（35 万×512 tolist ≈ 2-3GB） |
| 记忆去重缺失 | 同一失败重复入库（7×2=14 条） | 待修：根因哈希去重 |

---

## 6. 成本

| 项 | 值 |
|---|---|
| E1 两轮（DeepSeek） | ~95 万字符 token，≈1.5 元 |
| E2 两轮（DeepSeek 作答 + GLM 判分） | ~79 万字符 token，≈1.3 元 |
| 检索层评测 | 0 元（本地嵌入，无 LLM） |
| 累计评测成本 | **< 3 元** |

## 7. 复现命令

```bash
# 检索层基线（零 LLM 成本）
uv run python scripts/run_eval.py --tag base42

# 答案层：记忆消融 + Judge 仲裁
uv run python scripts/run_eval.py --tag mem_off_v2 --with-llm --memory off --judge
uv run python scripts/run_eval.py --tag mem_on_v2  --with-llm --memory on  --judge

# 扩容评测集
uv run python evals/convert_crud.py --repo /tmp/CRUD_RAG --n1 25 --n2 15
uv run python scripts/index_extras.py   # 补索引后需重跑检索基线
```

## 8. 结论与下一步

**已证明**：① 全量 8.9 万篇知识库上检索 Hit@5=0.95；② 系统语义准确率（跨源 Judge）1.0、拒答准确率 1.0；③ 失败记忆机制全链路生效（归因→入库→相似召回→注入改变回答）；④ 关键词规则评测有假阴性且运行间方差 ~5pp，必须与语义 Judge 组合使用。

**待证明**：记忆模块的**增益**。当前系统在此评测集上真失败≈0，记忆无原料可用。下一步：① 构造陷阱题/同知识点改写题子集制造真实失败；② 检索失败（hit@5=0）入库沉淀 query 改写教训；③ 教训升级为实体级并双注入点（query 改写 + 作答前）。
