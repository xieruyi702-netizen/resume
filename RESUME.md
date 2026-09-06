# 【姓名】

> 求职意向：后端开发 / 推荐策略工程 / AI 应用工程师  
> 手机：【电话】｜邮箱：【邮箱】｜GitHub：【链接】｜城市：【城市】

---

## 教育背景

**【学校】｜【专业】｜本科/硕士**　　【入学年】—【毕业年】

---

## 实习经历

**2026.05 — 至今　　　　　　　　上海任意门科技有限公司（Soul）　　　　　　　　服务端开发实习生**

**实习背景**：负责语音匹配服务（商业化 reco 链路）后端迭代与治理：并发正确性、在线性能、Kafka 消费改造，并基于 **Spring AI** 开发实验策略取数 Agent（双知识库 + RAG + Hive 试跑）。

**核心工作**：

- **超级匹配延迟弹窗**：付费玩法——购卡后延迟推送高质量候选。Redis ZSet（score=触发时刻）做延迟队列，入队顺带清过期脏任务；**Lua 原子批量弹出**（ZRANGEBYSCORE + 按 member ZREM）防多实例重复消费，任务入独立线程池并发推荐。候选互斥用 **SET NX EX** 原子加锁，释放前 Lua 校验锁归属，延迟期续期 TTL。发送/曝光双 key 频控，pipeline 前置读做主叫拦截与候选过滤；
- **UCB 延迟匹配工程化**：将「延迟多久再匹配」后端落地——按付费等级 **HIGH/MID/LOW** 分段独立统计（VIP/SVIP→HIGH、NP→MID、其余→LOW）；臂数据存**独立 Redis Hash**。支付/挂断回调异步结算，**Lua** 单次往返完成查 pending → 累加臂统计 → 清 pending，防回调与过期清理双结算；pending **TTL + 定时批量清理**保统计窗口无脏数据。冷启动次数不足优先试满，再 UCB 打分；链路失败降级固定延迟并分原因打点；
- **优化链路逻辑**：Kafka 消费侧——debug 消息含上千行候选明细，同 poll 批内合并后分块批量 INSERT，Hive 回写迁出消费线程异步化；子批失败整批不 ack、DuplicateKey 幂等成功，止血改 pause/resume（offset 不动），避免静默丢数；DAG 编排侧——算子 metadata 声明 import/export 属性依赖，启动期贪心分层生成串行/并行执行组并按 requestType 缓存执行计划：组间串行保数据依赖、组内独立线程池并行，无依赖算子耗时由串行加和降为最慢节点，组超时取组内最大节点超时做有界等待，超时/异常按算子维度分类打点，timeout/batchSize/skip 支持 Apollo 热更新；列表查询 N+1 改批量 IN；参与链路可观测建设——算子分段耗时打点 + 白名单请求全链路快照经独立线程池异步上报 Kafka，主流程零阻塞；
- **实验策略 Agent**：基于 **Spring AI** 在 SQL 跑批平台上构建取数 Agent。多线程跑批 + 异步飞书，端到端约 **−89%**；定时任务 cron 存 MySQL，**ThreadPoolTaskScheduler + CronTrigger** 动态注册/重载，改周期不重启。知识层职责分离——业务表摘要向量化按需检索，指标口径常驻提示词防漂移；生成 SQL 直查 Hive 试跑；评测准确率约 **90%**、越界拒答约 **95%**。

---

## 项目经历

### 美食汇｜高并发美食优惠券 + 食谱 RAG Agent

**项目简介**：Spring Boot 美食优惠券秒杀（Redis 一主两从 + Kafka + MySQL，Docker Compose + Nginx 双实例）与 HowToCook 食谱 Agentic RAG；JMeter / 组件压测 + 检索与 LLM-as-Judge 评测可复现。

**工作**：

- **券详情多级缓存**：布隆本地快照拦非法 id；**L1 Caffeine** → **L2 Redis 从库**（轮询读、失败回主）→ MySQL；互斥重建 + 空值缓存防击穿，TTL 抖动防雪崩；更新删 L2 + pub/sub 踢各实例 L1；
- **秒杀与限流**：Nginx 轮询双实例；**令牌桶 / 漏桶**可切换、惰性补充、fail-fast（令牌桶偏突发、漏桶更平滑，同参压测验证实现）；热路径 **Redis Lua** 一次完成库存判定、一人一单与扣减；
- **订单闭环**：抢券同事务落待支付订单，返回即可支付；状态机 待支付/已支付/超时取消/用户取消；**支付幂等**（已付再调仍成功；CAS 防双扣）；ZSet 延迟取消 + CAS/Lua 回补；支付成功后 **Kafka 异步加积分**（订单 `points_status` 兼投递标记，消费幂等/死信）；用户余额 **Redisson**（读 Cache Aside，写改 MySQL 后删缓存）+ 锁内扣充，支付入账商铺；雪花订单号，时钟回拨拒发号；
- **食谱 Agentic RAG**：HowToCook 切片建库；关键词 + 向量双路召回；**LangChain / LangGraph ReAct Agent**（**DeepSeek V4 Flash**）；记忆**按用户隔离**（短期 **Redis** TTL+条数淘汰，长期向量召回 + 每用户上限淘汰，上下文按预算比例组装）；库外菜与非美食问题 OOD 拒答；
- **指标**：JMeter 64 线程真实漏斗约 **3.1k req/s**、P99 **112ms**（Err 0%，余额对账 OK）；Agent Hit@1≈0.90 / Hit@5≈0.96 / MRR≈0.91，Judge≈0.82，拒答≈0.90。

---

## 个人技能

- 荣誉证书：大学英语六级；
- 语言基础：熟悉 Java 并发编程（线程池、JUC、AQS/锁机制）、JVM、常用集合底层；
- 中间件：熟悉 Redis（主从、Lua、缓存三防）、Kafka（分区/消费组/死信）、MySQL 索引与事务；
- 系统设计：限流（令牌桶/漏桶）、多级缓存、MQ 削峰、支付后积分最终一致；
- LLM / RAG：文档切片、BM25 + 向量混合检索、分层记忆、Agent 拒答与 Hit@k/MRR 评测；
- 工程工具：Docker Compose、Git、JMeter。

