# 【姓名】

> 求职意向：后端开发 / 推荐策略工程 / AI 应用工程师  
> 手机：【电话】｜邮箱：【邮箱】｜GitHub：【链接】｜城市：【城市】

---

## 教育背景

**【学校】｜【专业】｜本科/硕士**　　【入学年】—【毕业年】

---

## 实习经历

**2026.05 — 至今　　　　　　　　上海任意门科技有限公司（Soul）　　　　　　　　服务端开发实习生**

**实习背景**：语音匹配商业化 reco 链路服务端开发——延迟弹窗 / UCB 延迟匹配工程化、可视化平台 Kafka 消费与 DAG 编排治理，并落地数据提效取数 Agent（双知识库 + RAG + Hive 试跑）。

**核心工作**：

- **超级匹配延迟弹窗**：新增链路 ——购卡后延迟推送高质量候选。Redis ZSet（score=触发时刻）做延迟队列，入队顺带清过期脏任务；**Lua 原子批量弹出**（ZRANGEBYSCORE + 按 member ZREM）防多实例重复消费，任务入独立线程池并发推荐。候选互斥用 **SET NX EX** 原子加锁，释放前 Lua 校验锁归属，延迟期续期 TTL。发送/曝光双 key 频控，pipeline 前置读做主叫拦截与候选过滤；
- **UCB 延迟匹配工程化**：采用UCB方法动态决定延迟时间 ——按付费等级 **HIGH/MID/LOW** 分段独立统计（VIP/SVIP→HIGH、NP→MID、其余→LOW）；臂数据存**独立 Redis Hash**。支付/挂断回调异步结算，**Lua** 单次往返完成查 pending → 累加臂统计 → 清 pending，防回调与过期清理双结算；pending **TTL + 定时批量清理**保统计窗口无脏数据。冷启动次数不足优先试满，再 UCB 打分；链路失败降级固定延迟并分原因打点；
- **优化链路逻辑**：
  - 可视化平台——debug 消息含上千行候选明细，同 poll 批内合并后分块批量 INSERT，Hive 回写迁出消费线程异步化；子批失败整批不 ack、DuplicateKey 幂等成功；失败时 pause 分区、不提交 offset，恢复后再 resume，避免失败仍推进导致丢数；DAG 编排——算子 metadata 声明 import/export 依赖，启动期贪心分层生成串行/并行执行组并按 requestType 缓存计划：组间串行、组内线程池并行，无依赖耗时由加和降为最慢节点，组超时取组内最大节点超时，超时/异常按算子打点；列表查询 N+1 改批量 IN；可观测——算子分段耗时 + 白名单全链路快照异步上报 Kafka，主流程零阻塞；
- **数据提效 Agent**：在 SQL 跑批平台上落地取数 Agent。**多线程 + 异步** 执行sql和生成文档，端到端约 **−89%**；cron 存 MySQL，**ThreadPoolTaskScheduler + CronTrigger** 动态注册/重载，改周期不重启。知识层拆分——业务表摘要向量化按需检索，指标口径常驻提示词防漂移；生成 SQL 直查 Hive 试跑；评测准确率约 **90%**、越界拒答约 **95%**。

---

## 项目经历

### 美食汇｜高并发美食优惠券 + 食谱 RAG Agent

**项目简介**：基于 Spring Boot 的高并发美食优惠券秒杀（Redis 一主两从 + Kafka + MySQL，Docker Compose + Nginx 双实例），并集成 HowToCook 食谱 Agentic RAG；秒杀侧 JMeter 压测可复现，RAG 侧检索与 LLM-as-Judge 评测可复现。

**工作**：

- **券详情多级缓存**：**布隆过滤器**本地快照拦非法 id；**L1 Caffeine** → **L2 Redis 从库**（轮询读、失败回主）→ MySQL；互斥重建 + 空值缓存防击穿，TTL 抖动防雪崩；更新时删 L2 + pub/sub 踢各实例 L1；
- **秒杀与限流**：Nginx 轮询双实例；**令牌桶 / 漏桶**可切换（分片、fail-fast）；JMeter 测试峰值令牌桶约 **6.0k**、漏桶约 **5.3k** req/s；热路径 **Redis Lua** 一次完成库存判定、一人一单与扣减；
- **订单闭环**：抢券同事务落待支付订单，返回即可支付；状态机 待支付/已支付/超时取消/用户取消；**支付幂等**（已付再调仍成功；CAS 防双扣）；ZSet 延迟取消 + CAS/Lua 回补；支付成功后 **Kafka 异步加积分**（订单 `points_status` 兼投递标记，消费幂等/死信）；用户余额 **Redisson**（读 Cache Aside，写改 MySQL 后删缓存）+ 锁内扣充，支付入账商铺；雪花订单号，时钟回拨拒发号；
- **食谱 Agentic RAG**：HowToCook 切片建库；关键词 + 向量双路召回；**LangChain / LangGraph ReAct Agent**（**DeepSeek V4 Flash**）；记忆**按用户隔离**（短期 **Redis** TTL+条数淘汰，长期向量召回 + 每用户上限淘汰，上下文按预算比例组装）；库外菜与非美食问题 OOD 拒答；
- **指标**：JMeter 64 线程真实漏斗约 **3.1k req/s**、P99 **54ms**（Err 0%）；Agent Hit@1≈0.90 / Hit@5≈0.96 / MRR≈0.91，Judge≈0.82，拒答≈0.90。

---

## 个人技能

- 荣誉证书：大学英语六级；
- 语言基础：熟悉 Java 并发编程（线程池、JUC、AQS/锁机制）、JVM；常见集合与并发容器（ArrayList和ConcurrentHashMap等）底层原理；
- 中间件：熟悉 Redis（主从、持久化、缓存三防、删除淘汰等）、Kafka（分区/消费组/）、MySQL 索引与事务；
- 大模型应用开发：RAG、ReAct Agent、LLM-as-Judge、Plan-and-Execute 等；

