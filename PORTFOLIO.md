# 电商用户智能画像与AI分析平台 — 面试 Portfolio

> **一句话**：运营用自然语言提问，AI Agent 自主调工具查数据库、分析数据、生成答案，三层纯规则核查确保不编造数据。

---

## 架构全景

```
┌─────────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Vue 3 商城 :5173     │    │ Gradio 管理台 :7860│    │ Swagger :8000    │
│ 顾客购物 + AI 导购    │    │ 运营分析 + 监控    │    │ 开发者调试        │
└────────┬────────────┘    └────────┬─────────┘    └────────┬────────┘
         │                          │                       │
         └──────────────────────────┼───────────────────────┘
                                    │
                          FastAPI :8000
                    ┌───────┼───────┐
                    │       │       │
              Agent 引擎  数据流水线  后台自主服务
             (LangGraph)  (RFM+KMeans) (Watcher+飞轮)

Agent 六节点图:
  START → preprocess → llm_decide → [tools → reflect → llm_decide]
                                        │
                                        └──→ respond → fact_check → END
```

---

## 技术栈

| 层 | 选型 | 为什么 |
|----|------|--------|
| Agent 编排 | LangGraph 1.2 | 显式条件边 + 状态透传，ReAct 的自由循环做不到 |
| LLM | qwen2.5:7b (Ollama) | 零成本本地运行，function calling 最稳定的中文开源模型 |
| Embedding | nomic-embed-text 768维 | Ollama 生态最成熟的嵌入模型 |
| 稀疏检索 | BM25 + jieba 分词 | IDF 自动加权，稀有信号词自动提升，常见词自动抑制 |
| 向量检索 | Milvus Lite (COSINE) | 持久化不丢数据，接口兼容生产级 Milvus，挂了自动回退内存 |
| 融合排序 | RRF (k=60) | 排名层面公平合并，不依赖原始分数量纲 |
| 精排 | LLM Cross-Encoder | 有 LLM 可用就物尽其用，挂了跳过 |
| 幻觉治理 | 三层纯规则 FactCheck | 零 LLM 调用，毫秒级，数值层/规则层/逻辑层 |
| API | FastAPI + Pydantic v2 | 自动 OpenAPI 文档，类型安全 |
| 数据存储 | 内存 dict + SQLite WAL + Pickle TTL + Milvus | 按热/温/冷分层，各取所长 |
| 数据源 | MySQL → Pickle缓存 → Mock随机 | 三级降级，没 MySQL 也能跑 |
| 电商前端 | Vue 3 + Vite + TypeScript + Pinia | 完整购物链路，证明不只是 AI Demo |
| 部署 | Docker Compose + GitHub Actions | docker compose up -d 一键启动 |
| 日志 | python-json-logger + RotatingFileHandler | 结构化 JSON，按模块分文件 |

---

## 核心创新（面试展开点）

### 1. 三层纯规则 FactCheck

**问题**：LLM 编造数据——实际 50 人说成 500 人，决策树阈值 recency≤45.5 说成 ≤10。

**方案**：回答必须经过三层检查，零 LLM 调用，纯 Python 规则：

| 层 | 检查什么 | 方法 |
|----|---------|------|
| Layer 1 数值 | 回答中的数字 vs Skill 返回的真实数据 | 正则提取 → 逐项对比 |
| Layer 2 规则 | LLM 提到的阈值 vs 决策树原文 | 解析决策树 → 字符串匹配 |
| Layer 3 逻辑 | 运营建议 vs 聚类结论 | 关键词 → 矛盾检测 |

双档位：`relaxed`（追加 ⚠️ 警告）/ `strict`（拦截重生成，最多 2 次）。

**面试说法**："Self-Reflection 不可靠——LLM 也会骗自己。我改成了纯规则引擎，事实核查 10ms 完成，不消耗额外 Token。"

### 2. BM25 + Milvus 混合检索

**问题**：只用语义向量会漏掉精确关键词匹配，只用关键词会把"用户"这种高频词当信号。

**四阶段管线**：

```
用户提问
  ├─ 阶段1: SQLite 取候选池（200条正样本）
  ├─ 阶段2: 双路并发（ThreadPoolExecutor）
  │    ├─ BM25 + jieba: IDF 自动抑制"用户"，提升"流失""召回"
  │    └─ Milvus 768维: 语义相近但用词不同的也能召回
  ├─ 阶段3: RRF 融合（k=60，排名空间公平合并）
  ├─ 阶段4: LLM Cross-Encoder 精排（1-5分打分）
  └─ 质量过滤: 惩罚 Watcher 模板文本 + 纯表格回答
```

**降级链**：Ollama 挂 → bigram 回退；Milvus 挂 → 内存 numpy 余弦；全部挂 → 纯 BM25。

**面试说法**："双路互补——BM25 抓精确术语，语义向量抓意图相近。RRF 在排名层面融合避免分数量纲不一致。整个管线每步都有降级。"

### 3. 工具物理隔离

**问题**：购物场景的 LLM 能看到分群分析工具，可能误调。

**方案**：按 prompt 场景过滤工具列表，确定性隔离：

```python
if prompt == PROMPT_SHOPPING:
    scoped_names = ["search_products", "get_categories"]     # 2个
else:
    scoped_names = ["get_user_segment_stats", ...]           # 5个
llm = build_llm_with_tools(scoped_names)  # LLM 物理上无法调用看不到的工具
```

**面试说法**："不是建议 LLM 别乱调，是它根本看不到不该调的工具。确定性的。"

### 4. Agent 六节点 StateGraph

节点顺序不是"自由对话"，而是**强制流水线**：

```
preprocess  → 意图分类 + 实体抽取（product_insight / user_segment）
llm_decide  → 选 prompt + 选工具范围 + LLM 决策调哪个工具
tools       → 执行 Skill，返回 SkillResult（含结构化 data 给 FactCheck 用）
reflect     → LLM 自检数据完整度，最多 2 轮自动补查
respond     → LLM 基于真实数据生成 Markdown 回答
fact_check  → 三层规则核查，strict 模式不通过就回到 llm_decide 重生成
```

**为什么是 LangGraph 不是 while 循环**：条件边显式定义在图上，状态自动传递，每个节点的耗时/输入/输出自动记录（Tracer → Gradio Trace 面板可视化）。

### 5. 数据飞轮正循环

```
用户 👍/👎 + Watcher 自动分析报告
        ↓
  scorer.py（11条纯规则评分，全部配置化）
        ↓
  正样本（score≥0.5）→ SQLite + Milvus 向量入库
  负样本（score<0.5）→ 只写 SQLite，不建索引
        ↓
  下次 Agent 回答 → 检索高质量样本 → 注入系统提示作为"范文"
        ↓
  回答质量更高 → 用户更愿意点赞 → 更多高质量样本
```

### 6. 三级数据降级

```
MySQL :3306 → 连不上？
  Pickle TTL 缓存 → 过期了？
    np.random 生成 Mock 数据（150用户 × 50商品 × 随机订单）
```

保证任何人 clone 后不配置数据库也能跑全功能。

---

## 项目演进（7 文件 → 65 文件，12 个 Phase）

| Phase | 做了什么 | 解决了什么 |
|-------|---------|-----------|
| 0 | 8 包分层 + 配置中心 + 日志 + 异常体系 + LLM 封装 | 硬编码、无日志、裸 except |
| 1 | LangGraph StateGraph + Skill 框架 + Preprocess + Reflect | ReAct 自由循环不可控 |
| 2 | 三层纯规则 FactCheck | LLM 编造数据 |
| 3 | 自动选 K + 增量 RFM + 决策树剪枝 + 快照 | 固定 K=4 不科学 |
| 4 | Watcher 引擎 + 6 条检测规则 + 事件降噪/冷却 | 只能被动触发 |
| 5 | 滑动窗口限流 + 异常标准化 + 线程池隔离 | 无保护措施 |
| 6 | Gradio 解耦 → 全部通过 httpx 调 API | Gradio 直连业务代码 |
| 7 | 飞轮：反馈采集 → 评分 → 向量入库 → 检索反哺 | 每次从零开始 |
| 8 | 25 QA + 10 Event 评测 + LLM-Judge | 无量化的手段 |
| 9 | 12 bug + 21 API 冒烟 + 29 模块零循环依赖 | 累积边界 bug |
| 10 | Vue 商城 + 三 Agent 编排 + CDC | 无真实电商流程 |
| 11 | 真 Embedding + 混合检索 + Agent Trace + 全中文化 | 功能可用但不够"作品级" |
| 12 | 8 严重 bug + 5 性能优化 + 66 单测 + 流式 SSE + 可观测性 | 收官质量打磨 |

---

## 项目结构速览

```
agent/       10 files  Agent核心（图 + 核查 + 编排 + Trace + Memory）
api/          8 files  FastAPI（路由 + 中间件 + 电商 + 数据存储）
pipeline/     4 files  数据流水线（加载 → 清洗 → RFM → 聚类 → 画像）
skills/       3 files  可插拔 Skill（7个工具，2购物 + 5分析）
watcher/      6 files  事件检测 + CDC + 任务管理
flywheel/     7 files  飞轮（采集 → 评分 → BM25+Milvus检索 → 调度）
eval/         6 files  评测（LLM-Judge + Hit/MRR + 检索评测）
llm/          2 files  LLM客户端（连接池 + 重试 + Token统计）
config/       2 files  Pydantic-settings（50+配置项）
log/          2 files  JSON结构化日志
tests/        5 files  66单元测试（fact_checker 24 + data_store 25 + events 17）
frontend/     -        Vue 3 电商商城（6页面 + Pinia状态管理）
```

---

## 面试常见追问速查

### Q: 为什么用 LangGraph 而不是直接循环？
> "六节点、三条条件边、两条回环、一条重试路径——手写状态机很快就会不可维护。LangGraph 把图的结构和节点的逻辑分开了，改流程不影响业务代码。而且每个节点的输入/输出/耗时自动记录，Gradio 面板直接可视化。"

### Q: RAG 召回策略是什么？
> "双路并发——BM25 + jieba 做稀疏检索，Milvus 768 维做语义检索。BM25 的 IDF 自动抑制高频词、提升稀有信号词。两路通过 RRF 在排名层面融合，最后 LLM 做 cross-encoder 精排。每步都有自动降级。"

### Q: BM25 和 bigram 有什么区别？
> "bigram 把所有字符片段等权处理，'用户'和'流失'各一票。BM25 的 IDF 让'用户'这种每篇都有的词权重趋零，'召回'这种稀有信号词权重放大。同样的查询'流失用户怎么召回'，BM25 能把真正相关的文档排到第一，bigram 排不出来。"

### Q: 怎么防止 LLM 幻觉？
> "做了好几轮迭代。一开始 Self-Reflection 让 LLM 自己检查，发现它会骗自己。改成了三层纯规则核查——数值层正则对比数据库、规则层校验决策树阈值、逻辑层检查聚类一致性。零 LLM 调用，毫秒级完成。relaxed 模式追加警告，strict 模式拦截
### Q: 怎么保证生产稳定？
> "滑动窗口限流 60s/20次 + 相似问句 MD5 去重。LLM 全局并发 Semaphore(3)。异常统一转 JSON 返回。SQLite 全部 WAL 模式。独立线程池处理后台任务不抢占用户请求。API 请求全量 JSON 结构化日志。"

### Q: 测试怎么设计的？
> "挑了三个最核心的纯函数模块——fact_checker（24用例）测正则匹配和边界、data_store（25用例）测 CRUD 和 10 线程并发安全、events（17用例）测快照对比和事件检测。纯逻辑不需要启动 LLM，秒级跑完。CI 在每次 push 自动触发。"

### Q: 怎么部署？
> "docker compose up -d 一键启动全部 4 个服务。Ollama 拆了独立的 GPU profile——纯 CPU 环境用 OpenAI 兼容 API 也能跑。前端用 Nginx 做反向代理，/api/* 自动转发。GitHub Actions 在每次 push 自动跑测试和导入校验。"

### Q: 最大的技术难点是什么？
> "LLM 幻觉问题经过了好几轮迭代——Self-Reflection → 发现不靠谱 → 纯规则核查 → strict 模式死循环 → 加最大重试次数。还有 Ollama 模型不支持 function calling 的问题，切到 qwen2.5:7b 又加了 text→tool_call 桥接。Milvus-lite 在 Windows 上有文件锁问题，从 upsert 改成 delete+insert，flush 改成批量。这些都是在真实环境中踩出来的。"

### Q: 有什么不足？
> "当前测试覆盖了最核心的 3 个模块，但 agent 图、检索管线、Skill 执行等模块还没有单元测试——这些依赖 LLM，需要 mock。API 没有认证中间件，本地项目不需要但生产环境必须加。前端 AI 对话还没做成流式——流式端点 `/ask/stream` 后端已经写好了，前端还没接。"

---

## 面试自我介绍（STAR 法则，2-3 分钟）

**Situation**：电商运营每天要做用户分析报告，手动查数据写总结耗时且容易出错。

**Task**：做一个人工智能问答平台——运营用自然语言提问，系统自动查数据、分析、生成可核查的回答。

**Action**：
1. 用 LangGraph 搭了六节点 Agent 图——意图识别 → LLM 决策 → 工具执行 → 数据自检 → 回答生成 → 事实核查，三条条件边两条回环
2. 做了三层纯规则 FactCheck 解决幻觉——数值层正则对比数据库、规则层校验决策树、逻辑层检查一致性。零 LLM 调用毫秒级完成
3. 混合检索双路召回——BM25 + jieba 关键词 + Milvus 768 维语义向量，RRF 融合，LLM 精排。评测 Hit@1=70%、MRR=0.75
4. 后台 Watcher 每 5 分钟比快照 → 6 条规则检测异常 → 自动触发三 Agent（Monitor→Analysis→Strategy）流水线分析
5. 数据飞轮——用户反馈 + 自动评分 → 高质量样本入库 → 向量索引 → 反哺 Agent 系统提示
6. Docker Compose 一键部署 + GitHub Actions CI 自动跑 66 个测试

**Result**：
- 三层核查下事实准确率接近 100%
- 混合检索 Hit@1=70%、MRR=0.75
- 7 个可插拔 Skill，购物/分析双场景工具物理隔离
- 流式 SSE + Agent Trace 可观测性面板
- 完整电商链路（商品浏览→购物车→下单→支付）+ AI 导购

---

## 复习清单（面试前 1 小时）

- [ ] 六节点图拓扑能画出来
- [ ] 三层 FactCheck 每层怎么做的说得清
- [ ] BM25 + Milvus + RRF + LLM 重排的四阶段检索能讲
- [ ] 飞轮正循环的闭环逻辑
- [ ] Agent 工具物理隔离的做法
- [ ] 数据源三级降级策略
- [ ] 7 个 Skill 分别是什么、分属哪个场景
- [ ] 66 测试覆盖了哪 3 个模块
- [ ] Docker Compose 的 4 个服务
- [ ] 项目从 7 文件到 65 文件的演进故事
