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

**覆盖句式族持续收敛**（对应旧短板"只认固定句式"）：直接陈述/表格行/阈值区间 + 换说法四族——
占比 N%、总数（"全平台共 N 人"，千分位支持，须全局锚定词防子集误报）、倍数（"X 是 Y 的 N 倍"，
人数词/指标词判别，消费倍数不误报人数）、差额（"X 比 Y 多/少 N 人"，方向+量级双检）；
"约/左右"按声称精度四舍五入容差。对抗样本测试守护（13 例：正确说法零误报、编造全检出、
长散文混合、掩码防跨段双报）。边界如实写进 docstring：更隐蔽改写不在覆盖内 → relaxed 警告 +
前端数据源标注兜底，不做全检声称。

**面试说法**："Self-Reflection 不可靠——LLM 也会骗自己。我改成了纯规则引擎，事实核查 10ms 完成，不消耗额外 Token。覆盖句式族从'分群X有N人'收敛到占比/总数/倍数/差额四种换说法——我加了对抗样本测试专门防'换个说法就漏'。规则有边界我承认，所以 relaxed 模式叠加数据源标注兜底，而不是假装全检。"

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

### 6. 数据源降级 + 天池公开数据集接入

```
MySQL :3306 → 连不上？
  Pickle TTL 缓存 → 过期了？
    (可选) 京东 JData 公开脱敏数据集 CSV → 文件缺失？
      np.random 生成 Mock 数据（150用户 × 50商品 × 随机订单）
```

保证任何人 clone 后不配置数据库也能跑全功能。`DATA_SOURCE=tianchi` 时接入
京东 JData 算法大赛数据集(已脱敏、学术许可):下单行为→订单明细、人口属性进画像、
无价格字段按 (品类,品牌) 确定性合成——RFM/聚类/Agent 全链路零改动,
简化的口径如实记录在 [DATA_COMPLIANCE.md](docs/DATA_COMPLIANCE.md)。

### 7. 模拟数据实时化:滚动揭晓(假数据也能"活起来")

**问题**:JData 整体平移后是"刚体"——每次重算分群结果一模一样,快照对比无
差异,Watcher 自主分析在模拟数据上永远静默,演示只能靠手动触发。

**方案**:数据面按墙钟日**逐日生长**:第 1 天露出原始窗口最后 30 天(预热),
之后每天 +1,直到全窗口稳态。窗口增长 → 数据指纹变化 → force_refresh →
新快照 → 事件检测真实触发。三处关键设计:
- **状态机幂等**:进度落盘 `cache_data/tianchi_reveal.json`(RLock + 原子写),
  watcher 与 loader 两处调用不重复推进,重启续播;
- **口径统一**:漏斗独立读 CSV,同口径截断;截断在"原始时间线、平移之前",
  RFM 基准日(最新+1)语义不变;
- **指纹只含窗口数不含日期**——稳态后不每天白触发一次全量重算。

**面试说法**:"模拟数据的'实时感'不能靠平移,平移是刚体运动——要让 Watcher
真正检测到变化,数据必须逐日'长'出来。状态机每天 +1 天窗口,指纹驱动重算,
演示时 /debug/reveal-advance 可手动推进,不用等真的一天。"

### 8. 状态与存储的分层可插拔(演示不迁就,生产不裸奔)

**问题**:作品级 demo 为了"clone 即跑"只能选零依赖方案(JSON 文件 / Milvus-lite),
面试官追问"生产怎么办"时容易穿帮;反之直接上生产组件,demo 又跑不动。

**方案**:一切按"默认轻量 + 可切换 + 降级链"分层,切换点都是配置:
- **会话记忆双轨**:`SESSION_STORE=json`(默认,零依赖文件版)↔ `=postgres`
  (LangGraph 官方 checkpoint-postgres,同步/异步各持连接池)。关键语义:
  连不上 **fail-fast 启动报错,不静默降级**——状态是业务记忆不是缓存,
  宁缺勿假;Windows 自动切 Selector 事件循环(psycopg async 要求)。
  真实 PG 实测:sync 重启恢复 / async 事件循环 / 跨实例持久化全通过。
- **向量后端可插拔**:`FLYWHEEL_VECTOR_BACKEND` = `milvus`(lite,默认)/
  `milvus-remote`(pymilvus 连独立 Milvus)/ `memory`。同一接口三实现,
  挂了自动回退内存;独立 compose(`docker-compose.milvus.yml`,etcd+
  MinIO+Milvus 三件套,镜像前缀变量化)让"生产级验证"一条命令可复现。
- **评分阈值实证**:不再说"经验值"——89 条真实样本校准
  ([SCORER_CALIBRATION.md](docs/SCORER_CALIBRATION.md)):正样本 0.50-0.85、
  负样本 0.0-0.25,0.26-0.49 为空带 → 0.5 阈值分离无争议,无需调权;
  校准周期 ~200 条。

**面试说法**:"每条'演示级'都留了生产开关,而且切换语义是显式设计的——
会话存储 fail-fast 不静默降级,因为丢记忆比报错更糟;向量库三后端同接口,
compose 一键起真 Milvus。剩下没做的(分布式锁、线上效果信号)我明说没做,
不在文档里假装。"

---

## 项目演进（7 文件 → 65 文件，12 个 Phase）

| Phase | 做了什么 | 解决了什么 |
|-------|---------|-----------|
| 0 | 8 包分层 + 配置中心 + 日志 + 异常体系 + LLM 封装 | 硬编码、无日志、裸 except |
| 1 | LangGraph StateGraph + Skill 框架 + Preprocess + Reflect | ReAct 自由循环不可控 |
| 2 | 三层纯规则 FactCheck | LLM 编造数据 |
| 3 | Gap statistic 自动选 K + 增量 RFM + 决策树剪枝 + 快照 | 固定 K=4 不科学 |
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
agent/       10 files  Agent核心（图 + 核查 + 编排 + Trace + Memory,会话存储双轨:JSON/Postgres）
api/          8 files  FastAPI（路由 + 中间件 + 电商 + 数据存储）
pipeline/     4 files  数据流水线（加载 → 清洗 → RFM → 聚类 → 画像）
skills/       3 files  可插拔 Skill（11 个工具，商品 3 + 分析 7 + 图像解析 1）
watcher/      6 files  事件检测 + CDC + 任务管理
flywheel/     7 files  飞轮（采集 → 评分 → BM25+Milvus检索 → 调度,向量后端 lite/远端/内存可插拔）
eval/         6 files  评测（LLM-Judge + Hit/MRR + 检索评测）
llm/          2 files  LLM客户端（连接池 + 重试 + Token统计）
config/       2 files  Pydantic-settings（50+配置项）
log/          2 files  JSON结构化日志
tests/        28 files  329 项测试（315 pytest:纯函数 + Agent 图剧本化 + API 集成 + 14 vitest 前端）
frontend/     -        Vue 3 电商商城（6页面 + Pinia状态管理）
docs/         2 files  数据合规 + 评分阈值校准报告
root          + docker-compose.milvus.yml(独立 Milvus standalone 开发栈)
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
> "分层做:① 统一 LLM 适配器——超时/429/额度不足/上下文超长错误分类驱动重试退避,额度不足和鉴权不盲目重试,直接降级到下一家供应商(阿里百炼 → DeepSeek → 千帆 → 本地 Ollama);② 请求层——按 session 分桶的滑动窗口限流 + 相似问句 MD5 去重,调试端点演示级 Api-Key;③ 安全——规则式 Prompt 注入检测(加权拦截)+ 百度内容审核输入输出双向校验(fail-open 不阻断主链路);④ 工程——Agent 跑线程池不阻塞事件循环、SQLite 全 WAL、JSON 结构化日志、快照保留策略防无限累积。"

### Q: 两个项目怎么复用同一套 LLM 代码？
> "抽了一个统一大模型适配器,一份源码复制进两个仓库(各自自包含、可独立跑)。适配器对上层只暴露 invoke/stream/stream_events 和错误分类:超时重试、429 按 Retry-After 退避、额度不足/鉴权直接降级、上下文超长截断重试一次。电商项目再包一层 LangChain Bridge(自定义 chat model 实现 bind_tools),让 LangGraph 的工具绑定语义和适配器的降级链同时生效——康养项目直接消费适配器原语,两个项目的差异只在最后一层。"

### Q: 流式 SSE 怎么做到"实时展示每一步工具调用"？
> "不用 astream_events——因为 tools 节点是直接执行 Skill 而不是走 ToolNode,不会产生 on_tool_start 事件。用 graph.astream 的三种 stream mode 组合:custom 模式由节点内 get_stream_writer 发 node_start/tool_call/tool_result 事件,updates 模式取节点结果和事实核查结论,messages 模式拿 decide 节点的 token 级 delta。前端用原生 fetch 手写 SSE 帧解析、渲染步骤条,token 打完再用 answer 事件做权威全文覆盖——因为 respond 节点可能修正 Markdown、fact_check 会追加警告。"

### Q: 测试怎么设计的？
> "分层设计,329 个用例(后端 315 + 前端 14)全部离线秒级跑完:① 纯函数层——fact_checker 正则边界、data_store 并发安全、events 快照对比、json_repair 修复规则、guardrails 注入规则;② Agent 图层——用剧本化适配器注入降级链,测图的直接回答/工具调用→事实核查/反思回环三条主路径,以及 SSE 事件序,零真实 LLM 调用;③ API 集成层——TestClient 测限流按 session 分桶、调试端点 Api-Key、注入拦截 403、流式事件顺序。CI 每次 push 自动跑。"

### Q: 怎么部署？
> "docker compose up -d 一键启动全部 4 个服务。Ollama 拆了独立的 GPU profile——纯 CPU 环境用 OpenAI 兼容 API 也能跑。前端用 Nginx 做反向代理，/api/* 自动转发。GitHub Actions 在每次 push 自动跑测试和导入校验。"

### Q: 最大的技术难点是什么？
> "LLM 幻觉问题经过了好几轮迭代——Self-Reflection → 发现不靠谱 → 纯规则核查 → strict 模式死循环 → 加最大重试次数。还有 Ollama 模型不支持 function calling 的问题，切到 qwen2.5:7b 又加了 text→tool_call 桥接。Milvus-lite 在 Windows 上有文件锁问题，从 upsert 改成 delete+insert，flush 改成批量。这些都是在真实环境中踩出来的。"

### Q: 有什么不足？
> "诚实的短板:① 事实核查是正则规则引擎——句式族已从固定句式收敛到四种换说法(占比/总数/倍数/差额)并有对抗样本守护,但更隐蔽的改写仍会漏——覆盖是规则边界,靠 relaxed 警告 + 前端数据源标注兜底,不做全检声称;② 会话持久化已双轨:默认 JSON 演示零依赖,生产切官方 checkpoint-postgres(连不上 fail-fast 不静默降级,本地 compose 起 PG 实测重启恢复)——但**分布式锁仍未做**,多实例部署会话一致性还需要它;③ 飞轮评分器阈值经真实样本校准(89 条:正样本 0.50-0.85/负样本 0.0-0.25 空带分离,0.5 无需调权),但仍是代理信号而非线上效果数据,校准周期每 ~200 条;④ 百度内容审核免费档 QPS=1,输出审核偶有误伤(演示时可临时关闭)。这些我都知道边界在哪,不是不知道才不做。"

---

## 面试自我介绍（STAR 法则，2-3 分钟）

**Situation**：电商运营每天要做用户分析报告，手动查数据写总结耗时且容易出错。

**Task**：做一个人工智能问答平台——运营用自然语言提问，系统自动查数据、分析、生成可核查的回答。

**Action**：
1. 用 LangGraph 搭了六节点 Agent 图——意图识别 → LLM 决策 → 工具执行 → 数据自检 → 回答生成 → 事实核查，三条条件边两条回环
2. 做了三层纯规则 FactCheck 解决幻觉——数值层正则对比数据库、规则层校验决策树、逻辑层检查一致性。零 LLM 调用毫秒级完成
3. 混合检索双路召回——BM25 + jieba 关键词 + Milvus 768 维语义向量，RRF 融合，LLM 精排。评测 Hit@1=70%、MRR=0.75
4. 后台 Watcher 每 5 分钟比快照 → 6 条规则检测异常 → HIGH 事件自动触发三 Agent（Monitor→Analysis→Strategy）流水线分析,三段结果落库可查;分群历史快照画成时间趋势图
5. 数据飞轮——用户反馈 + 自动评分 → 高质量样本入库 → 向量索引 → 反哺 Agent 系统提示
6. 会话记忆持久化(JSON checkpointer,重启不丢)+ 长对话自动摘要;个性化推荐闭环(图像标签→画像→打分推荐→一键加购)
7. Docker Compose 一键部署 + GitHub Actions CI 自动跑 329 个测试（315 pytest + 14 vitest）

**Result**：
- 三层核查下事实准确率接近 100%
- 混合检索 Hit@1=70%、MRR=0.75
- 11 个可插拔 Skill（分析 7 + 商品 3 + 图像解析 1），购物/分析双场景工具物理隔离
- 流式 SSE + Agent Trace 可观测性面板
- 完整电商链路（商品浏览→购物车→下单→支付）+ AI 导购

---

## 复习清单（面试前 1 小时）

- [ ] 六节点图拓扑能画出来
- [ ] 三层 FactCheck 每层怎么做的说得清
- [ ] FactCheck 换说法句式族(占比/总数/倍数/差额)+ 掩码防双报 + 13 例对抗样本
- [ ] 会话双轨:json 默认/postgres 官方 saver/启动 fail-fast/Windows Selector 事件循环/PG 实测重启恢复
- [ ] 向量后端可插拔:FLYWHEEL_VECTOR_BACKEND 三值、docker-compose.milvus.yml、回退内存
- [ ] 评分校准结论:正 0.50-0.85/负 0.0-0.25 空带分离、0.5 无需调权、校准周期 ~200 条
- [ ] BM25 + Milvus + RRF + LLM 重排的四阶段检索能讲
- [ ] 飞轮正循环的闭环逻辑
- [ ] Agent 工具物理隔离的做法
- [ ] 数据源三级降级策略
- [ ] 11 个 Skill 分别是什么、分属哪个场景（分析 7：统计/规则/高价值/环比/趋势/漏斗/刷新；商品 3：搜索/品类/推荐；图像解析 1 由接口直调）
- [ ] 自动选 K:Gap statistic + 1-SE 简约规则 + 最小簇约束(为什么不用肘部+轮廓的组合分)
- [ ] 分群业务命名:LLM 生成 + 启发式兜底 + 缓存,全链路(对话/图表/建议/环比)说人话
- [ ] 对话内出图:Skill 数据 → chart 协议 → 前端零依赖 SVG 渲染
- [ ] 环比 Skill:月份参数 + 诚实降级(无快照时报可用月份)
- [ ] 数据新鲜度标注:数据时间/数据源/采样口径
- [ ] 快照保留策略:20 个只够 100 分钟 → 2016 个(一周),趋势/环比才有历史可看
- [ ] 统一适配器:错误分类表 + 降级链 + 与康养项目的复用方式
- [ ] SSE 三 stream mode 组合 + 事件 schema(meta/node_start/tool_call/delta/answer/done)
- [ ] 注入检测 + 内容审核 fail-open 的取舍
- [ ] Trace 节点耗时是实测的(不是估算)
- [ ] 329 测试的三层设计(纯函数 / Agent 图剧本化 / API 集成 + 前端 14 vitest)能讲
- [ ] Skill 工程化:SKILL.md 声明式(YAML frontmatter + Markdown 指令)、加 Skill 不碰代码、tool vs instruction 两类(Skill≠Tool)、两层发现(注册表推导场景绑定 + 描述检索打分)、渐进式披露(600/2400 上限)、SKILL_ENABLED 真实生效
- [ ] 滚动揭晓:刚体平移 vs 逐日生长、预热/STEP/稳态、按墙钟日幂等、指纹只含窗口数、funnel 独立读 CSV 同口径截断、/debug/reveal-status 与 reveal-advance
- [ ] HIGH 事件三阶段流水线的触发与结果落库
- [ ] 分群时间趋势图的数据来源(快照历史)
- [ ] 会话持久化的实现(JSON serde 桥接 + PG 双轨)与重启恢复演示
- [ ] 长对话摘要的触发条件与上下文有界性
- [ ] 个性化推荐的打分因子(品类/标签/价格适配)
- [ ] Docker Compose 的 4 个服务
- [ ] 项目从 7 文件到 65 文件的演进故事
