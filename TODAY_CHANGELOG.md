# 电商用户智能画像与AI分析平台 — 优化日志

> 日期：2026-07-23 ~ 2026-07-26  
> 起点：7 个 Python 文件的扁平 MVP  
> 终点：AI-Native 智能电商平台（流式Agent + 持久化向量检索 + 可观测性 + 66单测）

---

## Phase 12：代码质量与性能深度优化

> 日期：2026-07-25 ~ 2026-07-26  
> 目标：26 项代码级优化 + 66 单元测试 + 检索评测体系 + 流式响应 + 可观测性

### 12.1 严重 Bug 修复 (8 项)

| 优化项 | 文件 | 详情 |
|--------|------|------|
| 线程安全修复 | `api/data_store.py` | 锁未覆盖全部写操作，`add_product`/`checkout` 存在竞态条件。抽取 `_next_oid()`/`_next_pid()` 生成器，所有写操作纳入 `_lock` 保护 |
| 内存泄漏修复 | `api/middleware.py` | RateLimiter 的 `_windows`/`_query_hashes` 字典无限增长。新增定期清理 + 10k 硬上限 + LRU 逐出 |
| 模块副作用消除 | `watcher/engine.py` | `ThreadPoolExecutor` 在 import 时创建且永不关闭。移至实例属性懒初始化，`stop()` 时正确 shutdown |
| LogRecord 冲突 | `flywheel/vector_store.py` | `logger.extra={"name": ...}` 与 Python LogRecord 保留属性冲突 → KeyError 导致 Milvus 静默失败回退到内存 |
| Tracer 多线程安全 | `agent/tracer.py` | `_trace_store`/`_trace_history` 被多线程并发读写无锁 → 数据损坏。加 `RLock` 保护全部读写 |
| 单例竞态 | `flywheel/vector_store.py` | `get_vector_store()` 无锁 check-then-set → 可能创建多个 Milvus 实例。改为双重检查锁 |
| Agent NameError | `agent/agent.py` | `_llm_decide_node` 中 `settings.TOOL_CALL_GUARD_ENABLED` 引用了未定义的 `settings` 变量 → Agent 返回空回复 |
| 缩进解析 bug | `agent/fact_checker.py` | `_parse_decision_tree` 使用 `re.match` 导致 sklearn 缩进格式的规则文本无法解析 → Layer2 校验全部跳过 |

### 12.2 性能优化 (5 项)

| 优化项 | 文件 | 详情 |
|--------|------|------|
| HTTP 连接池 | `llm/client.py` | 每次 `_call_api` 创建新 `httpx.Client` → 复用单例（keepalive 5 连接），连续调用延迟降 ~40% |
| O(1) 索引查找 | `api/data_store.py` | `_products_by_id`/`_users_by_id`/`_orders_by_id` 三个 dict 索引，`get_product` 等从 O(n) 扫描变常数查找 |
| Embedding LRU 缓存 | `flywheel/vector_store.py` | `@lru_cache(maxsize=512)` 缓存 Ollama embedding 结果含负向缓存 |
| 检索并行化 | `flywheel/retriever.py` | Sparse + Dense 检索从串行改为 `ThreadPoolExecutor(2)` 并发 |
| 线程池复用 | `flywheel/retriever.py` | 每次检索创建/销毁线程池 → 模块级单例复用 |

### 12.3 Milvus 持久化向量存储

| 优化项 | 详情 |
|--------|------|
| `MilvusVectorStore` | 新增持久化后端，数据跨进程重启存活，接口与 `InMemoryVectorStore` 一致 |
| 透明回退 | Milvus 不可用时自动切换 InMemoryVectorStore |
| 批量刷盘 | 每 20 次插入自动 flush + close 时最终刷盘（解决 Windows milvus-lite 3.1 连续 flush 的 `FileExistsError`） |
| 配置化 | `FLYWHEEL_VECTOR_BACKEND=milvus|memory`，`FLYWHEEL_MILVUS_DATA_DIR` |

### 12.4 工程质量 (13 项)

| 优化项 | 文件 | 详情 |
|--------|------|------|
| SQLite WAL 模式 | `api/feedback.py` `watcher/task_manager.py` `flywheel/store.py` | 所有 SQLite 连接开启 `PRAGMA journal_mode=WAL`，并发读写性能提升 |
| Pipeline 线程安全 | `skills/user_segment.py` | `_cache_lock = RLock()` + 双重检查，防止 API 线程和 Watcher 线程竞争 |
| 模块副作用消除 | `api/feedback.py` | `_init_db()` 从 import 时执行改为首次调用时懒初始化 |
| 评分权重配置化 | `flywheel/scorer.py` + `config/settings.py` | 11 个评分权重从硬编码移至 `SCORER_*` 配置项 |
| CORS 配置化 | `api/main.py` + `config/settings.py` | `CORS_ORIGINS` 从硬编码列表移至配置 |
| 工具守卫配置化 | `agent/agent.py` + `config/settings.py` | `TOOL_CALL_GUARD_ENABLED` + `TOOL_CALL_GUARD_KEYWORDS` |
| 飞轮异常处理 | `agent/agent.py` | `except Exception: pass` → 细分 `ImportError`/`Exception` + 日志 |
| `conn_exec` 反模式 | `watcher/task_manager.py` | 删除模块级函数访问私有方法，`list_tasks` 改为实例方法 |
| 重复代码删除 | `log/middleware.py` | 已删除 — 与 `api/middleware.py` 完全重复，无人引用 |
| 依赖补全 | `pyproject.toml` | 补充 gradio/numpy/scikit-learn/pymysql/sqlalchemy/langchain-openai |
| 过期文件清理 | `requirements.txt` `schema.sql` | 删除旧依赖文件；schema.sql 分区日期 2020→2024 更新为 2024→2030 |
| `atexit` 清理 | `app.py` `llm/client.py` | Gradio 和 LLM 客户端在进程退出时释放连接池 |
| 单例工具 | `common/singleton.py` | 新增 `@singleton` 装饰器，自动提供 `.reset()`/`.close()` |

### 12.5 API 规范

| 优化项 | 详情 |
|--------|------|
| HTTP 方法修正 | `POST /tasks/{id}/retry` `POST /tasks/{id}/ignore` → `PATCH` |
| 分页一致性 | `GET /api/orders` 补全 `total_pages` 字段 |
| 响应模型 | 所有 debug 端点新增专用 Pydantic 响应模型 + Example 值，OpenAPI 文档不再显示 `"string"` |
| X-User-ID | 电商 API 支持 `X-User-ID` Header 替代硬编码 `user_id=1` |
| 流式响应 | 新增 `POST /ask/stream` SSE 端点，打字机效果逐字输出 |

### 12.6 单元测试 (66 用例)

| 模块 | 文件 | 用例数 | 覆盖内容 |
|------|------|--------|---------|
| 事实核查 | `tests/test_fact_checker.py` | 24 | 三层校验 + 编排器 + 边界条件 |
| 数据存储 | `tests/test_data_store.py` | 25 | CRUD + 索引 + 10 线程并发安全 |
| 事件检测 | `tests/test_events.py` | 17 | 快照 diff + 6 条规则 + 合并窗口 + 虚拟查询 |

### 12.7 检索评测体系

| 优化项 | 详情 |
|--------|------|
| 评测模块 | `eval/retrieval_eval.py` — Hit Rate @1/@3/@5 + MRR + RAGAS LLM-as-Judge |
| 人工标注真值 | 10 条查询 × 精确样本 ID 映射，消除关键词匹配假阳性 |
| 飞轮样本治理 | 新增 8 条手工标注样本；scorer 增加 watcher_alert 检测(-0.40)和模板检测；14 条 auto_task 模板自动翻为 negative |
| 评测结果 | Hit@1=70%、Hit@3=80%、Hit@5=80%、MRR=0.75（严格人工标注） |
| 结果留存 | `eval/results/latest_report.txt` + `latest_report.json` |

### 12.8 LLM 可观测性

| 优化项 | 详情 |
|--------|------|
| 统计收集 | `agent/observability.py` — 请求数/Token/延迟/工具调用/事实核查 |
| 可观测性 API | `GET /stats/observability` |
| Gradio 面板 | 新增 Tab「📈 可观测性」— P50/P95/P99 延迟、工具调用率、最近 20 条请求 |

### 12.9 一键启动与文档

| 优化项 | 详情 |
|--------|------|
| 一键启动 | `start.bat` — 双击弹出三个窗口，10 秒后三个服务全部就绪 |
| Mermaid 架构图 | README 新增完整 Mermaid 流程图（Frontend→API→Agent→Skills→Pipeline→Watcher→Flywheel→Eval） |
| .env.example | 新开发者引导文件，列出所有可配置环境变量 |

---

## Phase 11：面试级打磨 — AI 能力可视化与生产就绪

> 日期：2026-07-25  
> 目标：从"传统前后端项目"升级为"AI 应用工程师的作品集"

### 11.1 Agent 可观测性

| 优化项 | 详情 |
|--------|------|
| Trace 系统 | 新增 `agent/tracer.py` — 轻量级 StateGraph 执行链路追踪，记录每个节点耗时、token、输入输出 |
| Trace API | `GET /traces`、`GET /traces/{id}`、`DELETE /traces` — 通过 HTTP 暴露 Trace 数据 |
| Trace 面板 | Gradio 新增 Tab「🔍 Agent Trace」— 下拉选择会话 → 查看完整 6 节点执行链路 |
| 工具调用修复 | 发现 `deepseek-r1:7b` 不支持 function calling → 切换到 `qwen2.5:7b` |
| text→tool 桥接 | Ollama 模型有时输出工具名文本而非 tool_calls，新增自动检测转换 |
| reflect 循环修复 | `_reflect_condition` 检查最后一条 AI 消息是否有 tool_calls，有则必须回 llm_decide 生成文本回复 |

### 11.2 真实 Embedding + 混合检索

| 优化项 | 详情 |
|--------|------|
| 真 Embedding | `hash(bigram) % 128` 假 embedding → `nomic-embed-text` 768 维真实语义向量 |
| 混合检索 | `flywheel/retriever.py` 重写 — Dense(语义) + Sparse(关键词) 双路召回 |
| RRF 融合 | Reciprocal Rank Fusion 合并两路排序结果 |
| LLM Re-rank | 用 LLM cross-encoder 对候选精排 |
| 向量入库 | `store.add_sample()` 自动调 `vector_store.insert()` 同步向量索引 |

### 11.3 LLM-as-Judge 评测

| 优化项 | 详情 |
|--------|------|
| Judge 模块 | 新增 `eval/judge.py` — 3 维度评分（忠实度/完整性/可读性，1-5→0-1） |
| 评测集成 | `eval/runner.py` 新增 `run_judge_suite()`，`--judge` CLI 参数 |
| 报告增强 | 报告中新增 LLM-Judge 评分表 |
| 用例扩容 | qa_test_set 从 20 条扩到 25 条，新增 5 条对比分析用例 |

### 11.4 数字飞轮闭环

| 优化项 | 详情 |
|--------|------|
| 飞轮面板 | Gradio 新增「🔁 数字飞轮」— 手动触发 + 统计面板 |
| 飞轮 API | `GET /flywheel/stats`、`POST /flywheel/trigger` |
| 反馈面板 | Gradio 新增「📋 反馈记录」— 查看 👍/👎 历史 |
| 反馈 API | `GET /feedback/history` — 最近 20 条反馈 |
| 向量入库串联 | sample_library 入库 → 自动索引到向量存储 → Agent 检索反哺 |

### 11.5 分群增长指标

| 优化项 | 详情 |
|--------|------|
| 增长计算 | `pipeline/profile.py` 新增 `compute_segment_growth()` — 销售额增长%/转化率变化%/GMV提升% |
| 增长 Skill | 新增 `SegmentGrowthSkill` — 调用快照对比 → 生成专业对比分析表 + 数据解读 |
| 对比 Prompt | 新增 `PROMPT_COMPARISON` 模板 + `COMPARISON_KEYWORDS` 路由 |

### 11.6 图表与数据联动

| 优化项 | 详情 |
|--------|------|
| 中文字体 | matplotlib 配置 Microsoft YaHei，解决图表中文乱码 |
| 图表说明 | 4 张图表下方新增 📖 说明文字（指标含义、数据解读） |
| 数据联动 | `_try_mock_data()` 合并 data_store 中的订单，图表随下单变化 |
| 强制刷新 | `/stats/*` 新增 `force=true` 参数，每次刷新图表强制重算 pipeline |
| 随机数据 | `np.random.seed(42)` → `np.random.seed(None)`，每次重置用户数/分群数不同 |

### 11.7 全链路中文化

| 优化项 | 详情 |
|--------|------|
| 数据源 | 20 商品名 + 5 品类 + 4 城市 → 全中文 |
| API 错误 | 所有 HTTP 错误消息中文化 |
| 前端 | Vue 全界面中文化，`$` → `¥`，订单状态/品类翻译 |
| Gradio | 图表标签/下拉框/调试消息/任务状态 → 全中文 |
| 清理 | 删除 HelloWorld.vue + Vite 脚手架资源 |

### 11.8 生产就绪打磨

| 优化项 | 详情 |
|--------|------|
| README | 新增专业 README.md（架构图/快速启动/技术栈/核心能力） |
| 线程安全 | `data_store.py` 加 `threading.RLock()` 保护并发读写 |
| 密码移除 | `DB_PASSWORD: "123456"` → `""`，运行时从 .env 读取 |
| 私有变量封装 | 消除 `from api.data_store import _orders`，新增 `create_order_from_items()` |
| 错误处理 | 18 处裸 `except` 全部加 logger；4 处 `pass` 改为日志记录 |
| CORS | 添加 `CORSMiddleware`，允许 5173/7860 跨域 |
| XSS | AiChat.vue `v-html` → `v-text` + `white-space:pre-wrap` |
| 硬编码消除 | Gradio API URL 从 settings 读取，版本号统一为 `settings.APP_VERSION` |
| 脚手架清理 | `style.css` 297 行 Vite 模板 → 5 行全局 reset |
| 网络兼容 | `0.0.0.0` → `127.0.0.1` 修复 WinError 10049 |

---

## Phase 0：基线底层改造

**问题**：项目是 7 文件扁平结构，硬编码配置、裸异常捕获、全量数据加载、无日志体系。

| 优化项 | 详情 |
|--------|------|
| 目录分层 | 拆分为 `api/` `agent/` `pipeline/` `config/` `log/` `llm/` `cache/` `errors/` 8 个包 |
| 配置中心 | `config/settings.py` — Pydantic-settings 统一管理，消除 agent.py 硬编码 `os.environ` 覆盖 |
| 日志体系 | JSON 结构化日志 → `logs/user_chat_log/` + `logs/auto_task_log/`，RotatingFileHandler 轮转 |
| 三级降级 | MySQL → TTL 文件缓存 → Mock 模拟数据，任意环境可运行 |
| 分页加载 | 全表 JOIN → `LIMIT/OFFSET` 分页查询，避免内存溢出 |
| 异常体系 | Database / Computation / Parameter / API 四类异常 + context 上下文 |
| LLM 封装 | `llm/client.py`：超时 + 自动重试(tenacity) + Token 统计 |

---

## Phase 1：Agent 架构升级

**问题**：基础 ReAct Agent，3 个硬编码工具，无预处理/自反思，消息取值靠数组下标。

| 优化项 | 详情 |
|--------|------|
| Skill 框架 | `skills/base.py`：BaseSkill + SkillResult + SkillRegistry，配置开关启停 |
| Agentic RAG | `create_react_agent` → 自定义 StateGraph（preprocess→llm_decide→tools→reflect→respond） |
| Query 预处理 | `agent/preprocessor.py`：5 类电商意图分类 + 实体抽取 + 子任务拆分 |
| Self-Reflection | `agent/reflector.py`：LLM-as-judge 校验完整性/冲突，最多 2 轮自动重调 |
| 消息取值修复 | `extract_ai_response()` 按 `type=='ai'` 筛选，不再用 `[-1]` |
| 双层记忆 | InMemorySaver（聊天会话）+ BusinessMemory（业务上下文 TTL） |

---

## Phase 2：幻觉治理

**问题**：LLM 编造数值、篡改规则阈值、运营建议与聚类结果矛盾。

| 优化项 | 详情 |
|--------|------|
| 三层 Fact-Check | 纯规则引擎：Layer1 数值校验 / Layer2 规则校验 / Layer3 逻辑校验 |
| 双档位管控 | relaxed（用户问答→追加警告）/ strict（自主分析→阻断重试） |
| 三套 Prompt | DATA_STATS / RULE_INTERPRET / STRATEGY，关键词路由选择 |
| Markdown 强制 | `enforce_markdown()` + Prompt 内置格式指令 |

---

## Phase 3：数据流水线优化

**问题**：固定 K=4、RFM 全量重算、3 维特征单调、无异常检测、缓存永不过期。

| 优化项 | 详情 |
|--------|------|
| 增量 RFM | `compute_rfm_incremental()` 仅对新增订单重算 |
| 自动最优 K | 肘部法则 + 轮廓系数联合判定，替换固定 K=4 |
| 优化决策树 | ccp_alpha 剪枝 + 流转特征（recency_change_rate / frequency_trend） |
| 画像扩容 | 新增复购间隔、价格敏感度、品类偏好、品类多样性 4 个维度 |
| 异常检测 | IQR / Z-score 自动识别过滤极端消费值 |
| 快照持久化 | `ClusterSnapshot` + save/load/compare，支持跨时间人群对比 |
| 缓存 TTL | `_load_and_process()` TTL 过期自动重算 + `RefreshPipelineSkill` |

---

## Phase 4：事件驱动自主 Agent

**问题**：所有分析被动触发，无法自动感知数据变化。

| 优化项 | 详情 |
|--------|------|
| 后台轮询 | FastAPI lifespan 启动 asyncio 后台任务，定时比对快照 |
| 事件降噪 | 6 条检测规则 + 冷却机制 + 低优先级合并 |
| 事件分级 | HIGH（流失/暴跌）优先调度，NORMAL 延后批处理 |
| 虚拟 Query | 业务事件自动封装为中文 Query，复用同一套 Agent |
| 安全限制 | 最大工具轮次 + 超时管控 |
| 生命周期 | SQLite 持久化：pending→running→completed/failed/timeout/ignored |

---

## Phase 5：网关与稳定性

**问题**：无限流保护、异常返回格式不统一、无健康检查、线程池混用。

| 优化项 | 详情 |
|--------|------|
| 滑动窗口限流 | 60s/20次 per session + 相似问句 MD5 去重 |
| LLM 并发管控 | `asyncio.Semaphore(3)` 全局并发上限 |
| 异常标准化 | 所有错误 → `{"error":{"code":"...","message":"..."}}` |
| 健康检查 | `/health/full`：MySQL 连通性 + LLM API 状态 + Watcher 状态 |
| 线程池隔离 | 自主任务独立 `ThreadPoolExecutor(2)`，与用户请求互不抢占 |

---

## Phase 6：Gradio 前端改造

**问题**：Gradio 直连业务代码，仅单一聊天界面。

| 优化项 | 详情 |
|--------|------|
| 前后端解耦 | Gradio 全部通过 httpx 调用 FastAPI，零业务模块导入 |
| 三面板布局 | 💬 AI对话 / 🛠️ 调试操作台 / 📊 任务监控+图表 |
| 反馈系统 | 👍/👎 按钮 → SQLite 落库 |
| 调试操作台 | 商品/订单/用户 CRUD + 一键重置 + 手动触发事件 |
| 可视化 | RFM 柱状图 / 分群饼图 / 雷达图 / 流转分布图 |

---

## Phase 7：数字飞轮

**问题**：每次问答"从零开始"，反馈无后续利用。

| 优化项 | 详情 |
|--------|------|
| 三源采集 | 用户反馈 + 自主任务报告 + 人工标注 |
| 自动评分 | 7 条纯规则评分，幻觉/低质量→负样本库 |
| 增量更新 | 游标式增量入库，无需全量重建 |
| 检索反哺 | 相似历史优质 QA → Agent 系统提示注入 |

---

## Phase 8：自动化评测

**问题**：无量化评测手段，每次改动无法客观衡量效果。

| 优化项 | 详情 |
|--------|------|
| 两套测试集 | 20 条 QA 用例 + 10 条事件检测用例 |
| 6 项指标 | 工具准确率 / 数据忠实度 / 业务相关性 / 事件准确率 / Token 消耗 / 响应耗时 |
| 一键对比 | `python -m eval` → Markdown 对比报告 + JSON 结果留存 |

---

## Phase 9：联调回归

**问题**：多 Phase 累积的边界 bug 和未覆盖场景。

| 优化项 | 详情 |
|--------|------|
| 12 个 bug 修复 | 缓存击穿、空回复误判、限流内存泄漏、参数错误码等 |
| 21 API 全通 | 全部端点冒烟测试通过 |
| 29 模块全量导入 | 零循环依赖，零断裂引用 |

---

## Phase 10：生产级架构演进

**问题**：无真实电商流程、定时轮询延迟高、知识库检索弱、单 Agent 瓶颈。

| 优化项 | 详情 |
|--------|------|
| Vue 电商商城 | Vue 3 + Vite + Pinia：首页→商品列表→详情→购物车→下单→支付 + AI 助手 |
| 电商 API | 7 个新端点：商品搜索/分类、购物车 CRUD、下单/支付、用户中心 |
| 多智能体编排 | Monitor → Analysis → Strategy 三 Agent 流水线 |
| Milvus 向量检索 | 128 维 bigram embedding + COSINE 相似度，替代关键词匹配 |
| CDC 消费者 | Binlog → asyncio.Queue → debounce → detect，为实时感知做准备 |

---

## 最终项目全貌

```
                    ┌──────────────────────────┐
                    │   Vue 电商商城 (:5173)     │  用户端
                    │   商品/购物车/下单/支付     │
                    └──────────┬───────────────┘
                               │ httpx
                    ┌──────────┴───────────────┐
                    │   FastAPI 后端 (:8000)     │  核心层
                    │   21 API + 5 中间件        │
                    └──────────┬───────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
  ┌──────┴──────┐    ┌────────┴────────┐    ┌───────┴───────┐
  │ Agentic RAG  │    │  Watcher Engine │    │  Flywheel     │
  │ StateGraph   │    │  事件检测/CDC    │    │  数据飞轮     │
  │ Skill Pool   │    │  Task Manager   │    │  Milvus 检索  │
  │ FactCheck    │    │  降噪/分级       │    │  样本评分     │
  │ 多智能体      │    │                 │    │               │
  └──────┬───────┘    └────────┬────────┘    └───────┬───────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                    ┌──────────┴───────────────┐
                    │   Gradio 管理后台 (:7860)  │  运营端
                    │   AI对话/调试/监控/图表    │
                    └──────────────────────────┘
```

### 技术栈全览

| 层 | 技术 |
|----|------|
| 前端 | Vue 3 + Vite + TypeScript + Pinia |
| 管理端 | Gradio 6 |
| API | FastAPI + Pydantic |
| Agent | LangGraph 1.2 + LangChain 1.3 |
| 数据处理 | Pandas + NumPy + scikit-learn |
| 向量检索 | Milvus Lite |
| 存储 | SQLite (任务/反馈/样本) + TTL Pickle 缓存 |
| LLM | Ollama + OpenAI 兼容 API |
| 日志 | python-json-logger + RotatingFileHandler |
| 评测 | 自研 eval 框架 (CLI) |
