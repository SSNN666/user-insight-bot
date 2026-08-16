# 数据源合规整改说明

> 面试口述素材:数据来源、版权与爬虫合规边界。

## 1. 现状(开发/演示阶段)

- 开发与 Demo 全部使用**内置 mock 数据**(`pipeline/data_loader.py` 的 `generate_mock_orders`,
  固定随机种子 `MOCK_SEED=42` → 演示可复现、Watcher 事件反映真实业务变化而非随机噪声)。
- **不爬取任何网站**、不采集任何真实用户数据;调试台/Vue 商城的订单数据仅存在于本机内存与本地 SQLite。
- 电商 CRUD 与 pipeline 之间通过共享内存存储联动,无任何外部数据上传。

## 2. 生产数据源方案(已实现:京东 JData 数据源)

### 2.1 数据集选型与下载渠道

首选 **「京东 JData 算法大赛——高潜用户购买意向预测」数据集**
(2017 年竞赛,数据覆盖 2016-02 ~ 2016-04;已脱敏、竞赛/学术研究许可):

| 文件 | 行数(官方口径) | 字段 | 本项目用途 |
|------|--------------|------|-----------|
| JData_User.csv | 105,321 | user_id(脱敏)、age、sex、user_lv_cd、user_reg_tm | 画像人口属性(年龄/性别/等级) |
| JData_Product.csv | 24,187 | sku_id、a1-a3 属性、cate、brand | 商品品类/品牌 |
| JData_Action_20160{2,3,4}.csv | 1150万/2592万/1320万 | user_id、sku_id、time、model_id、type、cate、brand | 行为时序 → 订单 |
| JData_Comment.csv | 558,552 | dt、sku_id、评论数、差评率 | (预留)商品口碑信号 |

行为类型官方编码:**1=浏览 2=加购 3=删除 4=下单 5=关注 6=点击**(下单 = type 4)。

**下载渠道**(比赛已结束,以下渠道任选):

1. **Kaggle 数据集页**(推荐,浏览器点击 Download 即可,已验证无需登录即可下载):
   [owincontext/jdata2016](https://www.kaggle.com/datasets/owincontext/jdata2016) —
   官方 6 文件完整版,行数与官方口径一致(User 105,321 / Product 24,187 /
   Comment 558,552 / Action 1,148万+2,591万+1,319万)
2. **官方竞赛页**:[DataFountain 京东JData算法大赛](https://www.datafountain.cn/competitions/247)
   (注册后下载;FAQ 页提供各文件 MD5 校验值)
3. **网盘镜像**:CSDN 博客整理版,百度网盘 `https://pan.baidu.com/s/1ojjVqjXS0cP2KAAyC-tsxg`
   提取码 `semp`(下载后建议用官方 MD5 校验)
4. **GitHub 复现仓库**(数据需按仓库 README 自行放入):[paristsai/jdata](https://github.com/paristsai/jdata)、
   [guangningyu/JD-prediction](https://github.com/guangningyu/JD-prediction)

> ⚠️ 许可边界:竞赛数据仅限**学术研究/学习用途**,不可商用;生产必须切企业自有订单库。

### 2.2 使用方式

```bash
# 1. 下载 6 个 CSV 放入 data/tianchi/(文件名保持原样即可,加载器按关键字匹配)
# 2. .env 切换数据源
DATA_SOURCE=tianchi
# 3. 重启后降级链变为:MySQL → TTL 缓存 → JData CSV → mock(文件缺失自动回退)
```

官方数据已知的"脏点"加载器已自动处理:Action 表 user_id/sku_id 是浮点
("1.0" 格式,自动归一化 int 才能与 User 表 join)、个别行字段数不一致(自动跳过)、
user_reg_tm 空值(保留 NaT)。

已实现的工程接入(见 `pipeline/data_loader.py` Tier 2.5):
- 文件名关键字自动匹配(支持分月 CSV),行为 type=4(下单)→ 订单明细;
- 采样上限 `TIANCHI_MAX_ACTIONS` / `TIANCHI_MAX_USERS`(演示用,None=全量);
- RFM/聚类/Agent/推荐全链路零改动(下游只消费 canonical orders DataFrame)。

### 2.3 如实记录的简化口径(面试追问素材)

- **JData 不含价格字段** → 按 (cate, brand) 做 md5 确定性合成价格(50-5000 元,
  同一商品跨运行稳定,快照对比不受影响);换含金额的数据集只需替换 `_synth_price`;
- 一行下单行为 = 一条订单明细(quantity=1),F=下单次数、M=Σ合成价格;
- 分类/品牌是数字编码,不做中文映射(推荐 Skill 品类不匹配时自动回退热门商品);
- 许可范围:**学术研究用途**,不可商用 → 生产必须切企业自有订单库。

### 2.4 真实业务数据(生产)

接入企业自有订单库(MySQL,降级链第一层已具备,分页 JOIN 查询即 `_try_mysql_joined`)。

## 3. 爬虫与数据合规边界(口述要点)

- **授权渠道优先**:电商数据应走官方开放平台 API,例如淘宝开放平台(TOP)——
  需应用审核、按类目申请权限、遵守 API 调用频率与数据使用协议,禁止超范围留存/二次分发。
- **技术边界不等于法律边界**:即使技术上可爬,未经授权的规模化抓取存在法律风险:
  - 违反 robots 协议、绕过反爬措施 → 可能触犯《反不正当竞争法》第十二条(互联网专条);
  - 爬取个人信息(收货地址、手机号等)→ 触犯《个人信息保护法》,即使公开可见也不可随意收集;
  - 司法判例(如淘宝诉美景案、微博诉脉脉案等)确立了平台数据权益保护口径。
- **本项目立场**:知识库/画像数据全部来自公开许可或自建数据,不使用爬虫获取电商网页数据;
  面试表述统一为"数据源使用公开脱敏数据集,生产走官方开放平台与自有订单库"。

## 4. 面试通用话术(两项目共用)

> 我聚焦 AI 应用开发,没有做模型 LoRA/SFT 微调。依靠应用层架构——检索编排、
> Agent 工作流、多模型调度、异常降级、安全校验、评测体系——解决幻觉与 API 不稳定等问题。
> 当前系统为本地 Demo,没有正式上线生产;但我梳理了完整的生产改造方案
> (JWT 权限体系、熔断机制、模型灰度切换、数据库连接池优化)。
> 适配器同时支持文本模型与多模态 VL 模型,二者职责分离:文本模型负责对话与工具调用;
> VL 模型专门处理图像理解,不会用来批量提取文字,遵循工程选型最优方案。
