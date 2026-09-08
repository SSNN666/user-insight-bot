"""Data loading with pagination and three-tier fallback.

Tier 1: MySQL paginated query
Tier 2: Local TTL file cache
Tier 3: Built-in mock dataset

Phase 3: incremental loading + expanded mock data with product/category.
"""

import hashlib
import os

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from config.settings import get_settings
from errors.exceptions import DatabaseError, ComputationError
from cache.ttl_cache import TTLCache
from log.logger import get_logger

logger = get_logger(__name__)
_cache = TTLCache(namespace="orders")

CACHE_KEY_JOINED = "orders_joined"
CACHE_KEY_RFM = "rfm_from_db"


def get_db_engine():
    """Create SQLAlchemy engine for MySQL."""
    settings = get_settings()
    conn_str = (
        f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        f"?charset={settings.DB_CHARSET}"
    )
    return create_engine(conn_str)


# ── Tier 1: MySQL ───────────────────────────────────────────────


def _try_mysql_joined(since_date: str | None = None) -> pd.DataFrame:
    """Tier 1: Paginated MySQL query with JOIN, optionally since a date."""
    settings = get_settings()
    engine = get_db_engine()
    page_size = settings.DATA_PAGE_SIZE

    where_clause = ""
    params: dict = {"limit": page_size, "offset": 0}
    if since_date:
        where_clause = "WHERE o.order_date > :since_date"
        params["since_date"] = since_date

    try:
        with engine.connect() as conn:
            count_query = text(f"SELECT COUNT(*) FROM orders o {where_clause}")
            total = conn.execute(count_query, params).scalar()
            logger.info("mysql_count", extra={"total_orders": total, "since": since_date})

        dfs: list[pd.DataFrame] = []
        for offset in range(0, total, page_size):
            params["offset"] = offset
            query_str = f"""
                SELECT
                    o.order_id, o.user_id, u.username, u.reg_date,
                    u.city, u.age, u.gender,
                    o.product_id, p.product_name, p.category,
                    p.price AS unit_price, p.price AS price,
                    o.quantity, o.total_amount, o.order_date
                FROM orders o
                JOIN users u ON o.user_id = u.user_id
                JOIN products p ON o.product_id = p.product_id
                {where_clause}
                ORDER BY o.order_date
                LIMIT :limit OFFSET :offset
            """
            chunk = pd.read_sql(text(query_str), engine, params=params)
            dfs.append(chunk)
            logger.debug("page_loaded", extra={"offset": offset, "rows": len(chunk)})

        engine.dispose()
        result = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if not since_date and len(result) > 0:
            try:
                _cache.set(CACHE_KEY_JOINED, result)
            except Exception:
                logger.warning("cache_write_failed")
        logger.info("tier1_success", extra={"rows": len(result)})
        return result

    except (SQLAlchemyError, OSError) as e:
        engine.dispose()
        raise DatabaseError(f"MySQL query failed: {e}", context={"tier": 1}) from e


# ── Tier 2: TTL cache ───────────────────────────────────────────


def _try_cache_joined() -> pd.DataFrame | None:
    """Tier 2: Read from local TTL file cache.

    Returns None on cache miss so the caller can fall through to Tier 3
    without treating a cache miss as an error.
    """
    data = _cache.get(CACHE_KEY_JOINED)
    if data is None:
        logger.debug("cache_miss")
        return None
    logger.info("tier2_success", extra={"rows": len(data)})
    return data


# ── Tier 2.5: 京东 JData 公开脱敏数据集(可选) ──────────────────
#
# 「京东 JData 算法大赛——高潜用户购买意向预测」数据集(官方发布在 DataFountain
# 竞赛平台,学术研究许可、已脱敏;下载渠道见 docs/DATA_COMPLIANCE.md):
#   JData_User.csv    user_id(脱敏), age, sex, user_lv_cd, user_reg_tm
#   JData_Product.csv sku_id, a1, a2, a3, cate, brand
#   JData_Action.csv  user_id, sku_id, time, model_id, type, cate, brand(2016-02~04 分月)
#   JData_Comment.csv dt, sku_id, comment_num, has_bad_comment, bad_comment_rate
#
# 映射口径(如实记录,面试可讲):
#   - type=4(下单)行为行 → 一条订单明细(quantity=1);F=下单次数,M=Σ价格
#   - JData 不含价格字段 → 按 (cate, brand) 做 md5 确定性合成价格(50-5000 元,
#     同一商品跨运行稳定,快照对比不受影响)
#   - JData 无商品名字段、品类为数字编码 → 按 (品类码 → 中文品类 + 品名词库)
#     确定性合成可检索的商品名(如 "耳机-9001"/品类"影音数码"),与价格合成
#     同级演示口径(商城中文搜索/品类筛选/漏斗分组因此可用;原数字码商品名
#     无法支撑任何自然语言检索)。真实性与合成边界如实记录在 DATA_COMPLIANCE.md
#   - 合成映射集中在 _JD_CATE_ZH,唯一事实源,全部消费方同源

# JData 品类码(实测 8 个:4-11)→ (中文品类, 品名词库);词按 sku 哈希确定性分配
_JD_CATE_ZH: dict[str, tuple[str, tuple[str, ...]]] = {
    "4":  ("影音数码", ("耳机", "音箱", "麦克风", "便携播放器")),
    "5":  ("电脑办公", ("键盘", "鼠标", "显示器", "笔记本支架")),
    "6":  ("手机配件", ("手机壳", "充电器", "数据线", "移动电源")),
    "7":  ("家用电器", ("台灯", "电水壶", "吸尘器", "电风扇")),
    "8":  ("时尚服饰", ("运动鞋", "T恤", "休闲外套", "双肩背包")),
    "9":  ("美妆个护", ("洗面奶", "面膜", "电动牙刷", "护肤套装")),
    "10": ("食品生鲜", ("坚果礼盒", "茶叶", "咖啡豆", "零食组合")),
    "11": ("图书文娱", ("小说精选", "工具书", "文具套装", "儿童绘本")),
}
_JD_CATE_ZH_FALLBACK: tuple[str, tuple[str, ...]] = ("其他", ("精选好物",))


def _jdata_zh_product(cate: object, sku: int) -> tuple[str, str]:
    """(品类码, sku) → (中文品类, 可检索中文商品名),确定性(跨运行稳定)。"""
    zh_cate, words = _JD_CATE_ZH.get(str(cate), _JD_CATE_ZH_FALLBACK)
    h = int(hashlib.md5(f"sku:{sku}".encode("utf-8")).hexdigest()[:4], 16)
    word = words[h % len(words)]
    return zh_cate, f"{word}-{sku}"
#   - Action 表的 user_id/sku_id 是浮点("1.0"),加载时归一化为 int 才能与 User 表 join

JDATA_ACTION_TYPE_ORDER = 4      # 官方行为编码:1=浏览 2=加购 3=删除 4=下单 5=关注 6=点击
JDATA_SEX_MAP = {0: "男", 1: "女", 2: "保密"}


def jdata_date_shift(dates: "pd.Series") -> pd.Timedelta:
    """JData 数据平移到当前时间线:让数据最新日期 = 今天 - 1 天。

    原始 JData 是 2016-02~04 的历史数据;运行时订单(商城)是真实系统时间。
    两者若在同一 RFM 里,基准日会被 2026 年订单拉走 → 全部 JData 用户
    recency 变成 3000+ 天,分群失真。平移后:JData 代表"最近 ~90 天、
    截止昨天"的订单,商城订单(今天)自然衔接 —— 下单用户 recency=1,
    dormant 复活,漏斗"最近 N 天"与分群口径完全一致。
    """
    max_date = pd.to_datetime(dates, errors="coerce").max()
    if pd.isna(max_date):
        return pd.Timedelta(0)
    target = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    return target - max_date.normalize()


def original_span_days(dates: "pd.Series") -> int:
    """原始时间线跨度(天)= (max-min).days+1;空/全 NaN → 0(NaN 安全)。"""
    parsed = pd.to_datetime(dates, errors="coerce").dropna()
    if parsed.empty:
        return 0
    return int((parsed.max().normalize() - parsed.min().normalize()).days) + 1


def apply_reveal_cut(df: "pd.DataFrame") -> "pd.DataFrame":
    """滚动揭晓截断:保留原始时间线"最后 revealed_days 天"(推进/初始化进度)。

    必须在时间线平移**之前**调用(基于原始日期);关闭开关时调用方不进入本函数。
    状态文件缺失 → ensure_state 完成首启(预热 PREHEAT 天);窗口已到顶/数据
    异常 → 原样返回,不截断。
    """
    from pipeline import reveal_state

    span = original_span_days(df["order_date"])
    if span <= 0:
        return df
    st = reveal_state.ensure_state(span)
    revealed = int(st["revealed_days"])
    logger.info("reveal_cut", extra={"revealed": revealed, "span": span,
                                     "rows": len(df)})
    if revealed <= 0 or revealed >= span:
        return df
    cutoff = (pd.to_datetime(df["order_date"], errors="coerce").max().normalize()
              - pd.Timedelta(days=revealed - 1))
    return df[pd.to_datetime(df["order_date"], errors="coerce") >= cutoff]


def _synth_price(cate, brand) -> float:
    """JData 无价格字段:(品类,品牌) → md5 → 50-5000 元确定性合成。"""
    h = int(hashlib.md5(f"{cate}|{brand}".encode("utf-8")).hexdigest()[:8], 16)
    return round(50 + (h % 4950), 2)


def _find_jdata_files(data_dir: str) -> dict[str, list[str]]:
    """按文件名关键字匹配 JData 四个表(支持 JData_Action_201602.csv 分月格式)。"""
    found: dict[str, list[str]] = {"user": [], "product": [], "action": [], "comment": []}
    if not os.path.isdir(data_dir):
        return found
    for fn in sorted(os.listdir(data_dir)):
        if not fn.lower().endswith(".csv"):
            continue
        low = fn.lower()
        if "user" in low:
            found["user"].append(fn)
        elif "product" in low or "sku" in low:
            found["product"].append(fn)
        elif "action" in low:
            found["action"].append(fn)
        elif "comment" in low:
            found["comment"].append(fn)
    return found


def _read_jdata_csvs(data_dir: str, names: list[str], nrows: int | None = None) -> pd.DataFrame:
    """拼接同一表的分月 CSV;任一文件读取失败抛 ComputationError(整档跳过)。"""
    frames = []
    for fn in names:
        # on_bad_lines="skip":官方数据存在个别字段数不一致的行(6 列 vs 7 列)
        # 编码:官方文件混用 utf-8 与 GBK(实测 JData_User.csv 为 GBK)→ 依次尝试
        path = os.path.join(data_dir, fn)
        try:
            df = pd.read_csv(path, nrows=nrows, encoding="utf-8-sig",
                             low_memory=False, on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(path, nrows=nrows, encoding="gbk",
                             low_memory=False, on_bad_lines="skip")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_orders_from_jdata(
    users: pd.DataFrame,
    actions: pd.DataFrame,
    max_users: int | None = None,
    action_type_order: int = JDATA_ACTION_TYPE_ORDER,
) -> pd.DataFrame | None:
    """JData 原始表 → canonical orders DataFrame(纯函数,便于单测)。

    Returns None when no order actions exist.
    """
    if users.empty or actions.empty or "type" not in actions.columns:
        return None

    orders = actions[actions["type"] == action_type_order].copy()
    if orders.empty:
        return None

    # 官方 Action 表 user_id/sku_id 为浮点("1.0"),归一化为 int 才能与 User 表 join
    for col in ("user_id", "sku_id"):
        orders[col] = pd.to_numeric(orders[col], errors="coerce")
        orders = orders[orders[col].notna()].copy()
        orders[col] = orders[col].astype(int)

    # 按时间排序后分配自增 order_id(同一用户的时间顺序即订单顺序)
    orders = orders.sort_values("time").reset_index(drop=True)
    orders["order_id"] = orders.index + 1

    # 用户采样:按出现顺序取前 N 个去重用户(确定性)
    if max_users and orders["user_id"].nunique() > max_users:
        keep = orders["user_id"].drop_duplicates().head(max_users)
        orders = orders[orders["user_id"].isin(keep)]

    # 价格:JData 无价格字段 → 确定性合成;一行下单行为 = 一条订单明细
    price_col = orders.apply(lambda r: _synth_price(r.get("cate"), r.get("brand")), axis=1)
    orders["unit_price"] = price_col
    orders["price"] = price_col
    orders["quantity"] = 1
    orders["total_amount"] = price_col
    orders["order_date"] = pd.to_datetime(orders["time"], errors="coerce")
    orders["product_id"] = orders["sku_id"]
    # 中文商品名/品类:JData 无名称字段 → 按品类码词库确定性合成(商城搜索/漏斗可演示)
    if "cate" in orders.columns:
        zh = orders.apply(
            lambda r: _jdata_zh_product(r.get("cate"), r["sku_id"]), axis=1)
        orders["category"] = [c for c, _ in zh]
        orders["product_name"] = [n for _, n in zh]
    else:
        orders["category"] = "其他"
        orders["product_name"] = "精选好物-" + orders["sku_id"].astype(str)

    # 用户维:age / 性别(sex 0/1/2)/ 注册时间(user_reg_tm,空值保留 NaT)
    u = users[["user_id", "age"]].copy()
    u["user_id"] = pd.to_numeric(u["user_id"], errors="coerce")
    u = u[u["user_id"].notna()]
    u["user_id"] = u["user_id"].astype(int)
    if "sex" in users.columns:
        # 官方数据 sex 读入为 float(2.0/0.0/1.0),显式归一化为 int 再查字典
        sex = pd.to_numeric(users["sex"], errors="coerce").astype("Int64")
        u["gender"] = sex.map(JDATA_SEX_MAP).fillna("保密")
    else:
        u["gender"] = "保密"
    u["reg_date"] = pd.to_datetime(
        users["user_reg_tm"] if "user_reg_tm" in users.columns else pd.NaT,
        errors="coerce",
    )
    orders = orders.merge(u, on="user_id", how="left")
    # 官方口径:age 是分桶字符串("26-35岁"/"56岁以上"),-1 = 未知(不是数值!)
    orders["age"] = orders["age"].fillna("未知").astype(str).replace("-1", "未知")
    orders["city"] = "未知"

    keep_cols = ["order_id", "user_id", "product_id", "product_name", "category",
                 "price", "unit_price", "quantity", "total_amount", "order_date",
                 "reg_date", "age", "gender", "city"]
    return orders[[c for c in keep_cols if c in orders.columns]]


def load_tianchi_orders(data_dir: str | None = None, since_date: str | None = None) -> pd.DataFrame | None:
    """加载京东 JData CSV 并映射为 canonical orders。

    Returns None when files missing (caller falls through to mock tier).
    """
    settings = get_settings()
    data_dir = data_dir or settings.TIANCHI_DATA_DIR
    found = _find_jdata_files(data_dir)
    if not (found["user"] and found["action"]):
        logger.debug("tianchi_files_missing", extra={"dir": data_dir})
        return None

    try:
        max_actions = settings.TIANCHI_MAX_ACTIONS
        users = _read_jdata_csvs(data_dir, found["user"])
        actions = _read_jdata_csvs(data_dir, found["action"], nrows=max_actions)
        df = _build_orders_from_jdata(users, actions, max_users=settings.TIANCHI_MAX_USERS)
        if df is None:
            return None
        # 滚动揭晓:在原始时间线上截取"最后 revealed_days 天"(保留最新尾部),
        # 再平移 → 露出子集最新 = 今天-1,RFM 基准日口径不变。关闭时零改动。
        if settings.DATA_SOURCE == "tianchi" and settings.TIANCHI_REVEAL_ENABLED:
            df = apply_reveal_cut(df)
        # 时间线平移:JData(2016)→ 当前时间线(最新 = 今天-1),注册时间同步平移
        shift = jdata_date_shift(df["order_date"])
        if shift != pd.Timedelta(0):
            df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce") + shift
            df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce") + shift
        if since_date:
            df = df[pd.to_datetime(df["order_date"], errors="coerce") > pd.Timestamp(since_date)]
        # 合并共享存储的运行时订单(商城下单即入分析,真实日期与平移后时间线衔接)
        df = _merge_store_orders(df)
        logger.info("tier_tianchi_success", extra={
            "dir": data_dir, "rows": len(df), "users": df["user_id"].nunique(),
        })
        return df
    except (OSError, ValueError, pd.errors.ParserError) as e:
        logger.warning("tianchi_load_failed", extra={"dir": data_dir, "error": str(e)})
        return None


def _merge_store_orders(df: pd.DataFrame) -> pd.DataFrame:
    """把共享存储(data_store)的运行时订单并入 JData 订单。

    商城/Vue 下单的用户是 data_store 按 tianchi 模式播种的 JData 用户、
    商品是 JData SKU → 与 CSV 天然同源,合并后这些订单进入 RFM/分群,
    "下单即入分析"闭环成立(下单用户 recency 更新、dormant 复活)。
    品类从共享商品索引查询;data_store 不可用时跳过合并(不阻塞主链路)。
    """
    try:
        # ⚠️ 必须读模块私有状态而非 list_orders()/list_products():
        # 懒播种期间(seed → load_orders_with_join → 本函数)调用公开入口会
        # 再次触发 _ensure_seeded → 无限递归。_orders/_products_by_id 是
        # 运行时数据,与播种(仅重建 users/products 索引)互不依赖。
        import api.data_store as ds
        store_orders = ds._orders
        if not store_orders:
            return df
        try:
            cat_by_pid = {p["product_id"]: p.get("category", "未知")
                          for p in ds._products_by_id.values()}
        except Exception:
            cat_by_pid = {}

        # 时序口径:JData 已平移到当前时间线(最新 = 今天-1),运行时订单
        # 使用真实 created_at(今天)自然衔接 → 下单用户 recency=1,dormant 复活
        extra_rows = []
        for o in store_orders:
            pid = o.get("product_id", 1)
            extra_rows.append({
                "order_id": o.get("order_id", 0),
                "user_id": o.get("user_id", 1),
                "product_id": pid,
                "product_name": o.get("product_name", "未知"),
                "category": cat_by_pid.get(pid, "未知"),
                "price": o.get("total_amount", 0),
                "unit_price": o.get("total_amount", 0),
                "quantity": o.get("quantity", 1),
                "total_amount": o.get("total_amount", 0),
                "order_date": pd.Timestamp(o.get("created_at", pd.Timestamp.now())),
                "reg_date": pd.NaT,
                "age": "未知",
                "gender": "未知",
                "city": "未知",
            })
        out = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
        logger.info("tianchi_merged_store_orders", extra={
            "store_orders": len(store_orders), "total": len(out),
        })
        return out
    except Exception as e:
        logger.debug("store_merge_skipped", extra={"error": str(e)[:150]})
        return df


def _try_tianchi(since_date: str | None = None) -> pd.DataFrame | None:
    """Tier 2.5: 京东 JData 数据源(仅 DATA_SOURCE=tianchi 时进降级链)。"""
    return load_tianchi_orders(since_date=since_date)


def data_fingerprint() -> str:
    """订单数据变化指纹(纯函数,Watcher 增量重算用)。

    覆盖三类变化源:
      1. 数据源切换(DATA_SOURCE);
      2. JData CSV 文件更新(mtime);
      3. 共享存储运行时订单(商城下单)—— 数量 + 最新订单时间。

    读 data_store 私有 _orders(与 _merge_store_orders 同理由:避免懒播种
    期间调用公开入口触发递归)。指纹不变 → 流水线无需强制重算(TTL 兜底)。
    """
    import hashlib

    settings = get_settings()
    parts = [f"src={settings.DATA_SOURCE}"]

    if settings.DATA_SOURCE == "tianchi":
        try:
            found = _find_jdata_files(settings.TIANCHI_DATA_DIR)
            mt = [os.path.getmtime(os.path.join(settings.TIANCHI_DATA_DIR, f))
                  for f in found["action"] if f]
            if mt:
                parts.append(f"csv={max(mt):.0f}")
        except OSError:
            pass
        # 滚动揭晓进度:窗口增长 = 数据变化(只读状态、不推进;不含
        # last_advanced —— 稳态后日期照走,避免每天白触发一次全量重算)
        if settings.TIANCHI_REVEAL_ENABLED:
            try:
                from pipeline import reveal_state
                st = reveal_state.load_state()
                parts.append(f"reveal={st['revealed_days']}/{st['total_days']}")
            except Exception:
                pass

    try:
        import api.data_store as ds
        parts.append(f"orders={len(ds._orders)}")
        if ds._orders:
            parts.append(f"last={ds._orders[-1].get('created_at', '')}")
    except Exception:
        pass

    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:16]


# ── Tier 3: mock data ───────────────────────────────────────────


def generate_mock_orders() -> pd.DataFrame:
    """Generate synthetic orders for development / demo.

    Phase 3: includes product_id, category, and unit_price for
    extended profile computation.
    """
    # ── Random seed ──
    # MOCK_SEED 固定时数据可复现(默认 42):Watcher 事件反映真实业务变化而非随机噪声;
    # 设为 None 时每次生成不同数据(演示随机性用)。
    settings = get_settings()
    np.random.seed(settings.MOCK_SEED if settings.MOCK_SEED is not None else None)

    n_users = np.random.randint(120, 350)
    n_orders = np.random.randint(600, 2000)

    # Products with categories
    products = pd.DataFrame({
        'product_id': range(1, 21),
        'product_name': [
            '笔记本电脑', '鼠标', '图书', '耳机', '键盘',
            '显示器', '台灯', '办公椅', '笔记本', '钢笔套装',
            '背包', '水杯', 'U盘', '摄像头', '音箱',
            '平板电脑', '手机壳', '充电器', '运动鞋', 'T恤',
        ],
        'category': [
            '电子产品', '电子产品', '教育用品', '电子产品', '电子产品',
            '电子产品', '家具家居', '家具家居', '教育用品', '教育用品',
            '时尚服饰', '生活用品', '电子产品', '电子产品', '电子产品',
            '电子产品', '电子产品', '电子产品', '时尚服饰', '时尚服饰',
        ],
        'price': [
            5999, 99, 59, 299, 199,
            1299, 89, 899, 29, 49,
            159, 39, 79, 249, 399,
            2599, 49, 69, 329, 79,
        ],
    })

    users = pd.DataFrame({
        'user_id': range(1, n_users + 1),
        'reg_date': pd.date_range('2020-01-01', periods=n_users, freq='D'),
        'age': np.random.randint(18, 65, n_users),
        'city': np.random.choice(
            ['北京', '上海', '广州', '深圳', '杭州', '成都'], n_users
        ),
    })

    # Orders with realistic inter-order intervals per user
    orders = pd.DataFrame({
        'order_id': range(1, n_orders + 1),
        'user_id': np.random.choice(users['user_id'], n_orders),
        'order_date': pd.date_range('2021-01-01', periods=n_orders, freq=f'{np.random.randint(2,12)}h'),
        'total_amount': np.random.uniform(30, 5000, n_orders),
        'quantity': np.random.randint(1, 5, n_orders),
    })

    # Assign random products
    orders['product_id'] = np.random.choice(products['product_id'], n_orders)

    # Merge products and users
    result = orders.merge(products, on='product_id').merge(users, on='user_id')
    return result


def _try_mock_data() -> pd.DataFrame:
    """Tier 3: Built-in mock dataset, enriched with shared-store orders.

    Merges orders from the in-memory data store (created via debug console
    or Vue ecommerce) into the mock dataset so charts reflect real user activity.
    """
    logger.info("tier3_fallback")
    base = generate_mock_orders()

    # ── Merge shared-store orders so charts reflect actual activity ──
    try:
        from api.data_store import list_orders as ds_orders
        store_orders = ds_orders()
        if store_orders:
            extra_rows = []
            for o in store_orders:
                extra_rows.append({
                    "order_id": o.get("order_id", 0),
                    "user_id": o.get("user_id", 1),
                    "product_id": o.get("product_id", o.get("items", [{}])[0].get("product_id", 1) if o.get("items") else 1),
                    "product_name": o.get("product_name", "未知"),
                    "category": "电子产品",
                    "price": o.get("total_amount", 99),
                    "order_date": pd.Timestamp(o.get("created_at", pd.Timestamp.now())),
                    "total_amount": o.get("total_amount", 99),
                    "quantity": o.get("quantity", 1),
                    "unit_price": o.get("total_amount", 99),
                    "reg_date": pd.Timestamp.now() - pd.Timedelta(days=365),
                    "age": 30,
                    "city": "北京",
                })
            if extra_rows:
                extra_df = pd.DataFrame(extra_rows)
                base = pd.concat([base, extra_df], ignore_index=True)
                logger.info("mock_enriched_with_store_orders", extra={
                    "store_orders": len(store_orders),
                    "total_rows": len(base),
                })
    except Exception as e:
        logger.debug("store_merge_skipped", extra={"error": str(e)})

    return base


# ── Public API ──────────────────────────────────────────────────


def load_orders_with_join() -> pd.DataFrame:
    """降级链:MySQL → TTL 缓存 → (京东 JData,可选) → mock。"""
    settings = get_settings()
    tiers = [
        ("MySQL", _try_mysql_joined),
        ("cache", _try_cache_joined),
    ]
    if settings.DATA_SOURCE == "tianchi":
        tiers.append(("tianchi", _try_tianchi))
    tiers.append(("mock", _try_mock_data))
    for name, loader in tiers:
        try:
            result = loader()
            if result is None:
                # Tier 2 can return None on cache miss → skip to next tier
                logger.debug("tier_skipped", extra={"tier": name})
                continue
            # 成本观测:记录本轮实际选中的降级层(best-effort)
            try:
                from llm.metrics import record_data_tier
                record_data_tier(name, len(result))
            except Exception:
                pass
            return result
        except (DatabaseError, ComputationError) as e:
            logger.warning("tier_failed", extra={"tier": name, "error": str(e)})
            if name == "mock":
                raise
    raise DatabaseError("All three data tiers failed")


def load_new_orders_since(since_date: str) -> pd.DataFrame:
    """Load only orders newer than ``since_date`` (ISO format).

    Three-tier fallback; Tier 2 cache is bypassed for delta loads.
    """
    settings = get_settings()
    tiers = [
        ("MySQL", lambda: _try_mysql_joined(since_date=since_date)),
    ]
    if settings.DATA_SOURCE == "tianchi":
        tiers.append(("tianchi", lambda: _try_tianchi(since_date=since_date)))
    tiers.append(("mock", _try_mock_data))
    for name, loader in tiers:
        try:
            df = loader()
            logger.info("delta_loaded", extra={"since": since_date, "rows": len(df)})
            try:
                from llm.metrics import record_data_tier
                record_data_tier(name, len(df))
            except Exception:
                pass
            return df
        except (DatabaseError, ComputationError) as e:
            logger.warning("tier_failed", extra={"tier": name, "error": str(e)})
            if name == "mock":
                raise
    raise DatabaseError("All data tiers failed for delta load")


def load_rfm_from_db():
    """Compute RFM directly in MySQL via GROUP BY (server-side aggregation)."""
    engine = get_db_engine()
    try:
        query = text("""
            SELECT
                user_id,
                MAX(order_date) AS last_order_date,
                DATEDIFF(CURDATE(), MAX(order_date)) AS recency,
                COUNT(DISTINCT order_id) AS frequency,
                SUM(total_amount) AS monetary
            FROM orders
            GROUP BY user_id
        """)
        df = pd.read_sql(query, engine)
        engine.dispose()
        return df
    except (SQLAlchemyError, OSError) as e:
        engine.dispose()
        raise DatabaseError(
            f"RFM SQL query failed: {e}", context={"query": "load_rfm_from_db"}
        ) from e
