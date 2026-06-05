import pandas as pd
from sqlalchemy import create_engine
from config import DB_CONFIG

def get_db_engine():
    """创建 SQLAlchemy 引擎连接 MySQL"""
    conn_str = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset={DB_CONFIG['charset']}"
    return create_engine(conn_str)

def load_orders_with_join():
    """
    多表查询：订单关联用户和商品，返回宽表 DataFrame
    同时演示分组聚合在 SQL 层面完成部分工作
    """
    engine = get_db_engine()
    query = """
    SELECT 
        o.order_id, o.user_id, u.username, u.reg_date, u.city, u.age, u.gender,
        o.product_id, p.product_name, p.category, p.price AS unit_price,
        o.quantity, o.total_amount, o.order_date
    FROM orders o
    JOIN users u ON o.user_id = u.user_id
    JOIN products p ON o.product_id = p.product_id
    ORDER BY o.order_date
    """
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df

def load_rfm_from_db():
    """
    直接在 MySQL 中计算 RFM 指标，利用分组聚合减少 Python 内存压力
    """
    engine = get_db_engine()
    query = """
    SELECT 
        user_id,
        MAX(order_date) AS last_order_date,
        DATEDIFF(CURDATE(), MAX(order_date)) AS recency,
        COUNT(DISTINCT order_id) AS frequency,
        SUM(total_amount) AS monetary
    FROM orders
    GROUP BY user_id
    """
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df

# 备用：如果无数据库环境，生成模拟数据
def generate_mock_orders():
    import numpy as np
    np.random.seed(42)
    n_users = 200
    n_orders = 1000
    users = pd.DataFrame({
        'user_id': range(1, n_users+1),
        'reg_date': pd.date_range('2020-01-01', periods=n_users, freq='D'),
        'age': np.random.randint(18, 65, n_users),
        'city': np.random.choice(['Beijing','Shanghai','Guangzhou','Shenzhen'], n_users)
    })
    orders = pd.DataFrame({
        'order_id': range(1, n_orders+1),
        'user_id': np.random.choice(users['user_id'], n_orders),
        'order_date': pd.date_range('2021-01-01', periods=n_orders, freq='10h'),
        'total_amount': np.random.uniform(30, 5000, n_orders)
    })
    return orders.merge(users, on='user_id')