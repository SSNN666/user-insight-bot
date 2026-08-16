"""京东 JData 数据源:canonical 映射纯函数 + 文件加载 + 降级链接入。

全部离线,不下载真实数据集(用合成 CSV 模拟 JData 表结构,含官方浮点 ID 口径)。
"""
import pandas as pd
import pytest

from pipeline import data_loader as dl
from errors.exceptions import DatabaseError


def _users_df() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [101, 102, 103],
        "age": ["26-35岁", "36-45岁", "-1"],   # 官方口径:分桶字符串,-1=未知
        "sex": [0, 1, 2],
        "user_lv_cd": [1, 3, 2],
        "user_reg_tm": ["2016-01-01", "2016-01-15", ""],
    })


def _actions_df() -> pd.DataFrame:
    return pd.DataFrame({
        # 官方口径:Action 表 user_id/sku_id 为浮点("1.0"),加载时归一化为 int
        "user_id": [101.0, 101.0, 102.0, 103.0, 103.0],
        "sku_id": [9001.0, 9002.0, 9001.0, 9003.0, 9004.0],
        "time": ["2016-02-01 10:00:00", "2016-02-02 10:00:00",
                 "2016-02-03 10:00:00", "2016-02-04 10:00:00",
                 "2016-02-05 10:00:00"],
        "model_id": [0] * 5,
        "type": [1, 4, 4, 4, 4],      # 官方编码:1=浏览 4=下单;第 1 行浏览被过滤
        "cate": [8, 8, 4, 8, 5],
        "brand": [100, 100, 200, 300, 400],
    })


class TestBuildOrdersFromJdata:
    def test_order_actions_mapped_to_canonical(self):
        df = dl._build_orders_from_jdata(_users_df(), _actions_df())
        assert df is not None
        assert len(df) == 4                       # 浏览行为被过滤
        assert list(df["order_id"]) == [1, 2, 3, 4]   # 按时间排序自增
        assert (df["quantity"] == 1).all()
        assert (df["total_amount"] == df["unit_price"]).all()
        assert (df["total_amount"] >= 50).all() and (df["total_amount"] <= 5000).all()
        assert df.iloc[0]["category"] == "8"      # 分类保留原编码
        assert df.iloc[0]["product_name"] == "SKU-9002"

    def test_user_attributes_mapped(self):
        df = dl._build_orders_from_jdata(_users_df(), _actions_df())
        gender_by_user = dict(zip(df["user_id"], df["gender"]))
        assert gender_by_user[101] == "男"
        assert gender_by_user[102] == "女"
        assert gender_by_user[103] == "保密"
        assert df[df["user_id"] == 101]["age"].iloc[0] == "26-35岁"  # 分桶原样保留
        assert df[df["user_id"] == 103]["age"].iloc[0] == "未知"     # -1 → 未知
        assert df[df["user_id"] == 103]["reg_date"].isna().all()   # 空注册时间 → NaT

    def test_no_order_actions_returns_none(self):
        actions = _actions_df()
        actions["type"] = 1                        # 全部浏览
        assert dl._build_orders_from_jdata(_users_df(), actions) is None

    def test_empty_input_returns_none(self):
        assert dl._build_orders_from_jdata(pd.DataFrame(), _actions_df()) is None
        assert dl._build_orders_from_jdata(_users_df(), pd.DataFrame()) is None


class TestSynthPrice:
    def test_deterministic_across_calls(self):
        """JData 无价格字段:同一(品类,品牌)合成价格跨调用稳定(快照对比依赖)。"""
        assert dl._synth_price("8", 100) == dl._synth_price("8", 100)

    def test_within_price_range(self):
        for cate, brand in [("4", 200), ("11", 12345), (None, None)]:
            assert 50 <= dl._synth_price(cate, brand) <= 5000


class TestLoadTianchiOrders:
    def _write_jdata_files(self, tmp_path, data_dir: str) -> None:
        import os
        os.makedirs(data_dir, exist_ok=True)
        _users_df().to_csv(os.path.join(data_dir, "JData_User.csv"), index=False)
        _actions_df().to_csv(os.path.join(data_dir, "JData_Action_201602.csv"),
                             index=False)

    def test_loads_from_csv_files(self, tmp_path, monkeypatch):
        data_dir = str(tmp_path / "tianchi")
        self._write_jdata_files(tmp_path, data_dir)
        monkeypatch.setenv("TIANCHI_DATA_DIR", data_dir)
        from config.settings import get_settings
        get_settings.cache_clear()

        df = dl.load_tianchi_orders()
        assert df is not None and len(df) == 4
        assert {"order_id", "user_id", "product_name", "category", "gender",
                "order_date", "total_amount"} <= set(df.columns)

    def test_missing_dir_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TIANCHI_DATA_DIR", str(tmp_path / "not_exist"))
        from config.settings import get_settings
        get_settings.cache_clear()
        assert dl.load_tianchi_orders() is None

    def test_gbk_encoded_user_file(self, tmp_path, monkeypatch):
        """官方 JData_User.csv 实测为 GBK 编码 → 加载器自动回退解码。"""
        import os
        data_dir = str(tmp_path / "tianchi")
        os.makedirs(data_dir, exist_ok=True)
        users = _users_df()
        users.loc[0, "user_reg_tm"] = "2016-01-01 注册备注"
        users.to_csv(os.path.join(data_dir, "JData_User.csv"),
                     index=False, encoding="gbk")
        _actions_df().to_csv(os.path.join(data_dir, "JData_Action_201602.csv"),
                             index=False)
        monkeypatch.setenv("TIANCHI_DATA_DIR", data_dir)
        from config.settings import get_settings
        get_settings.cache_clear()

        df = dl.load_tianchi_orders()
        assert df is not None and len(df) == 4
        assert (df["gender"] == "男").any()          # join 成功,GBK 解码无损

    def test_since_date_filters(self, tmp_path, monkeypatch):
        data_dir = str(tmp_path / "tianchi")
        self._write_jdata_files(tmp_path, data_dir)
        monkeypatch.setenv("TIANCHI_DATA_DIR", data_dir)
        from config.settings import get_settings
        get_settings.cache_clear()

        df = dl.load_tianchi_orders(since_date="2016-02-04 12:00")
        assert df is not None and len(df) == 1       # 只剩 02-05 那一单


class TestFallbackChainIntegration:
    def _setup(self, tmp_path, monkeypatch, data_source: str) -> str:
        """写 JData 文件 + 切断 MySQL/缓存两层,返回数据目录。"""
        data_dir = str(tmp_path / "tianchi")
        import os
        os.makedirs(data_dir, exist_ok=True)
        _users_df().to_csv(os.path.join(data_dir, "JData_User.csv"), index=False)
        _actions_df().to_csv(os.path.join(data_dir, "JData_Action_201602.csv"),
                             index=False)

        def _boom():
            raise DatabaseError("mysql down")
        monkeypatch.setattr(dl, "_try_mysql_joined", _boom)
        monkeypatch.setattr(dl, "_try_cache_joined", lambda: None)
        monkeypatch.setenv("DATA_SOURCE", data_source)
        monkeypatch.setenv("TIANCHI_DATA_DIR", data_dir)
        from config.settings import get_settings
        get_settings.cache_clear()
        return data_dir

    def test_tianchi_mode_inserts_jdata_tier(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch, data_source="tianchi")
        df = dl.load_orders_with_join()
        assert df["product_name"].str.startswith("SKU-").any()      # JData 数据特征
        assert df["category"].astype(str).str.isdigit().all()       # 数字编码分类
        assert "gender" in df.columns

    def test_auto_mode_skips_jdata_falls_to_mock(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch, data_source="auto")
        df = dl.load_orders_with_join()
        assert df["category"].astype(str).str.contains(
            "电子|教育|时尚|家居|生活").any()                        # mock 中文分类
