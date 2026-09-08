"""商品搜索 token 级匹配测试(中文复合词 vs 合成商品名)。

背景:商品名为词库合成("耳机-36692"),LLM 常拿整串复合词
("降噪耳机")搜索 → 子串不命中。token 兜底:任一词命中即可。
"""
import pytest

from skills.user_segment import ProductSearchSkill

_PRODUCTS = [
    {"product_id": 1, "product_name": "耳机-9001", "category": "影音数码", "price": 199.0},
    {"product_id": 2, "product_name": "音箱-9002", "category": "影音数码", "price": 399.0},
    {"product_id": 3, "product_name": "运动鞋-9003", "category": "时尚服饰", "price": 159.0},
]


@pytest.fixture
def search_skill(monkeypatch):
    monkeypatch.setattr(
        "api.data_store.list_products", lambda: _PRODUCTS)
    return ProductSearchSkill()


class TestSearchTokenMatching:
    def test_full_phrase_substring_hits(self, search_skill):
        r = search_skill.execute(keyword="耳机")
        assert len(r.data) == 1 and r.data[0]["product_id"] == 1

    def test_compound_word_falls_back_to_token(self, search_skill):
        """整串"降噪耳机"不命中 → token 级(降噪/耳机任一词)命中耳机商品。"""
        r = search_skill.execute(keyword="降噪耳机")
        assert r.status.value == "success" or r.data
        assert any(p["product_id"] == 1 for p in r.data)

    def test_unrelated_word_returns_empty(self, search_skill):
        r = search_skill.execute(keyword="航天飞机")
        assert r.data == []

    def test_category_filter_still_applies(self, search_skill):
        r = search_skill.execute(keyword="耳机", category="时尚服饰")
        assert r.data == []

    def test_category_only_search(self, search_skill):
        r = search_skill.execute(category="影音数码")
        assert len(r.data) == 2
