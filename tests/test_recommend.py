"""skills/recommend 单测:画像驱动打分(无画像兜底 / 偏好品类优先)。"""
import pytest

from skills.recommend import PersonalRecommendationSkill
from skills.base import SkillStatus


class TestPersonalRecommendation:
    def test_no_profile_fallback(self):
        """无任何画像 → 按热度兜底,仍返回商品列表。"""
        skill = PersonalRecommendationSkill()
        r = skill.execute(user_id=999999)   # 不存在/无偏好用户
        assert r.status == SkillStatus.SUCCESS
        assert r.data and len(r.data) >= 1
        assert all("product_id" in d and "product_name" in d for d in r.data)
        assert r.confidence == 0.5

    def test_preferred_category_first(self, monkeypatch):
        """用户偏好品类「电子产品」→ 推荐首位应为该品类。"""
        import api.data_store as ds
        user_id = 999998
        monkeypatch.setitem(
            ds._users_by_id, user_id,
            {"user_id": user_id, "age": 30, "city": "北京",
             "preferences": {"category": "电子产品", "tags": ["耳机"]}},
        )
        skill = PersonalRecommendationSkill()
        r = skill.execute(user_id=user_id)
        assert r.status == SkillStatus.SUCCESS
        assert r.data[0]["category"] == "电子产品"
        assert r.confidence == 0.85
        # 标签命中加成:「耳机」商品应进入推荐
        names = [d["product_name"] for d in r.data]
        assert any("耳机" in n for n in names)

    def test_empty_product_library(self, monkeypatch):
        """商品库为空 → PARTIAL 明确提示。"""
        import api.data_store as ds
        monkeypatch.setattr(ds, "_products_by_id", {})
        skill = PersonalRecommendationSkill()
        r = skill.execute(user_id=1)
        assert r.status == SkillStatus.PARTIAL
        assert r.data == []
