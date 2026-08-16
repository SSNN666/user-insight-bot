"""skills/product_image 单测:文件缺失 / 无多模态配置的降级路径(不发真实请求)。"""
import pytest

from skills.product_image import ProductImageSkill
from skills.base import SkillStatus


class TestProductImageSkill:
    def test_missing_file(self):
        skill = ProductImageSkill()
        r = skill.execute(image_path="不存在的路径.jpg")
        assert r.status == SkillStatus.MISSING

    def test_no_vision_provider(self, tmp_path, monkeypatch):
        from config.settings import get_settings
        # 云 Key 置空 → vision 链无可用供应商
        monkeypatch.setattr(get_settings(), "DASHSCOPE_API_KEY", "")
        monkeypatch.setattr(get_settings(), "LLM_LOCAL_FALLBACK_ENABLED", False)
        from llm.bridge import reset_fallback_chains
        reset_fallback_chains()

        img = tmp_path / "product.jpg"
        img.write_bytes(b"\xff\xd8\xff fake jpeg")

        skill = ProductImageSkill()
        r = skill.execute(image_path=str(img))
        assert r.status == SkillStatus.MISSING
        assert "多模态" in r.summary
        reset_fallback_chains()
