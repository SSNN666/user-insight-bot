"""商品图像解析 Skill(VL 扩展)— qwen3-vl-plus 识别品类/外观特征。

职责边界(与康养项目同一口径):
  - VL 只做**图像内容理解**(品类/外观/颜色/材质猜测 → 结构化标签),不做文字提取;
  - 主 Agent 仍由文本模型驱动;本 Skill 由上传接口直接调用(文件路径只能由服务端落盘后提供),
    不进 Agent 的工具绑定名单。
标签结果存入用户画像(data_store preferences),支撑个性化推荐。
"""

import os

from pydantic import BaseModel, Field

from skills.base import BaseSkill, SkillResult, SkillStatus
from log.logger import get_logger

logger = get_logger(__name__)

MAX_IMAGE_BYTES = 8 * 1024 * 1024   # 与康养 /v1/vision 同口径

VISION_PROMPT = """这是一张商品图片。请识别并输出结构化标签(严格 JSON):
{
  "category": "商品品类(如 电子产品/时尚服饰/家具家居/生活用品/教育用品)",
  "appearance": "外观特征一句话描述",
  "color": "主色调",
  "material_guess": "材质猜测(不确定写'不确定')",
  "tags": ["2-5 个关键词标签"]
}
只输出 JSON 对象,不要 Markdown、代码块或解释文字。"""

_DEFAULT_TAGS = {
    "category": "未知", "appearance": "", "color": "",
    "material_guess": "不确定", "tags": [],
}


class ProductImageInput(BaseModel):
    image_path: str = Field(..., description="本地图片文件路径(由上传接口落盘后传入)")
    mime: str = Field(default="image/jpeg", description="图片 MIME 类型")


class ProductImageSkill(BaseSkill):
    name = "analyze_product_image"
    description = (
        "解析商品图片,输出结构化标签(品类/外观特征/颜色/材质猜测),"
        "用于用户画像补充与个性化推荐。需要本地图片路径参数。"
    )
    input_schema = ProductImageInput

    def execute(self, image_path: str = "", mime: str = "image/jpeg") -> SkillResult:
        try:
            if not image_path or not os.path.isfile(image_path):
                return SkillResult(
                    status=SkillStatus.MISSING, data=[],
                    summary="图片文件不存在,请先上传图片。", confidence=0.0,
                )
            with open(image_path, "rb") as f:
                data = f.read()
            if len(data) > MAX_IMAGE_BYTES:
                return SkillResult(
                    status=SkillStatus.ERROR, data=[],
                    error="图片过大(>8MB)", confidence=0.0,
                )

            from llm.bridge import _get_chain
            chain = _get_chain("vision")
            if not chain.active_providers:
                return SkillResult(
                    status=SkillStatus.MISSING, data=[],
                    summary="未配置多模态模型(需 DASHSCOPE_API_KEY 启用 qwen3-vl-plus)。",
                    confidence=0.0,
                )

            resp = chain.invoke_vision(data, VISION_PROMPT, mime=mime)
            if resp.error_kind or not resp.content:
                return SkillResult(
                    status=SkillStatus.ERROR, data=[],
                    error=resp.error_kind or "no_vision_provider", confidence=0.0,
                )

            from common.json_repair import load_json_or_default
            tags, repair = load_json_or_default(
                resp.content, default=_DEFAULT_TAGS,
                hint='{"category":"...","appearance":"...","color":"...","material_guess":"...","tags":[...]}',
                logger=logger,
            )
            status = SkillStatus.SUCCESS if repair.ok else SkillStatus.PARTIAL
            return SkillResult(
                status=status,
                data=tags,
                summary=(
                    f"图像解析完成: 品类={tags.get('category', '未知')} | "
                    f"外观={tags.get('appearance', '')[:60]} | "
                    f"标签={', '.join(tags.get('tags', []))}"
                ),
                confidence=0.85 if repair.ok else 0.4,
            )
        except Exception as e:
            logger.error("product_image_skill_error", extra={"error": str(e)})
            return SkillResult(status=SkillStatus.ERROR, error=str(e), confidence=0.0)
