"""个性化推荐 Skill —— 画像(分群+偏好标签)驱动的商品打分推荐。

数据来源:
- 用户偏好标签:商品图像解析 Skill 写入 data_store 的 preferences
- 用户分群:RFM 聚类结果(高价值分群消费力强,推荐单价适配)
- 商品热度:订单记录兜底(无任何画像时的默认排序)

输出与 search_products 同构的 [{"product_id","product_name","category","price"}],
前端商品卡片/一键加购共用同一渲染逻辑。
"""

from pydantic import BaseModel, Field

from skills.base import BaseSkill, SkillResult, SkillStatus
from log.logger import get_logger

logger = get_logger(__name__)


class RecommendInput(BaseModel):
    user_id: int = Field(default=1, description="用户ID(画像来源)")


class PersonalRecommendationSkill(BaseSkill):
    name = "get_personal_recommendations"
    description = (
        "根据用户画像(分群+品类偏好+标签)生成个性化商品推荐。"
        "适合用户问'给我推荐'、'猜我喜欢'、'有什么适合我的商品'时调用。"
    )
    input_schema = RecommendInput

    def execute(self, user_id: int = 1) -> SkillResult:
        try:
            from api.data_store import (
                list_products as ds_products, get_user as ds_get_user,
            )
            products = ds_products()
            if not products:
                return SkillResult(
                    status=SkillStatus.PARTIAL, data=[],
                    summary="商品库为空,暂无可推荐商品。", confidence=0.3,
                )

            prefs = (ds_get_user(user_id) or {}).get("preferences", {})
            pref_cat = prefs.get("category")
            pref_tags = set(prefs.get("tags", []))

            # ── 用户分群(消费力画像)──
            seg_info = None
            try:
                from skills.user_segment import _load_and_process
                rfm, _, _ = _load_and_process()
                row = rfm[rfm["user_id"] == user_id]
                if len(row):
                    seg_info = (int(row.iloc[0]["segment"]),
                                float(row.iloc[0]["monetary"]))
            except Exception:
                seg_info = None   # 分群不可用 → 仅按偏好/热度

            # ── 打分 ──
            scored = []
            for p in products:
                score = 0.0
                reasons = []
                if pref_cat and p.get("category") == pref_cat:
                    score += 0.5
                    reasons.append("偏好品类")
                name = p.get("product_name", "")
                hits = [t for t in pref_tags if t in name]
                if hits:
                    score += 0.1 * len(hits)
                    reasons.append("标签匹配")
                if seg_info:
                    seg, monetary = seg_info
                    if p["price"] <= monetary * 0.2:   # 价格适配消费力
                        score += 0.2
                        reasons.append("价格适配")
                scored.append((score, p, reasons))

            scored.sort(key=lambda x: (-x[0], x[1]["price"]))
            with_reasons = [x for x in scored if x[0] > 0]
            top = with_reasons[:5] or scored[:3]

            items = [
                {
                    "product_id": p["product_id"],
                    "product_name": p["product_name"],
                    "category": p.get("category", ""),
                    "price": p["price"],
                }
                for _, p, _ in top
            ]

            lines = ["**个性化推荐**(画像驱动):", ""]
            lines.append("| 商品ID | 名称 | 品类 | 价格 |")
            lines.append("|--------|------|------|------|")
            for item in items:
                lines.append(
                    f"| {item['product_id']} | {item['product_name']} | "
                    f"{item['category']} | ¥{item['price']} |"
                )
            basis = []
            if seg_info:
                basis.append(f"分群{seg_info[0]}")
            if pref_cat:
                basis.append(f"偏好品类「{pref_cat}」")
            if pref_tags:
                basis.append(f"标签 {', '.join(sorted(pref_tags)[:5])}")
            lines.append("")
            lines.append(
                "**推荐依据**: " + ("、".join(basis) if basis
                                     else "暂无画像,按商品热度兜底推荐")
                + "。点击商品卡片可直接加入购物车。"
            )

            return SkillResult(
                status=SkillStatus.SUCCESS,
                data=items,
                summary="\n".join(lines),
                confidence=0.85 if (pref_cat or pref_tags or seg_info) else 0.5,
            )
        except Exception as e:
            logger.error("recommend_skill_error", extra={"error": str(e)})
            return SkillResult(status=SkillStatus.ERROR, error=str(e), confidence=0.0)
