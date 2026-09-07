---
name: get_personal_recommendations
description: 根据用户画像(分群+品类偏好+标签)生成个性化商品推荐。适合用户问'给我推荐'、'猜我喜欢'、'有什么适合我的商品'时调用。
group: analysis
kind: tool
handler: recommend.PersonalRecommendationSkill
version: 1.0.0
enabled: true
tags: [推荐, 画像, 偏好, 个性化]
---
## 使用说明
- 打分因子:偏好品类(0.5) + 标签命中(0.1/个) + 价格适配消费力(0.2);
- 无画像时按商品热度兜底,confidence 降低(0.5),回答时如实说明依据;
- 参数 user_id 默认 1;AI 导购场景务必传入当前登录用户。

## 示例
Q: 给我推荐点东西 → get_personal_recommendations(user_id=当前用户)
