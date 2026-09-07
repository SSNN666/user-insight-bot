---
name: analyze_product_image
description: 解析商品图片,输出结构化标签(品类/外观特征/颜色/材质猜测),用于用户画像补充与个性化推荐。需要本地图片路径参数。
group: vision
kind: tool
handler: product_image.ProductImageSkill
version: 1.0.0
enabled: true
tags: [图像, 视觉, 图片]
---
## 使用说明
- ⚠️ 本 Skill 不进 Agent 工具绑定名单(group=vision):LLM 无法获得本地文件路径,
  只能由上传接口(/api/product-image/analyze)落盘后直调;
- VL 只做图像内容理解(品类/外观/颜色/材质),不做文字提取;
- 标签结果写入用户画像 preferences,支撑个性化推荐。
