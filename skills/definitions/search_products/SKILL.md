---
name: search_products
description: "搜索电商平台商品。根据关键词匹配商品名称，或按品类筛选。返回匹配的商品列表（含ID、名称、品类、价格）。适用于用户想找特定商品、浏览某品类、或对比同类商品价格的场景。参数 keyword: 商品名关键词（如'手机'）; category: 品类名（如'电子产品'）"
group: shopping
kind: tool
handler: user_segment.ProductSearchSkill
version: 1.0.0
enabled: true
tags: [商品, 搜索, 购物, 价格, 对比]
---
## 使用说明
- keyword 与 category 可组合;keyword 先匹配商品名,无结果时降级匹配品类名;
- 返回字段:product_id / product_name / category / price;
- 结果按品类分布给出摘要,LLM 挑 3-5 件推荐并说明理由。

## 示例
Q: 有没有蓝牙耳机 → search_products(keyword="蓝牙耳机")
Q: 500 以内的电子产品 → search_products(category="电子产品") + 价格筛选后解读
