---
name: get_categories
description: 获取平台所有商品品类列表。在用户不确定具体商品名、想浏览某个品类时优先调用此工具。
group: shopping
kind: tool
handler: user_segment.GetCategoriesSkill
version: 1.0.0
enabled: true
tags: [品类, 浏览, 分类]
---
## 使用说明
- 返回品类 → 件数映射,按件数降序;
- 用户没有明确需求时优先调用本工具展示品类,引导用户选择。

## 示例
Q: 你们有什么卖 → get_categories() 后按品类引导
