---
name: get_segment_rules
description: 获取决策树导出的用户分群规则，可帮助理解不同群体的特征。返回文本形式的 if-then 规则。
group: analysis
kind: tool
handler: user_segment.UserSegmentRulesSkill
version: 1.0.0
enabled: true
tags: [规则, 决策树, 边界, 特征, 怎么划分]
---
## 使用说明
- 返回决策树原文(export_text 格式),回答时必须代码块原文引用;
- 禁止改写阈值(如 <= 2.50 写成 <= 3),事实核查 Layer 2 校验阈值一致性;
- 规则之外的特征不要为分群添加条件。

## 示例
Q: 分群是怎么划分的 → 本工具 + 逐条解释边界
