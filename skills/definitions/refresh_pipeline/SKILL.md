---
name: refresh_pipeline
description: 刷新数据流水线缓存。强制重新加载数据、重算RFM、重新聚类。在数据更新后使用此工具获取最新分群结果。
group: analysis
kind: tool
handler: user_segment.RefreshPipelineSkill
version: 1.0.0
enabled: true
tags: [刷新, 重算, 缓存, 最新]
---
## 使用说明
- 全量重算(RFM + 聚类 + 决策树 + 快照),耗时较慢,仅在数据变更确认后使用;
- 会联动清空漏斗行为缓存,保证全链路数据一致。

## 示例
Q: 数据更新了吗,重新算一下 → refresh_pipeline() 后重新查询
