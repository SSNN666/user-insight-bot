---
name: get_segment_trend
description: 获取各分群人数随时间的变化趋势(历史快照时间序列)。返回每个快照时间点各分群的人数与平均消费。适用于'分群人数趋势''最近分群有什么变化'类问题。
group: analysis
kind: tool
handler: user_segment.SegmentTrendSkill
version: 1.0.0
enabled: true
tags: [趋势, 时间, 走势, 变化]
---
## 使用说明
- 返回近 10 个快照的结构化时序行(timestamp/segment/user_count/avg_monetary),前端折线图直接消费;
- 摘要给出首末快照各分群人数变化,带业务命名;
- 快照不足 2 个时返回 PARTIAL 提示,等 Watcher 轮询后再问。

## 示例
Q: 最近分群有什么变化 → get_segment_trend() 后按分群解读走势
