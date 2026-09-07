---
name: get_high_value_users
description: 获取高价值用户列表（分群编号最大的群体），返回用户ID和他们的近度、频次、消费金额（RFM）指标以及流转标签。
group: analysis
kind: tool
handler: user_segment.HighValueUsersSkill
version: 1.0.0
enabled: true
tags: [高价值, 用户名单, 召回, 流失]
---
## 使用说明
- 返回字段:user_id / recency / frequency / monetary / flow_tag;
- summary 仅展示前 20 人(上下文有界),完整名单在结构化 data 中;
- 配合流失预警场景:flow_tag=dormant/churned 的用户是召回对象。

## 示例
Q: 高价值用户有哪些 → 本工具,列表 + 流转标签解读
