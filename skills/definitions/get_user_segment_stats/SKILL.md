---
name: get_user_segment_stats
description: 获取各用户分群的人数、平均近度、平均频次、平均消费金额。返回分群统计表格。
group: analysis
kind: tool
handler: user_segment.UserSegmentStatsSkill
version: 1.2.0
enabled: true
tags: [分群, 统计, rfm, 人数, 画像]
---
## 使用说明
- 返回字段:segment / user_count / recency_avg / frequency_avg / monetary_avg;
- 数据附带分群业务命名(如"高价值核心用户"),回答时优先使用业务名;
- 数值必须原样引用,禁止四舍五入或编造(事实核查会逐项比对)。

## 示例
Q: 各分群的人数是多少 → 调用本工具,Markdown 表格输出
Q: 高价值用户有哪些特征 → 本工具 + get_segment_rules 结合解读
