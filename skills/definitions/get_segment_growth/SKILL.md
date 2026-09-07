---
name: get_segment_growth
description: 获取各用户分群的环比增长指标:销售额增长、转化率变化、GMV提升、人数与平均消费变化。对比两个时间段的快照;默认最近两期,可用 period1/period2 指定月份(如 '2016-02' vs '2016-03')。适用于'2月比3月怎么样'这类环比对比问题。
group: analysis
kind: tool
handler: user_segment.SegmentGrowthSkill
version: 1.1.0
enabled: true
tags: [增长, 环比, 对比, gmv, 转化率, 趋势]
---
## 使用说明
- 默认对比最近两期快照;指定月份时若该月无快照,返回可用月份提示(诚实降级);
- 返回字段:sales_growth_pct / conversion_change_pct / gmv_lift_pct / user_count / avg_monetary / user_count_delta;
- 回答需输出对比表 + 数据说明,百分比带正负号。

## 示例
Q: 2月比3月增长怎么样 → get_segment_growth(period1="2016-02", period2="2016-03")
Q: 最近有什么变化 → get_segment_growth()
