---
name: get_funnel_analysis
description: "获取最近N天 浏览→加购→下单 转化漏斗:每步去重用户数与逐级转化率。适用于'加购到下单的流失率''转化率为什么这么低''漏斗'类问题。参数 days: 窗口天数(1-90),默认7。"
group: analysis
kind: tool
handler: funnel.FunnelAnalysisSkill
version: 1.0.0
enabled: true
tags: [漏斗, 转化率, 流失, 加购, 下单]
---
## 使用说明
- 返回步骤:浏览 → 加购 → 下单,每步 user_count / conversion_rate;
- summary 自动定位最大流失环节(流失率最高台阶),优先解读;
- 数据带来源与时间窗口标注,引用时如实说明口径。

## 示例
Q: 加购到下单的流失率是多少 → get_funnel_analysis(days=7)
Q: 转化率为什么这么低 → 本工具 + 流失环节解读
