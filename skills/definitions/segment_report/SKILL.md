---
name: segment_report
description: 生成一份完整的用户分群综合分析报告:先取分群统计与规则,再补环比增长与时间趋势,最后给出分群诊断与运营建议。
group: analysis
kind: instruction
version: 1.0.0
enabled: true
tags: [报告, 综合分析, 全览]
---
## 使用说明
本 Skill 是纯指令型(无绑定工具),命中时其指令注入 LLM 上下文,指导工具编排:

1. 先调用 `get_user_segment_stats` 拿分群全貌(人数/近度/频次/消费);
2. 再调用 `get_segment_rules` 解释分群划分依据;
3. 如快照充足,补 `get_segment_growth` 与 `get_segment_trend` 看环比与走势;
4. 输出结构:分群概览表 → 划分规则 → 增长与趋势 → 诊断与运营建议;
5. 所有数值必须来自工具返回,禁止编造;数据不足的维度如实标注。

## 示例
Q: 给我一份用户分群分析报告 → 依次编排 stats/rules/growth/trend 四个工具后汇总
