"""chart_extract:Skill 结构化数据 → 前端图表协议(纯函数,零渲染)。"""
from agent.chart_extract import build_charts_from_skills
from skills.base import SkillResult, SkillStatus

STATS_ROWS = [
    {"segment": 0, "用户数": 1852, "平均近度": 46.9, "平均频次": 1.09, "平均消费": 2478.82},
    {"segment": 1, "用户数": 746, "平均近度": 1.3, "平均频次": 1.07, "平均消费": 2591.91},
    {"segment": 2, "用户数": 262, "平均近度": 37.2, "平均频次": 2.78, "平均消费": 7678.67},
]

TREND_ROWS = [
    {"timestamp": "2016-02-01 10:00", "segment": 0, "user_count": 60},
    {"timestamp": "2016-02-01 10:00", "segment": 2, "user_count": 10},
    {"timestamp": "2016-02-02 10:00", "segment": 0, "user_count": 61},
    {"timestamp": "2016-02-02 10:00", "segment": 2, "user_count": 11},
]

NAMES = {0: "低活长尾", 1: "新晋活跃", 2: "高价值核心"}


def _sr(data):
    return SkillResult(status=SkillStatus.SUCCESS, data=data, summary="t", confidence=0.9)


class TestBarChart:
    def test_stats_rows_become_bar_with_names(self):
        charts = build_charts_from_skills([_sr(STATS_ROWS)], NAMES)
        assert len(charts) == 1
        c = charts[0]
        assert c["type"] == "bar"
        assert c["categories"] == ["分群0·低活长尾", "分群1·新晋活跃", "分群2·高价值核心"]
        assert [s["name"] for s in c["series"]] == ["用户数", "平均消费(元)"]
        assert c["series"][0]["data"] == [1852, 746, 262]
        assert c["series"][1]["data"] == [2478.82, 2591.91, 7678.67]

    def test_fallback_label_without_names(self):
        charts = build_charts_from_skills([_sr(STATS_ROWS)], {})
        assert charts[0]["categories"][0] == "分群0"


class TestLineChart:
    def test_trend_rows_become_line(self):
        charts = build_charts_from_skills([_sr(TREND_ROWS)], NAMES)
        assert len(charts) == 1
        c = charts[0]
        assert c["type"] == "line"
        assert c["categories"] == ["2016-02-01 10:00", "2016-02-02 10:00"]
        assert len(c["series"]) == 2                     # 两个分群
        assert c["series"][0]["name"] == "分群0·低活长尾"
        assert c["series"][0]["data"] == [60, 61]

    def test_missing_point_becomes_none(self):
        rows = [
            {"timestamp": "t1", "segment": 0, "user_count": 60},
            {"timestamp": "t1", "segment": 1, "user_count": 20},
            {"timestamp": "t2", "segment": 0, "user_count": 61},   # 分群1 在 t2 无数据
        ]
        charts = build_charts_from_skills([_sr(rows)], {})
        assert charts[0]["series"][1]["data"] == [20, None]


class TestCombination:
    def test_stats_then_trend_gives_two_charts(self):
        charts = build_charts_from_skills([_sr(STATS_ROWS), _sr(TREND_ROWS)], NAMES)
        assert [c["type"] for c in charts] == ["bar", "line"]

    def test_product_rows_ignored(self):
        products = [{"product_id": 1, "product_name": "耳机", "price": 299}]
        assert build_charts_from_skills([_sr(products)], {}) == []

    def test_categories_capped(self):
        rows = [{"timestamp": f"t{i}", "segment": 0, "user_count": i}
                for i in range(30)]
        c = build_charts_from_skills([_sr(rows)], {})[0]
        assert len(c["categories"]) <= 24


class TestFunnelChart:
    FUNNEL_ROWS = [
        {"step": "浏览", "user_count": 1475, "conversion_rate": 1.0},
        {"step": "加购", "user_count": 1056, "conversion_rate": 0.72},
        {"step": "下单", "user_count": 1000, "conversion_rate": 0.95},
    ]

    def test_funnel_rows_become_bar(self):
        charts = build_charts_from_skills([_sr(self.FUNNEL_ROWS)], {})
        assert len(charts) == 1
        c = charts[0]
        assert c["type"] == "bar"
        assert c["title"] == "转化漏斗(浏览→加购→下单)"
        assert c["categories"] == ["浏览", "加购", "下单"]
        assert [s["name"] for s in c["series"]] == ["用户数", "转化率(%)"]
        assert c["series"][0]["data"] == [1475, 1056, 1000]
        assert c["series"][1]["data"] == [100.0, 72.0, 95.0]

    def test_funnel_preferred_over_stats_in_first_slot(self):
        charts = build_charts_from_skills([_sr(self.FUNNEL_ROWS), _sr(STATS_ROWS)], NAMES)
        assert charts[0]["title"].startswith("转化漏斗")

    def test_funnel_then_trend_gives_two_charts(self):
        charts = build_charts_from_skills([_sr(self.FUNNEL_ROWS), _sr(TREND_ROWS)], NAMES)
        assert [c["type"] for c in charts] == ["bar", "line"]
        assert charts[0]["categories"] == ["浏览", "加购", "下单"]
