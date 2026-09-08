# 产品截图拍摄清单(README 展示用)

> 目标:3 张图讲完"这是能跑的产品"。每张 1200px 宽、PNG/JPG ≤ 300KB,
> 存到 `docs/screenshots/`,文件名按下面命名,README 的「产品演示画面」段
> 就绪后改为 `![说明](docs/screenshots/xxx.png)` 即可。

## 开服务

```bash
start.bat            # 或手动: run_api.py + app.py + frontend npm run dev
```

## 1. `chat-steps.png` — AI 导购实时步骤条(最能打的一张)

- 打开 http://localhost:5173 → 切到「AI 导购」页
- 输入:「各分群分别有多少人?」
- 在回答流式输出**进行中**截屏:右侧/下方应有步骤条
  (意图识别 → 模型决策 → 工具调用 → 数据校验 → 事实核查),带进度感

## 2. `gradio-console.png` — 运营分析台全景

- 打开 http://localhost:7860
- 选「分析」Tab,问「高价值用户有哪些特征?」
- 截包含:回答 + 对话内出的**图表(SVG 柱状/饼图)** + 底部 👍/👎
- 加分:同 Tab 截「Trace」面板(六节点链路耗时)拼成一张宽图

## 3. `segment-trend.png` — 自主监测与趋势(证明"不是静态 demo")

- Gradio → 「监测」Tab:任务列表应有 Watcher 自动分析记录
- 「分群趋势」图:各分群人数随时间变化的折线(有快照历史才会出现)
- 截一张图;若为空,先问一句分析类问题触发快照,或开滚动揭晓
  (`.env` 设 `TIANCHI_REVEAL_ENABLED=true` + `POST /debug/reveal-advance {"days":30}`)

## 可选第 4 张

- 商城首页商品列表/购物车结算页(证明完整电商链路不是空壳)

## 压缩建议

- Windows 自带「画图」另存即可压到 <300KB;或
  `python -c "from PIL import Image; Image.open('a.png').convert('RGB').save('a.jpg', quality=82)"`
