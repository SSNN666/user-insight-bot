# 电商用户智能画像与AI问答平台

基于 MySQL + K-Means + 决策树 + LLM Agent 的用户分群与智能问答系统。  
支持自然语言查询用户群体特征、人数、价值分布等。

## 功能
- MySQL 数据库设计、分区表、RFM 指标提取
- 数据清洗与特征工程（Pandas）
- K-Means 用户聚类 + 决策树规则提取
- LLM Agent 自动调用工具回答运营问题
- Gradio 交互式对话 Demo

## 本地运行
1. 安装依赖：`pip install -r requirements.txt`
2. 启动本地模型（如 Ollama）：`ollama pull qwen2.5:7b && ollama serve`
3. 配置 `.env`（参考 `.env.example`）
4. 运行：`python app.py`

## 技术栈
Python · MySQL · Pandas · Scikit-learn · LangChain · LangGraph · Gradio · Ollama
