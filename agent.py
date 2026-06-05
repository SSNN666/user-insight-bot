from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from config import LLM_MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL
from tools import tools
import os
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["LLM_MODEL"] = "qwen2.5:7b"
# 初始化 LLM
llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL
)

# 修改系统提示，LangGraph 的 create_react_agent 会自动处理 ReAct 格式
system_prompt = "你是一个电商用户画像分析助手，可以使用工具查询用户分群信息。请用中文回答。"

# 创建 Agent
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt
)

# 设置记忆（对话历史管理）
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_memory = RunnableWithMessageHistory(
    agent,
    get_session_history,
    input_messages_key="messages",
    history_messages_key="history",
)

def ask_agent(question: str, session_id: str = "default") -> str:
    """调用 Agent 获取回答"""
    result = agent_with_memory.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"session_id": session_id}}
    )
    # 从返回的消息列表中提取最后一条 AI 回复
    return result["messages"][-1].content