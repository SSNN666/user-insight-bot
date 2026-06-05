import gradio as gr
from agent import ask_agent
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

def respond(message, history):
    reply = ask_agent(message)
    return reply

# 创建对话界面
demo = gr.ChatInterface(
    fn=respond,
    title="电商用户智能画像问答",
    description="向我询问用户分群信息，例如：'各分群的人数是多少？'、'高价值用户有哪些特征？'",
)

if __name__ == "__main__":
    # 把主题设置在 launch 里
    demo.launch(theme=gr.themes.Soft())