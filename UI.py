# UI.py
import gradio as gr
import uuid
from chat import conversational_rag_chain

def get_response(user_message, chat_history, session_id):
    if not session_id:
        session_id = str(uuid.uuid4())
    
    response = conversational_rag_chain.invoke(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}},
    )["answer"]
    
    chat_history.append((user_message, response))
    return "", chat_history, session_id

with gr.Blocks(title="Bajra Leave Policy Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍚 Bajra Leave Policy Chatbot")
    gr.Markdown("Ask me about the company's leave policies and procedures!")
    
    with gr.Row():
        session_id = gr.State()
        chatbot = gr.Chatbot(height=500, avatar_images=("user.png", "bot.png"))
        
    with gr.Row():
        msg = gr.Textbox(
            label="Your Question",
            placeholder="Type your question about leave policies here...",
            container=False,
            scale=7
        )
        clear = gr.Button("Clear Chat", variant="stop", scale=1)
    
    msg.submit(
        get_response,
        inputs=[msg, chatbot, session_id],
        outputs=[msg, chatbot, session_id]
    )
    
    clear.click(lambda: (None, [], None), None, [msg, chatbot, session_id], queue=False)

if __name__ == "__main__":
    demo.launch()