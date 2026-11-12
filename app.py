"""Interface web local (Gradio): python app.py  ->  http://127.0.0.1:7860"""
import gradio as gr

from rag import RagChatbot

bot = RagChatbot()


def responder(pergunta: str):
    if not pergunta.strip():
        return "", ""
    resposta, hits = bot.ask(pergunta)
    fontes = "\n".join(f"- {chunk.source} (similaridade {score:.2f})" for chunk, score in hits)
    return resposta, fontes


demo = gr.Interface(
    fn=responder,
    inputs=gr.Textbox(label="Pergunta", lines=2),
    outputs=[gr.Textbox(label="Resposta"), gr.Textbox(label="Fontes recuperadas")],
    title="Chatbot RAG local (CPU)",
    description="Responde perguntas com base nos documentos da pasta data/docs.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
