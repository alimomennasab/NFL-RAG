import gradio as gr
from gemini import ask

def respond(query: str, history: str):
    return ask(query)

if __name__ == "__main__":
    gr.ChatInterface(
        fn=respond,
        title="NFL Stats RAG",
        examples=[
            "Who led Week 1 in passing yards?",
            "Top 3 receivers by yards in Week 1",
        ],
    ).launch()
