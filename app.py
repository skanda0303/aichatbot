import os
import asyncio
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Set HuggingFace embedding fallback if Ollama is not present
os.environ["USE_HUGGINGFACE_EMBEDDINGS"] = "1"

# Import multi_agent components
from multi_agent.retrieval.ingestion import load_and_index_documents
from multi_agent.retrieval.retriever import build_retriever
from multi_agent.agents import supervisor_agent

print("[HF SPACE] Preparing document index...")
chunks = load_and_index_documents()
print("[HF SPACE] Preparing hybrid retriever...")
retriever = build_retriever(chunks)
print("[HF SPACE] Multi-agent pipeline ready!")

async def predict(message, history):
    # Convert Gradio message history to LangChain message format
    history_messages = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            history_messages.append(AIMessage(content=content))

    full_response = ""
    async for token in supervisor_agent.run_streaming(
        query=message,
        history_messages=history_messages,
        retriever=retriever,
        chunks=chunks,
    ):
        full_response += token
        yield full_response

demo = gr.ChatInterface(
    fn=predict,
    title="🤖 Multi-Agent RAG Chatbot",
    description="Fact-grounded multi-agent system with Context Evaluation & Fact-checking Critic.",
    type="messages",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
