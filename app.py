import gradio as gr
from transformers import pipeline
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 1. Load models
#   • QA pipeline (DistilBERT fine‑tuned on SQuAD)
qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad",
              tokenizer="distilbert-base-uncased-distilled-squad")

#   • SentenceTransformer for semantic search
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Load your knowledge base
with open("knowledgebase.txt", "r", encoding="utf-8") as f:
    data = f.read().strip().split("---")
    qa_pairs = [(block.split("\n")[0].replace("Q: ",""),
                 block.split("\n")[1].replace("A: ",""))
                for block in data if block]

questions = [q for q, _ in qa_pairs]
answers   = [a for _, a in qa_pairs]
embs      = embedder.encode(questions).astype("float32")

# 3. Build FAISS index
index = faiss.IndexFlatL2(embs.shape[1])
index.add(embs)

def bot_response(user_message):
    # A. Retrieve best matching QA
    q_emb = embedder.encode([user_message]).astype("float32")
    _, I = index.search(q_emb, k=1)
    base_ans = answers[I[0][0]]
    # B. Optionally refine via QA pipeline over the base answer as context
    out = qa(question=user_message, context=base_ans)
    return out["answer"]

# 4. Build Gradio interface
iface = gr.Interface(
    fn=bot_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask me anything…"),
    outputs="text",
    title="Mahat.ai Chatbot",
    description="Answers from Mahat.ai knowledge base."
)

if __name__ == "__main__":
    iface.launch()
