# app.py
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import gradio as gr
import os
from dotenv import load_dotenv
load_dotenv()

print("📂 Loading all components...\n")

with open('pdf_corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)
print(f"✅ Corpus loaded    : {len(corpus)} chunks")

embedder = SentenceTransformer(
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)
print(f"✅ Embedder loaded")

index = faiss.read_index('disease.index')
print(f"✅ FAISS index loaded: {index.ntotal} vectors")

GROQ_API_KEY = os.getenv("groq_key", "").strip().strip("'\"")
client       = Groq(api_key=GROQ_API_KEY)
print(f"✅ Groq client ready")
print(f"\n🎉 All components loaded!\n")

# ─────────────────────────────────────────────────────
# RETRIEVER
# ─────────────────────────────────────────────────────
def retrieve(query, top_k=5):
    query_embedding = embedder.encode(
        [query], convert_to_numpy=True
    )
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, top_k)
    return [{
        'text'  : corpus[idx]['text'],
        'source': corpus[idx]['source'],
        'pubid' : corpus[idx]['pubid'],
        'score' : float(scores[0][i])
    } for i, idx in enumerate(indices[0])]

# ─────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MedAI, a friendly and knowledgeable medical assistant.
You talk like a real doctor having a warm conversation with a patient.
Be caring, clear and easy to understand.

When a patient describes symptoms:
- Respond conversationally like a real doctor
- Ask follow up questions naturally if needed
- Give possible conditions in simple language
- Mention severity and urgency naturally
- Suggest what type of doctor to see
- Give basic care tips
- Always remind them to see a real doctor for final diagnosis

Never prescribe specific drug names or dosages.
If symptoms sound like emergency, strongly urge them to go to ER immediately."""

WELCOME = "👋 Hello! I'm MedAI, your personal medical assistant.\n\nTell me how you're feeling — describe your symptoms and I'll help you understand what might be going on. I'm here to guide you! 😊"

# ─────────────────────────────────────────────────────
# CHAT FUNCTION — Gradio 3.x uses tuples (user, bot)
# ─────────────────────────────────────────────────────
def chat(message, history):
    if not message.strip():
        return "", history

    # Retrieve medical context
    contexts     = retrieve(message, top_k=5)
    context_text = "\n\n".join([
        f"[Medical Evidence]: {ctx['text']}"
        for ctx in contexts
    ])

    # Build messages for Groq
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Use this medical evidence:\n{context_text}"}
    ]

    # ✅ Gradio 3.x history is list of tuples (user, bot)
    for human, assistant in history:
        if human:
            messages.append({"role": "user",      "content": human})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})

    # Groq response
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=1000
    )

    reply = response.choices[0].message.content

    # ✅ Gradio 3.x format — append tuple
    history.append((message, reply))
    return "", history

# ─────────────────────────────────────────────────────
# UI — Gradio 3.x compatible
# ─────────────────────────────────────────────────────

# ✅ Gradio 3.x initial value — list of tuples
INITIAL_HISTORY = [(None, WELCOME)]

with gr.Blocks(title="MedAI Chat") as app:

    gr.Markdown("""
    # 🏥 MedAI — Your Personal Medical Assistant
    *Powered by RAG + Llama3 + BioBERT + 37,714 Medical Records*
    > ⚠️ For educational purposes only. Always consult a real doctor.
    """)

    # ✅ Gradio 3.x Chatbot — no type parameter
    chatbot = gr.Chatbot(
        value=INITIAL_HISTORY,
        height=550,
        show_label=False
    )

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Type your symptoms... e.g. I have fever and headache since yesterday",
            show_label=False,
            scale=9,
            container=False
        )
        send_btn = gr.Button(
            "Send 🚀", scale=1, variant="primary"
        )

    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear Chat")

    gr.Examples(
        label="💡 Try these",
        examples=[
            ["I have fever, headache and body pain since 2 days"],
            ["I have chest pain and difficulty breathing"],
            ["I feel very tired, always thirsty and urinating frequently"],
            ["I have cough, weight loss and night sweats for 3 weeks"],
            ["Can I take creatine for bodybuilding? Is it safe?"],
            ["I have joint pain and swelling in my knees"],
            ["I feel anxious, heart is racing and sweating a lot"],
            ["I have yellow skin, dark urine and stomach pain"],
        ],
        inputs=msg_input
    )

    # ── Events ────────────────────────────────────────
    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot]
    )
    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot]
    )
    clear_btn.click(
        fn=lambda: ("", INITIAL_HISTORY),
        outputs=[msg_input, chatbot]
    )

if __name__ == "__main__":
    print("🚀 Launching MedAI Chat...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )