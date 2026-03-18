# build_index.py
import pickle
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

print("📂 Loading corpus...")
with open('pdf_corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)
print(f"✅ Corpus loaded: {len(corpus)} chunks")

# ── Load embedder ─────────────────────────────────────
print("\n📥 Loading BioBERT embedder...")
embedder = SentenceTransformer(
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)
print("✅ Embedder loaded!")

# ── Generate embeddings ───────────────────────────────
print("\n🔢 Generating embeddings for 37,714 chunks...")
print("⏳ Takes 10-15 minutes on CPU...\n")

texts      = [doc['text'] for doc in corpus]
embeddings = embedder.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)
print(f"\n✅ Embeddings shape: {embeddings.shape}")

# ── Build FAISS index ─────────────────────────────────
print("\n🏗️  Building FAISS index...")
dimension = embeddings.shape[1]
index     = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print(f"✅ FAISS index built: {index.ntotal} vectors")

# ── Save everything ───────────────────────────────────
print("\n💾 Saving to disk...")

np.save('embeddings.npy', embeddings)
faiss.write_index(index, 'disease.index')

print("✅ Embeddings saved : embeddings.npy")
print("✅ FAISS index saved: disease.index")

print(f"\n{'='*50}")
print(f"✅ INDEX BUILT SUCCESSFULLY!")
print(f"{'='*50}")
print(f"   Corpus size : {len(corpus)} chunks")
print(f"   Index size  : {index.ntotal} vectors")
print(f"   Dimension   : {dimension}")
print(f"{'='*50}")
print(f"\n▶️  Now run: python3 app.py")
