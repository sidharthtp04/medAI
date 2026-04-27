# add_pdfs_to_db.py
import pdfplumber
import pickle
import re
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

pdf_dir    = '/home/sidharth/medAI/pdfs'
output_dir = '/home/sidharth/medAI'

# ── Topic labels for each PDF ─────────────────────────
PDF_TOPICS = {
    'infectious_dengue.pdf'      : 'Fever_Infectious',
    'infectious_tuberculosis.pdf': 'Fever_Infectious',
    'infectious_malaria.pdf'     : 'Fever_Infectious',
    'diabetes_who_report.pdf'    : 'Diabetes',
    'diabetes_prevention.pdf'    : 'Diabetes',
    'heart_cvd_prevention.pdf'   : 'Cardiology',
    'heart_hypertension.pdf'     : 'Cardiology',
    'cancer_guide_who.pdf'       : 'Oncology',
    'cancer_prevention.pdf'      : 'Oncology',
    'neuro_epilepsy.pdf'         : 'Neurology',
    'neuro_dementia.pdf'         : 'Neurology',
    'neuro_stroke.pdf'           : 'Neurology',
    'mental_health_atlas.pdf'    : 'Mental_Health',
    'mental_depression.pdf'      : 'Mental_Health',
    'mental_anxiety.pdf'         : 'Mental_Health',
    'bone_arthritis.pdf'         : 'Bone_Joint',
    'bone_osteoporosis.pdf'      : 'Bone_Joint',
    'respiratory_copd.pdf'       : 'Respiratory',
    'respiratory_asthma.pdf'     : 'Respiratory',
    'respiratory_pneumonia.pdf'  : 'Respiratory',
    'kidney_disease.pdf'         : 'Kidney_Liver',
    'liver_hepatitis.pdf'        : 'Kidney_Liver',
    'liver_hepatitis_b.pdf'      : 'Kidney_Liver',
    'skin_leprosy.pdf'           : 'Dermatology',
    'skin_conditions.pdf'        : 'Dermatology',
    'skin_leishmaniasis.pdf'     : 'Dermatology',
}

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\d+\s*$', '', text)
    return text.strip()

def chunk_text(text, source, pubid, chunk_size=200, overlap=50):
    words  = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunk = clean_text(chunk)
        if len(chunk.split()) > 40:
            chunks.append({
                'text'  : chunk,
                'source': source,
                'pubid' : pubid
            })
    return chunks

def extract_pdf(path):
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        print(f"     ⚠️  PDF read error: {e}")
    return text

# ── Step 1: Load existing corpus ──────────────────────
corpus_path = os.path.join(output_dir, 'disease_corpus.pkl')
if not os.path.exists(corpus_path):
    corpus_path = os.path.join(output_dir, 'pdf_corpus.pkl')

with open(corpus_path, 'rb') as f:
    corpus = pickle.load(f)

print(f"📦 Existing corpus: {len(corpus)} chunks")

# ── Step 2: Process all PDFs ──────────────────────────
print(f"\n📄 Processing PDFs from: {pdf_dir}\n")
new_chunks = []
processed  = 0
skipped    = 0

for filename in os.listdir(pdf_dir):
    if not filename.endswith('.pdf'):
        continue

    filepath = os.path.join(pdf_dir, filename)
    size_mb  = os.path.getsize(filepath) / 1e6

    if size_mb < 0.05:
        print(f"  ⚠️  {filename} too small ({size_mb:.2f}MB) — skipping")
        skipped += 1
        continue

    topic  = PDF_TOPICS.get(filename, 'Medical_PDF')
    source = f'PDF_{topic}'

    print(f"  📄 {filename} ({size_mb:.1f}MB)...")
    text   = extract_pdf(filepath)

    if len(text.split()) < 100:
        print(f"     ⚠️  Very little text extracted — skipping")
        skipped += 1
        continue

    chunks = chunk_text(text, source=source, pubid=filename.replace('.pdf',''))
    new_chunks.extend(chunks)
    processed += 1
    print(f"     ✅ {len(chunks)} chunks extracted")

print(f"\n{'='*55}")
print(f"📊 PDF Processing Summary:")
print(f"   Processed : {processed} PDFs")
print(f"   Skipped   : {skipped} PDFs")
print(f"   New chunks: {len(new_chunks)}")

# ── Step 3: Merge with existing corpus ────────────────
combined = corpus + new_chunks
print(f"\n📦 Combined corpus: {len(combined)} chunks")

# Show breakdown
sources = {}
for item in combined:
    src = item.get('source', 'Unknown')
    sources[src] = sources.get(src, 0) + 1

print(f"\n📊 Source Breakdown:")
for src, count in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"   {src:<30}: {count:>6} chunks")

# ── Step 4: Save combined corpus ──────────────────────
save_path = os.path.join(output_dir, 'disease_corpus.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(combined, f)
print(f"\n💾 Corpus saved: {save_path}")

# ── Step 5: Rebuild FAISS index ───────────────────────
print(f"\n🔢 Rebuilding FAISS index for {len(combined)} chunks...")
print(f"⏳ This takes ~5-10 mins...\n")

embedder = SentenceTransformer(
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)

texts      = [doc['text'] for doc in combined]
embeddings = embedder.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
index     = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save
np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)
faiss.write_index(index, os.path.join(output_dir, 'disease.index'))

print(f"\n{'='*55}")
print(f"✅ FAISS INDEX REBUILT!")
print(f"{'='*55}")
print(f"   Total vectors : {index.ntotal:,}")
print(f"   Corpus size   : {len(combined):,}")
print(f"   Dimensions    : {dimension}")
print(f"{'='*55}")
print(f"\n🚀 Now run: python3 app.py")