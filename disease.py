# build_disease_db.py
from datasets import load_dataset
import pickle
import os

print("📥 Loading medical datasets from HuggingFace...\n")

output_dir = '/home/nasc/medAI'
corpus     = []

# ── Dataset 1: Disease & Symptoms ────────────────────
print("1️⃣  Loading Disease-Symptoms dataset...")
try:
    disease_ds = load_dataset(
        "QuyenAnhDE/Diseases_Symptoms",
        split="train"
    )
    for item in disease_ds:
        text = f"Disease: {item['Name']}. Symptoms: {item['Symptoms']}. Treatment: {item.get('Treatments', 'Consult a doctor')}"
        corpus.append({
            'text'   : text,
            'source' : 'Disease_DB',
            'pubid'  : item['Name']
        })
    print(f"   ✅ {len(disease_ds)} diseases loaded!")
    print(f"   📋 Sample: {corpus[0]['text'][:150]}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# ── Dataset 2: Medical QA ─────────────────────────────
print("\n2️⃣  Loading Medical QA dataset...")
try:
    medqa_ds = load_dataset(
        "medalpaca/medical_meadow_medical_flashcards",
        split="train"
    )
    for item in medqa_ds:
        text = f"Question: {item['input']} Answer: {item['output']}"
        corpus.append({
            'text'  : text,
            'source': 'Medical_QA',
            'pubid' : 'MedFlashcard'
        })
    print(f"   ✅ {len(medqa_ds)} QA pairs loaded!")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# ── Dataset 3: PubMedQA ───────────────────────────────
print("\n3️⃣  Loading PubMedQA dataset...")
try:
    pubmed_ds = load_dataset(
        "pubmed_qa",
        "pqa_labeled",
        split="train"
    )
    import re
    for item in pubmed_ds:
        for ctx in item['context']['contexts']:
            cleaned = re.sub(r'\s+', ' ', ctx).strip()
            if len(cleaned.split()) > 30:
                corpus.append({
                    'text'  : cleaned,
                    'source': 'PubMedQA',
                    'pubid' : str(item['pubid'])
                })
    print(f"   ✅ PubMedQA loaded!")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# ── Dataset 4: Symptom2Disease ────────────────────────
print("\n4️⃣  Loading Symptom2Disease dataset...")
try:
    sym_ds = load_dataset(
        "gretelai/symptom_to_diagnosis",
        split="train"
    )
    for item in sym_ds:
        text = f"Patient symptoms: {item['input_text']}. Diagnosis: {item['output_text']}"
        corpus.append({
            'text'  : text,
            'source': 'Symptom2Disease',
            'pubid' : item['output_text']
        })
    print(f"   ✅ {len(sym_ds)} symptom cases loaded!")
    print(f"   📋 Sample: {corpus[-1]['text'][:150]}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# ── Save corpus ───────────────────────────────────────
print(f"\n{'='*50}")
print(f"📦 CORPUS SUMMARY:")
print(f"{'='*50}")

sources = {}
for item in corpus:
    src = item['source']
    sources[src] = sources.get(src, 0) + 1

for src, count in sources.items():
    print(f"   {src:<20}: {count:>6} chunks")

print(f"{'─'*50}")
print(f"   {'TOTAL':<20}: {len(corpus):>6} chunks")
print(f"{'='*50}")

output_path = os.path.join(output_dir, 'disease_corpus.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(corpus, f)

print(f"\n💾 Corpus saved to: {output_path}")
print(f"✅ Ready to build FAISS index!")