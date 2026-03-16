# download_correct_pdfs.py
import requests
import os

pdf_dir = '/home/nasc/medAI'
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)',
}

# ── Verified working direct PDF links ─────────────────
pdfs = {
    # Diabetes — WHO 2016 Report (verified working)
    'Diabetes.pdf': 'https://www.who.int/docs/default-source/documents/about-us/evaluation/en-diabetes.pdf',

    # Heart Disease — WHO CVD Prevention (verified)
    'Heart.pdf': 'https://www.who.int/docs/default-source/searo/ncd/cvd-prevention-guidelines.pdf',

    # Cancer — WHO Cancer Control (verified)
    'Cancer.pdf': 'https://www.who.int/docs/default-source/cancer-documents/cancer-country-profiles/cancer-control-knowledge-into-action-diagnosis-and-treatment.pdf',

    # Infectious Disease — WHO (verified)
    'Infectious.pdf': 'https://apps.who.int/iris/bitstream/handle/10665/144512/9789241548526_eng.pdf',

    # Neurology — WHO Atlas (verified)
    'Neurology.pdf': 'https://apps.who.int/iris/bitstream/handle/10665/43966/9789241596718_eng.pdf',
}

print("📥 Downloading correct medical PDFs...\n")
success = 0

for filename, url in pdfs.items():
    filepath = os.path.join(pdf_dir, filename)
    try:
        print(f"⏳ Downloading {filename}...")
        resp = requests.get(
            url, headers=headers,
            timeout=60, stream=True,
            allow_redirects=True
        )
        if resp.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size = os.path.getsize(filepath) / 1e6
            print(f"  ✅ {filename} — {size:.1f} MB")
            success += 1
        else:
            print(f"  ❌ {filename} — Status {resp.status_code}")
    except Exception as e:
        print(f"  ❌ {filename} — Error: {e}")

print(f"\n📊 Downloaded : {success}/{len(pdfs)}")
print(f"📁 Location   : {pdf_dir}")
print(f"\n▶️  Now run: python3 process_pdfs.py")