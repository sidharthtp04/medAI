# download_pdfs.py
import requests, os, time

pdf_dir = '/home/sidharth/medAI/pdfs'
os.makedirs(pdf_dir, exist_ok=True)

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
    'Accept'    : 'application/pdf,*/*',
}

# ── All 10 topics — verified free PMC/WHO/NIH PDFs ───
pdfs = {

    # 🦠 Fever & Infectious Diseases
    'infectious_dengue.pdf'     : 'https://apps.who.int/iris/bitstream/handle/10665/117098/9789241507493_eng.pdf',
    'infectious_tuberculosis.pdf': 'https://apps.who.int/iris/bitstream/handle/10665/44165/9789241547833_eng.pdf',
    'infectious_malaria.pdf'    : 'https://apps.who.int/iris/bitstream/handle/10665/274339/9789241565523-eng.pdf',

    # 🩸 Diabetes & Endocrinology
    'diabetes_who_report.pdf'   : 'https://apps.who.int/iris/bitstream/handle/10665/204871/9789241565257_eng.pdf',
    'diabetes_prevention.pdf'   : 'https://apps.who.int/iris/bitstream/handle/10665/43943/9789241563567_eng.pdf',

    # ❤️ Heart & Cardiology
    'heart_cvd_prevention.pdf'  : 'https://apps.who.int/iris/bitstream/handle/10665/43685/9789241547123_eng.pdf',
    'heart_hypertension.pdf'    : 'https://apps.who.int/iris/bitstream/handle/10665/254882/9789241512824-eng.pdf',

    # 🎗️ Cancer & Oncology
    'cancer_guide_who.pdf'      : 'https://apps.who.int/iris/bitstream/handle/10665/43743/9789241547338_eng.pdf',
    'cancer_prevention.pdf'     : 'https://apps.who.int/iris/bitstream/handle/10665/44009/9789241563666_eng.pdf',

    # 🧠 Neurology & Brain
    'neuro_epilepsy.pdf'        : 'https://apps.who.int/iris/bitstream/handle/10665/164073/9789241564687_eng.pdf',
    'neuro_dementia.pdf'        : 'https://apps.who.int/iris/bitstream/handle/10665/75263/9789241564458_eng.pdf',
    'neuro_stroke.pdf'          : 'https://apps.who.int/iris/bitstream/handle/10665/43733/9789241547543_eng.pdf',

    # 🧘 Mental Health & Psychiatry
    'mental_health_atlas.pdf'   : 'https://apps.who.int/iris/bitstream/handle/10665/345946/9789240036703-eng.pdf',
    'mental_depression.pdf'     : 'https://apps.who.int/iris/bitstream/handle/10665/254610/9789241512855-eng.pdf',
    'mental_anxiety.pdf'        : 'https://apps.who.int/iris/bitstream/handle/10665/204764/9789241549905_eng.pdf',

    # 🦴 Bone & Joint Diseases
    'bone_arthritis.pdf'        : 'https://apps.who.int/iris/bitstream/handle/10665/42407/WHO_MSD_MSB_00.1.pdf',
    'bone_osteoporosis.pdf'     : 'https://apps.who.int/iris/bitstream/handle/10665/43551/9789241563369_eng.pdf',

    # 🫁 Respiratory & Lung
    'respiratory_copd.pdf'      : 'https://apps.who.int/iris/bitstream/handle/10665/44428/9789241500463_eng.pdf',
    'respiratory_asthma.pdf'    : 'https://apps.who.int/iris/bitstream/handle/10665/67384/WHO_HTM_TB_2006.368_eng.pdf',
    'respiratory_pneumonia.pdf' : 'https://apps.who.int/iris/bitstream/handle/10665/255888/9789241512336-eng.pdf',

    # 🫘 Kidney & Liver
    'kidney_disease.pdf'        : 'https://apps.who.int/iris/bitstream/handle/10665/44585/9789241502542_eng.pdf',
    'liver_hepatitis.pdf'       : 'https://apps.who.int/iris/bitstream/handle/10665/255017/9789241565455-eng.pdf',
    'liver_hepatitis_b.pdf'     : 'https://apps.who.int/iris/bitstream/handle/10665/246177/9789241549981-eng.pdf',

    # 🌿 Skin & Dermatology
    'skin_leprosy.pdf'          : 'https://apps.who.int/iris/bitstream/handle/10665/77771/9789241503228_eng.pdf',
    'skin_conditions.pdf'       : 'https://apps.who.int/iris/bitstream/handle/10665/205003/9789241565189_eng.pdf',
    'skin_leishmaniasis.pdf'    : 'https://apps.who.int/iris/bitstream/handle/10665/44412/9789241548595_eng.pdf',
}

print(f"📥 Downloading {len(pdfs)} medical PDFs...\n")
success = 0
failed  = []

for filename, url in pdfs.items():
    filepath = os.path.join(pdf_dir, filename)

    # Skip if already downloaded
    if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
        print(f"  ⏭️  {filename} already exists — skipping")
        success += 1
        continue

    try:
        print(f"  ⏳ {filename}...")
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
            if size < 0.01:
                print(f"  ⚠️  {filename} — too small ({size:.2f}MB), may be empty")
                failed.append(filename)
            else:
                print(f"  ✅ {filename} — {size:.1f} MB")
                success += 1
        else:
            print(f"  ❌ {filename} — HTTP {resp.status_code}")
            failed.append(filename)
        time.sleep(0.5)
    except Exception as e:
        print(f"  ❌ {filename} — {e}")
        failed.append(filename)

print(f"\n{'='*55}")
print(f"✅ Downloaded : {success}/{len(pdfs)} PDFs")
print(f"❌ Failed     : {len(failed)}")
if failed:
    print(f"\nFailed files:")
    for f in failed:
        print(f"  • {f}")
print(f"📁 Location   : {pdf_dir}")
print(f"\n▶️  Now run: python3 add_pdfs_to_db.py")