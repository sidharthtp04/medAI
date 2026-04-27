# medai_v6.py  —  MedAI  |  Fixed: Delete works, Load chat works, Voice status small, Input pinned bottom
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import gradio as gr
import os, json, datetime
from dotenv import load_dotenv
load_dotenv()

try:
    import speech_recognition as sr
    VOICE_INPUT_OK = True
except:
    VOICE_INPUT_OK = False

print("⏳ Loading components...\n")

corpus_path = 'disease_corpus.pkl' if os.path.exists('disease_corpus.pkl') else 'pdf_corpus.pkl'
with open(corpus_path, 'rb') as f:
    corpus = pickle.load(f)
print(f"✅ Corpus: {len(corpus)} chunks")

embedder = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
print("✅ Embedder loaded")

index = faiss.read_index('disease.index')
print(f"✅ FAISS: {index.ntotal} vectors")

GROQ_API_KEY = os.getenv("groq_key", "").strip().strip("'\"")
client = Groq(api_key=GROQ_API_KEY)
print(f"✅ Groq ready | Voice: {'ON' if VOICE_INPUT_OK else 'OFF'}\n🚀 Ready!\n")

HISTORY_FILE = "chat_histories.json"

def load_histories():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except:
            pass
    return {}

def save_histories(h):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(h, f)

def retrieve(query, top_k=5):
    qe = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qe)
    _, idxs = index.search(qe, top_k)
    return [{'text': corpus[i]['text'], 'source': corpus[i].get('source', 'Unknown')} for i in idxs[0]]

def voice_to_text():
    if not VOICE_INPUT_OK:
        return "", "⚠️ SpeechRecognition not installed"
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as src:
            recognizer.adjust_for_ambient_noise(src, duration=0.8)
            audio = recognizer.listen(src, timeout=8, phrase_time_limit=15)
        text = recognizer.recognize_google(audio, language="en-IN")
        return text, f'🎤 "{text}"'
    except sr.WaitTimeoutError:
        return "", "⏱️ No speech detected"
    except sr.UnknownValueError:
        return "", "❓ Could not understand"
    except Exception as e:
        return "", f"❌ {e}"

SYSTEM_PROMPT = """You are MedAI, a warm expert medical AI assistant.

When a patient describes symptoms, ALWAYS respond using this exact structure:

**Symptom Summary**
One warm sentence acknowledging what they described.

**Possible Conditions**
- **Condition Name** — High / Medium / Low likelihood
  → Why this matches their symptoms

**Key Questions to Narrow Diagnosis**
- How long have you had these symptoms?
- Is it constant or intermittent?
- Severity from 1 (mild) to 10 (severe)?
- Any fever? Temperature?
- Associated symptoms (nausea, dizziness, fatigue)?
- Existing medical conditions?
- Current medications?
- [2 specific questions for their exact symptoms]
- Recent travel, injury, or sick contact?
- What makes it better or worse?

**Severity**
🟢 Mild — home care  |  🟡 Moderate — doctor soon  |  🟠 Severe — today  |  🔴 Emergency — ER now

**Recommended Specialist**
Name the specialist.

**Red Flags — Go to ER if:**
- 3-4 specific emergency signs

**Home Care Tips**
- 2-3 safe non-medication steps

**Disclaimer**
Always consult a qualified doctor for proper diagnosis.

RULES: Never prescribe medicines. Flag emergencies immediately. Be warm and professional."""

WELCOME = ("MedAI", "Hello! I'm **MedAI**, your AI medical assistant powered by RAG + BioBERT + Llama 3.3 70B.\n\nDescribe your symptoms and I'll help identify possible conditions, ask key diagnostic questions, and recommend the right specialist.\n\n💬 Type your symptoms below — I'm here to guide you, not replace your doctor! 🏥")

def chat(message, history, session_id, all_histories):
    if not message.strip():
        return "", history, session_id, all_histories
    contexts = retrieve(message)
    ctx_text = "\n\n".join([f"[{c['source']}]: {c['text']}" for c in contexts])
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Medical evidence:\n{ctx_text}"}
    ]
    for h, a in history:
        if h: msgs.append({"role": "user", "content": h})
        if a: msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": message})
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=msgs,
        temperature=0.3, max_tokens=1500
    )
    reply = resp.choices[0].message.content
    history.append((message, reply))
    histories = json.loads(all_histories) if all_histories else {}
    if session_id and session_id in histories:
        histories[session_id]["messages"] = history
        # Move the newly interacted chat to the end of the dict, so it appears at top
        val = histories.pop(session_id)
        histories[session_id] = val
        save_histories(histories)
    return "", history, session_id, json.dumps(histories), build_sidebar_html(histories, session_id)

def new_chat(all_histories):
    histories = json.loads(all_histories) if all_histories else {}
    sid = datetime.datetime.now().strftime("Chat %d %b, %H:%M:%S")
    histories[sid] = {"title": sid, "messages": []}
    save_histories(histories)
    return [WELCOME], sid, json.dumps(histories), build_sidebar_html(histories, sid)

# FIX 1: load_chat now also returns all_histories_state so session switches properly
def load_chat(session_id, all_histories):
    histories = json.loads(all_histories) if all_histories else {}
    if session_id and session_id in histories:
        msgs = histories[session_id]["messages"]
        return (msgs if msgs else [WELCOME]), session_id, json.dumps(histories), build_sidebar_html(histories, session_id)
    return [WELCOME], session_id, all_histories, build_sidebar_html(histories, session_id)

def delete_chat(target_payload, current_sid, all_histories):
    # Failsafe for empty payload
    if not target_payload or "||" not in target_payload:
        histories = json.loads(all_histories) if all_histories else {}
        msgs = histories.get(current_sid, {}).get("messages") if current_sid in histories else [WELCOME]
        return msgs, current_sid, all_histories, build_sidebar_html(histories, current_sid)

    target_sid = target_payload.split("||")[0]
    if not target_sid:
        histories = json.loads(all_histories) if all_histories else {}
        msgs = histories.get(current_sid, {}).get("messages") if current_sid in histories else [WELCOME]
        return msgs, current_sid, all_histories, build_sidebar_html(histories, current_sid)

    histories = json.loads(all_histories) if all_histories else {}

    if target_sid in histories:
        del histories[target_sid]
        save_histories(histories)

    new_histories_json = json.dumps(histories)

    if current_sid == target_sid:
        if histories:
            new_sid = list(histories.keys())[-1]
            msgs = histories[new_sid]["messages"] or [WELCOME]
        else:
            new_sid = ""
            msgs = [WELCOME]
    else:
        new_sid = current_sid
        msgs = histories.get(current_sid, {}).get("messages") or [WELCOME]

    return msgs, new_sid, new_histories_json, build_sidebar_html(histories, new_sid)

def build_sidebar_html(histories, active_sid=""):
    if not histories:
        return (
            '<div style="padding:32px 16px;text-align:center;">'
            '<div style="font-size:36px;margin-bottom:12px;opacity:0.3;">💬</div>'
            '<div class="no-hist-text">No consultations yet.<br>Tap <b>＋ New Chat</b> to begin.</div>'
            '</div>'
        )
    items = ""
    for sid, data in reversed(list(histories.items())):
        is_active = sid == active_sid
        ac = "hist-active" if is_active else ""
        safe_sid = sid.replace("'", "\\'")
        items += (
            f'<div class="hist-item {ac}" onclick="selectChat(\'{safe_sid}\')" title="{sid}">'
            f'<span class="hist-icon">{"🟢" if is_active else "💬"}</span>'
            f'<span class="hist-label">{sid}</span>'
            f'<button class="hist-del-btn" onclick="event.stopPropagation(); openDelModal(\'{safe_sid}\')" title="Delete Chat">✕</button>'
            f'</div>'
        )
    return f'<div class="hist-list">{items}</div>'

# ── Initial state ──
stored       = load_histories()
init_sid     = list(stored.keys())[-1] if stored else ""
init_msgs    = stored[init_sid]["messages"] if init_sid and stored[init_sid]["messages"] else [WELCOME]
init_sidebar = build_sidebar_html(stored, init_sid)

# ══════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg:            #ffffff;
    --bg-sidebar:    #f9f9f9;
    --bg-hover:      #ececec;
    --bg-active:     #e3e3e3;
    --bg-user:       #f4f4f4;
    --bg-input:      #ffffff;
    --bg-send:       #10a37f;
    --bg-send-h:     #0d8f6e;
    --border:        #e5e5e5;
    --border-inp:    #d1d1d1;
    --text:          #0d0d0d;
    --text-sub:      #6b6b6b;
    --text-faint:    #b0b0b0;
    --accent:        #10a37f;
    --red:           #ef4444;
    --shadow-inp:    0 0 0 1px rgba(0,0,0,0.07), 0 1px 4px rgba(0,0,0,0.05);
}
[data-theme="dark"] {
    --bg:            #212121;
    --bg-sidebar:    #171717;
    --bg-hover:      #2a2a2a;
    --bg-active:     #323232;
    --bg-user:       #2f2f2f;
    --bg-input:      #2f2f2f;
    --bg-send:       #10a37f;
    --bg-send-h:     #0d8f6e;
    --border:        #3a3a3a;
    --border-inp:    #484848;
    --text:          #ececec;
    --text-sub:      #8e8e9a;
    --text-faint:    #555;
    --accent:        #10a37f;
    --red:           #f87171;
    --shadow-inp:    0 0 0 1px rgba(255,255,255,0.05), 0 1px 4px rgba(0,0,0,0.25);
}

/* ── Reset ── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body,.gradio-container,#root,.main,.wrap,.contain,.block,gradio-app{
    background:var(--bg)!important;
    font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important;
    color:var(--text)!important;
}
footer,.footer,.built-with,.svelte-1rjryqp{display:none!important}
.gap,.form,.gr-group{gap:0!important;padding:0!important}
.gradio-container{padding:0!important;margin:0!important;max-width:100%!important}
.gr-block.gr-box{padding:0!important;border:none!important}
.hidden-input{display:none!important}

/* ── App shell: sidebar + main side by side, full viewport ── */
.app-shell{
    display:flex!important;
    width:100vw!important;
    height:100dvh!important;
    overflow:hidden!important;
    background:var(--bg)!important;
    gap:0!important;padding:0!important;
}

/* ══ SIDEBAR ══ */
.sidebar{
    width:260px!important;min-width:260px!important;max-width:260px!important;
    height:100dvh!important;
    background:var(--bg-sidebar)!important;
    border-right:1px solid var(--border)!important;
    display:flex!important;flex-direction:column!important;
    overflow:hidden!important;z-index:200!important;flex-shrink:0!important;
    transition:transform .28s cubic-bezier(.4,0,.2,1),background .2s!important;
}
.sb-header{padding:16px 14px 10px}
.sb-brand{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.sb-icon{
    width:34px;height:34px;
    background:linear-gradient(135deg,#10a37f,#059669);
    border-radius:9px;display:flex;align-items:center;justify-content:center;
    font-size:17px;box-shadow:0 2px 8px rgba(16,163,127,.35);flex-shrink:0;
}
.sb-name{font-size:15px;font-weight:700;color:var(--text);letter-spacing:-.3px}
.sb-sub{font-size:9.5px;color:var(--text-sub);margin-top:1px}
.new-chat-btn button{
    width:100%!important;background:transparent!important;
    border:1px solid var(--border)!important;border-radius:9px!important;
    color:var(--text)!important;font-size:13px!important;font-weight:500!important;
    padding:9px 13px!important;cursor:pointer!important;text-align:left!important;
    transition:background .15s!important;display:flex!important;align-items:center!important;
    gap:8px!important;font-family:'Inter',sans-serif!important;
}
.new-chat-btn button:hover{background:var(--bg-hover)!important}
.sb-sec-label{
    font-size:10px;font-weight:600;color:var(--text-faint);
    text-transform:uppercase;letter-spacing:.09em;padding:10px 14px 4px;
}
.sb-scroll{flex:1;overflow-y:auto}
.sb-scroll::-webkit-scrollbar{width:3px}
.sb-scroll::-webkit-scrollbar-thumb{background:var(--border);border-radius:6px}
.hist-list{padding:4px 8px;display:flex;flex-direction:column;gap:1px}
.hist-item{
    display:flex;align-items:center;gap:8px;
    padding:8px 10px;border-radius:8px;cursor:pointer;transition:background .12s;
    position: relative;
}
.hist-item:hover{background:var(--bg-hover)}
.hist-active{background:var(--bg-active)!important}
.hist-icon{font-size:13px;flex-shrink:0;opacity:.6}
.hist-label{
    font-size:13px;font-weight:400;color:var(--text-sub);
    flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;
}
.hist-active .hist-label{color:var(--text)!important;font-weight:500!important}
.hist-del-btn {
    background: transparent;
    border: none;
    color: var(--text-faint);
    cursor: pointer;
    font-size: 13px;
    padding: 2px 4px;
    border-radius: 4px;
    transition: all .15s;
    opacity: 0;
}
.hist-item:hover .hist-del-btn, .hist-active .hist-del-btn {
    opacity: 1;
}
.hist-del-btn:hover {
    background: var(--red);
    color: white !important;
}
.no-hist-text{font-size:12px;color:var(--text-faint);line-height:1.9}
.sb-footer{border-top:1px solid var(--border);padding:12px 14px}
.sb-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px}
.sb-card{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:7px 10px}
.sb-card-label{font-size:9px;font-weight:700;color:var(--text-faint);text-transform:uppercase;letter-spacing:.06em;margin-bottom:2px}
.sb-card-val{font-size:11.5px;font-weight:600;color:var(--text)}
.disclaimer{
    background:#fefce8;border:1px solid #fde047;
    border-radius:8px;padding:9px 11px;
    display:flex;align-items:flex-start;gap:7px;
}
[data-theme="dark"] .disclaimer{background:#1c1500;border-color:#78350f}
.disclaimer-text{font-size:10px;color:#92400e;line-height:1.65;font-weight:500}
[data-theme="dark"] .disclaimer-text{color:#fbbf24}

/* ══ MAIN COLUMN ══ */
.chat-main{
    flex:1 1 0!important;
    display:flex!important;
    flex-direction:column!important;
    height:100dvh!important;
    max-height:100dvh!important;
    min-width:0!important;
    overflow:hidden!important;
    background:var(--bg)!important;
    position:relative!important;
}

/* ── Topbar ── */
.topbar{
    flex-shrink:0!important;
    height:52px!important;min-height:52px!important;
    padding:0 14px!important;
    border-bottom:1px solid var(--border)!important;
    background:var(--bg)!important;
    display:flex!important;align-items:center!important;justify-content:space-between!important;
    z-index:10!important;
}
.tb-left{display:flex;align-items:center;gap:10px;min-width:0;flex:1}
.tb-right{display:flex;align-items:center;gap:6px;flex-shrink:0}
.tb-title{font-size:14px;font-weight:600;color:var(--text);white-space:nowrap;letter-spacing:-.2px}
.tb-sub{font-size:11px;color:var(--text-sub);white-space:nowrap}
.live-badge{
    display:flex;align-items:center;gap:5px;
    background:rgba(16,163,127,.10);border:1px solid rgba(16,163,127,.22);
    border-radius:20px;padding:3px 10px;font-size:10px;font-weight:700;color:#10a37f;
}
.live-dot{
    width:5px;height:5px;background:#10a37f;border-radius:50%;
    animation:pulse 2s ease-in-out infinite;
}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}

/* ── Topbar icon/text buttons ── */
.tb-icon-btn{
    background:transparent;border:1px solid var(--border);border-radius:7px;
    color:var(--text-sub);font-size:15px;height:32px;width:32px;
    padding:0;cursor:pointer;display:flex;align-items:center;justify-content:center;
    transition:all .15s;flex-shrink:0;
}
.tb-icon-btn:hover{background:var(--bg-hover)}

#del-gradio-btn { display: none !important; }

/* ══ CHATBOT — takes ALL remaining vertical space ══ */
.chat-box{
    flex:1 1 0!important;
    min-height:0!important;
    overflow:hidden!important;
    display:flex!important;
    flex-direction:column!important;
    background:var(--bg)!important;
    border:none!important;
}
.chat-box>div,
.chat-box .wrap,
.chat-box .block{
    flex:1 1 0!important;min-height:0!important;height:100%!important;
    background:var(--bg)!important;border:none!important;border-radius:0!important;
    display:flex!important;flex-direction:column!important;
}
.chat-box .overflow-y-auto,
.chat-box .scroll-hide{
    flex:1 1 0!important;min-height:0!important;
    overflow-y:auto!important;
    padding-bottom:120px!important;
    background:var(--bg)!important;
}
.chat-box .overflow-y-auto::-webkit-scrollbar{width:4px}
.chat-box .overflow-y-auto::-webkit-scrollbar-thumb{background:var(--border);border-radius:6px}

.chat-box .message-wrap {
    max-width: 90% !important;
    margin: 0 auto !important;
}
.chat-box .message-wrap > div {
    max-width: 100% !important;
    width: 100% !important;
}
.chat-box .message.bot {
    font-size: 15.5px !important;
    letter-spacing: 0.01em;
}

/* Bot message */
.chat-box .message-wrap .message.bot{
    background:transparent!important;color:var(--text)!important;
    border:none!important;border-radius:0!important;padding:0!important;
    font-size:14.5px!important;line-height:1.9!important;
    max-width:100%!important;box-shadow:none!important;
}
.chat-box .message-wrap .message.bot p{
    color:var(--text)!important;font-size:14.5px!important;
    line-height:1.9!important;margin-bottom:6px!important;
}
.chat-box .message-wrap .message.bot strong{color:var(--text)!important}
.chat-box .message-wrap .message.bot ul,
.chat-box .message-wrap .message.bot ol{padding-left:20px!important;margin:8px 0!important}
.chat-box .message-wrap .message.bot li{margin-bottom:4px!important}
.chat-box .message-wrap .message.bot h1,
.chat-box .message-wrap .message.bot h2,
.chat-box .message-wrap .message.bot h3{
    color:var(--text)!important;margin:12px 0 5px!important;font-size:15px!important;
}

/* User message */
.chat-box .message-wrap .message.user{
    background:var(--bg-user)!important;color:var(--text)!important;
    border:none!important;border-radius:16px 16px 4px 16px!important;
    padding:12px 16px!important;font-size:14px!important;line-height:1.65!important;
    max-width:62%!important;margin-left:auto!important;
    box-shadow:none!important;word-break:break-word!important;
}
.chat-box .message-wrap .message.user p{
    color:var(--text)!important;font-size:14px!important;margin:0!important;
}

/* ══ INPUT BAR — pinned to bottom ══ */
.input-bar{
    position:absolute!important;
    bottom:0!important;
    left:0!important;
    right:0!important;
    z-index:100!important;
    background:var(--bg)!important;
    border-top:1px solid var(--border)!important;
    padding:12px 16px 14px!important;
    padding-bottom:max(14px,env(safe-area-inset-bottom,14px))!important;
}
.input-inner{max-width:780px;margin:0 auto}

.input-pill{
    display:flex;align-items:flex-end;gap:0;
    background:var(--bg-input);
    border:1.5px solid var(--border-inp);
    border-radius:14px;
    padding:5px 7px 5px 3px;
    box-shadow:var(--shadow-inp);
    transition:border-color .2s,box-shadow .2s;
}
.input-pill:focus-within{
    border-color:var(--accent)!important;
    box-shadow:0 0 0 3px rgba(16,163,127,.12),var(--shadow-inp)!important;
}

.mic-btn button{
    background:transparent!important;border:none!important;
    border-radius:9px!important;color:var(--text-sub)!important;
    font-size:18px!important;min-width:38px!important;width:38px!important;height:38px!important;
    cursor:pointer!important;transition:all .15s!important;
    flex-shrink:0!important;display:flex!important;
    align-items:center!important;justify-content:center!important;padding:0!important;
    box-shadow:none!important;
}
.mic-btn button:hover{background:var(--bg-hover)!important;color:var(--accent)!important}

.msg-input{flex:1!important;min-width:0!important}
.msg-input textarea{
    background:transparent!important;border:none!important;border-radius:0!important;
    color:var(--text)!important;padding:8px!important;
    font-size:15px!important;font-family:'Inter',sans-serif!important;
    resize:none!important;line-height:1.55!important;
    box-shadow:none!important;outline:none!important;min-height:38px!important;
}
.msg-input textarea:focus{outline:none!important;box-shadow:none!important}
.msg-input textarea::placeholder{color:var(--text-faint)!important}

.send-btn button{
    background:var(--bg-send)!important;border:none!important;
    border-radius:9px!important;color:#fff!important;
    font-size:16px!important;min-width:34px!important;width:34px!important;height:34px!important;
    cursor:pointer!important;transition:all .15s!important;
    flex-shrink:0!important;display:flex!important;
    align-items:center!important;justify-content:center!important;padding:0!important;
    box-shadow:none!important;
}
.send-btn button:hover{background:var(--bg-send-h)!important;transform:scale(1.06)!important}
.send-btn button:active{transform:scale(.96)!important}

/* ── Voice status ── */
.voice-status{
    min-height:0!important;
    height:20px!important;
    overflow:hidden!important;
    margin-top:4px!important;
}
.voice-status textarea,
.voice-status input{
    background:transparent!important;border:none!important;
    color:var(--accent)!important;font-size:11px!important;font-weight:500!important;
    text-align:center!important;resize:none!important;
    padding:0 8px!important;line-height:20px!important;
    height:20px!important;min-height:20px!important;max-height:20px!important;
    box-shadow:none!important;outline:none!important;
    overflow:hidden!important;
}
.voice-status .wrap,
.voice-status>div{
    padding:0!important;border:none!important;background:transparent!important;
    min-height:0!important;height:20px!important;
}

.input-footer{
    display:flex;align-items:center;justify-content:center;
    padding-top:6px;gap:6px;
}
.footer-hint{font-size:10.5px;color:var(--text-faint)}
.hint-sep{color:var(--border);font-size:10px}
kbd{
    background:var(--bg-hover);border:1px solid var(--border);
    border-radius:4px;padding:1px 5px;font-size:10px;font-family:inherit;
}

/* ══ DELETE CONFIRM MODAL ══ */
.del-modal{
    display:none;position:fixed;inset:0;
    background:rgba(0,0,0,.55);
    z-index:2000;
    align-items:center;justify-content:center;
    backdrop-filter:blur(4px);
}
.del-modal.open{display:flex!important}
.del-card{
    background:var(--bg);
    border:1px solid var(--border);
    border-radius:16px;padding:26px 26px 20px;
    max-width:340px;width:calc(100% - 40px);
    box-shadow:0 24px 64px rgba(0,0,0,.28);
    animation:modal-in .18s ease;
}
@keyframes modal-in{from{transform:scale(.93);opacity:0}to{transform:scale(1);opacity:1}}
.del-icon{font-size:34px;margin-bottom:12px}
.del-title{font-size:16px;font-weight:700;color:var(--text);margin-bottom:7px}
.del-body{font-size:13px;color:var(--text-sub);line-height:1.65;margin-bottom:20px}
.del-btns{display:flex;gap:9px}
.del-cancel{
    flex:1;background:var(--bg-hover);
    border:1px solid var(--border);border-radius:10px;
    color:var(--text);font-size:13px;font-weight:500;
    padding:9px;cursor:pointer;font-family:'Inter',sans-serif;transition:background .15s;
}
.del-cancel:hover{background:var(--bg-active)}
.del-confirm-btn{
    flex:1;background:var(--red);border:none;border-radius:10px;
    color:#fff;font-size:13px;font-weight:600;
    padding:9px;cursor:pointer;font-family:'Inter',sans-serif;transition:opacity .15s;
}
.del-confirm-btn:hover{opacity:.88}

/* ══ Mobile overlay ══ */
.mob-overlay{
    display:none;position:fixed;inset:0;
    background:rgba(0,0,0,.45);z-index:150;backdrop-filter:blur(2px);
}
.mob-overlay.open{display:block!important}
.ham-btn{display:none!important}

/* ══ Responsive ══ */
@media(max-width:768px){
    .sidebar{
        position:fixed!important;left:0!important;top:0!important;
        transform:translateX(-100%)!important;
        box-shadow:4px 0 24px rgba(0,0,0,.2)!important;
    }
    .sidebar.open{transform:translateX(0)!important}
    .ham-btn{display:flex!important}
    .tb-sub{display:none!important}
    .live-badge{display:none!important}
    .chat-box .message-wrap>div{padding:12px 14px!important}
    .chat-box .message-wrap .message.user{max-width:82%!important}
    .input-bar{padding:10px 10px 12px!important}
}
"""

# FIX 3: confirmDelModal now resets the input to "" after triggering delete,
#         so the same session ID can trigger .change again if needed.
JS = """
<script>
/* ── Theme ── */
function _applyTheme(t,save){
    [document.documentElement,document.body,
     ...document.querySelectorAll('gradio-app,.gradio-container,#root,.main,.wrap')]
    .forEach(el=>el&&el.setAttribute('data-theme',t));
    const b=document.getElementById('theme-btn');
    if(b)b.textContent=t==='dark'?'☀️':'🌙';
    if(save!==false)localStorage.setItem('medai_theme',t);
}
function toggleTheme(){
    _applyTheme(document.documentElement.getAttribute('data-theme')==='dark'?'light':'dark');
}
(()=>_applyTheme(localStorage.getItem('medai_theme')||'light',false))();
new MutationObserver(()=>{
    const s=localStorage.getItem('medai_theme')||'light';
    if(document.documentElement.getAttribute('data-theme')!==s)_applyTheme(s,false);
}).observe(document.body,{childList:true,subtree:false});

/* ── Mobile sidebar ── */
function toggleSidebar(){
    const sb=document.getElementById('sidebar'),ov=document.getElementById('mob-overlay');
    const open=sb&&sb.classList.toggle('open');
    ov&&ov.classList.toggle('open',open);
}
function closeSidebar(){
    document.getElementById('sidebar')?.classList.remove('open');
    document.getElementById('mob-overlay')?.classList.remove('open');
}

/* ── Universal Trigger Helper ── */
function triggerGradio(id, value) {
    const inp = document.getElementById(id);
    if (!inp) {
        console.warn('MedAI:', id, 'not found');
        return;
    }
    const el = inp.querySelector('textarea, input') || inp;
    const desc = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), 'value');
    if (desc && desc.set) {
        desc.set.call(el, value);
    } else {
        el.value = value;
    }
    el.dispatchEvent(new Event('input', { bubbles: true }));
}

/* ── Select chat from sidebar ── */
function selectChat(sid){
    closeSidebar();
    triggerGradio('chat-select-input', sid);
}

/* ── Delete modal ── */
function openDelModal(sid){
    window.chatToDelete = sid;
    document.getElementById('del-modal')?.classList.add('open');
}
function closeDelModal(){
    document.getElementById('del-modal')?.classList.remove('open');
}
function confirmDelModal(){
    closeDelModal();
    const sid = window.chatToDelete;
    if(!sid) return;

    triggerGradio('chat-delete-input', sid + "||" + Date.now());

    window.chatToDelete = null;
}
function openTopDelModal() {
    const activeEl = document.querySelector('.hist-active');
    if (activeEl) {
        window.chatToDelete = activeEl.getAttribute('title');
        document.getElementById('del-modal')?.classList.add('open');
    } else {
        alert("No active chat to delete.");
    }
}
</script>
"""

# ══════════════════════════════════════════════════════════
#  UI BUILD
# ══════════════════════════════════════════════════════════
with gr.Blocks(css=CSS, title="MedAI — Medical AI Assistant") as app:

    session_id_state    = gr.State(init_sid)
    all_histories_state = gr.State(json.dumps(stored))

    gr.HTML('<div class="mob-overlay" id="mob-overlay" onclick="closeSidebar()"></div>')

    gr.HTML("""
    <div class="del-modal" id="del-modal">
      <div class="del-card">
        <div class="del-icon">🗑️</div>
        <div class="del-title">Delete this conversation?</div>
        <div class="del-body">This chat and all its messages will be permanently removed. This cannot be undone.</div>
        <div class="del-btns">
          <button class="del-cancel" onclick="closeDelModal()">Cancel</button>
          <button class="del-confirm-btn" onclick="confirmDelModal()">Delete</button>
        </div>
      </div>
    </div>
    """)

    with gr.Row(elem_classes="app-shell"):

        # ══ SIDEBAR ══
        with gr.Column(elem_classes="sidebar", scale=0, elem_id="sidebar"):

            gr.HTML(f"""
            <div class="sb-header">
              <div class="sb-brand">
                <div class="sb-icon">⚕️</div>
                <div>
                  <div class="sb-name">MedAI</div>
                  <div class="sb-sub">Medical Intelligence</div>
                </div>
              </div>
              <div style="display:inline-flex;align-items:center;gap:5px;
                          background:rgba(16,163,127,.10);border:1px solid rgba(16,163,127,.22);
                          border-radius:20px;padding:3px 10px;
                          font-size:9.5px;font-weight:700;color:#10a37f;margin-top:2px;">
                <span style="width:5px;height:5px;background:#10a37f;border-radius:50%;
                             display:inline-block;animation:pulse 2s ease-in-out infinite;"></span>
                Live · Groq API
              </div>
            </div>
            """)

            with gr.Column(elem_classes="sb-new-chat-col"):
                new_chat_btn = gr.Button("＋  New Chat", elem_classes="new-chat-btn")

            gr.HTML('<div class="sb-sec-label">Recent Chats</div>')

            with gr.Column(elem_classes="sb-scroll"):
                sidebar_html = gr.HTML(init_sidebar)

            chat_select_input = gr.Textbox(elem_classes="hidden-input", elem_id="chat-select-input", show_label=False)

            gr.HTML(f"""
            <div class="sb-footer">
              <div class="sb-grid">
                <div class="sb-card"><div class="sb-card-label">Embedder</div><div class="sb-card-val">BioBERT</div></div>
                <div class="sb-card"><div class="sb-card-label">Search</div><div class="sb-card-val">FAISS</div></div>
                <div class="sb-card"><div class="sb-card-label">LLM</div><div class="sb-card-val">Llama 3.3</div></div>
                <div class="sb-card"><div class="sb-card-label">Voice</div><div class="sb-card-val">{'Web STT' if VOICE_INPUT_OK else 'Off'}</div></div>
              </div>
              <div class="disclaimer">
                <span style="font-size:15px;flex-shrink:0;">⚠️</span>
                <span class="disclaimer-text">Educational use only. Always consult a qualified physician.</span>
              </div>
            </div>
            """)

        # ══ MAIN ══
        with gr.Column(elem_classes="chat-main", scale=1):

            with gr.Row(elem_classes="topbar", equal_height=True):
                gr.HTML("""
                <div style="display:flex; width:100%; align-items:center; justify-content:space-between;">
                    <div class="tb-left">
                      <button class="tb-icon-btn ham-btn" id="ham-btn" onclick="toggleSidebar()">☰</button>
                      <div style="width:30px;height:30px;flex-shrink:0;
                                  background:linear-gradient(135deg,#10a37f,#059669);
                                  border-radius:50%;display:flex;align-items:center;
                                  justify-content:center;font-size:15px;
                                  box-shadow:0 1px 6px rgba(16,163,127,.3);">⚕️</div>
                      <div style="min-width:0;">
                        <div class="tb-title">MedAI</div>
                        <div class="tb-sub">RAG · BioBERT · Llama 3.3 70B · Groq</div>
                      </div>
                    </div>
                    <div class="tb-right">
                      <div class="live-badge"><div class="live-dot"></div>Live</div>
                      <button class="tb-icon-btn" id="del-top-btn" onclick="openTopDelModal()" title="Delete Current Chat" style="color:var(--red); border-color:var(--border);">🗑️</button>
                      <button class="tb-icon-btn" id="theme-btn" onclick="toggleTheme()" title="Toggle theme">🌙</button>
                    </div>
                </div>
                """)

            # Hidden Gradio trigger for chat deletion
            chat_delete_input = gr.Textbox(elem_classes="hidden-input", elem_id="chat-delete-input", show_label=False)

            chatbot = gr.Chatbot(
                value=init_msgs,
                height=None,
                show_label=False,
                elem_classes="chat-box",
                bubble_full_width=False,
                avatar_images=(None, None),
            )

            with gr.Column(elem_classes="input-bar"):
                gr.HTML('<div class="input-inner"><div class="input-pill">')

                with gr.Row(equal_height=True):
                    mic_btn   = gr.Button("🎤", scale=0, min_width=38, elem_classes="mic-btn")
                    msg_input = gr.Textbox(
                        placeholder="Describe your symptoms…",
                        show_label=False, scale=1, container=False,
                        lines=1, max_lines=6, elem_classes="msg-input"
                    )
                    send_btn  = gr.Button("↑", scale=0, min_width=34, elem_classes="send-btn")

                gr.HTML("</div>")  # close input-pill

                voice_status = gr.Textbox(
                    value="", show_label=False, interactive=False,
                    lines=1, max_lines=1,
                    elem_classes="voice-status",
                    container=False
                )

                gr.HTML("""
                <div class="input-footer">
                  <span class="footer-hint">Press <kbd>Enter</kbd> to send</span>
                  <span class="hint-sep">·</span>
                  <span class="footer-hint">Shift+Enter for new line</span>
                  <span class="hint-sep">·</span>
                  <span class="footer-hint">🎤 tap for voice</span>
                </div>
                </div>
                """)

    gr.HTML(JS)

    # ══ Event wiring ══
    new_chat_btn.click(
        fn=new_chat,
        inputs=[all_histories_state],
        outputs=[chatbot, session_id_state, all_histories_state, sidebar_html]
    )
    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot, session_id_state, all_histories_state],
        outputs=[msg_input, chatbot, session_id_state, all_histories_state, sidebar_html]
    )
    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot, session_id_state, all_histories_state],
        outputs=[msg_input, chatbot, session_id_state, all_histories_state, sidebar_html]
    )
    mic_btn.click(
        fn=voice_to_text,
        inputs=[],
        outputs=[msg_input, voice_status]
    )
    # FIX: delete now outputs all 4 needed states including all_histories_state
    chat_delete_input.change(
        fn=delete_chat,
        inputs=[chat_delete_input, session_id_state, all_histories_state],
        outputs=[chatbot, session_id_state, all_histories_state, sidebar_html]
    )
    # FIX: load_chat now outputs all_histories_state too so active session updates
    chat_select_input.change(
        fn=load_chat,
        inputs=[chat_select_input, all_histories_state],
        outputs=[chatbot, session_id_state, all_histories_state, sidebar_html]
    )

if __name__ == "__main__":
    print("🚀 MedAI v6 — Delete fixed · Load chat fixed · Voice status tiny · Input pinned bottom")
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
    