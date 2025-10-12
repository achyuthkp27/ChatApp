from __future__ import annotations
import os, re, time, json, uuid, logging
from typing import List, Dict, Tuple
import fitz
import requests
import numpy as np
import warnings
from urllib3.exceptions import InsecureRequestWarning
from flask import Flask, request, jsonify
from flask_cors import CORS

warnings.filterwarnings("ignore", category=InsecureRequestWarning)
# Optional libs (faiss or qdrant)
USE_QDRANT = bool(os.environ.get("QDRANT_URL"))
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "bank_kb")
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# qdrant-client (optional)
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, Distance
    HAS_QDRANT_CLIENT = True
except Exception:
    HAS_QDRANT_CLIENT = False

# ---------------- CONFIG ----------------
GEMINI_API_KEY = "AIzaSyB-QQVwdFb33KqPsbeJxbtnlVJkfoOrSAY"
GEMINI_GEN_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
VERIFY_SSL =  False

PDF_PATH = os.environ.get("RETAIL_PDF", "retail_faq.pdf")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 800))       # characters per chunk (~400-800 is common)
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200)) # overlap characters
TOP_K = int(os.environ.get("TOP_K", 5))

# Logging (PII-safe)
logging.basicConfig(level=logging.INFO, filename="chatbot.log",
                    format="%(asctime)s %(levelname)s %(message)s")
app = Flask(__name__)
CORS(app)

# In-memory stores
SESSIONS: Dict[str, Dict] = {}
CHUNKS: List[Dict] = []       # each item: {id, text, meta}
EMBED_MATRIX = None           # numpy array of shape (n, dim)
FAISS_INDEX = None
QDRANT_CLIENT = None
EMBED_DIM = None

# ---------------- Helpers: text sanitization ----------------
PII_PATTERNS = [
    re.compile(r"\b\d{12}\b"),  # Aadhaar-like
    re.compile(r"\b\d{16}\b"),  # card-like
    re.compile(r"\b\d{10}\b"),  # phone-like
    re.compile(r"[\w\.-]+@[\w\.-]+")  # emails
]
INJECTION_BAITS = [
    re.compile(r"(?i)ignore previous instructions"),
    re.compile(r"(?i)you are now .* assistant"),
    re.compile(r"(?i)reveal the system prompt"),
]

def redact_pii(text: str) -> Tuple[str, List[str]]:
    hits = []
    out = text
    for p in PII_PATTERNS:
        for m in p.findall(out):
            hits.append(m)
        out = p.sub("[REDACTED]", out)
    return out, hits

def strip_injection_bait(text: str) -> str:
    t = text
    for p in INJECTION_BAITS:
        t = p.sub("", t)
    return t.strip()

# ---------------- PDF loader & chunker ----------------
def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    return "\n".join(texts)

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# ---------------- Gemini embedding ----------------
def gemini_embed(texts: List[str]) -> List[List[float]]:
    GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}

    embeddings = []
    for text in texts:
        body = {
            "model": "models/gemini-embedding-001",
            "content": {
                "parts": [
                    {"text": text}
                ]
            }
        }

        try:
            resp = requests.post(GEMINI_EMBED_URL, headers=headers, params=params, json=body, timeout=20, verify=VERIFY_SSL)
            resp.raise_for_status()
            j = resp.json()
            embeddings.append(j["embedding"]["values"])
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error: {e}")
            logging.error(f"Response content: {resp.text}")
            dim = 1536
            embeddings.append(np.random.randn(dim).tolist())
    return embeddings


# ---------------- Vector index management (FAISS or Qdrant) ----------------
def init_qdrant_client():
    global QDRANT_CLIENT
    if not HAS_QDRANT_CLIENT:
        raise RuntimeError("qdrant-client not installed")
    url = QDRANT_URL
    QDRANT_CLIENT = QdrantClient(url=url)
    # create collection if not exists (simple on-disk/hardcoded config)
    try:
        QDRANT_CLIENT.recreate_collection(collection_name=QDRANT_COLLECTION,
                                          vectors_config={"size": EMBED_DIM, "distance": "Cosine"})
    except Exception:
        pass

def index_to_qdrant(vectors: List[List[float]], chunks_meta: List[Dict]):
    points = []
    for vec, meta in zip(vectors, chunks_meta):
        points.append(PointStruct(id=meta["id"], vector=vec, payload={"text": meta["text"]}))
    QDRANT_CLIENT.upsert(collection_name=QDRANT_COLLECTION, points=points)

def init_faiss_index(vectors: np.ndarray):
    global FAISS_INDEX
    dim = vectors.shape[1]
    idx = faiss.IndexFlatIP(dim)  # inner-product for cosine if vectors normalized
    FAISS_INDEX = idx
    FAISS_INDEX.add(vectors)

def search_faiss(query_vec: np.ndarray, top_k=TOP_K) -> List[Tuple[int, float]]:
    # returns list of (idx, score)
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    # if using IP, ensure vectors normalized
    faiss.normalize_L2(query_vec)
    if FAISS_INDEX is None:
        return []
    D, I = FAISS_INDEX.search(query_vec, top_k)
    return list(zip(I[0].tolist(), D[0].tolist()))

def search_qdrant(query_vec: List[float], top_k=TOP_K):
    res = QDRANT_CLIENT.search(collection_name=QDRANT_COLLECTION, query_vector=query_vec, limit=top_k)
    # return list of (id, score, payload)
    out = []
    for r in res:
        out.append((r.id, r.score, r.payload))
    return out

# ---------------- Build index at startup ----------------
def build_index_from_pdf(pdf_path: str):
    global CHUNKS, EMBED_MATRIX, EMBED_DIM
    full_text = load_pdf(pdf_path)
    raw_chunks = chunk_text(full_text)
    CHUNKS = []
    for i, c in enumerate(raw_chunks):
        CHUNKS.append({"id": f"chunk_{i}", "text": c})
    # batch embed
    texts = [c["text"] for c in CHUNKS]
    vectors = gemini_embed(texts)
    # convert to numpy matrix
    vecs = np.array(vectors, dtype=np.float32)
    # normalize for cosine
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    EMBED_MATRIX = vecs
    EMBED_DIM = vecs.shape[1]
    # index
    if USE_QDRANT and HAS_QDRANT_CLIENT:
        init_qdrant_client()
        index_to_qdrant(vectors.tolist(), CHUNKS)
    else:
        if not HAS_FAISS:
            logging.error("FAISS not available; install faiss-cpu or configure Qdrant")
        else:
            init_faiss_index(vecs)
    logging.info(f"Indexed {len(CHUNKS)} chunks, dim={EMBED_DIM}")

# ---------------- RAG retrieval using embeddings ----------------
def retrieve_chunks_by_embedding(query: str, top_k=TOP_K) -> List[Dict]:
    q_vecs = gemini_embed([query])
    qv = np.array(q_vecs, dtype=np.float32)[0]
    # normalize
    qv = qv / (np.linalg.norm(qv) + 1e-12)
    if USE_QDRANT and HAS_QDRANT_CLIENT:
        results = search_qdrant(qv.tolist(), top_k=top_k)
        out = []
        for id_, score, payload in results:
            out.append({"id": id_, "text": payload.get("text", ""), "score": score})
        return out
    else:
        if not HAS_FAISS:
            return []
        hits = search_faiss(qv, top_k=top_k)
        out = []
        for idx, score in hits:
            if idx < 0 or idx >= len(CHUNKS):
                continue
            out.append({"id": CHUNKS[idx]["id"], "text": CHUNKS[idx]["text"], "score": float(score)})
        return out

# ---------------- Gemini generator for answer ----------------
SYSTEM_PROMPT = (
    "You are a banking assistant. Answer user question ONLY using the provided CONTEXT. "
    "Do NOT hallucinate. If the context doesn't contain the answer, say you don't know and offer to escalate."
)

ANNIE_PROMPT_TEMPLATE = """
You are Annie, a highly professional and friendly banking assistant. Your primary role is to provide customers with clear, comprehensive, and empathetic support for their banking inquiries.

**Core Principles:**
1. **Empathy and Professionalism:** Always begin your response by acknowledging the user's query in a helpful and polite manner. Your tone should be reassuring and professional, just like a human customer service representative.
2. **Provide Detailed and Clear Information:** When an answer is available, provide a complete and easy-to-understand explanation. Avoid jargon and break down complex information into simple steps or bullet points where appropriate. Your goal is to ensure the user has a full understanding of the topic.
3. **Handle Ambiguity with Grace:** If a query is vague or requires more information, kindly and professionally ask for the necessary details. Frame your request as a way to "better assist" the user.
4. **Manage Unavailable Information:** If the information is not within your knowledge base, do not guess or provide a fabricated response. Politely inform the user of this limitation and direct them to the appropriate channel for further assistance.
5. **Always Offer Further Assistance:** Conclude every interaction by offering to help with anything else uniquely.

**Context:**
{context_text}

**User Query:**
{user_question}

**Assistant's Response:**
"""

def call_gemini_generation(context: str, user_question: str) -> str:
    context_prompt = ANNIE_PROMPT_TEMPLATE.format(
        context_text=context,
        user_question=user_question
    )
    prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{user_question}\n\nAnswer concisely."
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    body = {"contents": [{"parts": [{"text": context_prompt}]}]}
    try:
        resp = requests.post(GEMINI_GEN_URL, params=params, json=body, timeout=20, verify=VERIFY_SSL)
        resp.raise_for_status()
        j = resp.json()
        text = j.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return text
    except Exception as e:
        logging.exception("Gemini generation failed")
        return "Service temporarily unavailable."

# ---------------- Simple intent detection (unchanged) ----------------
FUNCTIONAL_INTENTS = ["account_balance_check", "fund_transfer", "order_cheque_book", "stop_cheque"]
def detect_intent(text: str) -> str:
    tl = text.lower()
    if "balance" in tl: return "account_balance_check"
    if "transfer" in tl or "send money" in tl: return "fund_transfer"
    if "cheque book" in tl or "checkbook" in tl: return "order_cheque_book"
    if "stop cheque" in tl or "stop check" in tl: return "stop_cheque"
    return "faq"

# ---------------- Session management (in-memory) ----------------
def init_session(sid: str):
    if sid not in SESSIONS:
        SESSIONS[sid] = {"history": []}

# ---------------- Flask routes ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "ts": int(time.time())})

@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True)
    user_text_raw = payload.get("message", "")
    session_id = payload.get("session_id") or str(uuid.uuid4())[:16]

    # sanitize & strip injection bait
    user_text, pii_hits = redact_pii(user_text_raw)
    user_text = strip_injection_bait(user_text)
    if not user_text.strip():
        return jsonify({"error": "empty_input"}), 400

    init_session(session_id)
    SESSIONS[session_id]["history"].append({"role": "user", "text": user_text, "ts": int(time.time())})

    intent = detect_intent(user_text)

    # Functional intents -> return intent only; UI will call banking APIs
    if intent in FUNCTIONAL_INTENTS:
        # Minimal slots detection (simple heuristics), return to UI
        slots = {}
        # naive amount extraction
        m = re.search(r"(?:rs\.?|₹)\s*([0-9,]+(?:\.[0-9]{1,2})?)", user_text, re.I)
        if m:
            slots["amount"] = float(m.group(1).replace(",", ""))
        # detect transfer type keywords
        if "neft" in user_text.lower(): slots["transfer_type"] = "NEFT"
        if "imps" in user_text.lower(): slots["transfer_type"] = "IMPS"
        if "rtgs" in user_text.lower(): slots["transfer_type"] = "RTGS"
        # account type
        if "savings" in user_text.lower(): slots["from_account_type"] = "savings"
        if "current" in user_text.lower(): slots["from_account_type"] = "current"

        res = {"type": "functional_intent", "intent": intent, "slots": slots, "session_id": session_id, "pii_redacted": bool(pii_hits)}
        SESSIONS[session_id]["history"].append({"role": "assistant", "text": json.dumps(res), "ts": int(time.time())})
        logging.info(f"session={session_id} intent={intent} slots={list(slots.keys())}")
        return jsonify(res)

    # FAQ path -> embedding retrieval + generation
    candidates = retrieve_chunks_by_embedding(user_text, top_k=TOP_K)
    if not candidates:
        return jsonify({"type": "faq", "answer": "I couldn't find relevant info in retail docs.", "citations": [], "session_id": session_id})

    # join top-k text as context (careful with length; here we send as-is)
    context_text = "\n\n---\n\n".join([c["text"] for c in candidates])
    answer = call_gemini_generation(context_text, user_text)
    SESSIONS[session_id]["history"].append({"role": "assistant", "text": answer, "ts": int(time.time())})
    citations = [c["id"] for c in candidates]
    logging.info(f"session={session_id} faq retrieved_chunks={len(candidates)}")
    return jsonify({"type": "faq", "answer": answer, "citations": citations, "session_id": session_id, "pii_redacted": bool(pii_hits)})

# ---------------- Startup: build index ----------------
if __name__ == "__main__":
    # Basic sanity
    if not os.path.exists(PDF_PATH):
        raise SystemExit(f"Please put retail PDF at {PDF_PATH} or set RETAIL_PDF env var")

    logging.info("Loading and indexing PDF...")
    build_index_from_pdf(PDF_PATH)
    logging.info("Ready. Serving on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
