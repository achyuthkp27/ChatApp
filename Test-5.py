from __future__ import annotations

import difflib
import json
import logging
import os
import random
import re
import time
import uuid
import warnings
from typing import List, Dict, Tuple

import fitz
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)
USE_QDRANT = bool(os.environ.get("QDRANT_URL"))
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "bank_kb")
try:
    import faiss

    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, Distance

    HAS_QDRANT_CLIENT = True
except Exception:
    HAS_QDRANT_CLIENT = False

GEMINI_API_KEY = "AIzaSyCcYJknjfIS9VCFgrTainOfpDM3jim2fc0"
GEMINI_GEN_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
VERIFY_SSL = False

PDF_PATH = os.environ.get("RETAIL_PDF", "retail.pdf")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
TOP_K = int(os.environ.get("TOP_K", 5))
EMBEDDING_FILE = "doc_vectors.npy"
CHUNKS_FILE = "doc_chunks.json"

logging.basicConfig(level=logging.INFO, filename="chatbot.log", format="%(asctime)s %(levelname)s %(message)s")
app = Flask(__name__)
CORS(app)

SESSIONS: Dict[str, Dict] = {}
CHUNKS: List[Dict] = []
EMBED_MATRIX = None
FAISS_INDEX = None
QDRANT_CLIENT = None
EMBED_DIM = None

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

# --- Followup Detection Config Loading ---
FOLLOWUP_CONFIG_PATH = "followup_patterns.json"
DEFAULT_CONFIG = {
    "pronouns": [
        "same page", "there", "it", "that", "above", "as before", "update that"
    ],
    "trigger_phrases": [
        "can i", "how about", "what about"
    ],
    "min_length": 8,
    "similarity_threshold": 0.7
}
if os.path.exists(FOLLOWUP_CONFIG_PATH):
    with open(FOLLOWUP_CONFIG_PATH, "r", encoding="utf-8") as f:
        FOLLOWUP_CONFIG = json.load(f)
else:
    FOLLOWUP_CONFIG = DEFAULT_CONFIG
# ---------------------------------------

def redact_pii(text: str) -> Tuple[str, List[str]]:
    hits = []
    out = text
    for p in PII_PATTERNS:
        for m in p.findall(out):
            hits.append(m)
        out = p.sub("[REDACTED]", out)
    if hits:
        logging.info(f"PII redacted: {hits}")
    return out, hits

def strip_injection_bait(text: str) -> str:
    t = text
    for p in INJECTION_BAITS:
        t = p.sub("", t)
    if t != text:
        logging.warning("Injection bait detected and removed")
    return t.strip()

def load_pdf(path: str) -> str:
    logging.info(f"Loading PDF from {path}")
    doc = fitz.open(path)
    return "\n".join([page.get_text("text") for page in doc])

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
    logging.info(f"Chunked document into {len(chunks)} pieces")
    return chunks

def persist_embeddings_and_chunks(vectors: np.ndarray, chunks: List[Dict], pdf_path: str):
    np.save(EMBEDDING_FILE, vectors)
    meta = {"chunks": chunks, "pdf_path": pdf_path}
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    logging.info("Saved vectors and chunk metadata")

def load_embeddings_and_chunks():
    if not (os.path.exists(EMBEDDING_FILE) and os.path.exists(CHUNKS_FILE)):
        logging.info("No cached embeddings/chunks found")
        return None, None
    vecs = np.load(EMBEDDING_FILE)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        d = json.load(f)
        return vecs, d["chunks"]
    return None, None

def gemini_embed(texts: List[str]) -> List[List[float]]:
    GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    embeddings = []
    for i, text in enumerate(texts):
        for attempt in range(5):
            try:
                body = {
                    "model": "models/gemini-embedding-001",
                    "content": {"parts": [{"text": text}]}
                }
                logging.info(f"Embedding chunk {i} (attempt {attempt + 1})")
                resp = requests.post(
                    GEMINI_EMBED_URL,
                    headers=headers,
                    params=params,
                    json=body,
                    timeout=20,
                    verify=VERIFY_SSL
                )
                if resp.status_code == 429:
                    wait_time = (2 ** attempt) + random.random()
                    logging.warning(f"429 Too Many Requests, sleeping {wait_time}s")
                    time.sleep(wait_time)
                    continue
                resp.raise_for_status()
                j = resp.json()
                embeddings.append(j["embedding"]["values"])
                break
            except Exception as e:
                logging.warning(f"Chunk {i} embed failed: {e}")
                time.sleep(1.5 * (attempt + 1))
        else:
            logging.error(f"Chunk {i} failed all retries, using random vector")
            dim = 1536
            embeddings.append(np.random.randn(dim).tolist())
        time.sleep(0.5)
    logging.info(f"Finished embedding {len(embeddings)} chunks")
    return embeddings

def init_qdrant_client():
    global QDRANT_CLIENT
    if not HAS_QDRANT_CLIENT:
        logging.error("qdrant-client not installed")
        raise RuntimeError("qdrant-client not installed")
    QDRANT_CLIENT = QdrantClient(url=QDRANT_URL)
    try:
        QDRANT_CLIENT.recreate_collection(collection_name=QDRANT_COLLECTION,
                                          vectors_config={"size": EMBED_DIM, "distance": "Cosine"})
        logging.info("Qdrant collection ready")
    except Exception as e:
        logging.warning(f"Qdrant setup ex: {e}")

def index_to_qdrant(vectors: List[List[float]], chunks_meta: List[Dict]):
    points = [PointStruct(id=meta["id"], vector=vec, payload={"text": meta["text"]})
              for vec, meta in zip(vectors, chunks_meta)]
    QDRANT_CLIENT.upsert(collection_name=QDRANT_COLLECTION, points=points)
    logging.info(f"Upserted {len(points)} Qdrant points")

def init_faiss_index(vectors: np.ndarray):
    global FAISS_INDEX
    dim = vectors.shape[1]
    idx = faiss.IndexFlatIP(dim)
    FAISS_INDEX = idx
    FAISS_INDEX.add(vectors)
    logging.info(f"FAISS index populated ({vectors.shape[0]} entries)")

def search_faiss(query_vec: np.ndarray, top_k=TOP_K) -> List[Tuple[int, float]]:
    if query_vec.ndim == 1: query_vec = query_vec.reshape(1, -1)
    faiss.normalize_L2(query_vec)
    if FAISS_INDEX is None:
        logging.error("No FAISS index on search")
        return []
    D, I = FAISS_INDEX.search(query_vec, top_k)
    logging.info(f"FAISS search: {I[0].tolist()}")
    return list(zip(I[0].tolist(), D[0].tolist()))

def search_qdrant(query_vec: List[float], top_k=TOP_K):
    logging.info("Qdrant search requested")
    res = QDRANT_CLIENT.search(collection_name=QDRANT_COLLECTION, query_vector=query_vec, limit=top_k)
    return [(r.id, r.score, r.payload) for r in res]

def build_index_from_pdf(pdf_path: str):
    global CHUNKS, EMBED_MATRIX, EMBED_DIM
    vecs, chunks = load_embeddings_and_chunks()
    if vecs is not None and chunks is not None:
        CHUNKS = chunks
        EMBED_MATRIX = vecs
        EMBED_DIM = vecs.shape[1]
        logging.info("Loaded cached embeddings/chunks")
        if USE_QDRANT and HAS_QDRANT_CLIENT:
            init_qdrant_client()
            index_to_qdrant(vecs.tolist(), CHUNKS)
        elif HAS_FAISS:
            init_faiss_index(vecs)
        else:
            logging.error("No index option available")
        return
    doc = load_pdf(pdf_path)
    raw_chunks = chunk_text(doc)
    CHUNKS = [{"id": f"chunk_{i}", "text": c} for i, c in enumerate(raw_chunks)]
    texts = [c["text"] for c in CHUNKS]
    vectors = gemini_embed(texts)
    vecs = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    EMBED_MATRIX = vecs
    EMBED_DIM = vecs.shape[1]
    persist_embeddings_and_chunks(vecs, CHUNKS, pdf_path)
    if USE_QDRANT and HAS_QDRANT_CLIENT:
        init_qdrant_client()
        index_to_qdrant(vectors, CHUNKS)
    elif HAS_FAISS:
        init_faiss_index(vecs)
    else:
        logging.error("No index option available")
    logging.info(f"Document indexed: {len(CHUNKS)} chunks, dim={EMBED_DIM}")

def sentence_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def is_followup_question(text: str, session_id: str) -> bool:
    config = FOLLOWUP_CONFIG
    pronouns = config["pronouns"]
    triggers = config["trigger_phrases"]
    min_length = config.get("min_length", 8)
    similarity_threshold = config.get("similarity_threshold", 0.7)

    text_lower = text.lower()
    if any(p in text_lower for p in pronouns + triggers):
        return True

    hist = SESSIONS.get(session_id, {}).get("history", [])
    user_texts = [m["text"] for m in hist if m["role"] == "user"]
    if user_texts and len(user_texts) > 1:
        prev_q = user_texts[-2]
        sim = sentence_similarity(text, prev_q)
        if sim > similarity_threshold:
            return False

    if len(text.split()) < min_length and any(w in text_lower for w in triggers):
        return True

    return False

def update_last_topic(session_id: str, text: str):
    if "last_topic" not in SESSIONS[session_id]:
        SESSIONS[session_id]["last_topic"] = text
    else:
        if not is_followup_question(text, session_id):
            SESSIONS[session_id]["last_topic"] = text

def get_last_assistant_response(session_id: str) -> str:
    hist = SESSIONS.get(session_id, {}).get("history", [])
    for msg in reversed(hist):
        if msg["role"] == "assistant":
            return msg["text"]
    return ""

def retrieve_chunks_by_embedding(query: str, session_id: str, top_k=TOP_K) -> List[Dict]:
    include_last = is_followup_question(query, session_id)
    logging.info(f"[DEBUG] is_followup_question: {include_last}")
    chunks_out = []
    # Log last_topic for debug
    last_topic_val = SESSIONS[session_id].get("last_topic")
    logging.info(f"[DEBUG] last_topic: {last_topic_val}")
    if include_last:
        last_resp = get_last_assistant_response(session_id)
        if last_resp:
            chunks_out.append({"id": "last_assistant", "text": last_resp, "score": 1.0})
            logging.info("Using last assistant response for follow-up")
    # Use last topic for followup retrieval instead of vague query
    if include_last and "last_topic" in SESSIONS[session_id]:
        topic_query = SESSIONS[session_id]["last_topic"]
        q_vecs = gemini_embed([topic_query])
        logging.info(f"[DEBUG] query used for embedding: {topic_query}")
    else:
        q_vecs = gemini_embed([query])
        logging.info(f"[DEBUG] query used for embedding: {query}")
    qv = np.array(q_vecs, dtype=np.float32)[0]
    qv = qv / (np.linalg.norm(qv) + 1e-12)
    if USE_QDRANT and HAS_QDRANT_CLIENT:
        results = search_qdrant(qv.tolist(), top_k=top_k)
        for id_, score, payload in results:
            if include_last and ('last_assistant' == id_ or
                                 (chunks_out and sentence_similarity(payload.get("text", ""),
                                                                     chunks_out[0]["text"]) > 0.9)):
                continue
            chunks_out.append({"id": id_, "text": payload.get("text", ""), "score": score})
    elif HAS_FAISS:
        hits = search_faiss(qv, top_k=top_k)
        for idx, score in hits:
            if idx < 0 or idx >= len(CHUNKS): continue
            ct = CHUNKS[idx]["text"]
            if include_last and (chunks_out and sentence_similarity(ct, chunks_out[0]["text"]) > 0.9):
                continue
            chunks_out.append({"id": CHUNKS[idx]["id"], "text": ct, "score": float(score)})
    else:
        logging.error("No retrieval method available")
    return chunks_out[:top_k]

ANNIE_PROMPT_TEMPLATE = """
You are Annie, the official, friendly, and professional Virtual Assistant for First Citizens Bank.
Core Principles:
- Embody our brand values: **Committed to Excellence, Integrity, People, and Customers**.
- Maintain highest standards of trust, security, and compliance. Never request, reveal, or reference any confidential data (PIN, CVV, account numbers, passwords, OTP, address, etc.).
- Strictly answer only using the provided CONTEXT below and recent CONVERSATION HISTORY.
Customer Experience Rules:
- Treat every customer with respect, warmth, and professionalism.
- If the message includes appreciation, praise, or thanks, start your reply with a polite acknowledgment.
- For follow-up questions or references to recent answers (e.g., "same page", "as above", etc.), respond naturally and concisely—confirm or summarize info without repeating all details unless the user explicitly asks.
- Do not improvise, guess, or give answers not found in the CONTEXT; if unsure, politely recommend contacting a First Citizens banker, using online banking, or official support channels.
- Use clear steps, short paragraphs, or bullet points. Favor accuracy and helpfulness over excessive detail.
- Never repeat greetings if the user has already greeted in the session.
- Stay up to date with banking hours, digital access, and know when to route complex requests to live agents or secure channels.
Privacy & Security:
- Remind users about protecting their personal information as appropriate.
- If the user's question is too broad or vague (e.g., "tell me everything"), ask them to specify which area of banking or support they need help with before providing advice.
- Do not provide exhaustive lists or sensitive process details unless the customer has made a clearly relevant, specific request.
- If the query relates to privacy, security, compliance, or ethical concerns, refer to official First Citizens Bank policies and support channels only.
Brand Language & Voice:
- Reflect the “Forever First” mission: People first, money second. Prioritize customer well-being, clarity, reliability, and support.
- Comply with FDIC, privacy, and code-of-conduct requirements.

CONTEXT:
{context_text}
CONVERSATION HISTORY:
{conversation_history}
CUSTOMER MESSAGE:
{user_question}
Annie’s Response:
"""

def call_gemini_generation(context: str, user_question: str, session_id: str) -> str:
    history = SESSIONS.get(session_id, {}).get("history", [])
    formatted_history = "\n".join([f"{h['role'].capitalize()}: {h['text']}" for h in history[-8:]])
    prompt = ANNIE_PROMPT_TEMPLATE.format(
        context_text=context,
        conversation_history=formatted_history,
        user_question=user_question,
    )
    logging.info("Calling Gemini generation API")
    params = {"key": GEMINI_API_KEY}
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(GEMINI_GEN_URL, params=params, json=body, timeout=20, verify=VERIFY_SSL)
        resp.raise_for_status()
        j = resp.json()
        text = j.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        if not text.strip():
            text = "I couldn’t find an answer at the moment. Please try again later."
        return text.strip()
    except Exception as e:
        logging.exception("Gemini generation failed")
        return "I'm facing a temporary issue fetching details. Please try again shortly."

FUNCTIONAL_INTENTS = ["account_balance_check", "fund_transfer", "order_cheque_book", "stop_cheque"]

def detect_intent(text: str) -> str:
    tl = text.lower()
    if "balance" in tl: return "account_balance_check"
    if "transfer" in tl or "send money" in tl: return "fund_transfer"
    if "cheque book" in tl or "checkbook" in tl: return "order_cheque_book"
    if "stop cheque" in tl or "stop check" in tl: return "stop_cheque"
    return "faq"

def init_session(sid: str):
    if sid not in SESSIONS:
        logging.info(f"Initializing new session: {sid}")
        SESSIONS[sid] = {"history": []}

@app.route("/health", methods=["GET"])
def health():
    logging.info("Health check called")
    return jsonify({"ok": True, "ts": int(time.time())})

@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True)
    user_text_raw = payload.get("message", "")
    session_id = payload.get("session_id") or str(uuid.uuid4())[:16]

    logging.info(f"Chat request (session={session_id}): {user_text_raw}")

    user_text, pii_hits = redact_pii(user_text_raw)
    user_text = strip_injection_bait(user_text)
    if not user_text.strip():
        logging.error("Empty input after sanitization")
        return jsonify({"error": "empty_input"}), 400

    init_session(session_id)
    SESSIONS[session_id]["history"].append({"role": "user", "text": user_text, "ts": int(time.time())})

    update_last_topic(session_id, user_text)

    intent = detect_intent(user_text)
    logging.info(f"Intent detected: {intent}")

    if intent in FUNCTIONAL_INTENTS:
        slots = {}
        m = re.search(r"(?:rs\.?|₹)\s*([0-9,]+(?:\.[0-9]{1,2})?)", user_text, re.I)
        if m: slots["amount"] = float(m.group(1).replace(",", ""))
        if "neft" in user_text.lower(): slots["transfer_type"] = "NEFT"
        if "imps" in user_text.lower(): slots["transfer_type"] = "IMPS"
        if "rtgs" in user_text.lower(): slots["transfer_type"] = "RTGS"
        if "savings" in user_text.lower(): slots["from_account_type"] = "savings"
        if "current" in user_text.lower(): slots["from_account_type"] = "current"
        res = {"type": "functional_intent", "intent": intent, "slots": slots, "session_id": session_id,
               "pii_redacted": bool(pii_hits)}
        SESSIONS[session_id]["history"].append({"role": "assistant", "text": json.dumps(res), "ts": int(time.time())})
        logging.info(f"Functional intent handled: {res}")
        return jsonify(res)

    candidates = retrieve_chunks_by_embedding(user_text, session_id, top_k=TOP_K)
    logging.info(f"Retrieved {len(candidates)} context chunks for chat")
    if not candidates:
        logging.warning("No FAQ chunks found")
        return jsonify({"type": "faq", "answer": "I couldn't find relevant info in retail docs.", "citations": [],
                        "session_id": session_id})

    context_text = "\n\n---\n\n".join([c["text"] for c in candidates])
    answer = call_gemini_generation(context_text, user_text, session_id)
    SESSIONS[session_id]["history"].append({"role": "assistant", "text": answer, "ts": int(time.time())})
    citations = [c["id"] for c in candidates]
    logging.info(f"FAQ response sent: '{answer[:60]}...' citations={citations}")
    return jsonify({"type": "faq", "answer": answer, "citations": citations, "session_id": session_id,
                    "pii_redacted": bool(pii_hits)})

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        logging.error(f"Missing PDF at {PDF_PATH}")
        raise SystemExit(f"Please put retail PDF at {PDF_PATH} or set RETAIL_PDF env var")
    logging.info("Loading and indexing PDF or cached vectors/chunks...")
    build_index_from_pdf(PDF_PATH)
    logging.info("Ready. Serving on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
