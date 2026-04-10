import streamlit as st
from supabase import create_client
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import time

# venv
load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)

# ---------------- CONFIG ----------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(layout="wide")

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "user_email" not in st.session_state:
    st.session_state.user_email = None

if "current_session" not in st.session_state:
    st.session_state.current_session = None

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []

if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = None

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

# ---------------- AUTH ----------------
def login(email, password):
    try:
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        if res.user:
            st.session_state.logged_in = True
            st.session_state.user_id = str(res.user.id)
            st.session_state.user_email = res.user.email
            st.success("Login successful!")
            st.rerun()

    except Exception as e:
        st.error(f"Login failed: {e}")


def signup(email, password):
    try:
        supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        st.success("Account created. Now login.")
    except Exception as e:
        st.error(f"Signup failed: {e}")

# ---------------- DATABASE ----------------
def create_session():
    try:
        res = supabase.table("sessions").insert({
            "user_id": st.session_state.user_id,
            "title": "New Chat"
        }).execute()

        return res.data[0]["id"]
    except Exception as e:
        st.error(f"Session error: {e}")
        return None

def get_sessions():
    res = supabase.table("sessions")\
        .select("*")\
        .eq("user_id", st.session_state.user_id)\
        .order("created_at", desc=True)\
        .execute()
    return res.data

def save_message(role, content):
    supabase.table("chat_history").insert({
        "user_id": st.session_state.user_id,
        "session_id": st.session_state.current_session,
        "role": role,
        "content": content
    }).execute()

def load_messages():
    res = supabase.table("chat_history")\
        .select("*")\
        .eq("session_id", st.session_state.current_session)\
        .order("created_at")\
        .execute()
    return res.data

# ---------------- FILE ----------------
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for p in reader.pages:
        text += p.extract_text() or ""
    return text


def chunk_text(text, size=300):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]


def embed(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [r.embedding for r in res.data]


def search(query):
    if st.session_state.doc_embeddings is None:
        return []

    q = embed([query])
    scores = cosine_similarity(q, st.session_state.doc_embeddings)[0]
    idxs = np.argsort(scores)[-3:][::-1]

    return [st.session_state.doc_chunks[i] for i in idxs]

# ---------------- AI ----------------

def stream_answer(query, contexts, history):

    try:
        context_text = "\n\n".join(contexts) if contexts else ""

        messages = [{"role": "system", "content": "You are a helpful AI tutor."}]

        for msg in history[-5:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        messages.append({
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion:\n{query}"
        })

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    except Exception as e:
        yield f"⚠️ {str(e)}"


# ---------------- AUTH UI ----------------
def auth_ui():
    st.title("📘 StudyMate AI")

    col1, col2, col3 = st.columns([1, 1, 5])

    with col1:
       if st.button("Login", use_container_width=True):
           st.session_state.auth_mode = "login"

    with col2:
        if st.button("Sign Up", use_container_width=True):
            st.session_state.auth_mode = "signup"
   
    
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.session_state.auth_mode == "login":
        if st.button("Login Now"):
            login(email, password)
    else:
        if st.button("Create Account"):
            signup(email, password)

# ---------------- CHAT UI ----------------

def chat_ui():
    with st.sidebar:
        st.markdown(f"👤 {st.session_state.user_email}")

        if st.button("➕ New Chat"):
            sid = create_session()
            if sid:
                st.session_state.current_session = sid
                st.rerun()

        st.markdown("### Recent Chats")

        sessions = get_sessions()
        for s in sessions:
            if st.button(s["title"], key=s["id"]):
                st.session_state.current_session = s["id"]
                st.rerun()

        st.divider()

        if st.button(" Logout"):
            st.session_state.clear()
            st.rerun()

    st.title("💬 StudyMate AI")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded:
        text = read_pdf(uploaded)
        chunks = chunk_text(text)
        st.session_state.doc_chunks = chunks
        st.session_state.doc_embeddings = embed(chunks)
        st.success("PDF ready")

    if st.session_state.current_session:
        history = load_messages()

        for m in history:
            with st.chat_message(m["role"]):
                st.write(m["content"])

    query = st.chat_input("Ask something...")

    if query:
        with st.chat_message("user"):
            st.write(query)

        if not st.session_state.current_session:
            st.session_state.current_session = create_session()

        history = load_messages()
        contexts = search(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""

            for chunk in stream_answer(query, contexts, history):
                full += chunk
                placeholder.markdown(full + "▌")

            placeholder.markdown(full)

        save_message("user", query)
        save_message("assistant", full)

# ---------------- MAIN ----------------
def main():
    if not st.session_state.logged_in:
        auth_ui()
    else:
        chat_ui()

main()
