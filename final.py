import streamlit as st
from supabase import create_client 
from dotenv import load_dotenv
import os
# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)
# response = supabase.table("your_table_name").select("*").execute()
print("Supabase client initialized")
# ---------------- CONFIG ----------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- SESSION STATE ----------------
if "user" not in st.session_state:
    st.session_state.user = None

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "current_session" not in st.session_state:
    st.session_state.current_session = None

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []

if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = None

# ---------------- AUTH ----------------

# LOGIN
def login(email, password):
    try:
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        st.session_state.user = res.user
        st.session_state.user_id = res.user.id
    except:
        st.error("Login failed")


# # SIGNUP
def signup(email, password):
    try:
        res = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        st.success("Account created. Please login.")
        st.write(res)  # 👈 this shows what Supabase returns
    except Exception as e:
        st.error(f"Signup failed: {e}")  # 👈 THIS is the key fix
# def signup(email, password):
#     try:
#         supabase.auth.sign_up({
#             "email": email,
#             "password": password
#         })
#         st.success("Account created. Please login.")
#     except:
#         st.error("Signup failed")


# ---------------- SESSION ----------------
def create_session(user_id):
    res = supabase.table("sessions").insert({
        "user_id": user_id,
        "title": "New Chat"
    }).execute()
    return res.data[0]["id"]


def get_sessions(user_id):
    res = supabase.table("sessions")\
        .select("*")\
        .eq("user_id", user_id)\
        .order("created_at", desc=True)\
        .execute()
    return res.data


def update_session_title(session_id, title):
    supabase.table("sessions")\
        .update({"title": title[:40]})\
        .eq("id", session_id)\
        .execute()


# ---------------- CHAT HISTORY ----------------
def save_message(user_id, session_id, role, content):
    supabase.table("chat_history").insert({
        "user_id": user_id,
        "session_id": session_id,
        "role": role,
        "content": content
    }).execute()


def load_session_messages(session_id):
    res = supabase.table("chat_history")\
        .select("*")\
        .eq("session_id", session_id)\
        .order("created_at")\
        .execute()
    return res.data


# ---------------- STORAGE ----------------
def upload_to_supabase(file, user_id, session_id):
    path = f"{user_id}/{session_id}/{file.name}"
    supabase.storage.from_("documents").upload(path, file.getvalue())
    return path


def save_document(user_id, session_id, file_path):
    supabase.table("documents").insert({
        "user_id": user_id,
        "session_id": session_id,
        "file_path": file_path
    }).execute()


# ---------------- PDF ----------------
def read_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except:
        return ""


def chunk_text(text, size=300):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]


# ---------------- SEARCH ----------------
def search(query, top_k=3):
    try:
        if st.session_state.doc_embeddings is None:
            return [{"text": "No document uploaded.", "score": 0}]

        q = model.encode([query])
        scores = cosine_similarity(q, st.session_state.doc_embeddings)[0]
        idxs = np.argsort(scores)[-top_k:][::-1]

        return [{"text": st.session_state.doc_chunks[i], "score": scores[i]} for i in idxs]

    except:
        return [{"text": "Search failed.", "score": 0}]


# ---------------- AI ----------------
def generate_answer(query, context, mode):
    try:
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}"

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Answer using context."},
                {"role": "user", "content": prompt}
            ],
            timeout=30
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"


# ---------------- UI ----------------
def highlight(text, query):
    for w in query.split():
        text = text.replace(w, f"**{w}**")
    return text


def chat_app():

    st.title("🚀 StudyMate AI")

    # Sidebar
    st.sidebar.title("💬 Chats")

    if st.sidebar.button("➕ New Chat"):
        st.session_state.current_session = create_session(st.session_state.user_id)
        st.rerun()

    sessions = get_sessions(st.session_state.user_id)

    for s in sessions:
        if st.sidebar.button(s["title"]):
            st.session_state.current_session = s["id"]
            st.rerun()

    # Upload
    uploaded = st.file_uploader("📄 Upload PDF", type=["pdf"])

    if uploaded:

        if not st.session_state.current_session:
            st.session_state.current_session = create_session(st.session_state.user_id)

        path = upload_to_supabase(uploaded, st.session_state.user_id, st.session_state.current_session)
        save_document(st.session_state.user_id, st.session_state.current_session, path)

        text = read_pdf(uploaded)

        if not text.strip():
            st.error("PDF unreadable")
            return

        chunks = chunk_text(text)
        embeds = model.encode(chunks)

        st.session_state.doc_chunks = chunks
        st.session_state.doc_embeddings = embeds

        st.success("PDF processed!")

    # Load messages
    if st.session_state.current_session:
        msgs = load_session_messages(st.session_state.current_session)
        for m in msgs:
            st.chat_message(m["role"]).write(m["content"])

    # Chat input
    query = st.chat_input("Ask something...")

    if query:

        if not st.session_state.current_session:
            st.session_state.current_session = create_session(st.session_state.user_id)

        history = load_session_messages(st.session_state.current_session)

        if len(history) == 0:
            update_session_title(st.session_state.current_session, query)

        results = search(query)
        context = "\n\n".join([r["text"] for r in results])

        with st.spinner("Thinking..."):
            answer = generate_answer(query, context, "chat")

        save_message(st.session_state.user_id, st.session_state.current_session, "user", query)
        save_message(st.session_state.user_id, st.session_state.current_session, "assistant", answer)

        st.chat_message("assistant").write(answer)

        st.markdown("### 📚 Sources")
        for r in results:
            st.info(highlight(r["text"][:300], query))

# ---------------- MAIN ----------------

def main():

    # ------------------------
    # INIT MODE STATE
    # ------------------------
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

    # ------------------------
    # IF USER NOT LOGGED IN
    # ------------------------
    if not st.session_state.user:

        # Toggle buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Login Mode"):
                st.session_state.auth_mode = "login"

        with col2:
            if st.button("Signup Mode"):
                st.session_state.auth_mode = "signup"

        # ------------------------
        # LOGIN VIEW
        # ------------------------
        if st.session_state.auth_mode == "login":
            st.title("🔐 Login")

            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("Login"):
                login(email, password)

        # ------------------------
        # SIGNUP VIEW
        # ------------------------
        elif st.session_state.auth_mode == "signup":
            st.title("🆕 Signup")

            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")

            if st.button("Create Account"):
                signup(email, password)

    # ------------------------
    # IF LOGGED IN
    # ------------------------
    else:
        chat_app()


main()

# def main():

#     if not st.session_state.user:

#         st.title("🔐 Login")

#         email = st.text_input("Email")
#         password = st.text_input("Password", type="password")

#         if st.button("Login"):
#             login(email, password)

#         if st.button("Signup"):
#             signup(email, password)

#     else:
#         chat_app()


# main()
