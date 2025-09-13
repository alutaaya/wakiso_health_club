# app.py
# Streamlit version of the Uganda HIV/AIDS Assistant

import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader

# --- 0. Load environment variables (for local dev) ---
load_dotenv()

# --- 1. PDF Download from Google Drive ---
PDF_FILE_ID = "11V9eOH2XHYrPl0kRnJGbJeGGxxvLuPmZ"  # Google Drive file ID
PDF_PATH = "Wakiso_Health_Club.pdf"
FAISS_INDEX_PATH = "faiss_index"

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive given a file ID."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            f.write(chunk)

    # Verify PDF
    try:
        reader = PdfReader(destination)
        if len(reader.pages) == 0:
            raise ValueError("PDF has no pages")
    except Exception as e:
        st.error(f"Downloaded file is not a valid PDF: {e}")
        if os.path.exists(destination):
            os.remove(destination)
        raise

# Download PDF if it doesn't exist or is too small
if not os.path.exists(PDF_PATH) or os.path.getsize(PDF_PATH) < 1000:
    st.info("Downloading PDF…")
    download_file_from_google_drive(PDF_FILE_ID, PDF_PATH)
    st.success("PDF downloaded successfully!")

# --- 2. Streamlit Page Config ---
st.set_page_config(page_title="WHC Assistant", layout="wide")

# --- 3. Load or Create Resources with Caching ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def load_vectorstore():
    hf_embed = load_embeddings()
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            return FAISS.load_local(
                FAISS_INDEX_PATH, 
                hf_embed, 
                allow_dangerous_deserialization=True
            )
        except Exception:
            st.warning("Error loading FAISS index. Rebuilding index.")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    text_chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(text_chunks, hf_embed)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

@st.cache_resource
def load_llm():
    groq_api = st.secrets.get("groq_api") or os.getenv("groq_api")

    if not groq_api:
        st.error("❌ ERROR: groq_api not found. Please set it in Streamlit secrets or .env file.")
        return None

    return ChatGroq(
        model_name="openai/gpt-oss-120b",
        temperature=0,
        api_key=groq_api
    )

# --- 4. Core Functions ---
def retrieve_relevant_chunks(query, vectorstore):
    if vectorstore:
        return vectorstore.similarity_search(query, k=12)
    return []

def answer_query(query, llm, vectorstore):
    if not llm:
        return "Error: Language Model unavailable."
    if not vectorstore:
        return "Error: Vector store unavailable."

    docs = retrieve_relevant_chunks(query, vectorstore)
    if not docs:
        return "I could not find relevant information in the document."

    context = "\n\n".join([d.page_content for d in docs])
    prompt = (
        "You are a legal assistant. Answer using ONLY the provided context. "
        "If the answer is not in the context, say you don't know. Be concise.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{query}\nAnswer:"
    )
    try:
        response = llm.invoke(prompt)
        return response.content.strip() if response else "Error: Empty response."
    except Exception as e:
        return f"Error invoking LLM: {e}"

# --- 5. Streamlit UI ---
def main():
    st.title("WHC Assistant Chatbot")
    st.write("Built by **Alfred Lutaaya** | Based on *WHC Constitution*.")

    # Debug: see which secrets are available
    st.write("DEBUG: Available secrets →", list(st.secrets.keys()))

    vectorstore = load_vectorstore()
    llm = load_llm()

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # User input
    user_input = st.text_input("Enter your question:")

    if st.button("Ask") and user_input:
        answer = answer_query(user_input, llm, vectorstore)
        st.session_state["chat_history"].append((user_input, answer))

    if st.button("Clear Chat"):
        st.session_state["chat_history"] = []

    # Display chat history
    for q, a in st.session_state["chat_history"]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")
        st.markdown("---")

if __name__ == "__main__":
    main()
