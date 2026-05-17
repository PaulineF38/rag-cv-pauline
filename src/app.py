# Force Python to look at the project root folder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.core.rag_engine import query_rag

# ----configuration----
st.set_page_config(page_title="Pauline - Interactive CV", page_icon="💼", layout="centered")

# ----disclaimer----
st.warning(
    "⚠️ **Disclaimer:** This project is not intended to replace my traditional CV, "
    "but rather to demonstrate the technical skills acquired during my work-study program "
    "and training. Please note that this assistant relies on a Large Language Model (LLM), "
    "meaning it may generate inaccuracies."
)

st.title("RAG chatbot - The iteractive CV")
st.write("Ask my virtual assistant anything to discover my background and professional journey!")

# ----API and Provider Configuration----
st.sidebar.header("API Configuration")
provider = st.sidebar.selectbox(
    "Choose your LLM:", 
    ["OpenAI", "MistralAI", "Google Gemini"]
)
api_key = st.sidebar.text_input(f"Enter your {provider} API Key:", type="password")
st.sidebar.markdown("---")
st.sidebar.info(
    "**Security:** Your API key is processed solely within your browser session "
    "and is never saved or stored on any server."
)

# ---- Chat part ----

# save history
if "mssages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if user_query := st.chat_input("e.g., What are her Python and Data Science skills?"):
    
    # Strict validation: check if the API key is provided
    if not api_key:
        st.error("Please enter your API key in the sidebar to activate the chatbot.")
    else:
        # 1. Display user message and append to history
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # 2. Generate assistant response using the RAG core engine
        with st.chat_message("assistant"):
            with st.spinner("The assistant is searching the CV..."):
                # Call our modular RAG engine
                response = query_rag(user_query, provider, api_key)
                st.markdown(response)
                
        # 3. Append assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})