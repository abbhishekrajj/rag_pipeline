import streamlit as st
import requests
import hashlib

#langsmith setup
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG_APP"



API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="RAG Assistant 🤖")
st.title("🤖 RAG Document Assistant")

# -----------------------------
# 🔥 CACHE FUNCTION (MAIN FIX)
# -----------------------------
@st.cache_data(show_spinner=False)
def call_rag_api(question, evaluate_flag):
    payload = {
        "question": question,
        "evaluate": evaluate_flag
    }
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"API Error: {response.status_code}"}

# -----------------------------
# UI INPUT
# -----------------------------
question = st.text_input("How Can I help you?:")
evaluate = st.checkbox("Enable Evaluation (RAGAS)")

# -----------------------------
# SESSION STATE (history)
# -----------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# BUTTON ACTION
# -----------------------------
if st.button("Ask"):

    if question:
        with st.spinner("Working for you..."):
            try:
                # 🔥 Cached API call
                data = call_rag_api(question, evaluate)

                #response = requests.post(API_URL, json={"question": question})

                #if response.status_code == 200:
                    #data = response.json()
                if "error" not in data: #include

                    # Save history
                    st.session_state.history.append((question, data.get("answer"))) #include

                    # Answer
                    st.subheader("📌 Answer")
                    st.write(data.get("answer"))

                    # Summary
                    st.subheader("🧾 Summary")
                    st.write(data.get("summary"))

                    # Sources
                    st.subheader("📚 Sources")
                    for src in data.get("sources", []):
                        st.write(src)

                    # Evaluation (future)
                    if "evaluation" in data:
                            st.subheader("📊 Evaluation")
                            st.write(data["evaluation"])
                    #clear Cache
                    if st.button("🔄 Clear Cache"):
                            st.cache_data.clear()

                else:
                    st.error(data["error"]) #include
                    #st.error(f"API Error: {response.status_code}")


            except Exception as e:
                st.error(f"Error: {str(e)}")


# -----------------------------
# 🔥 CHAT HISTORY (UX BOOST)
# -----------------------------
if st.session_state.history:
    st.subheader("💬 Chat History")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")