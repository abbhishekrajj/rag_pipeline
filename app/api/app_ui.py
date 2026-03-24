import streamlit as st
import requests

#langsmith setup
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG_APP"



API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="RAG Assistant 🤖")

st.title("🤖 RAG Document Assistant")

question = st.text_input("How Can I help you?:")

if st.button("Ask"):

    if question:
        with st.spinner("Thinking..."):

            response = requests.post(API_URL, json={"question": question})

            if response.status_code == 200:
                data = response.json()

                st.subheader("📌 Answer")
                st.write(data.get("answer"))

                st.subheader("🧾 Summary")
                st.write(data.get("summary"))

                st.subheader("📚 Sources")
                for src in data.get("sources", []):
                    st.write(src)

            else:
                st.error("API Error")