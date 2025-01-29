import os
import re
import json
import uuid

import streamlit as st
from pathlib import Path

from llama_index.readers.file import PDFReader
from llama_index.llms.openai import OpenAI

from Resume import Resume
from dotenv import load_dotenv

load_dotenv()

def extract_links(text):
    url_pattern = r'(https?://[^\s]+)'
    return re.findall(url_pattern, text)

def clean_name_for_file(name):
    if '.' in name:
        name = name.rsplit('.', 1)[0]
    return '_'.join(name.split())


st.set_page_config(page_title="Resume Parser", page_icon="📄")
st.title("Structured Resume Parser")
st.write("#### This app extracts structured data from a resume in PDF format.")
st.divider()

if "structured_data" not in st.session_state:
    st.session_state.structured_data = None
    st.session_state.file_name = None

uploaded_file = st.file_uploader("Upload a Resume (PDF only)", type="pdf")

col1, col2, col3 = st.columns([3, 2, 3])

with col2:
    st.write("")
    read_button = st.button("Process Resume", use_container_width=True, type="primary")

if read_button and uploaded_file is None:
    st.info("Please upload a resume first to process.", icon="ℹ️")
elif read_button and uploaded_file is not None:
    try:
        temp_path = Path(f"temp_resume_{uuid.uuid4()}.pdf")
        with temp_path.open("wb") as f:
            f.write(uploaded_file.getbuffer())

        pdf_reader = PDFReader()
        documents = pdf_reader.load_data(file=temp_path)
        resume_text = documents[0].text

        links = extract_links(resume_text)

        openai_api_key = os.getenv("OPENAI_API_KEY", None)
        if not openai_api_key:
            st.error("OpenAI API key is missing. Please set it in secrets.toml.", icon="🔑")
        else:
            with st.spinner("Processing resume..."):
                llm = OpenAI(model="gpt-4o-mini", api_key=openai_api_key)
                sllm = llm.as_structured_llm(Resume)

                response = sllm.complete(resume_text)
                structured_data = response.raw

                structured_data.links = links

                full_name = clean_name_for_file(structured_data.full_name)
                st.session_state.file_name = f"{full_name}_resume_parsed.json"

                st.session_state.structured_data = json.loads(response.text)

                st.success("Resume processed successfully!", icon="✅")

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        temp_path.unlink()

st.divider()

if st.session_state.structured_data:
    
    col1, col2, col3 = st.columns([3, 1, 2])
    
    with col1:
        st.subheader("Extracted Resume Data")
    
    with col3:
        json_data = json.dumps(st.session_state.structured_data, indent=4)
        st.download_button(
            label="Download parsed JSON",
            data=json_data,
            file_name=st.session_state.file_name,
            mime="application/json",
            type="primary",
        )
    
    st.write("")
    st.json(st.session_state.structured_data)
    
    st.divider()