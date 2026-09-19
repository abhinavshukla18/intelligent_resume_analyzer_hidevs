import os
import tempfile
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page Configuration
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume & Job Description Analyzer")
st.markdown("Upload a candidate's resume and paste a job description to generate a semantic match analysis powered by **ChromaDB** and **Groq (Llama 3)**.")

# Sidebar - Configuration
with st.sidebar:
    st.header("🔑 Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your gsk_... key here")
    st.markdown("---")
    st.markdown("**Built with:**")
    st.markdown("- Python & Streamlit\n- LangChain & ChromaDB\n- HuggingFace Embeddings\n- Groq Llama 3")

# Main Input Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Candidate Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

with col2:
    st.subheader("2. Job Description")
    job_description = st.text_area("Paste Job Description Requirements Here", height=250)

# Process & Analyze Button
if st.button("🚀 Analyze Match", type="primary", use_container_width=True):
    if not groq_api_key:
        st.error("Please enter a valid Groq API Key in the sidebar.")
    elif not uploaded_file:
        st.error("Please upload a resume PDF file.")
    elif not job_description.strip():
        st.error("Please provide a job description.")
    else:
        with st.spinner("Processing PDF, generating vector embeddings, and querying Llama 3..."):
            try:
                # 1. Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_pdf_path = tmp_file.name

                # 2. Extract & Chunk PDF Text
                loader = PyPDFLoader(tmp_pdf_path)
                docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_documents(docs)

                # 3. Create In-Memory Vector Store
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_db = Chroma.from_documents(chunks, embeddings)

                # 4. Retrieve Top Relevant Chunks for Job Description
                retriever = vector_db.as_retriever(search_kwargs={"k": 5})
                matching_chunks = retriever.invoke(job_description)
                context_text = "\n\n".join([doc.page_content for doc in matching_chunks])

                # 5. Query Groq Llama 3 Model
                client = Groq(api_key=groq_api_key)
                
                # Fetch first valid text chat model
                active_models = client.models.list()
                chat_model = next(
                    (
                    m.id for m in active_models.data 
                    if any(keyword in m.id.lower() for keyword in ["llama", "qwen", "mixtral"])
                    and not any(exclude in m.id.lower() for exclude in ["guard", "vision", "whisper", "embed", "safeguard"])
                ),
                active_models.data[0].id
            )

                st.info(f"Using Groq model: `{chat_model}`")

                prompt = f"""
                You are an expert HR and Resume Analyzer. Compare the candidate's resume content against the Job Description.

                Job Description:
                {job_description}

                Candidate Resume Context:
                {context_text}

                CRITICAL INSTRUCTIONS:
                - DO NOT hallucinate or invent skills. 
                - Base your analysis STRICTLY on the text provided in the Candidate Resume Context.
                - If a skill is missing, explicitly state that it is missing.

                Please provide:
                1. Match Percentage (0-100%)
                2. Key Matching Skills
                3. Missing Skills / Gaps
                4. Final Recommendation
                """

                response = client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "user", "content": prompt}]
                )

                # Cleanup temporary file
                os.remove(tmp_pdf_path)

                # 6. Render Results
                st.success("Analysis Complete!")
                st.subheader("📊 AI Resume Analysis Report")
                st.markdown(response.choices[0].message.content)

            except Exception as e:
                st.error("An error occurred during analysis:")
                st.exception(e)  # <--- This will display the exact error traceback