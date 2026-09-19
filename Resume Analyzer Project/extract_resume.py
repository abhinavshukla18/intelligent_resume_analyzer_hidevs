import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq

reader = PdfReader("sample_resume.pdf")
resume_text = ""

for page in reader.pages:
    resume_text = resume_text + page.extract_text()

clean_text = resume_text.lower()
clean_text = re.sub(r'[^a-z0-9\s]', ' ', clean_text)
clean_text = " ".join(clean_text.split())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.create_documents([clean_text])

print(f"Total Chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.page_content)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)
print("\n Success! Resume chunks have been converted to embeddings and saved in ChromaDB.")


# --- STEP 5: QUERY PROCESSING & AI ENGINE ---

# 1. Read the sample Job Description
with open("sample_jd.txt", "r") as f:
    sample_jd = f.read()

# 2. Retrieve top matching resume chunks from ChromaDB
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
matching_chunks = retriever.invoke(sample_jd)

context_text = "\n\n".join([doc.page_content for doc in matching_chunks])

# 3. Connect to Groq
client = Groq(api_key="YOUR_GROQ_API_KEY")

# 4. Automatically find an active production model
available_models = client.models.list()
active_model = next(
    m.id for m in available_models.data 
    if "llama-3" in m.id.lower() or "mixtral" in m.id.lower() or "qwen" in m.id.lower()
)

print(f"Using active Groq model: {active_model}")

# 5. Create prompt instructions
prompt = f"""
You are an expert HR and Resume Analyzer. Compare the following candidate's resume content against the Job Description.

Job Description:
{sample_jd}

Candidate Resume Context:
{context_text}

Please provide:
1. Match Percentage (0-100%)
2. Key Matching Skills
3. Missing Skills / Gaps
4. Final Recommendation
"""

# 6. Generate and print AI analysis
print("\n--- AI RESUME ANALYSIS ---")
response = client.chat.completions.create(
    model=active_model,
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)