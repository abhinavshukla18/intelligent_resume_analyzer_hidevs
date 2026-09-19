#  AI-Powered Resume & Job Description Analyzer

**Live App:** : (https://resume-analyzer-ai-lynvrnotb44f46yjrshrxk.streamlit.app/)

##  Overview
This AI-Powered Resume Analyzer is a Retrieval-Augmented Generation (RAG) web application designed to evaluate candidate resumes against specific job descriptions. It automates the screening process by semantically matching skills and experience, providing recruiters and HR professionals with accurate, bias-reduced candidate assessments.

##  Tech Stack Used
* **Frontend:** Streamlit
* **LLM / Inference:** Groq (Llama 3)
* **Framework:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Document Processing:** PyPDFLoader

##  Features
* **Smart PDF Parsing:** Extracts and chunks text from uploaded candidate resumes.
* **Semantic Search:** Uses ChromaDB to retrieve the most relevant resume sections based on the job description requirements.
* **Optimized RAG Pipeline:** Fine-tuned chunk sizing and context retrieval to prevent AI hallucinations and strictly ground analysis in the candidate's actual text.
* **Automated Scoring:** Generates a 0-100% match score alongside key matching skills and explicitly identified gaps.

##  Local Installation
If you want to run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone [https://github.com/abhinavshukla18/resume-analyzer-ai.git](https://github.com/abhinavshukla18/resume-analyzer-ai.git)
   cd resume-analyzer-ai/Resume_Analyzer_Project
   ```

2. Install dependencies:
Make sure you have Python 3.8+ installed. Run the following command to install the required libraries:
    ``` bash
    pip install -r requirements.txt
    ```

3. Get a Groq API Key:

You will need a free API key from Groq to power the LLM.

1. Go to the Groq Console
2. Create an API key (gsk_...)

4. Run the application
    ```bash
    python -m streamlit run app.py
    ```
Note: Once the local server starts, you can paste your Groq API key directly into the sidebar of the web app.

## Under the Hood (RAG Pipeline Optimizations)

To ensure high accuracy and prevent the AI from inventing skills the candidate doesn't have, this pipeline uses the following parameters:

- **Chunk Size**: 1000 characters

- **Chunk Overlap**: 100 characters

- **Retrieval 'K'**: Top 5 most relevant chunks are fed to the LLM.

- **Prompting**: Employs strict "LLM-as-a-judge" style constraints forcing the model to rely only on the retrieved context.


## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.