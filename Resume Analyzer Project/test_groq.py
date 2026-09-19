
from langchain_groq import ChatGroq

# Replace with your actual key
my_key = "YOUR_GROQ_API_KEY"

try:
    llm = ChatGroq(groq_api_key=my_key, model="llama-3.1-8b-instant")
    response = llm.invoke("Hi! Are you ready to analyze resumes?")
    print("SUCCESS:")
    print(response.content)
except Exception as e:
    print("ERROR:")
    print(e)