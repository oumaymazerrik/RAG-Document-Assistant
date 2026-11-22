
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY introuvable dans .env")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=api_key,
    temperature=0
)

response = llm.invoke(
    "Explain the attention mechanism in Transformers in one sentence."
)

print("Réponse Groq :")
print(response.content)
