
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# 1. Charger la clé API
load_dotenv()

# 2. Charger le modèle d'embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}
)

# 3. Charger FAISS
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
# Uniquement pour notre index local de confiance.

# 4. Initialiser le LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# 5. Question utilisateur
question = "What is the exact population of Casablanca in 2026?"
# 6. Récupérer les 5 passages pertinents
documents = vectorstore.similarity_search(question, k=5)

# 7. Construire le contexte avec les sources
context = "\n\n".join(
    f"[Page {doc.metadata['page'] + 1}]\n{doc.page_content}"
    for doc in documents
)

# 8. Instructions du système
system_prompt = """
You are an educational assistant specialized in Transformers and LLMs.

Answer the user's question using ONLY the provided context.

Rules:
- Do not invent information.
- Cite the relevant page numbers using [Page X].
- If the context does not contain enough information,
  say: "The provided documents do not contain enough information."
- Answer clearly and concisely.
"""

# 9. Construire les messages
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(
        content=f"Context:\n{context}\n\nQuestion:\n{question}"
    )
]

# 10. Générer la réponse
response = llm.invoke(messages)

print("\nQUESTION:")
print(question)

print("\nRÉPONSE RAG:")
print(response.content)

print("\nSOURCES RÉCUPÉRÉES:")
for doc in documents:
    print(f"- Page {doc.metadata['page'] + 1}")
