
import os
import json
import time

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# 1. Charger le Golden Set
with open("golden_set.json", "r", encoding="utf-8") as f:
    golden_set = json.load(f)

print(f"Questions chargées : {len(golden_set)}")

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

# 4. Initialiser Groq
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

system_prompt = """
You are an educational assistant specialized in Transformers and LLMs.

Answer using ONLY the provided context.
Cite relevant page numbers using [Page X].
Do not invent information.
If the context is insufficient, say:
"The provided documents do not contain enough information."
"""

# 5. Exécuter le RAG sur les 15 questions
results = []

for item in golden_set:
    question = item["question"]

    print(f"\nQuestion {item['id']}/{len(golden_set)}")
    print(question)

    # Retrieval
    start = time.perf_counter()
    documents = vectorstore.similarity_search(question, k=5)

    context = "\n\n".join(
        f"[Page {doc.metadata['page'] + 1}]\n{doc.page_content}"
        for doc in documents
    )

    # Generation
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion:\n{question}"
            )
        ])

        answer = response.content
        duration = round(time.perf_counter() - start, 2)

        results.append({
            "question": question,
            "reference": item["reference"],
            "response": answer,
            "retrieved_contexts": [
                doc.page_content for doc in documents
            ],
            "retrieved_pages": [
                doc.metadata["page"] + 1 for doc in documents
            ],
            "reference_pages": item["source_pages"],
            "duration_seconds": duration
        })

        print("Réponse :", answer[:150])
        print("Durée :", duration, "s")

    except Exception as e:
        print("Erreur :", str(e))
        print("Arrêt : vérifie la connexion et les quotas Groq.")
        break

# 6. Sauvegarder les résultats
with open("rag_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nRésultats sauvegardés : {len(results)}/{len(golden_set)}")
