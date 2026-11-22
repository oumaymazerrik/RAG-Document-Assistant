
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Charger le même modèle d'embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}
)

# 2. Charger notre base FAISS
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
# À utiliser uniquement pour notre index local de confiance.

# 3. Poser une question
question = "What is the attention mechanism in Transformers?"

# 4. Recherche des 3 chunks les plus similaires
results = vectorstore.similarity_search(question, k=5)
# 5. Afficher les résultats
for i, doc in enumerate(results, 1):
    print(f"\n--- Résultat {i} ---")
    print("Page :", doc.metadata["page"] + 1)
    print("Contenu :", doc.page_content)
