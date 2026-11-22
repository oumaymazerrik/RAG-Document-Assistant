
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

# 1. Charger le modèle d'embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}
)

# 2. Charger FAISS
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 3. Charger le Cross-Encoder
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    device="cpu",
    max_length=512
)

# 4. Question
question = "What is the attention mechanism in Transformers?"

# 5. Baseline : FAISS top-5
baseline = vectorstore.similarity_search(question, k=5)

print("\n===== AVANT RERANKING =====")

for i, doc in enumerate(baseline, 1):
    print(f"{i}. Page {doc.metadata['page'] + 1}")
    print(doc.page_content[:120], "\n")

# 6. Amélioration : FAISS top-10
candidates = vectorstore.similarity_search(question, k=10)

# 7. Construire les paires (question, chunk)
pairs = [
    (question, doc.page_content)
    for doc in candidates
]

# 8. Calculer les scores du Cross-Encoder
scores = reranker.predict(pairs)

# 9. Classer les passages
ranked = sorted(
    zip(candidates, scores),
    key=lambda x: float(x[1]),
    reverse=True
)

# 10. Conserver les 5 meilleurs
top5 = ranked[:5]

print("\n===== APRES RERANKING =====")

for i, (doc, score) in enumerate(top5, 1):
    print(f"{i}. Page {doc.metadata['page'] + 1}")
    print(f"Score reranker : {float(score):.4f}")
    print(doc.page_content[:120], "\n")
