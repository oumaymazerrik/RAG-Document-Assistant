
import json
import csv
import statistics

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder


# ==========================================
# 1. CONFIGURATION
# ==========================================

TOP_K = 5
CANDIDATE_K = 10

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


# ==========================================
# 2. CHARGER LE GOLDEN SET
# ==========================================

with open("golden_set.json", "r", encoding="utf-8") as f:
    golden_set = json.load(f)

print("Questions :", len(golden_set))


# ==========================================
# 3. CHARGER LES MODELES
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
# Charger uniquement un index FAISS de confiance.

reranker = CrossEncoder(
    RERANKER_MODEL,
    device="cpu",
    max_length=512
)


# ==========================================
# 4. FONCTIONS D'EVALUATION
# ==========================================

def get_pages(documents):
    """Extraire les pages originales du PDF."""
    return [
        doc.metadata["page"] + 1
        for doc in documents
    ]


def calculate_metrics(retrieved_pages, reference_pages):
    """
    Mesurer la récupération des pages de référence.

    Recall@5 : proportion des pages attendues retrouvées.
    Hit@5 : au moins une page attendue retrouvée.
    MRR@5 : inverse du rang de la première page pertinente.
    """

    references = set(reference_pages)
    retrieved = retrieved_pages[:TOP_K]

    if not references:
        raise ValueError(
            "Chaque question doit avoir une page de référence."
        )

    found = references.intersection(set(retrieved))

    recall = len(found) / len(references)

    hit = 1.0 if found else 0.0

    mrr = 0.0

    for rank, page in enumerate(retrieved, start=1):
        if page in references:
            mrr = 1.0 / rank
            break

    return {
        "recall_at_5": recall,
        "hit_at_5": hit,
        "mrr_at_5": mrr
    }


# ==========================================
# 5. COMPARAISON SUR 15 QUESTIONS
# ==========================================

results = []

for index, item in enumerate(golden_set, start=1):

    question = item["question"]
    reference_pages = item["source_pages"]

    print(f"\nQuestion {index}/{len(golden_set)}")
    print(question)

    # ---------- BASELINE : FAISS TOP-5 ----------

    baseline_docs = vectorstore.similarity_search(
        question,
        k=TOP_K
    )

    baseline_pages = get_pages(baseline_docs)

    baseline_scores = calculate_metrics(
        baseline_pages,
        reference_pages
    )

    # ---------- IMPROVED : FAISS + RERANKER ----------

    candidate_docs = vectorstore.similarity_search(
        question,
        k=CANDIDATE_K
    )

    pairs = [
        (question, doc.page_content)
        for doc in candidate_docs
    ]

    rerank_scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidate_docs, rerank_scores),
        key=lambda pair: float(pair[1]),
        reverse=True
    )

    improved_docs = [
        doc
        for doc, score in ranked[:TOP_K]
    ]

    improved_pages = get_pages(improved_docs)

    improved_scores = calculate_metrics(
        improved_pages,
        reference_pages
    )

    # ---------- ENREGISTRER ----------

    row = {
        "id": item.get("id", index),
        "question": question,
        "reference_pages": json.dumps(reference_pages),
        "baseline_pages": json.dumps(baseline_pages),
        "reranked_pages": json.dumps(improved_pages),

        "baseline_recall": baseline_scores["recall_at_5"],
        "reranked_recall": improved_scores["recall_at_5"],

        "baseline_hit": baseline_scores["hit_at_5"],
        "reranked_hit": improved_scores["hit_at_5"],

        "baseline_mrr": baseline_scores["mrr_at_5"],
        "reranked_mrr": improved_scores["mrr_at_5"]
    }

    results.append(row)

    print("Pages de référence :", reference_pages)
    print("FAISS :", baseline_pages)
    print("Reranking :", improved_pages)

    print(
        "Recall@5 :",
        round(row["baseline_recall"], 3),
        "->",
        round(row["reranked_recall"], 3)
    )


# ==========================================
# 6. SAUVEGARDER LES RESULTATS
# ==========================================

output_file = "retrieval_comparison.csv"

with open(
    output_file,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# ==========================================
# 7. CALCULER LES MOYENNES
# ==========================================

print("\n========== COMPARAISON FINALE ==========")

comparisons = [
    ("Recall@5", "baseline_recall", "reranked_recall"),
    ("Hit@5", "baseline_hit", "reranked_hit"),
    ("MRR@5", "baseline_mrr", "reranked_mrr")
]

for name, before_key, after_key in comparisons:

    before = statistics.mean(
        row[before_key] for row in results
    )

    after = statistics.mean(
        row[after_key] for row in results
    )

    difference = after - before

    print(f"\n{name}")
    print(f"FAISS seul       : {before:.4f}")
    print(f"FAISS + Reranker : {after:.4f}")
    print(f"Difference       : {difference:+.4f}")


print("\nFichier cree :", output_file)
print("Comparaison terminee !")
