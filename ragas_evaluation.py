
import os
import json
import csv
import math

from dotenv import load_dotenv
from ragas import evaluate, EvaluationDataset
from ragas.run_config import RunConfig

from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY introuvable")

OUTPUT_FILE = "ragas_scores_progress.csv"

METRIC_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall"
]

FIELDNAMES = [
    "question_id",
    "question"
] + METRIC_COLUMNS


# ==========================================
# 2. CHARGEMENT DES QUESTIONS
# ==========================================

with open("rag_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

print("Questions disponibles :", len(results))


# ==========================================
# 3. CHARGEMENT DES SCORES EXISTANTS
# ==========================================

completed = set()

if os.path.exists(OUTPUT_FILE):
    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            try:
                valid = all(
                    math.isfinite(float(row[col]))
                    for col in METRIC_COLUMNS
                )

                if valid:
                    completed.add(int(row["question_id"]))

            except (ValueError, TypeError, KeyError):
                pass

else:
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()

print("Questions deja evaluees :", len(completed))


# ==========================================
# 4. CONFIGURATION GROQ
# ==========================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=API_KEY,
    temperature=0,
    max_tokens=8192,
    reasoning_effort="low",
    max_retries=0
)

evaluator_llm = LangchainLLMWrapper(llm)


# ==========================================
# 5. EMBEDDINGS
# ==========================================

embedding_model = HuggingFaceEmbeddings(
    model_name=(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    ),
    model_kwargs={"device": "cpu"}
)

evaluator_embeddings = LangchainEmbeddingsWrapper(
    embedding_model
)


# ==========================================
# 6. METRIQUES
# ==========================================

metrics = [
    Faithfulness(),
    ResponseRelevancy(strictness=1),
    LLMContextPrecisionWithReference(),
    LLMContextRecall()
]


# ==========================================
# 7. CONFIGURATION D'EXECUTION
# ==========================================

run_config = RunConfig(
    max_workers=1,
    timeout=180,
    max_retries=0
)


# ==========================================
# 8. EVALUATION PROGRESSIVE
# ==========================================

print("\n===== DEBUT EVALUATION RAGAS =====")

for index, item in enumerate(results, start=1):

    if index in completed:
        print(f"\nQuestion {index} deja evaluee. SKIP.")
        continue

    print(f"\n===== QUESTION {index}/{len(results)} =====")
    print(item["question"])

    dataset = EvaluationDataset.from_list([
        {
            "user_input": item["question"],
            "response": item["response"],
            "reference": item["reference"],
            "retrieved_contexts": item["retrieved_contexts"]
        }
    ])

    try:

        scores = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config,
            raise_exceptions=True
        )

        df = scores.to_pandas()
        result = df.iloc[0]

        row = {
            "question_id": index,
            "question": item["question"]
        }

        for metric in METRIC_COLUMNS:
            value = float(result[metric])

            if not math.isfinite(value):
                raise ValueError(
                    f"Score invalide : {metric}"
                )

            row[metric] = value

        # Sauvegarde apres chaque question
        with open(
            OUTPUT_FILE,
            "a",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=FIELDNAMES
            )

            writer.writerow(row)

        completed.add(index)

        print("Scores :", {
            metric: round(row[metric], 4)
            for metric in METRIC_COLUMNS
        })

        print("Sauvegarde OK !")

    except Exception as e:

        print("\nERREUR :", str(e))
        print("Les questions precedentes sont sauvegardees.")
        print("Relance le script pour reprendre.")
        break


# ==========================================
# 9. RESUME DES RESULTATS
# ==========================================

print("\n========== RESUME ==========")

saved_rows = []

with open(
    OUTPUT_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)
    saved_rows = list(reader)

valid_rows = []

for row in saved_rows:
    try:
        if all(
            math.isfinite(float(row[col]))
            for col in METRIC_COLUMNS
        ):
            valid_rows.append(row)
    except (ValueError, TypeError, KeyError):
        pass

print(
    f"Questions evaluees : "
    f"{len(valid_rows)}/{len(results)}"
)

if valid_rows:

    print("\nSCORES MOYENS :")

    for metric in METRIC_COLUMNS:
        moyenne = sum(
            float(row[metric])
            for row in valid_rows
        ) / len(valid_rows)

        print(f"{metric}: {moyenne:.4f}")

print("\nFichier :", OUTPUT_FILE)

if len(valid_rows) == len(results):
    print("\nEvaluation complete !")
else:
    print("\nEvaluation partielle, reprise possible.")
