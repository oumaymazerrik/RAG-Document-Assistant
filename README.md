
# RAG Document Assistant
### A Retrieval-Augmented Generation System with LangChain, FAISS, Groq and RAGAS

A document-based AI assistant that answers questions about **Transformers and Large Language Models (LLMs)** using Retrieval-Augmented Generation (RAG).

The system retrieves relevant passages from a collection of academic course materials, generates grounded answers using an LLM, and provides source citations.

The project also includes RAGAS evaluation, semantic retrieval experiments, and a Cross-Encoder reranking comparison.

**Academic context:** ENSAM Casablanca — Transformers & LLMs  
**Project type:** Academic Practical Work — Document Assistant (RAG)  
**Language:** Python  
**Repository:** https://github.com/oumaymazerrik/RAG-Document-Assistant

---

## 1. Project Overview

Large Language Models can generate fluent and informative answers, but they are not guaranteed to know the contents of a particular document.

They can also generate unsupported or inaccurate information, commonly referred to as hallucinations.

Retrieval-Augmented Generation (RAG) addresses this limitation by retrieving relevant information from an external knowledge base before generating an answer.

This project implements a complete RAG pipeline composed of:

1. PDF document ingestion
2. Text extraction and chunking
3. Semantic embedding generation
4. Vector indexing with FAISS
5. Semantic retrieval
6. LLM-based answer generation
7. Source citation
8. RAG quality evaluation
9. Retrieval improvement experiments

The assistant is designed to answer questions using information retrieved from the provided academic documents.

---

## 2. Project Objectives

The main objectives are:

- Build a document question-answering assistant.
- Extract information from PDF course materials.
- Divide documents into manageable text chunks.
- Generate semantic embeddings using a multilingual model.
- Store and retrieve document embeddings with FAISS.
- Generate context-grounded answers using an LLM.
- Include document page citations.
- Evaluate answer quality using RAGAS.
- Analyze retrieval performance.
- Experiment with Cross-Encoder reranking.
- Compare retrieval strategies using quantitative metrics.

---

## 3. Dataset and Document Corpus

The knowledge base consists of five academic lectures covering Transformers and LLM fundamentals.

| Document | Pages |
|---|---:|
| Lecture 1 — From Words to Vectors | 39 |
| Lecture 2 — The Attention Mechanism, Part A | 30 |
| Lecture 2 — Attention, Part B | 49 |
| Lecture 3 — Transformer Encoder–Decoder | 37 |
| LLM Foundations & Inference | 37 |
| **Total** | **192** |

The documents were combined into a single PDF corpus.

### Corpus preprocessing

The preprocessing pipeline performs:

- PDF text extraction
- Page-level metadata preservation
- Recursive text splitting
- Overlapping chunk creation
- Embedding generation
- Vector database indexing

### Dataset statistics

| Parameter | Value |
|---|---:|
| Total PDF pages | 192 |
| Total chunks | 283 |
| Maximum chunk size | 800 characters |
| Chunk overlap | 150 characters |
| Embedding dimensions | 384 |
| Vector database | FAISS |

The academic course PDF is excluded from the public repository.

Users must provide their own authorized course document to reproduce the experiment.

---

## 4. System Architecture

The project follows two main pipelines:

### 4.1 Offline Ingestion Pipeline

```text
Academic PDF Documents
         |
         v
     PyPDFLoader
         |
         v
    Text Extraction
         |
         v
RecursiveCharacterTextSplitter
         |
         v
       283 Chunks
         |
         v
Multilingual MiniLM Embeddings
         |
         v
  384-Dimensional Vectors
         |
         v
    FAISS Vector Store
```

This pipeline is executed when preparing the document knowledge base.

### 4.2 Online Question-Answering Pipeline

```text
       User Question
             |
             v
     Query Embedding
             |
             v
      FAISS Retrieval
          Top-K = 5
             |
             v
  Retrieved Text Passages
             |
             v
       Prompt Assembly
             |
             v
        Groq API
      GPT-OSS 20B LLM
             |
             v
   Grounded Answer with
       Page Citations
```

The embedding model is responsible for representing semantic meaning, while FAISS performs vector similarity search.

The LLM is responsible for generating the final natural-language answer.

---

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| LangChain | RAG pipeline orchestration |
| PyPDFLoader | PDF document loading |
| RecursiveCharacterTextSplitter | Document chunking |
| Sentence Transformers | Semantic embedding generation |
| Multilingual MiniLM | Embedding model |
| FAISS | Vector similarity search |
| Groq API | LLM inference provider |
| GPT-OSS 20B | Text generation |
| RAGAS 0.4.4 | RAG evaluation |
| Cross-Encoder MiniLM | Document reranking |
| Pandas / CSV / JSON | Evaluation result storage |
| Python dotenv | Environment variable management |

### Embedding model

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

This multilingual sentence embedding model generates 384-dimensional vector representations.

### Reranking model

`cross-encoder/ms-marco-MiniLM-L6-v2`

This Cross-Encoder jointly processes a question and a candidate passage to estimate their relevance.

### Generative model

`openai/gpt-oss-20b`

The model is accessed through the Groq API.

The LLM runs remotely, while document processing, embeddings and FAISS retrieval run locally.

---

## 6. Project Structure

```text
RAG-Document-Assistant/
|
|-- main.py
|-- embeddings_test.py
|-- create_vectorstore.py
|-- test_retrieval.py
|-- test_groq.py
|-- rag.py
|-- evaluate_rag.py
|-- ragas_evaluation.py
|-- test_reranking.py
|-- compare_retrieval.py
|
|-- golden_set.json
|-- rag_results.json
|-- ragas_scores_test.csv
|-- ragas_scores_test3.csv
|-- ragas_scores_progress.csv
|-- retrieval_comparison.csv
|
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- .env                 # Not committed
|-- .venv/               # Not committed
|-- data/                # Not committed
|   |-- cours_transformers.pdf
|
|-- faiss_index/         # Not committed
    |-- index.faiss
    |-- index.pkl
```

The `data/`, `faiss_index/`, `.venv/` and `.env` paths are excluded from version control.

---

## 7. Installation and Setup

### Prerequisites

- Python 3.10.11 or a compatible Python environment
- Git
- Internet access for model downloads and Groq API requests
- A Groq API key
- Academic documents in PDF format

The implementation was developed using Python 3.10.11 on Windows.

### Step 1 — Clone the repository

```bash
git clone https://github.com/oumaymazerrik/RAG-Document-Assistant.git

cd RAG-Document-Assistant
```

### Step 2 — Create a virtual environment

Using Python:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Alternatively, using uv:

```bash
uv venv --python 3.10
```

### Step 3 — Install dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Or using uv:

```bash
uv pip install -r requirements.txt
```

The `requirements.txt` file records the Python dependencies used during development.

### Step 4 — Configure the Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Groq Console:

https://console.groq.com/keys

Never commit API keys to GitHub.

### Step 5 — Add the document corpus

Create the following directory:

```text
data/
```

Place your authorized PDF inside it:

```text
data/cours_transformers.pdf
```

For comparable results, use the same corpus and preprocessing settings.

### Step 6 — Build the FAISS index

```bash
python create_vectorstore.py
```

Expected output for the original corpus:

```text
Nombre de chunks : 283
Base FAISS créée et sauvegardée !
Nombre de vecteurs : 283
Dimension des vecteurs : 384
```

### Step 7 — Test semantic retrieval

```bash
python test_retrieval.py
```

This script retrieves the top document passages for a sample question.

### Step 8 — Test the LLM connection

```bash
python test_groq.py
```

### Step 9 — Run the RAG assistant

```bash
python rag.py
```

The system retrieves relevant document chunks, sends them to the LLM and generates an answer with page citations.

---

## 8. Retrieval-Augmented Generation Workflow

### 8.1 Document Loading

`PyPDFLoader` extracts page contents and preserves metadata.

Important metadata includes:

- Document source
- PDF page index
- Page label
- Document information

These fields allow the assistant to associate retrieved text with its source.

### 8.2 Document Chunking

The corpus is divided using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

The chunk size controls the maximum text length of each passage.

The overlap helps preserve information near chunk boundaries.

The original experiment generated 283 chunks.

### 8.3 Embedding Generation

The MiniLM embedding model converts each chunk into a numerical vector.

```text
Text Chunk
    |
    v
MiniLM Encoder
    |
    v
384-Dimensional Embedding
```

The same embedding model is used for document chunks and user questions.

### 8.4 Vector Indexing

FAISS stores the generated vectors and allows similarity-based retrieval.

LangChain also maintains associations between vector IDs, document text and metadata.

### 8.5 Retrieval

For each incoming question:

1. The question is converted into an embedding.
2. FAISS searches for similar vectors.
3. The top five passages are selected.
4. The corresponding text and page metadata are retrieved.

The baseline configuration uses:

```python
k = 5
```

### 8.6 Prompt Augmentation

The retrieved passages are included in the LLM prompt.

The prompt instructs the model to:

- Answer using the supplied context.
- Avoid unsupported claims.
- Cite relevant document pages.
- Acknowledge insufficient information.

### 8.7 Answer Generation

The LLM generates the final answer using the question and retrieved context.

Source citations are generated using page metadata supplied in the prompt.

These citations should be verified against the retrieved passages; prompt instructions alone do not guarantee citation correctness.

---

## 9. Example RAG Query

### Question

> What is the attention mechanism in Transformers?

### Generated Answer — Summary

The attention mechanism uses learned Query, Key and Value projections to calculate relevance scores between tokens.

The scores are scaled, normalized with softmax and applied to the Value vectors to produce contextualized token representations.

The output is a weighted aggregation of token information.

### Retrieved source pages

The initial FAISS retrieval returned:

```text
Page 118
Page 2
Page 70
Page 59
Page 122
```

The generated response cited:

```text
[Page 118, 70, 59, 122]
```

Page 59 contained the complete scaled dot-product attention equation.

### Insufficient Context Test

Question:

> What is the exact population of Casablanca in 2026?

System response:

```text
The provided documents do not contain enough information.
```

This test showed that the assistant could acknowledge insufficient source context rather than generating an unsupported answer.

A single successful test does not guarantee the absence of hallucinations in other situations.

---

## 10. RAG Evaluation with RAGAS

RAGAS is used to evaluate the quality of the document-based assistant.

### Golden Set

The evaluation dataset contains:

- 15 questions
- Reference answers
- Reference page annotations

The questions cover tokenization, embeddings, attention, positional encoding, Transformer architectures and LLM prompting.

The evaluation input for each question includes:

```text
Question
Reference Answer
Generated Answer
Retrieved Contexts
```

### Evaluation Metrics

**Faithfulness**

Measures whether statements in the generated answer are supported by the retrieved context.

**Answer Relevancy**

Measures how relevant the generated answer is to the user's question.

**Context Precision**

Evaluates the relevance and ranking of retrieved passages.

**Context Recall**

Measures whether the retrieved context covers the information required by the reference answer.

### RAGAS Results

The complete RAGAS evaluation was successfully executed on a subset of three questions.

| Metric | Score |
|---|---:|
| Faithfulness | 1.0000 |
| Answer Relevancy | 0.9925 |
| Context Precision | 0.9722 |
| Context Recall | 1.0000 |

### Interpretation

The tested answers were judged highly faithful to the retrieved context and highly relevant to the questions.

The retrieved passages covered the reference information in the evaluated subset.

However, these results cover only three questions.

They should be interpreted as preliminary evaluation results rather than a reliable estimate of performance across the entire 15-question Golden Set.

### Evaluation Limitations

An attempted evaluation of all 15 questions was interrupted by Groq API rate limits.

The original 15 generated answers remain saved in:

```text
rag_results.json
```

A progressive evaluation script was introduced to support saving completed evaluations and resuming after interruptions.

The partial evaluation file is:

```text
ragas_scores_progress.csv
```

The successful three-question results are saved in:

```text
ragas_scores_test3.csv
```

The project reports only successfully completed evaluation results.

---

## 11. Retrieval Improvement Experiment

The baseline system relies on FAISS semantic retrieval.

To investigate retrieval quality, a second approach was tested:

**FAISS Candidate Retrieval + Cross-Encoder Reranking**

### Baseline

```text
Question
   |
   v
FAISS Retrieval
   |
   v
Top 5 Chunks
```

### Improved Candidate Pipeline

```text
Question
   |
   v
FAISS Retrieval
   |
   v
Top 10 Candidate Chunks
   |
   v
Cross-Encoder Reranking
   |
   v
Top 5 Reranked Chunks
```

### Evaluation Method

Both approaches were evaluated using the same 15 questions.

The evaluation compared retrieved page numbers with the reference page annotations.

Three page-based retrieval metrics were calculated:

**Recall@5**

Fraction of annotated reference pages found in the first five results.

**Hit@5**

Fraction of questions for which at least one reference page appears in the top five results.

**MRR@5**

Mean reciprocal rank of the first retrieved reference page, limited to the first five positions.

These metrics use page annotations and are distinct from RAGAS context precision and recall.

---

## 12. Experimental Retrieval Results

| Metric | FAISS Only | FAISS + Reranker | Difference |
|---|---:|---:|---:|
| Recall@5 | 0.7667 | 0.7000 | -0.0667 |
| Hit@5 | 0.8000 | 0.7333 | -0.0667 |
| MRR@5 | 0.5211 | 0.5444 | +0.0233 |

### Findings

The reranker slightly improved MRR@5, indicating that the first annotated relevant page tended to appear at a better rank.

However, Recall@5 and Hit@5 decreased.

This suggests that reranking improved some ranking decisions while removing some relevant pages from the first five results.

### Case Study — Attention Mechanism

For the question:

> What is the attention mechanism in Transformers?

The FAISS baseline returned:

```text
1. Page 118
2. Page 2
3. Page 70
4. Page 59
5. Page 122
```

After reranking:

```text
1. Page 62
2. Page 70
3. Page 118
4. Page 40
5. Page 122
```

The reranker promoted page 62, which directly explains self-attention.

However, page 59, containing the attention equation, was no longer in the top five.

### Experimental Conclusion

The reranking experiment did not demonstrate a consistent overall improvement.

Although MRR@5 increased, retrieval coverage decreased.

Therefore, FAISS top-5 is retained as the baseline configuration for the current version.

The experiment illustrates why retrieval improvements must be validated quantitatively rather than assumed.

Full results:

```text
retrieval_comparison.csv
```

---

## 13. Challenges and Solutions

### Challenge 1 — Gemini API Access

An initial Gemini API integration was tested.

However, the Google Cloud project did not have access to the Gemini API free tier.

**Solution:** The generative component was changed to Groq with GPT-OSS 20B.

### Challenge 2 — Retrieval Ranking Quality

The initial FAISS results sometimes included general introductions instead of passages directly answering a question.

**Solution:** Increasing the retrieval depth and experimenting with Cross-Encoder reranking.

### Challenge 3 — RAGAS Evaluation Completion

Some RAGAS executions failed because the LLM evaluation generation did not complete.

**Mitigation:**

- Increased token generation allowance
- Reduced reasoning effort
- Limited concurrent evaluation workers

These changes enabled a successful three-question evaluation.

### Challenge 4 — Groq API Rate Limits

The full 15-question evaluation encountered API token quota restrictions.

**Solution:** A progressive evaluation workflow was introduced to save results after successful questions and support resumption.

---

## 14. Limitations

The current system has several limitations:

1. Retrieval quality depends on the embedding model and chunking configuration.
2. Fixed-size chunking can separate closely related information.
3. The text extraction process does not fully capture the meaning of figures and diagrams.
4. Retrieved passages are not guaranteed to answer the question.
5. The LLM can still generate unsupported statements.
6. Citation correctness requires verification.
7. RAGAS results cover only three successfully evaluated questions.
8. Free API limits restrict larger evaluation experiments.
9. Page-based relevance annotations may not cover every valid passage.
10. Cross-Encoder reranking did not improve all retrieval metrics.

---

## 15. Future Improvements

Potential improvements include:

### Hybrid Search

Combine dense FAISS retrieval with BM25 keyword-based retrieval.

### Structure-Aware Chunking

Split documents by meaningful sections and preserve lecture headings.

### Improved Reranking

Experiment with alternative Cross-Encoder models and candidate retrieval sizes.

### Expanded RAGAS Evaluation

Complete the evaluation on the full 15-question Golden Set when sufficient API quota is available.

### Citation Validation

Automatically verify whether cited page numbers correspond to passages supporting the answer.

### OpenTelemetry Tracing

Measure the duration of individual pipeline operations:

```text
Query Embedding
Vector Retrieval
Prompt Construction
LLM Generation
```

### Conversational RAG

Support follow-up questions using conversation history and query rewriting.

### Local LLM Execution

Experiment with Ollama to reduce dependence on external APIs.

---

## 16. Reproducibility

The project includes scripts for rebuilding the FAISS index, generating answers and reproducing the retrieval comparison.

To obtain comparable results, preserve:

- The original corpus and page ordering
- Chunk size and overlap
- Embedding model
- FAISS retrieval configuration
- Golden Set annotations
- Reranking model
- Evaluation settings

Results involving an LLM may vary across runs because of model behavior, provider configuration and API availability.

The project should be considered an experimental academic implementation.

---

## 17. Project Status

| Component | Status |
|---|---|
| PDF ingestion | Completed |
| Text chunking | Completed |
| Multilingual embeddings | Completed |
| FAISS vector database | Completed |
| Semantic retrieval | Completed |
| Groq LLM integration | Completed |
| Document-based answers | Completed |
| Page citation generation | Completed |
| Insufficient context test | Completed |
| 15-question Golden Set | Completed |
| RAGAS evaluation on 3 questions | Completed |
| Full 15-question RAGAS evaluation | Incomplete |
| Cross-Encoder reranking experiment | Completed |
| Retrieval metric comparison | Completed |
| Hybrid retrieval | Future work |
| OpenTelemetry tracing | Future work |

---

## 18. References

- [LangChain Documentation](https://docs.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [Groq Documentation](https://console.groq.com/docs)
- [Hugging Face](https://huggingface.co/)
- [Retrieval-Augmented Generation — Lewis et al., 2020](https://arxiv.org/abs/2005.11401)

---

## 19. Author

**Oumayma Zerrik**

Master's Student — Big Data & Internet of Things  
ENSAM Casablanca, Morocco

**GitHub:** [oumaymazerrik](https://github.com/oumaymazerrik)

---

## License and Data Availability

This repository contains the project's implementation and experimental results.

The original academic course materials are not redistributed.

Users are responsible for obtaining permission to use any third-party documents and for respecting the applicable model and API terms.
