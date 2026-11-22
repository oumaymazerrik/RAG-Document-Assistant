
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Charger le PDF
loader = PyPDFLoader("data/cours_transformers.pdf")
documents = loader.load()

# 2. Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(documents)
print("Nombre de chunks :", len(chunks))

# 3. Charger le modèle d'embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}
)

# 4. Vectoriser les chunks et construire FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Sauvegarder la base sur le disque
Path("faiss_index").mkdir(exist_ok=True)
vectorstore.save_local("faiss_index")

print("Base FAISS créée et sauvegardée !")
print("Nombre de vecteurs :", vectorstore.index.ntotal)
print("Dimension des vecteurs :", vectorstore.index.d)
