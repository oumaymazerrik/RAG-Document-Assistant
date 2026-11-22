
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Charger les pages du PDF
loader = PyPDFLoader("data/cours_transformers.pdf")
documents = loader.load()

print("Nombre de pages :", len(documents))

# 2. Configurer le chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# 3. Découper les pages en chunks
chunks = text_splitter.split_documents(documents)

print("Nombre total de chunks :", len(chunks))

# 4. Examiner le premier chunk
print("\n--- Premier chunk ---")
print(chunks[0].page_content)

print("\n--- Métadonnées ---")
print(chunks[0].metadata)

print("\nTaille du chunk :", len(chunks[0].page_content))
