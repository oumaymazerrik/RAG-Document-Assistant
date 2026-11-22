
from langchain_huggingface import HuggingFaceEmbeddings

# Initialiser le modèle d'embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}
)

# Transformer une phrase en vecteur
phrase = "The Transformer uses an attention mechanism."

vecteur = embeddings.embed_query(phrase)

print("Dimension du vecteur :", len(vecteur))
print("Premières valeurs :", vecteur[:10])
