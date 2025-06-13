import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Load the cleaned dataset
CSV_PATH = "../../data/faq_dataset.csv"
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["question", "answer"])

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a persistent ChromaDB client (NEW WAY)
chroma_client = chromadb.PersistentClient(
    path="vectorstore/chroma"  # folder to store persistent DB
)

# Create/get collection
collection = chroma_client.get_or_create_collection("faq-collection")

# Prepare data
questions = df["question"].tolist()
answers = df["answer"].tolist()
embeddings = embedding_model.encode(questions).tolist()
ids = [f"faq-{i}" for i in range(len(questions))]

# Add to collection
collection.add(
    documents=questions,
    embeddings=embeddings,
    metadatas=[{"answer": ans} for ans in answers],
    ids=ids
)

print("Vector store created and saved using new ChromaDB API.")
