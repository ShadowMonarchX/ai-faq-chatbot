from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from transformers import pipeline

# 1. Setup embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load ChromaDB vector store
CHROMA_PATH = "vectorstore/chroma"
vectordb = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model,
    collection_name="faq-collection"
)

# 3. Setup retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 4. Setup LLM using HuggingFace pipeline
llm_pipeline = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# 5. Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 6. Function to be used in FastAPI
def ask_question(query: str) -> str:
    result = qa_chain(query)
    return result["result"]

if __name__ == "__main__":
    question = "What is your privacy policy?"
    answer = ask_question(question)
    print("Answer:", answer)
