from datasets import load_dataset

ds = load_dataset("MakTek/Customer_support_faqs_dataset", split="train")
ds.to_csv("faq_dataset.csv", index=False)



# from datasets import load_dataset

# # Load the dataset from Hugging Face Hub
# dataset = load_dataset("promptslab/faq-dataset")

# # View one example
# print(dataset["train"][0])


# | Dataset Name    | Description                          | Hugging Face Link                                                       |
# | --------------- | ------------------------------------ | ----------------------------------------------------------------------- |
# | **SQuAD v1/v2** | Real Q\&A from Wikipedia articles    | [`squad`](https://huggingface.co/datasets/squad)                        |
# | **HotpotQA**    | Multi-hop Q\&A                       | [`hotpot_qa`](https://huggingface.co/datasets/hotpot_qa)                |
# | **FAQ Dataset** | Customer support FAQs from companies | [`faq_dataset`](https://huggingface.co/datasets/promptslab/faq-dataset) |
# | **MS MARCO**    | Real web Q\&A from Bing search logs  | [`ms_marco`](https://huggingface.co/datasets/ms_marco)                  |
# | **OpenBookQA**  | Elementary science facts Q\&A        | [`openbookqa`](https://huggingface.co/datasets/openbookqa)              |

# def clean(example):
#     prompt = example["question"].strip()
#     response = example["answer"].strip()
#     return {"prompt": prompt, "response": response}

# # Apply cleaning
# cleaned_dataset = dataset["train"].map(clean)

# # Filter out very short examples
# cleaned_dataset = cleaned_dataset.filter(lambda x: len(x["prompt"]) > 10 and len(x["response"]) > 10)

# # Preview cleaned sample
# print(cleaned_dataset[0])

# import json

# with open("processed_faq.jsonl", "w") as f:
#     for item in cleaned_dataset:
#         json.dump({"prompt": item["prompt"], "response": item["response"]}, f)
#         f.write("\n")

# from datasets import load_dataset

# ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# import pandas as pd

# df = pd.read_csv("hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")


# finetune_mistral.py

# import re
# import torch
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from transformers import BitsAndBytesConfig

# # -------------------------------------------
# # Step 1: Load and clean dataset
# # -------------------------------------------

# def clean_text(text):
#     # Basic cleaning: remove multiple spaces, symbols, strip
#     text = re.sub(r"\s+", " ", text)
#     text = re.sub(r"[^\x00-\x7F]+", "", text)  # remove non-ascii
#     return text.strip()

# def format_conversation(example):
#     question = clean_text(example["prompt"])
#     answer = clean_text(example["response"])
#     return {"text": f"### Question: {question}\n### Answer: {answer}"}

# dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
# dataset = dataset["train"].map(format_conversation)

# # -------------------------------------------
# # Step 2: Tokenizer and model setup (4-bit)
# # -------------------------------------------

# model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token  # prevent padding issues

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     quantization_config=bnb_config,
#     trust_remote_code=True,
# )

# # -------------------------------------------
# # Step 3: Add LoRA adapter (PEFT)
# # -------------------------------------------

# model = prepare_model_for_kbit_training(model)

# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# model = get_peft_model(model, lora_config)

# # -------------------------------------------
# # Step 4: Tokenize dataset
# # -------------------------------------------

# def tokenize(example):
#     return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

# tokenized = dataset.map(tokenize, batched=True)
# tokenized.set_format("torch")

# # -------------------------------------------
# # Step 5: Training setup
# # -------------------------------------------

# training_args = TrainingArguments(
#     output_dir="./mistral-faq-lora",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=4,
#     logging_steps=10,
#     num_train_epochs=1,
#     learning_rate=2e-4,
#     fp16=True,
#     save_total_limit=2,
#     save_steps=50,
#     logging_dir="./logs",
#     report_to="none"
# )

# data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized,
#     tokenizer=tokenizer,
#     data_collator=data_collator
# )

# # -------------------------------------------
# # Step 6: Start Training
# # -------------------------------------------

# if __name__ == "__main__":
#     trainer.train()


# import os
# import json
# import pandas as pd
# import torch
# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from transformers import BitsAndBytesConfig

# # === Step 1: Load & Clean Dataset ===
# def load_and_clean_dataset(file_path):
#     if file_path.endswith(".csv"):
#         df = pd.read_csv(file_path)
#     elif file_path.endswith(".json") or file_path.endswith(".jsonl"):
#         df = pd.read_json(file_path, lines=True)
#     else:
#         raise ValueError("Unsupported file format. Use CSV or JSONL.")

#     df.dropna(subset=["prompt", "response"], inplace=True)
#     df["prompt"] = df["prompt"].str.strip()
#     df["response"] = df["response"].str.strip()
#     return Dataset.from_pandas(df[["prompt", "response"]])

# # === Step 2: Format for Causal LM ===
# def format_for_training(example):
#     example["text"] = f"<s>[INST] {example['prompt']} [/INST] {example['response']}</s>"
#     return example

# # === Step 3: Model & Tokenizer Setup ===
# def get_model_and_tokenizer():
#     model_name = "mistralai/Mistral-7B-Instruct-v0.1"
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float16,
#     )

#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#     tokenizer.pad_token = tokenizer.eos_token

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         device_map="auto",
#         quantization_config=bnb_config,
#         trust_remote_code=True
#     )

#     model = prepare_model_for_kbit_training(model)

#     lora_config = LoraConfig(
#         r=8,
#         lora_alpha=16,
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM"
#     )

#     model = get_peft_model(model, lora_config)

#     return model, tokenizer

# # === Step 4: Training ===
# def train_model(dataset, model, tokenizer):
#     tokenized_dataset = dataset.map(
#         lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length", max_length=512),
#         batched=True,
#         remove_columns=["prompt", "response", "text"]
#     )

#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#     training_args = TrainingArguments(
#         output_dir="./mistral_faq_finetuned",
#         per_device_train_batch_size=2,
#         gradient_accumulation_steps=4,
#         num_train_epochs=3,
#         learning_rate=2e-5,
#         fp16=True,
#         logging_steps=10,
#         save_strategy="epoch",
#         save_total_limit=2,
#         report_to="none"
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_dataset,
#         tokenizer=tokenizer,
#         data_collator=data_collator
#     )

#     trainer.train()
#     model.save_pretrained("./mistral_faq_finetuned")

# # === Main Script ===
# if __name__ == "__main__":
#     # Change to your file path on Colab or local
#     dataset_path = "faq_dataset.json"  # or .csv or .jsonl
#     dataset = load_and_clean_dataset(dataset_path)
#     dataset = dataset.map(format_for_training)

#     model, tokenizer = get_model_and_tokenizer()
# #     train_model(dataset, model, tokenizer)
# !pip install transformers datasets peft accelerate bitsandbytes


# import os
# import torch
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.llms import HuggingFacePipeline
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import TextLoader

# # === Step 1: Load and Chunk Documents ===
# def load_and_chunk(doc_path, chunk_size=300, chunk_overlap=50):
#     loader = TextLoader(doc_path)
#     documents = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
#     return text_splitter.split_documents(documents)

# # === Step 2: Create Embeddings & Vector Store ===
# def create_vector_db(chunks, db_path="chroma_db"):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_path)
#     vectordb.persist()
#     return vectordb

# # === Step 3: Load Fine-Tuned Mistral Model ===
# def load_finetuned_mistral():
#     model_name = "./mistral_faq_finetuned"  # path to your LoRA fine-tuned model
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=512,
#         temperature=0.7,
#         top_p=0.95,
#         repetition_penalty=1.1
#     )

#     return HuggingFacePipeline(pipeline=pipe)

# # === Step 4: Build RAG Pipeline ===
# def build_rag(llm, vectordb):
#     retriever = vectordb.as_retriever(search_kwargs={"k": 3})
#     qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
#     return qa_chain

# # === Step 5: Ask Questions ===
# def ask_question(rag_chain, query):
#     answer = rag_chain.run(query)
#     print(f"\n🧠 Q: {query}\n🤖 A: {answer}\n")

# # === Main ===
# if __name__ == "__main__":
#     # 1. Chunking
#     chunks = load_and_chunk("data/faqs.txt")

#     # 2. Embedding & Storing
#     vectordb = create_vector_db(chunks)

#     # 3. Load LLM
#     llm = load_finetuned_mistral()

#     # 4. RAG Setup
#     rag_chain = build_rag(llm, vectordb)

#     # 5. Ask
#     while True:
#         q = input("Ask me anything (or type 'exit'): ")
#         if q.lower() == "exit":
#             break
#         ask_question(rag_chain, q)
# pip install langchain transformers accelerate bitsandbytes sentence-transformers chromadb


# pip install fastapi uvicorn transformers accelerate bitsandbytes sentence-transformers langchain chromadb


# import sqlite3

# # --- Setup DB on startup ---
# @app.on_event("startup")
# def setup_db():
#     conn = sqlite3.connect("feedback.db")
#     c = conn.cursor()
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS feedback (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             question TEXT,
#             answer TEXT,
#             feedback TEXT,
#             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#         )
#     ''')
#     conn.commit()
#     conn.close()

# # --- Save feedback in DB ---
# @app.post("/feedback")
# async def collect_feedback(question: str = Form(...), answer: str = Form(...), feedback: str = Form(...)):
#     conn = sqlite3.connect("feedback.db")
#     c = conn.cursor()
#     c.execute("INSERT INTO feedback (question, answer, feedback) VALUES (?, ?, ?)", (question, answer, feedback))
#     conn.commit()
#     conn.close()

#     return HTMLResponse(f"<html><body><h3>Thanks for your feedback! 👍</h3><a href='/'>Back to Chat</a></body></html>")


# import sqlite3
# import pandas as pd

# def analyze_feedback():
#     conn = sqlite3.connect("feedback.db")
#     df = pd.read_sql_query("SELECT * FROM feedback ORDER BY timestamp DESC", conn)
#     conn.close()

#     # Filter 👎 responses
#     poor_responses = df[df["feedback"] == "down"]
#     print("❌ Poor Responses:")
#     print(poor_responses[["question", "answer"]])

#     # Save to CSV for review
#     poor_responses.to_csv("data/poor_responses.csv", index=False)

# analyze_feedback()


# import json
# import pandas as pd

# df = pd.read_csv("data/poor_responses.csv")
# new_data = []

# for _, row in df.iterrows():
#     new_data.append({
#         "prompt": row["question"],
#         "response": input(f"✅ Enter better response for:\nQ: {row['question']}\nA: {row['answer']}\n→ ")
#     })

# with open("data/new_finetune_data.jsonl", "a") as f:
#     for pair in new_data:
#         f.write(json.dumps(pair) + "\n")

