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