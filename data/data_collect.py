from datasets import load_dataset

ds = load_dataset("MakTek/Customer_support_faqs_dataset", split="train")
ds.to_csv("faq_dataset.csv", index=False)
