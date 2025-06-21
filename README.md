# 🧠 MCQ Generator using Mistral-7B + LoRA (Fine-tuned)

This repository contains training and inference notebooks for fine-tuning the Mistral 7B model using the [LoRA](https://arxiv.org/abs/2106.09685) technique to generate Multiple Choice Questions (MCQs) from academic paragraphs.

---

## 📦 Files Included

```
mcq-mistral-lora-finetune/
├── train_lora_colab.ipynb            # LoRA fine-tuning + inference notebook
├── README.md                         # This file
├── requirements.txt                  # Dependencies
```

---

## 🔗 Model on Hugging Face

👉 [Lingesh-S/mcq-mistral-lora](https://huggingface.co/Lingesh-S/mcq-mistral-lora)

You can directly run inference from the Hugging Face Hub or clone this repo to reproduce the fine-tuning process.

---

## 🚀 How to Use

### 🧪 Fine-tuning

Run this notebook in Google Colab:

* [`notebooks/train_lora_colab.ipynb`](notebooks/train_lora_colab.ipynb)

It will:

* Load the Mistral-7B model in 8-bit using `bitsandbytes`
* Apply LoRA adapters using `peft`
* Train on a custom dataset of (Paragraph → MCQ) examples

### 📘 Inference

Use this notebook to generate MCQs using your fine-tuned model:

* [`notebooks/inference_colab.ipynb`](notebooks/inference_colab.ipynb)

---

## 🧾 Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```txt
transformers
peft
datasets
torch
bitsandbytes
```

---

## 📊 Training Summary

* **Model**: `mistralai/Mistral-7B-Instruct-v0.1`
* **Adapter**: LoRA (`r=8`, `lora_alpha=32`, `dropout=0.05`)
* **Training samples**: \~500 JSONL examples
* **Epochs**: 3
* **Final loss**: \~0.23

---

## ✍️ Example Prompt

```text
### Instruction:
Generate a multiple choice question with 4 options and one correct answer based on the paragraph below.

Paragraph: The Lok Sabha is the House of the People in India. It is one of the two houses of Parliament.
```

### ✅ Output

```text
What is the name of the House of the People in India?

a) The Rajya Sabha
b) The Lok Sabha
c) The Supreme Court
d) The President's House

Correct answer: b) The Lok Sabha
```

---

## 📩 Author

This project is created and maintained by **Lingesh S**.
Feel free to connect via [Hugging Face](https://huggingface.co/Lingesh-S) or [LinkedIn](https://www.linkedin.com/in/lingesh-s/).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
